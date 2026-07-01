"""
Preprocessing: parse Azure DI JSON → parents → children → conflict map.
Zero LLM calls. All parsing is deterministic.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# HTML table parsing
# ---------------------------------------------------------------------------

def parse_html_table(table_tag) -> List[Dict[str, str]]:
    """
    Walk <tr><th><td> tags, expanding rowspan/colspan into a full virtual grid.

    Key correctness requirement: we pre-allocate the grid to exactly
    len(html_rows) rows so that html_row_idx i always maps to grid[i].
    A previous approach used len(grid) as the row index, which broke when
    rowspan pre-populated future rows and grew the grid beyond the html count.
    """
    for br in table_tag.find_all("br"):
        br.replace_with("\n")

    rows = table_tag.find_all("tr")
    if not rows:
        return []

    # Pre-allocate grid to number of HTML rows — no dynamic appending needed.
    grid: List[Dict[int, str]] = [{} for _ in range(len(rows))]

    for html_row_idx, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        col_cursor = 0

        for cell in cells:
            # Skip columns already claimed by a rowspan from a previous row
            while col_cursor in grid[html_row_idx]:
                col_cursor += 1

            text = cell.get_text(separator=" ").strip()
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            for cs in range(colspan):
                target_col = col_cursor + cs
                grid[html_row_idx][target_col] = text
                # Fill future rows for rowspan (guard against overrun)
                for rs in range(1, rowspan):
                    future = html_row_idx + rs
                    if future < len(grid):
                        grid[future][target_col] = text

            col_cursor += colspan

    num_cols = max((max(r.keys()) + 1 for r in grid if r), default=0)

    first_row_tags = rows[0].find_all(["th", "td"])
    is_header = all(c.name == "th" for c in first_row_tags) if first_row_tags else False

    if is_header:
        headers = [grid[0].get(i, f"Col{i}") for i in range(num_cols)]
        data_rows = grid[1:]
    else:
        headers = [f"Col{i}" for i in range(num_cols)]
        data_rows = grid

    result = []
    for row_dict in data_rows:
        row = {header: row_dict[i] for i, header in enumerate(headers) if row_dict.get(i)}
        if row:
            result.append(row)

    return result


def flatten_row(row: Dict[str, str]) -> str:
    """Convert row dict to 'Header: Value\\nHeader: Value' text, skipping empty."""
    parts = []
    for k, v in row.items():
        v = v.strip()
        if v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Block parsing
# ---------------------------------------------------------------------------

def parse_blocks(raw_content: str) -> List[Dict]:
    """
    Split Azure DI content string into ordered typed blocks:
      {"type": "heading", "text": ...}
      {"type": "table",   "rows": [...]}
      {"type": "text",    "text": ...}
    """
    blocks = []
    soup = BeautifulSoup(raw_content, "html.parser")

    # Split on table tags — keep surrounding text
    # We process the soup children at top level to maintain order.
    # BeautifulSoup parses the whole content; we walk it linearly.

    # Re-approach: split raw_content on <table>...</table> boundaries
    # to preserve ordering of text vs table blocks.
    table_pattern = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
    segments = table_pattern.split(raw_content)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        if seg.lower().startswith("<table"):
            table_soup = BeautifulSoup(seg, "html.parser")
            table_tag = table_soup.find("table")
            if table_tag:
                rows = parse_html_table(table_tag)
                if rows:
                    blocks.append({"type": "table", "rows": rows})
        else:
            # Text segment — split into lines and classify
            text_block_lines = []
            for line in seg.splitlines():
                line = line.strip()
                if not line:
                    if text_block_lines:
                        blocks.append({"type": "text", "text": "\n".join(text_block_lines)})
                        text_block_lines = []
                    continue
                if line.startswith("#"):
                    if text_block_lines:
                        blocks.append({"type": "text", "text": "\n".join(text_block_lines)})
                        text_block_lines = []
                    blocks.append({"type": "heading", "text": line.lstrip("#").strip()})
                else:
                    text_block_lines.append(line)
            if text_block_lines:
                blocks.append({"type": "text", "text": "\n".join(text_block_lines)})

    return blocks


# ---------------------------------------------------------------------------
# Parent building — section-aware
# ---------------------------------------------------------------------------

def _group_into_sections(blocks: List[Dict]) -> List[Dict]:
    """
    First pass: collect all blocks that belong to the same heading into
    sections so that preambles, tables, and trailing notes stay together
    even when they are separated by whitespace or sub-headings in the source.

    Returns list of {"heading": str, "ordered": [block, ...]}
    """
    sections = []
    current: Dict = {"heading": "", "ordered": []}
    for block in blocks:
        if block["type"] == "heading":
            if current["ordered"]:
                sections.append(current)
            current = {"heading": block["text"], "ordered": []}
        else:
            current["ordered"].append(block)
    if current["ordered"]:
        sections.append(current)
    return sections


def _merge_consecutive_tables(parents: List[Dict]) -> List[Dict]:
    """
    Post-pass: merge consecutive table parents that share the same heading
    and doc_name. This handles Azure DI splitting a single table across
    page boundaries, producing two sibling table blocks with identical context.
    """
    if len(parents) <= 1:
        return parents
    merged = [parents[0]]
    for p in parents[1:]:
        prev = merged[-1]
        if (
            p["type"] == "table"
            and prev["type"] == "table"
            and p["heading"] == prev["heading"]
            and p["doc_name"] == prev["doc_name"]
        ):
            prev["rows"].extend(p["rows"])
            prev["notes"].extend(p["notes"])
            # preamble already set on prev — discard duplicate
        else:
            merged.append(p)
    # Re-number IDs after merge so they stay dense
    for i, p in enumerate(merged):
        p["id"] = i
    return merged


def build_parents(blocks: List[Dict], doc_name: str, id_offset: int = 0) -> List[Dict]:
    """
    Section-aware parent construction.

    For each heading section:
      - All text before the first table  → section preamble (attached to every
        table in the section, not a separate orphan narrative parent)
      - Each table                       → one parent; carries the section preamble
      - Text between or after tables     → notes[] on the immediately preceding table
      - Sections with no table at all    → single narrative parent

    This means cross-section footnotes and intro paragraphs are always present
    in the LLM context alongside the table rows they describe, even when they
    appear several lines away in the source document.
    """
    sections = _group_into_sections(blocks)
    parents: List[Dict] = []
    parent_id = id_offset

    # If the first section has no table it is a doc-level preamble (the
    # introductory paragraph at the top of the document).  We attach it to
    # every table parent in this document so the LLM always has the global
    # defaults in context — e.g. "DEUTDEFFEEQ is the default BIC" applying
    # to all FX Spot rows even though it sits above the ## FX Spot section.
    doc_preamble = ""
    if sections and not any(b["type"] == "table" for b in sections[0]["ordered"]):
        doc_preamble = "\n".join(
            b["text"] for b in sections[0]["ordered"] if b["type"] == "text"
        )

    for section in sections:
        heading = section["heading"]
        ordered = section["ordered"]

        # Collect section preamble: everything before the first table
        preamble_parts: List[str] = []
        has_table = any(b["type"] == "table" for b in ordered)

        if not has_table:
            # Pure narrative section → single parent
            text = "\n".join(b["text"] for b in ordered if b["type"] == "text")
            if text.strip():
                parents.append({
                    "id": parent_id,
                    "type": "narrative",
                    "heading": heading,
                    "doc_name": doc_name,
                    "preamble": "",
                    "rows": [text],
                    "notes": [],
                })
                parent_id += 1
            continue

        # Gather preamble (text before first table in this section).
        # Prepend the doc-level preamble so global defaults (e.g. default BIC)
        # are always visible to the LLM alongside any table in this document.
        for block in ordered:
            if block["type"] == "table":
                break
            if block["type"] == "text":
                preamble_parts.append(block["text"])
        section_preamble = "\n".join(
            p for p in [doc_preamble, "\n".join(preamble_parts)] if p
        )

        # Walk ordered blocks; text after a table → notes of that table
        past_first_table = False
        for block in ordered:
            if block["type"] == "text":
                if past_first_table and parents and parents[-1]["type"] == "table":
                    parents[-1]["notes"].append(block["text"])
                # pre-table text already captured in section_preamble; skip here
            elif block["type"] == "table":
                past_first_table = True
                row_texts = [flatten_row(r) for r in block["rows"]]
                parents.append({
                    "id": parent_id,
                    "type": "table",
                    "heading": heading,
                    "doc_name": doc_name,
                    "preamble": section_preamble,
                    "rows": row_texts,
                    "notes": [],
                })
                parent_id += 1

    # Merge sibling tables that were split across pages in the source PDF
    parents = _merge_consecutive_tables(parents)

    # Re-apply id_offset after merge re-numbered from 0
    for p in parents:
        p["id"] += id_offset

    return parents


# ---------------------------------------------------------------------------
# Child building
# ---------------------------------------------------------------------------

def _contextual_prefix(parent: Dict) -> str:
    """
    Short context string prepended to every child's embed_text.

    Contextual retrieval (Anthropic, 2024): embedding a row alone loses
    document-level context — "Currency: CHF / SWIFT: CRESCHZZXXX" in
    fx_spot_ssi.json embeds nearly identically to the same row in
    correspondent_banking_ssi.json.  Including source and section lets the
    embedding model distinguish them as different settlement paths.
    """
    doc_label = parent["doc_name"].replace(".json", "").replace("_", " ")
    heading = parent["heading"]
    if heading:
        return f"[{doc_label} | {heading}] "
    return f"[{doc_label}] "


def build_children(parents: List[Dict]) -> List[Dict]:
    """
    One child per table row (or one child per narrative parent).

    embed_text = contextual_prefix + row_text
    The prefix carries document name and section so the embedding model can
    distinguish the same currency row appearing in different source files or
    sections (e.g. FX Spot vs Correspondent Banking for the same currency).
    """
    children = []
    for parent in parents:
        prefix = _contextual_prefix(parent)
        if parent["type"] == "narrative":
            row_text = parent["rows"][0] if parent["rows"] else ""
            children.append({
                "parent_id": parent["id"],
                "row_text": row_text,
                "doc_name": parent["doc_name"],
                "embed_text": prefix + row_text,
            })
        else:
            for row_text in parent["rows"]:
                children.append({
                    "parent_id": parent["id"],
                    "row_text": row_text,
                    "doc_name": parent["doc_name"],
                    "embed_text": prefix + row_text,
                })
    return children


# ---------------------------------------------------------------------------
# Conflict map
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b")


def _extract_codes(text: str) -> List[str]:
    # Do NOT uppercase — only match already-uppercase sequences so that
    # mixed-case field names like "Required", "Clearing" are excluded.
    return _CODE_PATTERN.findall(text)


def _extract_scope_value(row_text: str) -> Optional[str]:
    """
    Extract the first field value from a flattened row as the scope key.
    The first field (Currency, Country, etc.) is the natural scope.
    Returns the raw value string, uppercased.
    """
    first_line = row_text.split("\n")[0]
    if ":" in first_line:
        val = first_line.split(":", 1)[1].strip()
        # Take first word-token only (e.g. "UNITED STATES" → keep as-is)
        return val.upper() if val else None
    return None


def build_conflict_map(parents: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    Scan all parents across all docs at index time.
    For each (scope_value, doc_name) pair, collect all SWIFT/BIC codes found.
    If the same scope_value appears in 2+ docs with DIFFERENT codes → conflict.

    Returns:
        {"GBP": {"fx_spot_ssi.json": ["LOYDGB2L"], "correspondent_banking_ssi.json": ["BARCGB22"]}}
    """
    # scope_value → doc_name → set of codes
    scope_to_doc_codes: Dict[str, Dict[str, set]] = {}

    for parent in parents:
        doc = parent["doc_name"]
        for row_text in parent["rows"]:
            scope = _extract_scope_value(row_text)
            if not scope:
                continue
            # Exclude the scope value itself from extracted codes (country/currency
            # names like NETHERLANDS are 11 chars and would match the BIC pattern).
            codes = set(_extract_codes(row_text)) - {scope}
            if not codes:
                continue
            scope_to_doc_codes.setdefault(scope, {})
            scope_to_doc_codes[scope].setdefault(doc, set()).update(codes)

    conflict_map: Dict[str, Dict[str, List[str]]] = {}
    for scope, doc_codes in scope_to_doc_codes.items():
        if len(doc_codes) < 2:
            continue
        all_codes = [frozenset(v) for v in doc_codes.values()]
        if len(set(all_codes)) > 1:
            conflict_map[scope] = {doc: sorted(codes) for doc, codes in doc_codes.items()}

    return conflict_map


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess_all(json_dir: str) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Load all *.json files from directory.
    Returns (all_parents, all_children, conflict_map).
    """
    json_dir = Path(json_dir)
    all_parents: List[Dict] = []
    all_children: List[Dict] = []
    id_offset = 0

    for path in sorted(json_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        doc_name = doc.get("doc_name", path.name)
        content = doc.get("content", "")

        blocks = parse_blocks(content)
        parents = build_parents(blocks, doc_name, id_offset=id_offset)
        children = build_children(parents)

        all_parents.extend(parents)
        all_children.extend(children)
        id_offset += len(parents)

    conflict_map = build_conflict_map(all_parents)
    return all_parents, all_children, conflict_map


def parent_full_text(parent: Dict) -> str:
    """
    Render a parent to readable multi-line text for the LLM context block.
    Preamble (section intro text) appears before the table rows so the LLM
    sees it as contextual background, not a footnote.
    """
    lines = [f"[Section: {parent['heading']}]"] if parent["heading"] else []
    if parent.get("preamble"):
        lines.append(parent["preamble"])
    lines.extend(parent["rows"])
    if parent.get("notes"):
        lines.extend(parent["notes"])
    return "\n".join(lines)

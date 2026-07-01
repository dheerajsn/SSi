"""
PDF → Azure DI JSON converter for SSI documents.

Usage:
    python -m src.pdf_converter <pdf_path> [--out <json_path>] [--doc-name <name>]

Extracts tables and surrounding text from any SSI PDF using pdfplumber,
normalises the rows, and emits the same JSON schema the pipeline consumes:
    { doc_name, model_id, api_version, content }
where content is plain text with tables rendered as HTML.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("pip install pdfplumber")


# ---------------------------------------------------------------------------
# Cell cleaning
# ---------------------------------------------------------------------------

def _clean(cell) -> str:
    if cell is None:
        return ""
    text = str(cell)
    # Collapse whitespace / newlines inside a cell
    text = re.sub(r"\s+", " ", text).strip()
    # Remove spaced-out characters artifact: "S W I F T" → "SWIFT"
    text = re.sub(r"\b([A-Z]) ([A-Z]) ([A-Z]) ([A-Z])\b", r"\1\2\3\4", text)
    text = re.sub(r"\b([A-Z]) ([A-Z]) ([A-Z])\b", r"\1\2\3", text)
    return text


def _row_has_content(row: List) -> bool:
    return any(_clean(c) for c in row)


def _is_header_row(row: List) -> bool:
    """Heuristic: a header row has mostly short uppercase/title-case tokens."""
    cells = [_clean(c) for c in row if _clean(c)]
    if not cells:
        return False
    upper = sum(1 for c in cells if c.isupper() or c.istitle())
    return upper >= len(cells) * 0.6


# ---------------------------------------------------------------------------
# HTML table builder
# ---------------------------------------------------------------------------

def _table_to_html(rows: List[List]) -> str:
    """Convert pdfplumber rows to an HTML table string."""
    if not rows:
        return ""

    # Determine header row
    has_header = _is_header_row(rows[0])
    html_rows = []

    for i, row in enumerate(rows):
        if not _row_has_content(row):
            continue
        if i == 0 and has_header:
            cells = "".join(f"<th>{_clean(c)}</th>" for c in row if _clean(c) or True)
            # Use all columns even empty to preserve alignment
            cells = "".join(f"<th>{_clean(c)}</th>" for c in row)
            html_rows.append(f"<tr>{cells}</tr>")
        else:
            cells = "".join(f"<td>{_clean(c)}</td>" for c in row)
            html_rows.append(f"<tr>{cells}</tr>")

    return "<table>" + "".join(html_rows) + "</table>"


# ---------------------------------------------------------------------------
# Page content extractor
# ---------------------------------------------------------------------------

def _extract_page_content(page) -> str:
    """
    Extract a page's content as a mix of plain text (headings/narrative)
    and HTML tables.  Tables are replaced in-position relative to the text
    blocks that surround them, using bounding boxes to determine order.
    """
    words = page.extract_words(keep_blank_chars=False, use_text_flow=True)
    tables = page.find_tables()

    # Sort tables by top y position
    table_bboxes = [(t.bbox, t) for t in tables]

    # Group words into lines, skipping words that fall inside a table bbox
    def inside_any_table(w):
        for bbox, _ in table_bboxes:
            x0, y0, x1, y1 = bbox
            if x0 <= w["x0"] and w["x1"] <= x1 and y0 <= w["top"] and w["bottom"] <= y1:
                return True
        return False

    # Build text lines from words not inside tables
    line_map = {}  # round(top) → list of words
    for w in words:
        if inside_any_table(w):
            continue
        key = round(w["top"] / 4) * 4  # bucket to 4-pt grid
        line_map.setdefault(key, []).append(w)

    # Sort lines by vertical position
    text_lines = []
    for key in sorted(line_map):
        line_words = sorted(line_map[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        if text:
            text_lines.append((key, "text", text))

    # Insert tables at their vertical position
    table_entries = []
    for bbox, t in table_bboxes:
        rows = t.extract()
        if rows:
            html = _table_to_html(rows)
            if html:
                table_entries.append((bbox[1], "table", html))  # y0 as sort key

    # Merge and sort all content by vertical position
    all_content = text_lines + table_entries
    all_content.sort(key=lambda x: x[0])

    parts = []
    for _, kind, content in all_content:
        if kind == "table":
            parts.append(content)
        else:
            parts.append(content)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(r"^(#+)\s+(.+)$"),                     # markdown-style
    re.compile(r"^([A-Z][A-Z\s&/\-]+[A-Z])\s*$"),     # ALL CAPS line
    re.compile(r"^(\d+\.\s+.+)$"),                     # numbered
]


def _classify_line(line: str) -> str:
    """Return 'heading' or 'text'."""
    line = line.strip()
    if not line:
        return "blank"
    for pat in _HEADING_PATTERNS:
        if pat.match(line):
            return "heading"
    return "text"


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def pdf_to_azure_di_json(
    pdf_path: str,
    doc_name: Optional[str] = None,
    model_id: str = "prebuilt-document",
    api_version: str = "2023-07-31",
) -> dict:
    """
    Convert a PDF file to the Azure DI JSON schema used by the SSI pipeline.
    """
    pdf_path = Path(pdf_path)
    if doc_name is None:
        doc_name = pdf_path.name

    content_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_content = _extract_page_content(page)
            if page_content.strip():
                content_parts.append(page_content)

    content = "\n\n".join(content_parts)

    return {
        "doc_name": doc_name,
        "model_id": model_id,
        "api_version": api_version,
        "content": content,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert SSI PDF to Azure DI JSON")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--out", help="Output JSON path (default: same name .json)")
    parser.add_argument("--doc-name", help="Override doc_name field")
    args = parser.parse_args()

    result = pdf_to_azure_di_json(args.pdf, doc_name=args.doc_name)

    out_path = args.out or str(Path(args.pdf).with_suffix(".json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Written: {out_path}")
    print(f"Content length: {len(result['content'])} chars")


if __name__ == "__main__":
    main()

"""
Chunking strategies for SSI documents.

Each strategy implements ChunkStrategy.build(blocks, doc_name, id_offset)
returning (parents, children) using the same dict schema as preprocessing.py.

Add new strategies to STRATEGIES and they become available as pipeline parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from src.preprocessing import build_parents, build_children, flatten_row


class ChunkStrategy(ABC):
    name: str

    @abstractmethod
    def build(
        self,
        blocks: List[Dict],
        doc_name: str,
        id_offset: int = 0,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Return (parents, children)."""


class SectionTableStrategy(ChunkStrategy):
    """
    Default strategy — section-aware parent-child hierarchy.
    - Parents: one per table (carries preamble + notes)
    - Children: one per table row (what gets embedded)

    Best for structured SSI tables where rows represent individual instruments.
    """
    name = "section_table"

    def build(self, blocks, doc_name, id_offset=0):
        parents = build_parents(blocks, doc_name, id_offset)
        children = build_children(parents)
        return parents, children


class SlidingWindowStrategy(ChunkStrategy):
    """
    Fixed-size word-window chunks with overlap.
    No parent-child hierarchy — every window is its own retrieval unit.

    Better for long-form narrative docs (policy memos, prospectuses)
    where structure is prose rather than tables.
    """
    name = "sliding_window"

    def __init__(self, window_words: int = 150, overlap_words: int = 30):
        self.window = window_words
        self.overlap = overlap_words

    def build(self, blocks, doc_name, id_offset=0):
        parts = []
        for b in blocks:
            if b["type"] == "table":
                parts.append("\n".join(
                    " | ".join(f"{k}: {v}" for k, v in row.items())
                    for row in b.get("rows", [])
                ))
            elif b.get("text"):
                parts.append(b["text"])
        all_text = "\n".join(p for p in parts if p)

        words = all_text.split()
        step = max(1, self.window - self.overlap)
        parents: List[Dict] = []
        chunk_id = id_offset

        for start in range(0, max(1, len(words)), step):
            chunk_words = words[start: start + self.window]
            if not chunk_words:
                break
            parents.append({
                "id": chunk_id,
                "type": "narrative",
                "heading": "",
                "doc_name": doc_name,
                "preamble": "",
                "rows": [" ".join(chunk_words)],
                "notes": [],
            })
            chunk_id += 1

        children = build_children(parents)
        return parents, children


class ParagraphStrategy(ChunkStrategy):
    """
    One chunk per paragraph (blank-line-separated prose) or per table.
    Preserves natural paragraph boundaries without a fixed word budget.
    Good for mixed narrative + table docs where tables are small.
    """
    name = "paragraph"

    def build(self, blocks, doc_name, id_offset=0):
        parents: List[Dict] = []
        chunk_id = id_offset

        for block in blocks:
            if block["type"] == "table":
                rows = [flatten_row(r) for r in block.get("rows", [])]
                if rows:
                    parents.append({
                        "id": chunk_id,
                        "type": "table",
                        "heading": "",
                        "doc_name": doc_name,
                        "preamble": "",
                        "rows": rows,
                        "notes": [],
                    })
                    chunk_id += 1
            elif block.get("text", "").strip():
                for para in block["text"].split("\n\n"):
                    para = para.strip()
                    if para:
                        parents.append({
                            "id": chunk_id,
                            "type": "narrative",
                            "heading": "",
                            "doc_name": doc_name,
                            "preamble": "",
                            "rows": [para],
                            "notes": [],
                        })
                        chunk_id += 1

        children = build_children(parents)
        return parents, children


STRATEGIES: Dict[str, ChunkStrategy] = {
    SectionTableStrategy.name: SectionTableStrategy(),
    SlidingWindowStrategy.name: SlidingWindowStrategy(),
    ParagraphStrategy.name: ParagraphStrategy(),
}


def get_strategy(name: str) -> ChunkStrategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy '{name}'. Available: {list(STRATEGIES)}")
    return STRATEGIES[name]

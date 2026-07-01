"""
Jinja2-based prompt building for SSI RAG.

Separating template logic from pipeline code means you can iterate on
prompt wording without touching Python, and swap templates per domain.

Templates
---------
  rag_context  : assembles the LLM user message from retrieved context + question
  citation     : formats the source list appended to answers

Usage
-----
    from src.prompt import render_rag_prompt, render_citation

    user_msg = render_rag_prompt(question, matched_parents, matched_children)
    sources  = render_citation(matched_parents)
"""

from typing import Dict, List, Optional

try:
    from jinja2 import Environment, BaseLoader, StrictUndefined

    _JINJA_AVAILABLE = True
except ImportError:
    _JINJA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

_RAG_CONTEXT_TMPL = """\
{% for block in context_blocks %}
[Source: {{ block.doc_name }} | Section: {{ block.heading }}]
{% if block.preamble %}{{ block.preamble }}
{% endif %}
{% for row in block.rows %}{{ row }}
{% endfor %}
{% for note in block.notes %}{{ note }}
{% endfor %}
{% endfor %}
Question: {{ question }}"""

_CITATION_TMPL = """\
Sources:
{% for entry in sources %}
  - {{ entry }}
{% endfor %}"""


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _build_context_blocks(
    matched_parents: List[Dict],
    matched_children: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Merge retrieval results into template-ready block dicts.

    When matched_children is provided, only the retrieved rows are included
    per parent (focused mode). Otherwise the full parent row list is used.
    """
    relevant_by_parent: Dict[int, set] = {}
    if matched_children:
        for child in matched_children:
            relevant_by_parent.setdefault(child["parent_id"], set()).add(child["row_text"])

    blocks = []
    for parent in matched_parents:
        if relevant_by_parent and parent["id"] in relevant_by_parent:
            rows = [r for r in parent["rows"] if r in relevant_by_parent[parent["id"]]]
        else:
            rows = list(parent["rows"])

        blocks.append({
            "doc_name": parent["doc_name"],
            "heading": parent.get("heading", ""),
            "preamble": parent.get("preamble", ""),
            "rows": rows,
            "notes": parent.get("notes", []),
        })
    return blocks


def render_rag_prompt(
    question: str,
    matched_parents: List[Dict],
    matched_children: Optional[List[Dict]] = None,
) -> str:
    """
    Render the user message sent to the LLM.

    Uses Jinja2 when available; falls back to plain-string assembly so the
    pipeline works without jinja2 installed (optional dependency).
    """
    blocks = _build_context_blocks(matched_parents, matched_children)

    if _JINJA_AVAILABLE:
        env = Environment(loader=BaseLoader(), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string(_RAG_CONTEXT_TMPL)
        return tmpl.render(context_blocks=blocks, question=question).strip()

    # Fallback: plain string assembly (identical output, no jinja2 needed)
    parts = []
    for b in blocks:
        header = f"[Source: {b['doc_name']} | Section: {b['heading']}]"
        lines = [header]
        if b["preamble"]:
            lines.append(b["preamble"])
        lines.extend(b["rows"])
        lines.extend(b["notes"])
        parts.append("\n".join(lines))

    context_block = "\n\n".join(parts)
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def render_citation(matched_parents: List[Dict]) -> str:
    """
    Render a deduplicated source list.

    Uses Jinja2 when available; falls back to plain string.
    """
    seen = []
    for p in matched_parents:
        entry = f"{p['doc_name']} / {p['heading']}"
        if entry not in seen:
            seen.append(entry)

    if _JINJA_AVAILABLE:
        env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
        tmpl = env.from_string(_CITATION_TMPL)
        return tmpl.render(sources=seen).strip()

    return "Sources:\n" + "\n".join(f"  - {s}" for s in seen)


# ---------------------------------------------------------------------------
# Custom template support
# ---------------------------------------------------------------------------

def render_custom(template_str: str, **context) -> str:
    """
    Render an arbitrary Jinja2 template string with the given context.

    Useful for domain-specific prompt variations without touching pipeline code.

    Example
    -------
    tmpl = "For {{ market }}, extract: {{ fields | join(', ') }}"
    render_custom(tmpl, market="Germany-CLEARGER", fields=["Institution BIC", "Global Agent"])
    """
    if not _JINJA_AVAILABLE:
        raise ImportError(
            "jinja2 is required for render_custom(). pip install jinja2"
        )
    env = Environment(loader=BaseLoader(), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template_str).render(**context)

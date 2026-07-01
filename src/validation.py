"""
Validation layer: confidence gating, conflict detection, grounding check.

Design: NO hardcoded keyword lists or entity universes.
- Confidence gating is purely score-based (FAISS similarity).
- Conflict detection uses the data-derived conflict_map built at index time.
  Query-time lookup is: does any conflict_map key appear in the query text?
  The keys themselves come from the data, not from a hardcoded list.
- Grounding check uses a SWIFT/IBAN regex only to extract codes that the
  LLM has already placed in its answer — it doesn't gate on query parsing.
"""

import re
from typing import Dict, List


# ---------------------------------------------------------------------------
# Confidence gate — purely score-based, no keyword analysis
# ---------------------------------------------------------------------------

def confidence_gate(retrieval_result: Dict, threshold: float = 0.15) -> Dict:
    """
    Block the query if the best retrieval score is below threshold.
    This naturally handles queries about currencies/countries not in corpus
    (e.g. BRL) without needing a hardcoded currency list.
    """
    best = retrieval_result.get("best_score", 0.0)
    if best < threshold:
        return {
            "blocked": True,
            "reason": (
                f"No sufficiently relevant settlement instructions found "
                f"(best retrieval score {best:.3f} < threshold {threshold:.3f}). "
                "The queried instrument or currency may not be in the SSI corpus."
            ),
        }
    return {"blocked": False, "reason": None}


# ---------------------------------------------------------------------------
# Conflict detection — data-driven, not keyword-hardcoded
# ---------------------------------------------------------------------------

def detect_conflicts(query: str, conflict_map: Dict[str, Dict[str, List[str]]]) -> List[Dict]:
    """
    The conflict_map keys are scope values discovered from the data at index time
    (e.g. "GBP", "JPY", "UNITED STATES"). We check whether each known conflict
    key appears in the query — no hardcoded currency/country universe needed.
    """
    query_upper = query.upper()
    conflicts = []
    for scope_value, doc_codes in conflict_map.items():
        if scope_value in query_upper:
            docs_summary = "; ".join(
                f"{doc}: {', '.join(codes)}" for doc, codes in doc_codes.items()
            )
            conflicts.append({
                "scope_value": scope_value,
                "conflict": doc_codes,
                "message": (
                    f"Conflicting SSI found for '{scope_value}' across documents: {docs_summary}. "
                    "Verify with the responsible desk before settling."
                ),
            })
    return conflicts


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

_SWIFT_PATTERN = re.compile(r"\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?)\b")
_IBAN_PATTERN = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b")


def check_grounding(answer: str, context: str) -> Dict:
    """
    Extract SWIFT/BIC codes and IBANs from the LLM answer.
    Verify each appears verbatim in the retrieved context.
    The regex here operates on the LLM's output, not on the query.
    """
    answer_upper = answer.upper()
    context_upper = context.upper()

    swift_codes = set(_SWIFT_PATTERN.findall(answer_upper))
    iban_codes = set(_IBAN_PATTERN.findall(answer_upper))
    all_codes = swift_codes | iban_codes

    grounded = {}
    ungrounded = {}
    for code in all_codes:
        if code in context_upper:
            grounded[code] = True
        else:
            ungrounded[code] = False

    return {
        "grounded": grounded,
        "ungrounded": ungrounded,
        "passed": len(ungrounded) == 0,
    }

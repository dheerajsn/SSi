"""
Real SSI test suite — runs against actual public documents:
  - US Bank FX Standing Settlement Instructions
  - 26 Degrees Global Markets SSI
  - Cornèr Banca SA Switzerland SSI

Run: python tests/test_real_ssi.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import SSIPipeline

DATA_DIR = str(Path(__file__).parent.parent / "data" / "real_ssi")

# ---------------------------------------------------------------------------
# Test cases — all drawn from real document content
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "RT01",
        "query": "What are the US Bank settlement instructions for JPY?",
        "expect_gate": "pass",
        "expect_conflicts": False,
        "must_contain_doc": "usbank_fx_ssi.json",
        "must_contain_rows": ["BOTKJPJT"],
        "answer_must_contain": None,
        "description": "US Bank JPY — MUFG Bank Tokyo BOTKJPJT",
    },
    {
        "id": "RT02",
        "query": "What is the SWIFT code and account number for CHF settlement at Cornèr Banca?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "corner_banca_ssi.json",
        "must_contain_rows": ["CBLUCH22"],
        "answer_must_contain": None,
        "description": "Cornèr Banca CHF — direct SIC settlement via CBLUCH22; conflict: US Bank CHF uses different BIC",
    },
    {
        "id": "RT03",
        "query": "What are the settlement instructions for GBP at 26 Degrees Global Markets?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "26degrees_ssi.json",
        "must_contain_rows": ["CHASGB2L"],
        "answer_must_contain": None,
        "description": "26 Degrees GBP — JPMorgan London CHASGB2L; cross-doc conflict with US Bank GBP (NWBKGB2L)",
    },
    {
        "id": "RT04",
        "query": "What are the correspondent bank details for BHD (Bahraini Dinar) at US Bank?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "usbank_fx_ssi.json",
        "must_contain_rows": ["MIDLGB22"],
        "answer_must_contain": None,
        "description": "US Bank BHD — HSBC London intermediary; cross-doc conflict: Cornèr Banca BHD uses different correspondent",
    },
    {
        "id": "RT05",
        "query": "Show USD settlement instructions from US Bank, 26 Degrees, and Cornèr Banca",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": None,
        "must_contain_rows": ["CHASUS33", "SCBLUS33", "IRVTUS3N"],
        "answer_must_contain": None,
        "description": "Cross-doc USD: JPMorgan (26 Degrees) vs SCB (US Bank) vs BNY Mellon (Cornèr Banca)",
    },
    {
        "id": "RT06",
        "query": "What are the settlement instructions for precious metals XAU and XAG at Cornèr Banca?",
        "expect_gate": "pass",
        "expect_conflicts": False,
        "must_contain_doc": "corner_banca_ssi.json",
        "must_contain_rows": ["XAU", "RAIF"],
        "answer_must_contain": None,
        "description": "Cornèr Banca precious metals — Raiffeisen routing; 'RAIF' matches 'RAIF CH22' (PDF artifact space)",
    },
    {
        "id": "RT07",
        "query": "What are the US Bank FX settlement instructions for JPY, CHF, and AUD?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "usbank_fx_ssi.json",
        "must_contain_rows": ["BOTKJPJT"],
        "answer_must_contain": None,
        "description": "US Bank multi-currency: JPY (BOTKJPJT), CHF (UBSWCHZH80A), AUD (NATAAU3302R); CHF/AUD in conflict map",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(pipeline: SSIPipeline, tc: dict) -> bool:
    print(f"\n{'='*70}")
    print(f"[{tc['id']}] {tc['description']}")
    print(f"  Query: {tc['query']}")

    result = pipeline.query(tc["query"])

    gate_status = "block" if result["blocked"] else "pass"
    conflicts = result.get("conflicts", [])
    retrieval = result.get("retrieval", {})
    answer = result.get("answer", "")
    grounding = result.get("grounding", {})

    best_score = retrieval.get("best_score", 0.0)
    matched_docs = list({c["doc_name"] for c in retrieval.get("matched_children", [])})
    print(f"  Best score: {best_score:.3f} | Matched docs: {matched_docs}")

    if conflicts:
        for c in conflicts:
            print(f"  CONFLICT: {c['message'][:120]}")

    print(f"  Answer: {answer[:300]}...")
    print(f"  Grounding: passed={grounding.get('passed')} "
          f"ungrounded={list(grounding.get('ungrounded', {}).keys())}")
    print(f"  Latency: {result.get('latency_s', '?')}s")

    failures = []

    if gate_status != tc["expect_gate"]:
        failures.append(f"GATE: expected '{tc['expect_gate']}' got '{gate_status}'")

    if not result["blocked"]:
        has_conflicts = len(conflicts) > 0
        if has_conflicts != tc["expect_conflicts"]:
            failures.append(f"CONFLICT: expected {tc['expect_conflicts']} got {has_conflicts}")

        if tc.get("must_contain_doc"):
            all_docs = {c["doc_name"] for c in retrieval.get("matched_children", [])}
            if tc["must_contain_doc"] not in all_docs:
                failures.append(f"DOC: '{tc['must_contain_doc']}' not in {all_docs}")

        full_context = "\n".join(
            p.get("preamble", "") + "\n".join(p.get("rows", [])) + "\n".join(p.get("notes", []))
            for p in retrieval.get("matched_parents", [])
        )
        for must_row in tc.get("must_contain_rows", []):
            if must_row.upper() not in full_context.upper():
                failures.append(f"CONTEXT: '{must_row}' not found in retrieved context")

        if tc.get("answer_must_contain"):
            if tc["answer_must_contain"].lower() not in answer.lower():
                failures.append(f"ANSWER: '{tc['answer_must_contain']}' not in answer")

    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
        return False
    print("  [PASS]")
    return True


def main():
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not groq_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    pipeline = SSIPipeline(DATA_DIR, groq_key, groq_model=groq_model)
    pipeline.build()

    passed = 0
    for i, tc in enumerate(TEST_CASES):
        ok = run_test(pipeline, tc)
        if ok:
            passed += 1
        if i < len(TEST_CASES) - 1:
            print(f"  [pacing] sleeping 15s...")
            time.sleep(15)

    print(f"\n{'='*70}")
    print(f"REAL SSI RESULT: {passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    main()

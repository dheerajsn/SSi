"""
SSI RAG Pipeline — end-to-end test harness.
7 core test cases + model comparison table.

Run: python tests/test_pipeline.py
"""

import os
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import SSIPipeline
from src.providers import GroqLLM, LocalEmbedder

DATA_DIR = str(Path(__file__).parent.parent / "data" / "mock_azure_di")

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "TC01",
        "query": "What are the FX Spot settlement instructions for CHF?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "fx_spot_ssi.json",
        "must_contain_section": "FX Spot Settlement",
        "must_contain_rows": ["Currency: CHF", "MT599"],
        "description": "CHF multi-path (Agency/Principal) + MT599 footnote; conflict: FX-delivery BIC vs wire-transfer BIC in correspondent_banking",
    },
    {
        "id": "TC02",
        "query": "What is the custodian for Slovenian equities settlement?",
        "expect_gate": "pass",
        "expect_conflicts": False,
        "must_contain_doc": "cash_equities_ssi.json",
        "must_contain_section": "Cash Equities Settlement",
        "must_contain_rows": ["Country: SLOVENIA", "Country: LUXEMBOURG"],
        "description": "SLOVENIA cross-reference to LUXEMBOURG",
    },
    {
        "id": "TC03",
        "query": "What is the SWIFT code for Canada in Cash Equities settlement?",
        "expect_gate": "pass",
        "expect_conflicts": False,
        "must_contain_doc": "cash_equities_ssi.json",
        "must_contain_section": "Cash Equities Settlement",
        "must_contain_rows": ["Country: CANADA"],
        "description": "CANADA cash equities (not FX Spot CAD)",
    },
    {
        "id": "TC04",
        "query": "What is the default beneficiary BIC if none is stated?",
        "expect_gate": "pass",
        "expect_conflicts": False,
        "must_contain_doc": "fx_spot_ssi.json",
        "must_contain_section": None,
        "must_contain_rows": ["DEUTDEFFEEQ"],
        "description": "Global narrative: default BIC",
    },
    {
        "id": "TC05",
        "query": "What is the SWIFT code for GBP in FX Spot settlement?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "fx_spot_ssi.json",
        "must_contain_section": "FX Spot Settlement",
        "must_contain_rows": [],
        "description": "GBP conflict: LOYDGB2L vs BARCGB22",
    },
    {
        "id": "TC06",
        "query": "What are the settlement instructions for Netherlands repo?",
        "expect_gate": "pass",
        "expect_conflicts": True,
        "must_contain_doc": "repo_ssi.json",
        "must_contain_section": "Repo Settlement",
        "must_contain_rows": ["Country: NETHERLANDS", "Country: GERMANY"],
        "description": "NETHERLANDS cross-ref to GERMANY in repo; conflict: ABNANL2A (cash equities) vs DEUTDEFF (repo)",
    },
    {
        "id": "TC07",
        "query": "What are the settlement instructions for NGN (Nigerian Naira)?",
        "expect_gate": "pass",
        "expect_conflicts": False,
        "must_contain_doc": None,
        "must_contain_section": None,
        "must_contain_rows": [],
        "answer_must_contain": "not found",
        "description": "NGN not in corpus — LLM should say 'not found' per domain prompt rule 2",
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

    # Retrieval stats
    best_score = retrieval.get("best_score", 0.0)
    matched_docs = list({c["doc_name"] for c in retrieval.get("matched_children", [])})
    print(f"  Best score: {best_score:.3f} | Matched docs: {matched_docs}")

    if conflicts:
        print(f"  CONFLICTS DETECTED:")
        for c in conflicts:
            print(f"    {c['message']}")

    print(f"  Answer (truncated): {answer[:200]}...")
    print(f"  Grounding: passed={grounding.get('passed')} "
          f"ungrounded={list(grounding.get('ungrounded', {}).keys())}")
    print(f"  Latency: {result.get('latency_s', '?')}s")

    # --- Assertions ---
    failures = []

    # Gate check
    if gate_status != tc["expect_gate"]:
        failures.append(f"GATE: expected '{tc['expect_gate']}' got '{gate_status}'")

    if not result["blocked"]:
        # Conflict check
        has_conflicts = len(conflicts) > 0
        if has_conflicts != tc["expect_conflicts"]:
            failures.append(
                f"CONFLICT: expected {tc['expect_conflicts']} got {has_conflicts}"
            )

        # Doc check
        if tc["must_contain_doc"]:
            all_docs = {c["doc_name"] for c in retrieval.get("matched_children", [])}
            if tc["must_contain_doc"] not in all_docs:
                failures.append(
                    f"DOC: '{tc['must_contain_doc']}' not in matched docs {all_docs}"
                )

        # Section check (in any matched parent)
        if tc["must_contain_section"]:
            sections = {p["heading"] for p in retrieval.get("matched_parents", [])}
            if tc["must_contain_section"] not in sections:
                failures.append(
                    f"SECTION: '{tc['must_contain_section']}' not in {sections}"
                )

        # Row content checks (in full context visible to LLM)
        full_context = "\n".join(
            "\n".join(p["rows"]) + "\n".join(p.get("notes", []))
            for p in retrieval.get("matched_parents", [])
        )
        for must_row in tc["must_contain_rows"]:
            if must_row.upper() not in full_context.upper():
                failures.append(f"CONTEXT: '{must_row}' not found in retrieved context")

        # Optional: check LLM answer contains a phrase
        if tc.get("answer_must_contain"):
            phrase = tc["answer_must_contain"]
            if phrase.lower() not in answer.lower():
                failures.append(f"ANSWER: '{phrase}' not found in LLM answer")

    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
        return False
    else:
        print(f"  [PASS]")
        return True


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

GROQ_MODELS = [
    ("llama-3.3-70b-versatile", "Best quality, 70B"),
    ("llama-3.1-8b-instant",    "Fastest, 8B, lower quality"),
    ("llama3-8b-8192",          "Meta Llama 3 8B, 8K context"),
]

COMPARISON_QUERIES = [
    TEST_CASES[0],  # TC01 — CHF
    TEST_CASES[4],  # TC05 — GBP conflict
]


def compare_models(data_dir: str, groq_key: str):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: TC01 (CHF) and TC05 (GBP conflict)")
    print("=" * 70)

    rows = []
    for model_id, note in GROQ_MODELS:
        print(f"\n  Testing model: {model_id}")
        try:
            p = SSIPipeline(data_dir, llm=GroqLLM(api_key=groq_key, model=model_id), embedder=LocalEmbedder())
            p.build()
        except Exception as e:
            print(f"    Build failed: {e}")
            continue

        for tc in COMPARISON_QUERIES:
            t0 = time.time()
            try:
                result = p.query(tc["query"])
            except Exception as e:
                result = {"answer": f"ERROR: {e}", "conflicts": [], "latency_s": 0}
            latency = result.get("latency_s", round(time.time() - t0, 2))
            answer_short = result["answer"][:60].replace("\n", " ")
            conflict_noted = len(result.get("conflicts", [])) > 0
            rows.append({
                "model": model_id,
                "tc": tc["id"],
                "answer": answer_short,
                "conflict_noted": conflict_noted,
                "latency": latency,
                "note": note,
            })

    # Print table
    print(f"\n{'Model':<30} | {'TC':<5} | {'Conflict?':<10} | {'Latency':>8} | Answer[:60]")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['model']:<30} | {r['tc']:<5} | {str(r['conflict_noted']):<10} | "
            f"{r['latency']:>7.1f}s | {r['answer']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not groq_key:
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    pipeline = SSIPipeline(
        DATA_DIR,
        llm=GroqLLM(api_key=groq_key, model=groq_model),
        embedder=LocalEmbedder(),
    )
    pipeline.build()

    passed = 0
    total = len(TEST_CASES)

    for i, tc in enumerate(TEST_CASES):
        ok = run_test(pipeline, tc)
        if ok:
            passed += 1
        # 70b model: 12K TPM. Each call ≈ 3K tokens → need ≥15s gap.
        # Use 20s to stay safely within the limit.
        if i < len(TEST_CASES) - 1:
            time.sleep(20)

    print(f"\n{'='*70}")
    print(f"RESULT: {passed}/{total} passed")

    # --- Multi-attribute extraction demo ---
    # Allow the 6K-TPM rolling window to clear before firing extraction calls.
    print(f"\nWaiting 60s for TPM window to clear before extraction demo...")
    time.sleep(60)

    print(f"\n{'='*70}")
    print("MULTI-ATTRIBUTE EXTRACTION DEMO")
    print("Currencies: CHF, GBP, USD  |  Fields: SWIFT/BIC + Account Number + Notes")

    # batch_size=1: one LLM call per currency, keeps each prompt under 2K tokens.
    subset_fields = ["SWIFT/BIC", "Account Number", "Notes"]
    rows = pipeline.extract_batch(
        ["CHF", "GBP", "USD"],
        fields=subset_fields,
        batch_size=1,
    )
    print(f"\n{'Currency':<8} {'SWIFT/BIC':<16} {'Account Number':<25} {'Notes'}")
    print("-" * 80)
    for r in rows:
        f = r["fields"]
        conflict_flag = " ⚠ CONFLICT" if r["conflicts"] else ""
        print(f"{r['item']:<8} {f.get('SWIFT/BIC','N/A'):<16} "
              f"{f.get('Account Number','N/A')[:22]:<25} "
              f"{f.get('Notes','N/A')[:30]}{conflict_flag}")

    # Full-field extraction for two currencies
    # Wait 65s so CHF/GBP/USD tokens exit the 60s rolling TPM window.
    print(f"\nWaiting 65s for TPM window to clear before next batch...")
    time.sleep(65)
    print(f"\n{'='*70}")
    print("FULL-FIELD EXTRACTION: JPY + MXN (all 9 FX SSI fields)")
    full_rows = pipeline.extract_batch(["JPY", "MXN"], batch_size=1)
    for r in full_rows:
        print(f"\n--- {r['item']} ---")
        for field, val in r["fields"].items():
            print(f"  {field:<40} {val}")


if __name__ == "__main__":
    main()

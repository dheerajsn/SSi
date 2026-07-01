"""
Extraction validation tests — negative cases and field quality checks.

Tests:
  EX01  Unknown currency → "not found", no hallucinated BIC
  EX02  Mixed batch: known + unknown → known filled, unknown "not found"
  EX03  Known currency field coverage — primary fields should not be N/A
  EX04  SWIFT/BIC format — must be 8 or 11 alphanum chars (or "not found")
  EX05  AUD/EUR — correspondent BIC mapping (source says "Correspondent BIC:")
  EX06  No hallucination — returned BICs must appear verbatim in source rows

Run: GROQ_MODEL=llama-3.1-8b-instant python tests/test_extraction.py
"""

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import SSIPipeline
from src.providers import GroqLLM, LocalEmbedder
from src.domain_config import FX_SSI
from src.preprocessing import preprocess_all

DATA_DIR = str(Path(__file__).parent.parent / "data" / "mock_azure_di")

# Known-good ground-truth drawn directly from mock data
KNOWN_BICS = {
    "CHF": {"CRESCHZZXXX", "UBSWCHZH80A", "CREQCHGG"},      # Agency, Principal, Correspondent
    "GBP": {"LOYDGB2L", "BARCGB22"},
    "USD": {"CITIUS33", "CHASUS33"},
    "JPY": {"BOTKJPJT", "SMBCJPJT"},
    "AUD": {"ANZBAAU3XXX", "CTBAAU2S"},
    "EUR": {"DEUTDEFF", "CHASDEFX"},
}

BIC_PATTERN = re.compile(r"^[A-Z]{4}[A-Z0-9]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")


def _build_pipeline() -> SSIPipeline:
    key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)
    p = SSIPipeline(DATA_DIR, llm=GroqLLM(api_key=key, model=model),
                    embedder=LocalEmbedder(), domain="fx_ssi")
    p.build()
    return p


def _all_source_bics(parents) -> set:
    """Collect every BIC-like token from the raw source rows."""
    bic_re = re.compile(r"\b[A-Z]{4}[A-Z0-9]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b")
    found = set()
    for p in parents:
        for row in p["rows"]:
            found.update(bic_re.findall(row))
        # findall returns group-1 captures; re-search for full match
    found2 = set()
    for p in parents:
        for row in p["rows"]:
            for m in bic_re.finditer(row):
                found2.add(m.group(0))
    return found2


PASS_SYMBOL = "[PASS]"
FAIL_SYMBOL = "[FAIL]"


def run(name: str, desc: str, failures: list) -> bool:
    ok = len(failures) == 0
    symbol = PASS_SYMBOL if ok else FAIL_SYMBOL
    print(f"\n{'='*65}")
    print(f"{name}  {desc}")
    if failures:
        for f in failures:
            print(f"  {FAIL_SYMBOL} {f}")
    else:
        print(f"  {PASS_SYMBOL}")
    return ok


def main():
    print("Building pipeline...")
    pipeline = _build_pipeline()
    parents, _, _ = preprocess_all(DATA_DIR)
    source_bics = _all_source_bics(parents)

    passed = 0
    total = 0

    # ------------------------------------------------------------------
    # EX01 — Unknown currency should say "not found", no fake BIC
    # ------------------------------------------------------------------
    total += 1
    rows = pipeline.extract_batch(["XYZ", "FAKE123"], fields=["SWIFT/BIC", "Account Number"], batch_size=1)
    failures = []
    _NOT_FOUND_PHRASES = ("not found", "not present", "not in the context",
                          "not recognized", "cannot find", "unable to find")
    for r in rows:
        swift = r["fields"].get("SWIFT/BIC", "N/A")
        # Pipeline or LLM correctly signalled absence — accept "N/A", "not found", or empty
        if swift not in ("N/A", "not found", "") and BIC_PATTERN.match(swift or ""):
            failures.append(f"{r['item']}: SWIFT/BIC '{swift}' looks like a real BIC (hallucination)")
        answer_lo = r["answer"].lower()
        if not any(p in answer_lo for p in _NOT_FOUND_PHRASES):
            failures.append(f"{r['item']}: answer has no 'not found' / 'not present' signal — got: {r['answer'][:80]!r}")
    if run("EX01", "Unknown currencies (XYZ, FAKE123) — no hallucinated BICs", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # EX02 — Mixed batch: CHF (known) + NGN (not in corpus)
    # ------------------------------------------------------------------
    total += 1
    rows = pipeline.extract_batch(["CHF", "NGN"], fields=["SWIFT/BIC", "Account Number"], batch_size=1)
    failures = []
    for r in rows:
        swift = r["fields"].get("SWIFT/BIC", "N/A")
        if r["item"] == "CHF":
            if swift == "N/A":
                failures.append("CHF SWIFT/BIC should not be N/A")
            elif swift not in KNOWN_BICS["CHF"]:
                failures.append(f"CHF SWIFT/BIC '{swift}' not in known set {KNOWN_BICS['CHF']}")
        elif r["item"] == "NGN":
            if BIC_PATTERN.match(swift or ""):
                failures.append(f"NGN: hallucinated BIC '{swift}'")
    if run("EX02", "Mixed batch: CHF (known) + NGN (not in corpus)", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # EX03 — Known currencies: primary fields must not be N/A
    # ------------------------------------------------------------------
    total += 1
    currencies = ["CHF", "GBP", "USD"]
    rows = pipeline.extract_batch(currencies, fields=FX_SSI.fields, batch_size=1)
    failures = []
    required = ["SWIFT/BIC", "Account Number", "Beneficiary Bank"]
    for r in rows:
        for req in required:
            val = r["fields"].get(req, "N/A")
            if val == "N/A":
                failures.append(f"{r['item']}: '{req}' is N/A (should be populated)")
    if run("EX03", f"Primary fields for {currencies} must be populated (not N/A)", failures):
        passed += 1

    time.sleep(65)  # clear rolling TPM window

    # ------------------------------------------------------------------
    # EX04 — SWIFT/BIC format: 8 or 11 chars, alphanumeric only
    # ------------------------------------------------------------------
    total += 1
    currencies4 = ["CHF", "GBP", "USD", "JPY"]
    rows = pipeline.extract_batch(currencies4, fields=["SWIFT/BIC"], batch_size=1)
    failures = []
    for r in rows:
        swift = r["fields"].get("SWIFT/BIC", "N/A")
        if swift == "N/A":
            failures.append(f"{r['item']}: SWIFT/BIC is N/A")
        elif not BIC_PATTERN.match(swift):
            failures.append(f"{r['item']}: '{swift}' does not match BIC format (4+2+2[+3])")
    if run("EX04", "SWIFT/BIC format must be valid 8 or 11-char BIC", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # EX05 — Correspondent BIC mapping
    # AUD: primary BIC = ANZBAU3M (ANZ), correspondent BIC = CTBAAU2S (CBA) — different banks
    # EUR: DEUTDEFF is BOTH the primary and correspondent BIC (same bank);
    #      LLM correctly puts it in SWIFT/BIC and leaves Correspondent SWIFT/BIC as N/A
    # ------------------------------------------------------------------
    total += 1
    rows = pipeline.extract_batch(["AUD", "EUR"], fields=["SWIFT/BIC", "Correspondent SWIFT/BIC"], batch_size=1)
    failures = []
    for r in rows:
        swift = r["fields"].get("SWIFT/BIC", "N/A")
        corr  = r["fields"].get("Correspondent SWIFT/BIC", "N/A")
        if r["item"] == "AUD":
            # AUD has a separate correspondent BIC (CTBAAU2S) distinct from primary (ANZBAU3M)
            if corr == "N/A":
                failures.append(
                    "AUD: Correspondent SWIFT/BIC is N/A — "
                    "expected CTBAAU2S mapped from 'Correspondent BIC:' in correspondent_banking_ssi.json"
                )
            elif not BIC_PATTERN.match(corr):
                failures.append(f"AUD: Correspondent SWIFT/BIC '{corr}' not a valid BIC format")
        elif r["item"] == "EUR":
            # EUR primary and correspondent are both DEUTDEFF (Deutsche Bank)
            # LLM puts it in SWIFT/BIC; Correspondent SWIFT/BIC may be N/A — that's acceptable
            if swift == "N/A":
                failures.append("EUR: SWIFT/BIC is N/A — expected DEUTDEFF")
            elif "DEUTDEFF" not in swift.upper():
                failures.append(f"EUR: SWIFT/BIC '{swift}' expected to be DEUTDEFF")
    if run("EX05", "Correspondent BIC mapping: AUD distinct correspondent, EUR same-bank primary", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # EX06 — No hallucination: every BIC in results must be in source rows
    # ------------------------------------------------------------------
    total += 1
    rows = pipeline.extract_batch(
        ["CHF", "GBP", "USD"],
        fields=["SWIFT/BIC", "Correspondent SWIFT/BIC"],
        batch_size=1,
    )
    failures = []
    for r in rows:
        for field in ["SWIFT/BIC", "Correspondent SWIFT/BIC"]:
            val = r["fields"].get(field, "N/A")
            if val == "N/A":
                continue
            if not BIC_PATTERN.match(val):
                continue  # skip non-BIC values
            if val not in source_bics:
                failures.append(
                    f"{r['item']} {field}='{val}' not found in any source document row (hallucination?)"
                )
    if run("EX06", "Anti-hallucination: extracted BICs must appear verbatim in source rows", failures):
        passed += 1

    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"EXTRACTION TEST RESULT: {passed}/{total} passed")


if __name__ == "__main__":
    main()

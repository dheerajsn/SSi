"""
Markets SSI extraction test.

Extracts Global Agent Swift Agent, Local Agent Swift Agent,
Local Agent Account Number, Institution BIC for each market-method pair.

Run: GROQ_MODEL=llama-3.3-70b-versatile python tests/test_markets.py
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
from src.domain_config import MARKETS_SSI

DATA_DIR = str(Path(__file__).parent.parent / "data" / "mock_azure_di")

BIC_PATTERN = re.compile(r"^[A-Z]{4}[A-Z0-9]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")

# Ground truth drawn directly from markets_ssi.json
GROUND_TRUTH = {
    "Germany-CLEARGER": {
        "Institution BIC":         "DAKVDEFF",
        "Global Agent Swift Agent": "CHASDEFX",
        "Local Agent Swift Agent":  "DEUTDEFF",
    },
    "Italy-CLEARGER": {
        "Institution BIC":         "DAKVDEFF",
        "Global Agent Swift Agent": "UBSWDEFF",
        "Local Agent Swift Agent":  "DEUTDEFF",
        "Local Agent Account Number": "9876543",
    },
    "France-EUROCLEAR": {
        "Institution BIC":         "MGTCBEBE",
        "Global Agent Swift Agent": "BNPAFRPP",
        "Local Agent Swift Agent":  "AGRIFRPP",
    },
    "UK-CREST": {
        "Institution BIC":         "CRSTGB22",
        "Global Agent Swift Agent": "BARCGB22",
        "Local Agent Swift Agent":  "MIDLGB22",
    },
    "US-DTC": {
        "Institution BIC":         "DTCYUS33",
        "Global Agent Swift Agent": "SBOSUS3X",
        "Local Agent Swift Agent":  "IRVTUS3N",
    },
    "Japan-JASDEC": {
        "Institution BIC":         "JAESJPJT",
        "Global Agent Swift Agent": "BOTKJPJT",
        "Local Agent Swift Agent":  "SMBCJPJT",
    },
    "Spain-IBERCLEAR": {
        "Institution BIC":         "ESAEESBB",
        "Global Agent Swift Agent": "BBVAESBB",
        "Local Agent Swift Agent":  "CAIXESBB",
    },
}

PASS_SYMBOL = "[PASS]"
FAIL_SYMBOL = "[FAIL]"


def report(name: str, desc: str, failures: list) -> bool:
    ok = len(failures) == 0
    print(f"\n{'='*65}")
    print(f"{name}  {desc}")
    if failures:
        for f in failures:
            print(f"  {FAIL_SYMBOL} {f}")
    else:
        print(f"  {PASS_SYMBOL}")
    return ok


def build_pipeline() -> SSIPipeline:
    key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)
    p = SSIPipeline(DATA_DIR, key, groq_model=model, domain="markets_ssi")
    p.build()
    return p


def main():
    print("Building markets pipeline...")
    pipeline = build_pipeline()

    passed = 0
    total = 0
    FIELDS = MARKETS_SSI.fields  # all 8 fields

    # ------------------------------------------------------------------
    # MK01 — CLEARGER markets: Germany + Italy
    # ------------------------------------------------------------------
    total += 1
    markets = ["Germany-CLEARGER", "Italy-CLEARGER"]
    rows = pipeline.extract_batch(markets, fields=FIELDS, batch_size=1)
    failures = []
    for r in rows:
        gt = GROUND_TRUTH.get(r["item"], {})
        for field, expected in gt.items():
            got = r["fields"].get(field, "N/A")
            if got.upper() != expected.upper():
                failures.append(f"{r['item']} {field}: expected '{expected}', got '{got}'")
        # Institution BIC must be DAKVDEFF for both
        inst = r["fields"].get("Institution BIC", "N/A")
        if "DAKVDEFF" not in inst.upper():
            failures.append(f"{r['item']}: Institution BIC '{inst}' should contain DAKVDEFF")
    if report("MK01", "CLEARGER markets — Germany + Italy", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # MK02 — EUROCLEAR market: France
    # ------------------------------------------------------------------
    total += 1
    rows = pipeline.extract_batch(["France-EUROCLEAR"], fields=FIELDS, batch_size=1)
    failures = []
    for r in rows:
        gt = GROUND_TRUTH.get(r["item"], {})
        for field, expected in gt.items():
            got = r["fields"].get(field, "N/A")
            if got.upper() != expected.upper():
                failures.append(f"{r['item']} {field}: expected '{expected}', got '{got}'")
    if report("MK02", "Euroclear Belgium — France-EUROCLEAR", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # MK03 — Local CSD markets: UK-CREST + US-DTC
    # ------------------------------------------------------------------
    total += 1
    markets = ["UK-CREST", "US-DTC"]
    rows = pipeline.extract_batch(markets, fields=FIELDS, batch_size=1)
    failures = []
    for r in rows:
        gt = GROUND_TRUTH.get(r["item"], {})
        for field, expected in gt.items():
            got = r["fields"].get(field, "N/A")
            if got.upper() != expected.upper():
                failures.append(f"{r['item']} {field}: expected '{expected}', got '{got}'")
    if report("MK03", "Local CSDs — UK-CREST + US-DTC", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # MK04 — Unknown market: FAKE-MARKET should return not found
    # ------------------------------------------------------------------
    total += 1
    rows = pipeline.extract_batch(["FAKE-MARKET"], fields=FIELDS, batch_size=1)
    failures = []
    for r in rows:
        inst = r["fields"].get("Institution BIC", "N/A")
        global_agent = r["fields"].get("Global Agent Swift Agent", "N/A")
        # Should not hallucinate a valid BIC
        for field in ["Institution BIC", "Global Agent Swift Agent", "Local Agent Swift Agent"]:
            val = r["fields"].get(field, "N/A")
            if val not in ("N/A", "not found", "") and BIC_PATTERN.match(val or ""):
                failures.append(f"FAKE-MARKET {field}='{val}' — looks like a real BIC (hallucination)")
        # Answer should signal absence
        answer_lo = r["answer"].lower()
        if not any(sig in answer_lo for sig in ("not found", "not present", "not in the context", "not recognized")):
            failures.append(f"FAKE-MARKET: no not-found signal in answer: {r['answer'][:80]!r}")
    if report("MK04", "Unknown market — no hallucination", failures):
        passed += 1

    time.sleep(65)

    # ------------------------------------------------------------------
    # MK05 — BIC format: all extracted BICs must match 8/11-char pattern
    # ------------------------------------------------------------------
    total += 1
    markets = ["Germany-CLEARGER", "Italy-CLEARGER", "France-EUROCLEAR", "UK-CREST"]
    rows = pipeline.extract_batch(markets, fields=FIELDS, batch_size=1)
    failures = []
    bic_fields = ["Institution BIC", "Global Agent Swift Agent", "Local Agent Swift Agent"]
    for r in rows:
        for field in bic_fields:
            val = r["fields"].get(field, "N/A")
            if val in ("N/A", "not found", ""):
                continue
            # Some values may be comma-separated (multi-path) — check first token
            first = val.split(",")[0].strip().split()[0]
            if not BIC_PATTERN.match(first):
                failures.append(f"{r['item']} {field}: '{first}' not a valid BIC format")
    if report("MK05", "BIC format check for CLEARGER + EUROCLEAR + UK-CREST", failures):
        passed += 1

    time.sleep(20)

    # ------------------------------------------------------------------
    # MK06 — Japan + Spain
    # ------------------------------------------------------------------
    total += 1
    markets = ["Japan-JASDEC", "Spain-IBERCLEAR"]
    rows = pipeline.extract_batch(markets, fields=FIELDS, batch_size=1)
    failures = []
    for r in rows:
        gt = GROUND_TRUTH.get(r["item"], {})
        for field, expected in gt.items():
            got = r["fields"].get(field, "N/A")
            if got.upper() != expected.upper():
                failures.append(f"{r['item']} {field}: expected '{expected}', got '{got}'")
    if report("MK06", "Asia + EMEA local CSDs — Japan-JASDEC + Spain-IBERCLEAR", failures):
        passed += 1

    print(f"\n{'='*65}")
    print(f"MARKETS TEST RESULT: {passed}/{total} passed")

    # ------------------------------------------------------------------
    # Display summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("FULL EXTRACTION TABLE — all markets")
    print(f"\n{'Market':<25} {'Inst BIC':<14} {'Global Agent':<14} {'Local Agent':<14} {'Acct No.'}")
    print("-" * 90)
    for r in rows:
        f = r["fields"]
        print(
            f"{r['item']:<25} "
            f"{f.get('Institution BIC','N/A'):<14} "
            f"{f.get('Global Agent Swift Agent','N/A'):<14} "
            f"{f.get('Local Agent Swift Agent','N/A'):<14} "
            f"{f.get('Local Agent Account Number','N/A')[:20]}"
        )


if __name__ == "__main__":
    main()

# SSI RAG Pipeline — Rebuild Prompt

Use this prompt to recreate the SSI RAG pipeline from scratch on a new machine.

**Constraints for this setup:**
- LLM and embeddings are served by an inhouse Orchestra gateway (no external downloads)
- No Jina, no sentence-transformers, no HuggingFace model downloads
- JSON source documents are already available — do not create mock data
- Python 3.10+

---

## What to build

A **Retrieval-Augmented Generation (RAG) pipeline** for Standard Settlement Instructions (SSI).

SSI documents are structured data (tables) that describe how financial trades settle — which bank, which SWIFT/BIC code, which account number. The pipeline:
1. Indexes those documents using embeddings
2. At query time, retrieves the relevant rows
3. Sends them as context to an LLM to answer natural-language questions or extract structured fields

---

## Folder structure to create

```
project_root/
  data/                  ← user already has JSON files here, do not touch
  src/
    __init__.py
    preprocessing.py
    chunking.py
    retrieval.py
    validation.py
    domain_config.py
    prompt.py
    pipeline.py
    providers/
      __init__.py
      base.py
      _rest.py
      orchestra.py       ← only provider needed; skip groq.py / jina.py / local.py
  tests/
    __init__.py
    test_pipeline.py     ← use OrchestraLLM + OrchestraEmbedder only; no Groq, no sentence-transformers
  requirements.txt
  .env
  .env.example
```

---

## Input document format

JSON files produced by Azure Document Intelligence (or matching schema). Each file:

```json
{
  "doc_name": "fx_spot_ssi.json",
  "model_id": "prebuilt-layout",
  "content": "# Section Heading\n\nPreamble text.\n\n<table>...</table>\n\nNotes text."
}
```

`content` is a markdown-like string with:
- `# Heading` lines marking sections
- Plain text paragraphs (preamble / notes)
- `<table>` HTML blocks with `<tr><th><td>` structure, including rowspan/colspan

---

## Module specifications

### `src/providers/base.py`

Two abstract base classes. No dependencies beyond `abc` and `numpy`.

```python
class LLMProvider(ABC):
    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048, temperature: float = 0) -> str: ...

class EmbedProvider(ABC):
    def embed(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], task="retrieval.query")[0]
```

### `src/providers/_rest.py`

Shared HTTP logic for all REST-based providers. Uses `requests`. No other provider-specific imports.

**`RestLLM(LLMProvider)`**
- `__init__(endpoint, api_key, model, max_retries=8)`
- `complete(prompt, system, max_tokens, temperature)` — sends OpenAI-compatible chat/completions request
- 429 handling: parse `"try again in Xm Y.Zs"` from error body, sleep exactly that long, retry
- 5xx handling: 2-second backoff, retry
- Auth: `Authorization: Bearer {api_key}` header; skip header if api_key is empty

**`RestEmbedder(EmbedProvider)`**
- `__init__(endpoint, api_key, model, batch_size=256, timeout=60)`
- `embed(texts, task)` — sends `{"model": ..., "input": [...]}`, parses `{"data": [{"embedding": [...], "index": N}]}`
- Batches automatically; L2-normalises output matrix
- `_call(batch, extra=None)` — internal single-batch method, `extra` dict merged into payload

**`_extract_json(text: str) -> dict`**
- Strips ```json code fences if present
- `json.loads()` — if fails, regex search for `{.*}` with DOTALL, try again
- Fallback: `{"answer": text, "source": None}`

### `src/providers/orchestra.py`

```python
class OrchestraLLM(RestLLM):
    """Inhouse AI gateway — POST {base_url}/v2/chat/completions"""
    def __init__(self, base_url, api_key="", model="llama-70b",
                 chat_path="/v2/chat/completions", max_retries=8):
        super().__init__(endpoint=base_url.rstrip("/") + chat_path, ...)

class OrchestraEmbedder(RestEmbedder):
    """Inhouse AI gateway — POST {base_url}/v2/embeddings"""
    def __init__(self, base_url, api_key="", model="text-embedding-large",
                 embed_path="/v2/embeddings", batch_size=256, timeout=60):
        super().__init__(endpoint=base_url.rstrip("/") + embed_path, ...)
```

Both drop the `Authorization` header when `api_key` is empty (unauthenticated gateway).

### `src/providers/__init__.py`

**Only** import from `orchestra.py` and `base.py`. Do NOT create or import from `groq.py`, `jina.py`, or `local.py` — those files do not exist in this setup.

```python
from .base import LLMProvider, EmbedProvider
from .orchestra import OrchestraLLM, OrchestraEmbedder

def make_reranker(model_or_provider=None, api_key=None):
    """Reranking is disabled in the Orchestra-only setup. Always returns None."""
    return None

__all__ = ["LLMProvider", "EmbedProvider", "OrchestraLLM", "OrchestraEmbedder", "make_reranker"]
```

### `src/preprocessing.py`

**`parse_html_table(table_tag) -> List[Dict[str, str]]`**
- Walk `<tr><th><td>` expanding rowspan/colspan into a full virtual grid
- Pre-allocate grid to `len(html_rows)` rows so rowspan never shifts row indices
- First row all-`<th>` → use as headers; else generate `Col0, Col1, ...`
- Return list of `{header: value}` dicts, skipping empty cells

**`flatten_row(row: Dict) -> str`**
- `"Header: Value\nHeader: Value"` — skips empty values

**`parse_blocks(raw_content: str) -> List[Dict]`**
- Split on `<table>...</table>` boundaries using regex
- Text segments: classify lines as `{"type": "heading", "text": ...}` (starts with `#`) or `{"type": "text", "text": ...}`
- Table segments: parse with `parse_html_table`, emit `{"type": "table", "rows": [...]}`

**`build_parents(blocks, doc_name, id_offset=0) -> List[Dict]`**

Section-aware parent construction:
- Group blocks by heading (`_group_into_sections`)
- Doc-level preamble (first section with no table) is attached to every table parent in the doc
- Per section: text before first table → `section_preamble`; each table → one parent carrying the preamble; text after a table → `notes[]` on that parent; section with no table → single narrative parent
- Parent schema: `{"id": int, "type": "table"|"narrative", "heading": str, "doc_name": str, "preamble": str, "rows": [str], "notes": [str]}`
- Post-pass `_merge_consecutive_tables`: merge adjacent table parents with same heading+doc (handles PDF page-split tables)
- Re-apply id_offset after merge renumbers from 0

**`build_children(parents) -> List[Dict]`**
- One child per row string in each parent
- `embed_text = "[{doc_label} | {heading}] " + row_text` (contextual prefix for disambiguation)
- Child schema: `{"parent_id": int, "row_text": str, "doc_name": str, "embed_text": str}`

**`build_conflict_map(parents) -> Dict`**
- Scan all parents; for each row extract scope value (first field value) and BIC-pattern codes
- If same scope appears in 2+ docs with different codes → conflict
- Returns `{"GBP": {"fx_spot.json": ["LOYDGB2L"], "corr_banking.json": ["BARCGB22"]}}`

**`parent_full_text(parent) -> str`**
- `[Section: heading]\npreamble\nrow1\nrow2\nnotes`

### `src/chunking.py`

Three strategies, all returning `(parents, children)`:

**`SectionTableStrategy`** (default, name=`"section_table"`)
- Calls `build_parents` + `build_children` directly

**`SlidingWindowStrategy`** (name=`"sliding_window"`, window=150 words, overlap=30)
- Flatten all blocks to text, split into word windows with overlap
- Each window is one parent with one row (the window text)

**`ParagraphStrategy`** (name=`"paragraph"`)
- One parent per table, one parent per paragraph (blank-line-separated)

`get_strategy(name)` returns the right instance. `STRATEGIES` dict maps name → instance.

### `src/retrieval.py`

**`SSIRetriever`**
- `__init__(embedder, parents, children, mode="dense")`
- Modes: `"dense"` (FAISS cosine), `"sparse"` (BM25 via rank-bm25), `"hybrid"` (RRF fusion of both)
- `build_index()`: embed all `child["embed_text"]`, build FAISS `IndexFlatIP`; also fit BM25 on row texts
- `retrieve(query, k, score_threshold) -> dict`:
  - Returns `{"matched_children": [...], "matched_parents": [...], "best_score": float, "low_confidence": bool}`
  - Deduplicates parents; preserves score order
  - Each child gets a `"score"` key

### `src/validation.py`

**`confidence_gate(retrieval, threshold=0.15) -> dict`**
- If `best_score < threshold` → `{"blocked": True, "reason": "..."}`
- Else → `{"blocked": False, "reason": "pass"}`

**`detect_conflicts(question, conflict_map) -> List[dict]`**
- Check if any conflict key appears in the question (case-insensitive)
- Return list of `{"scope": "GBP", "message": "Conflicting SSI found for 'GBP'..."}` dicts

**`check_grounding(answer, context) -> dict`**
- Extract BIC-like tokens (8/11-char uppercase) from the answer
- Check each against the context string
- Return `{"passed": bool, "grounded": {bic: True}, "ungrounded": {bic: False}}`

### `src/domain_config.py`

**`DomainConfig` dataclass**: `name`, `scope_label`, `fields: List[str]`, `system_prompt: str`

**`_make_prompt(scope_label, fields, context_description) -> str`** builds a system prompt with these rules:
1. Only use provided context — never use training data for BIC/account numbers
2. If instrument not in context: say exactly "Settlement instructions for [X] were not found"
3. Field absent from context → "N/A"
4. Follow cross-references within the same context block
5. If multiple settlement paths exist, list all with labels
6. Include operational notes
7. **Always respond with a single valid JSON object** — no text outside it
   - Conversational: `{"answer": "...", "source": "doc / section"}`
   - Unknown: `{"answer": "Settlement instructions for [X] were not found...", "source": null}`
8. Extraction queries specify their own schema in the question — follow it exactly; missing field → "N/A"

**Pre-built configs**: `FX_SSI`, `EQUITY_SSI`, `REPO_SSI`, `CORRESPONDENT_BANKING`, `MARKETS_SSI`

For **MARKETS_SSI** the system prompt must include field alias mappings:
- `"Institution BIC"` = PSET BIC = `:95P::PSET` — the CSD/ICSD (e.g. DAKVDEFF=Clearstream Germany)
- `"Global Agent Swift Agent"` = DEC/RECU BIC = column header may say `DEC/RECU`
- `"Local Agent Swift Agent"` = Local Settlement Agent BIC = column header may say `Local Settlement Agent`
- `"Local Agent Account Number"` = custodian's safe-custody account (`:97A::SAFE`)

`DOMAIN_REGISTRY: Dict[str, DomainConfig]` — maps name strings to configs.
`get_domain(name)` — lookup with ValueError for unknown names.

### `src/prompt.py`

Jinja2-based prompt builder. Falls back to plain string assembly if jinja2 is not installed.

**`render_rag_prompt(question, matched_parents, matched_children=None) -> str`**

Template logic:
- For each matched parent, include only the children rows that were retrieved (focused mode)
- Always include preamble and notes
- Format: `[Source: {doc_name} | Section: {heading}]\n{preamble}\n{rows}\n{notes}\nQuestion: {question}`

**`render_citation(matched_parents) -> str`**
- Deduplicated `"Sources:\n  - doc / section"` list

**`render_custom(template_str, **context) -> str`**
- Render arbitrary Jinja2 template (requires jinja2)

### `src/pipeline.py`

**`SSIPipeline`**

Constructor:
```python
def __init__(
    self,
    data_dir: str,
    llm: LLMProvider,
    embedder: EmbedProvider,
    reranker=None,              # pass None — no reranker in Orchestra-only setup
    domain=None,                # str key or DomainConfig; default FX_SSI
    chunk_strategy="section_table",
    retrieval_mode="dense",
    default_k=10,
    score_threshold=0.15,
    small_corpus_k=50,          # corpus smaller than this → direct-context mode
):
```

**`build()`** — index all `*.json` files in `data_dir`. Zero LLM calls.
- Parse each file → blocks → parents + children (via chosen strategy)
- Build conflict map
- If `len(children) < small_corpus_k` → `_direct_mode = True` (skip retrieval, send all context)
- Else build SSIRetriever and call `build_index()`

**`query(question, k=None, max_tokens=None) -> dict`**

Returns:
```python
{
  "question": str,
  "answer": str,           # extracted from LLM JSON response
  "llm_raw": str,          # raw LLM output before parsing
  "conflicts": List[dict],
  "grounding": dict,
  "citation": str,
  "retrieval": dict,       # matched_children, matched_parents, best_score
  "gate": dict,
  "latency_s": float,
  "blocked": bool,
  "mode": "direct" | "retrieval",
}
```

Key behaviours:
- Broad query detection: if question contains "all"/"every"/"full"/"complete"/"entire" → k = all children
- Direct-context mode: skip retrieval, send full doc text to LLM
- Retrieval mode: retrieve k children, confidence gate, build prompt, call LLM, parse JSON answer
- `_parse_llm_answer(raw)`: extract `parsed["answer"]` from JSON; if value is dict, join as `"k: v\n"`

**`extract_batch(items, fields=None, batch_size=5, max_workers=1) -> List[dict]`**

Structured field extraction for a list of scope items (currencies, markets, etc.):
- Builds extraction question requesting a JSON schema: `{"item": {"field": "value", ...}, ...}`
- Question wording must NOT contain "all" or "every" (avoids broad-query heuristic)
- Passes `k=default_k` explicitly so retrieval depth is fixed
- Parses LLM JSON, maps each item to its field dict
- If item absent from JSON → all fields set to `"not found"`
- Returns list of `{"item", "fields", "answer", "conflicts", "grounding", "citation", "latency_s", "blocked"}`

**`batch_query(questions, k, max_tokens, max_workers=1)`** — parallel free-form queries.

**`evaluate(questions, strategies, metrics)`** — retrieval-only strategy comparison, no LLM calls.

### `tests/test_pipeline.py`

**Only use OrchestraLLM and OrchestraEmbedder.** Read credentials from `.env` via `python-dotenv`.

```python
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.providers.orchestra import OrchestraLLM, OrchestraEmbedder
from src.pipeline import SSIPipeline

DATA_DIR = str(Path(__file__).parent.parent / "data")

def build_pipeline(domain="fx_ssi") -> SSIPipeline:
    base_url = os.getenv("ORCHESTRA_BASE_URL")
    api_key  = os.getenv("ORCHESTRA_API_KEY", "")
    p = SSIPipeline(
        data_dir = DATA_DIR,
        llm      = OrchestraLLM(base_url=base_url, api_key=api_key),
        embedder = OrchestraEmbedder(base_url=base_url, api_key=api_key),
        domain   = domain,
    )
    p.build()
    return p

def main():
    pipeline = build_pipeline()
    result = pipeline.query("What is the SWIFT code for CHF?")
    print("Answer:", result["answer"])
    print("Blocked:", result["blocked"])
    print("Latency:", result["latency_s"])

if __name__ == "__main__":
    main()
```

---

## Usage example (Orchestra setup)

```python
import os
from src.providers.orchestra import OrchestraLLM, OrchestraEmbedder
from src.pipeline import SSIPipeline

pipeline = SSIPipeline(
    data_dir = "data/",
    llm      = OrchestraLLM(
                   base_url = os.getenv("ORCHESTRA_BASE_URL"),  # e.g. https://ai.mywork.internal
                   api_key  = os.getenv("ORCHESTRA_API_KEY"),
                   model    = "llama-70b",
               ),
    embedder = OrchestraEmbedder(
                   base_url = os.getenv("ORCHESTRA_BASE_URL"),
                   api_key  = os.getenv("ORCHESTRA_API_KEY"),
                   model    = "text-embedding-large",
               ),
    domain   = "fx_ssi",
)
pipeline.build()

# Single query
result = pipeline.query("What is the SWIFT code for CHF?")
print(result["answer"])

# Structured extraction
rows = pipeline.extract_batch(
    items      = ["CHF", "GBP", "USD"],
    fields     = ["SWIFT/BIC", "Account Number"],
    batch_size = 3,
)
for r in rows:
    print(r["item"], r["fields"])
```

---

## `.env` file

```
ORCHESTRA_BASE_URL=https://ai.mywork.internal
ORCHESTRA_API_KEY=your-bearer-token
```

---

## `requirements.txt`

```
requests>=2.31.0
numpy>=1.24.0
faiss-cpu>=1.7.4
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
jinja2>=3.1.0
rank-bm25>=0.2.2
```

No sentence-transformers. No pdfplumber. No HuggingFace downloads.

---

## Key design decisions to preserve

1. **Parent-child chunking**: one parent per table (carries preamble + notes), one child per row. Only children are embedded; parents are fetched at query time to provide full context to the LLM.

2. **Contextual prefix on embed_text**: `"[doc_name | section] row_text"` — prevents embedding collision between the same BIC appearing in different docs/sections.

3. **JSON-enforced LLM responses**: system prompt rules 7-8 require the LLM to always respond with valid JSON. `_extract_json` handles code fences and plain-text fallback.

4. **Broad-query guard**: extraction questions must not contain "all"/"every" — these words trigger `_is_broad_query()` which sets k = all children, blowing up the context window.

5. **Doc-level preamble propagation**: if the first section of a document has no table, its text is treated as a doc-level preamble and attached to every table parent in that document. This ensures global defaults (e.g. "Default BIC: DEUTDEFFEEQ") are always visible to the LLM.

6. **`_merge_consecutive_tables`**: adjacent table parents with the same heading and doc_name are merged into one. Handles tables that Azure DI splits across page boundaries.

7. **Small corpus direct-context mode**: when total children < `small_corpus_k` (default 50), skip retrieval entirely and send all document text to the LLM in one call. Avoids empty-retrieval failures on tiny corpora.

---

## Also create: `README.md`

Generate a `README.md` at the project root covering these sections:

### Overview
One paragraph: RAG pipeline over SSI documents that answers natural-language settlement questions and extracts structured fields (SWIFT/BIC, account numbers, PSET BICs) using an inhouse Orchestra LLM and embedding gateway.

### Prerequisites
- Python 3.10+
- Access to Orchestra AI gateway — base URL and API key
- SSI documents as JSON files (Azure Document Intelligence output format)

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in ORCHESTRA_BASE_URL and ORCHESTRA_API_KEY
```

### Document format
Each JSON file must match this schema:
```json
{
  "doc_name": "my_ssi.json",
  "content": "# Section\n\nPreamble text.\n\n<table>...</table>\n\nNotes."
}
```
Drop all files into the `data/` directory. `content` is a markdown-like string with HTML `<table>` blocks and `#` headings. Azure Document Intelligence produces this automatically from PDFs.

### Running for currency-based SSI (FX Spot, Correspondent Banking)

```python
import os
from src.providers.orchestra import OrchestraLLM, OrchestraEmbedder
from src.pipeline import SSIPipeline

pipeline = SSIPipeline(
    data_dir = "data/",
    llm      = OrchestraLLM(base_url=os.getenv("ORCHESTRA_BASE_URL"),
                             api_key=os.getenv("ORCHESTRA_API_KEY")),
    embedder = OrchestraEmbedder(base_url=os.getenv("ORCHESTRA_BASE_URL"),
                                  api_key=os.getenv("ORCHESTRA_API_KEY")),
    domain   = "fx_ssi",        # or "correspondent_banking"
)
pipeline.build()

# Free-form question
result = pipeline.query("What is the SWIFT code for CHF?")
print(result["answer"])

# Structured extraction — returns one dict per currency
rows = pipeline.extract_batch(
    items  = ["USD", "EUR", "GBP", "CHF", "JPY"],
    fields = ["SWIFT/BIC", "Account Number", "Beneficiary Bank"],
)
for r in rows:
    print(r["item"], r["fields"])
```

Available currency domains: `"fx_ssi"`, `"correspondent_banking"`.

### Running for market-based SSI (Equities, Markets)

Change `domain` to `"markets_ssi"` or `"equity_ssi"`. The LLM is taught field alias mappings automatically:
- Column `DEC/RECU` in source documents → field `Global Agent Swift Agent`
- Column `Local Settlement Agent` → field `Local Agent Swift Agent`
- Column `Institution BIC` → PSET BIC (the CSD/ICSD where the trade settles)

```python
pipeline = SSIPipeline(
    data_dir = "data/",
    llm      = OrchestraLLM(...),
    embedder = OrchestraEmbedder(...),
    domain   = "markets_ssi",
)
pipeline.build()

rows = pipeline.extract_batch(
    items  = ["Germany-CLEARGER", "France-EUROCLEAR", "UK-CREST", "US-DTC"],
    fields = ["Institution BIC", "Global Agent Swift Agent",
              "Local Agent Swift Agent", "Local Agent Account Number"],
    batch_size = 2,
)
for r in rows:
    print(r["item"], r["fields"])
```

Markets are identified as `Country-Method` strings matching the `Market` column in your SSI tables.

### Adding a new domain at runtime

No code changes needed — use `custom_domain()`:

```python
from src.domain_config import custom_domain

my_domain = custom_domain(
    scope_label = "bond",
    fields      = ["ISIN", "Custodian", "Custodian BIC", "Account Number"],
    context_description = "You handle bond settlement SSI."
)
pipeline = SSIPipeline(data_dir="data/", llm=..., embedder=..., domain=my_domain)
```

### Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | — | Directory containing `*.json` SSI files |
| `llm` | — | Any `LLMProvider` (e.g. `OrchestraLLM`) |
| `embedder` | — | Any `EmbedProvider` (e.g. `OrchestraEmbedder`) |
| `reranker` | `None` | Optional reranker; `None` = disabled |
| `domain` | `"fx_ssi"` | Domain config: `"fx_ssi"`, `"markets_ssi"`, `"equity_ssi"`, `"repo_ssi"`, `"correspondent_banking"`, or a `DomainConfig` object |
| `chunk_strategy` | `"section_table"` | `"section_table"` \| `"sliding_window"` \| `"paragraph"` |
| `retrieval_mode` | `"dense"` | `"dense"` \| `"sparse"` \| `"hybrid"` |
| `default_k` | `10` | Retrieval depth for normal queries |
| `score_threshold` | `0.15` | Min cosine similarity to pass confidence gate |
| `small_corpus_k` | `50` | Below this child count, skip retrieval and send all context directly |

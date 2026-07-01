"""
SSI RAG Pipeline — scalable, strategy-configurable orchestrator.

Key design choices:
  chunk_strategy   — swap section_table / sliding_window / paragraph at build time
  retrieval_mode   — dense / sparse (BM25) / hybrid (RRF fusion)
  domain           — fx_ssi / equity_ssi / repo_ssi / correspondent_banking / custom
  small_corpus_k   — if children < this, skip retrieval and send all context to LLM
  k auto-scaling   — broad queries ("all", "download") → k = len(children)
  extract_batch()  — structured field extraction for a list of currencies/countries
  evaluate()       — compare chunking strategies on a question set
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from src.preprocessing import preprocess_all, parse_blocks, parent_full_text
from src.chunking import get_strategy, STRATEGIES
from src.domain_config import DomainConfig, FX_SSI, get_domain
from src.embeddings import make_embedder
from src.retrieval import SSIRetriever
from src.validation import confidence_gate, detect_conflicts, check_grounding
from src.llm_layer import LLMClient, build_prompt, build_citation, _GROQ_BASE_URL, _extract_json
from src.preprocessing import build_conflict_map
from src.reranker import make_reranker

import json
from pathlib import Path

_BROAD_QUERY_SIGNALS = {"all", "every", "full", "complete", "entire", "list all", "download"}


def _is_broad_query(question: str) -> bool:
    q = question.lower()
    return any(sig in q for sig in _BROAD_QUERY_SIGNALS)


class SSIPipeline:
    """
    End-to-end SSI RAG pipeline.

    Parameters
    ----------
    data_dir        : directory of Azure DI JSON files
    api_key         : API key (used for both LLM and REST embeddings if configured)
    model           : LLM model identifier sent to the chat/completions endpoint
    base_url        : LLM endpoint base URL; POST {base_url}/chat/completions
                      Default: Groq (https://api.groq.com/openai/v1)
    embed_model     : model name for embeddings
                      • When embed_base_url is None → local sentence-transformers model
                      • When embed_base_url is set  → sent in the REST request body
    embed_base_url  : REST embedding endpoint base; POST {embed_base_url}/embeddings
                      None (default) → use local sentence-transformers (no network call)
    chunk_strategy  : "section_table" | "sliding_window" | "paragraph"
    retrieval_mode  : "dense" | "sparse" | "hybrid"
    domain          : DomainConfig instance or str key ("fx_ssi", "equity_ssi", …)
    default_k       : k for normal queries
    score_threshold : confidence gate threshold (cosine similarity)
    small_corpus_k  : if len(children) < this, skip retrieval → full-context LLM call
    rerank_model    : HuggingFace cross-encoder model name for reranking, or None to skip.
                      Example: "cross-encoder/ms-marco-MiniLM-L-6-v2" (~80 MB, free)
                      When set, the retriever fetches default_k * 5 candidates and the
                      reranker selects the top default_k — significantly improves precision.

    Backwards-compatible aliases
    ----------------------------
    groq_api_key  → api_key
    groq_model    → model
    """

    def __init__(
        self,
        data_dir: str,
        api_key: str = "",
        model: str = "llama-3.3-70b-versatile",
        base_url: str = _GROQ_BASE_URL,
        embed_model: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        embed_provider: Optional[str] = None,
        chunk_strategy: str = "section_table",
        retrieval_mode: str = "dense",
        domain: Optional[object] = None,
        default_k: int = 10,
        score_threshold: float = 0.15,
        small_corpus_k: int = 50,
        rerank_model: Optional[str] = None,
        # --- pre-built client overrides (used by make_self_hosted_pipeline) ---
        _llm=None,
        _embedder=None,
        _reranker=None,
        # --- backwards-compat aliases ---
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.default_k = default_k
        self.score_threshold = score_threshold
        self.small_corpus_k = small_corpus_k
        self.retrieval_mode = retrieval_mode

        # Honour legacy parameter names
        resolved_key = groq_api_key or api_key
        resolved_model = groq_model or model

        # Resolve domain config
        if domain is None:
            self.domain: DomainConfig = FX_SSI
        elif isinstance(domain, str):
            self.domain = get_domain(domain)
        else:
            self.domain = domain

        self._api_key = resolved_key
        self.strategy = get_strategy(chunk_strategy)
        self.embedder = _embedder or make_embedder(
            embed_base_url=embed_base_url,
            api_key=resolved_key,
            embed_model=embed_model,
            provider=embed_provider,
        )
        self.llm = _llm or LLMClient(api_key=resolved_key, model=resolved_model, base_url=base_url)

        self.rerank_model = rerank_model

        self.parents: List[Dict] = []
        self.children: List[Dict] = []
        self.conflict_map: Dict = {}
        self.retriever: Optional[SSIRetriever] = None
        self._reranker = _reranker  # None unless injected by make_self_hosted_pipeline
        self._parent_by_id: Dict = {}
        self._built = False
        self._direct_mode = False  # True when corpus is too small for retrieval

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """
        Preprocess all docs, build embeddings + index. Zero LLM calls.
        If corpus is small (children < small_corpus_k), switches to direct
        mode — the full document text is injected directly into every query.
        """
        json_files = sorted(Path(self.data_dir).glob("*.json"))
        print(f"Preprocessing {len(json_files)} document(s) "
              f"[strategy={self.strategy.name}, mode={self.retrieval_mode}]...")

        id_offset = 0
        for path in json_files:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            doc_name = doc.get("doc_name", path.name)
            blocks = parse_blocks(doc.get("content", ""))
            parents, children = self.strategy.build(blocks, doc_name, id_offset)
            self.parents.extend(parents)
            self.children.extend(children)
            id_offset += len(parents)

        self.conflict_map = build_conflict_map(self.parents)
        if self.conflict_map:
            print(f"  Found {len(self.conflict_map)} cross-document conflict(s): "
                  f"{sorted(self.conflict_map.keys())}")

        if len(self.children) < self.small_corpus_k:
            print(f"  Small corpus ({len(self.children)} chunks < {self.small_corpus_k}) "
                  "— switching to direct-context mode (no retrieval index).")
            self._direct_mode = True
        else:
            self.retriever = SSIRetriever(
                self.embedder, self.parents, self.children, mode=self.retrieval_mode
            )
            self.retriever.build_index()

        self._parent_by_id = {p["id"]: p for p in self.parents}

        if self._reranker is None and self.rerank_model:
            self._reranker = make_reranker(self.rerank_model, api_key=self._api_key)

        self._built = True
        rerank_label = f", Reranker={self.rerank_model}" if self.rerank_model else ""
        print(f"  Ready. Parents={len(self.parents)}, Children={len(self.children)}, "
              f"Conflicts={len(self.conflict_map)}, DirectMode={self._direct_mode}"
              f"{rerank_label}")

    # ------------------------------------------------------------------
    # Query (single)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict:
        """
        Full pipeline for one question.

        k          : retrieval depth; auto-scales for broad queries if None
        max_tokens : LLM output budget; auto-set (4096 broad / 2048 normal) if None
        """
        if not self._built:
            raise RuntimeError("Call pipeline.build() before query().")

        t_start = time.time()
        conflicts = detect_conflicts(question, self.conflict_map)

        # ---- Direct-context mode (small corpus) -------------------------
        if self._direct_mode:
            full_ctx = "\n\n".join(parent_full_text(p) for p in self.parents)
            prompt = (
                f"Context:\n{full_ctx}\n\nQuestion: {question}"
            )
            retrieval = {
                "matched_children": [{"row_text": c["row_text"],
                                       "doc_name": c["doc_name"],
                                       "parent_id": c["parent_id"],
                                       "score": 1.0}
                                      for c in self.children],
                "matched_parents": self.parents,
                "best_score": 1.0,
                "low_confidence": False,
            }
            answer_raw = self.llm.call(
                prompt,
                system_prompt=self.domain.system_prompt,
                max_tokens=4096,
            )
            _parsed = _extract_json(answer_raw)
            _val = _parsed.get("answer", answer_raw)
            if isinstance(_val, str):
                answer = _val
            elif isinstance(_val, dict):
                answer = "\n".join(f"{k}: {v}" for k, v in _val.items())
            else:
                answer = answer_raw
            grounding = check_grounding(answer, full_ctx)
            return {
                "question": question,
                "gate": {"blocked": False, "reason": "direct-context mode"},
                "retrieval": retrieval,
                "conflicts": conflicts,
                "answer": answer,
                "llm_raw": answer_raw,
                "grounding": grounding,
                "citation": build_citation(self.parents),
                "latency_s": round(time.time() - t_start, 2),
                "blocked": False,
                "mode": "direct",
            }

        # ---- Retrieval mode --------------------------------------------
        if k is None:
            k = len(self.children) if _is_broad_query(question) else self.default_k

        # Fetch more candidates when reranking — reranker selects top-k from a
        # wider pool, which dramatically improves precision for structured docs.
        k_retrieve = min(k * 5, len(self.children)) if self._reranker else k

        retrieval = self.retriever.retrieve(
            question, k=k_retrieve, score_threshold=self.score_threshold
        )

        gate = confidence_gate(retrieval, threshold=self.score_threshold)
        if gate["blocked"]:
            return {
                "question": question,
                "gate": gate,
                "retrieval": retrieval,
                "conflicts": [],
                "answer": gate["reason"],
                "grounding": {"passed": True, "grounded": {}, "ungrounded": {}},
                "citation": "",
                "latency_s": round(time.time() - t_start, 2),
                "blocked": True,
                "mode": "retrieval",
            }

        # Apply cross-encoder reranking if configured
        if self._reranker:
            reranked_children = self._reranker.rerank(
                question, retrieval["matched_children"], top_n=k
            )
            # Rebuild matched_parents from reranked children (preserves score order)
            seen: set = set()
            reranked_parents = []
            for ch in reranked_children:
                pid = ch["parent_id"]
                if pid not in seen:
                    seen.add(pid)
                    p = self._parent_by_id.get(pid)
                    if p:
                        reranked_parents.append(p)
            retrieval["matched_children"] = reranked_children
            retrieval["matched_parents"] = reranked_parents

        matched_parents = retrieval["matched_parents"]
        matched_children = retrieval["matched_children"]
        prompt = build_prompt(question, matched_parents, matched_children=matched_children)

        if max_tokens is None:
            max_tokens = 4096 if _is_broad_query(question) else 2048
        answer_raw = self.llm.call(
            prompt,
            system_prompt=self.domain.system_prompt,
            max_tokens=max_tokens,
        )
        _parsed = _extract_json(answer_raw)
        _val = _parsed.get("answer", answer_raw)
        if isinstance(_val, str):
            answer = _val
        elif isinstance(_val, dict):
            answer = "\n".join(f"{k}: {v}" for k, v in _val.items())
        else:
            answer = answer_raw

        full_context = "\n".join(parent_full_text(p) for p in matched_parents)
        grounding = check_grounding(answer, full_context)
        citation = build_citation(matched_parents)

        return {
            "question": question,
            "gate": gate,
            "retrieval": retrieval,
            "conflicts": conflicts,
            "answer": answer,
            "llm_raw": answer_raw,
            "grounding": grounding,
            "citation": citation,
            "latency_s": round(time.time() - t_start, 2),
            "blocked": False,
            "mode": "retrieval",
        }

    # ------------------------------------------------------------------
    # Batch NL queries (parallel-capable)
    # ------------------------------------------------------------------

    def batch_query(
        self,
        questions: List[str],
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_workers: int = 1,
    ) -> List[Dict]:
        """
        Process a list of free-form questions, optionally in parallel.

        max_workers=1  → sequential (safe for Groq free-tier)
        max_workers>1  → parallel threads; use only with inhouse APIs that
                         have no hard TPM limits, or low-latency services.
        """
        if not self._built:
            raise RuntimeError("Call pipeline.build() before batch_query().")
        return self._run_parallel(
            [(q,) for q in questions],
            fn=lambda q: self.query(q, k=k, max_tokens=max_tokens),
            max_workers=max_workers,
            label="batch_query",
        )

    # ------------------------------------------------------------------
    # Structured multi-attribute extraction (currencies / countries / markets)
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        items: List[str],
        fields: Optional[List[str]] = None,
        batch_size: int = 5,
        max_workers: int = 1,
    ) -> List[Dict]:
        """
        Structured field extraction for a list of scope items.

        Parameters
        ----------
        items       : currencies, countries, or markets to look up
                      e.g. ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"]
        fields      : subset of domain fields to extract.
                      None → all domain fields (e.g. all 9 FX SSI fields).
                      e.g. ["SWIFT/BIC", "Account Number"] for a quick BIC sweep.
        batch_size  : how many items are packed into one LLM call.
                      5 works well for 70b; use 3 for 8b to stay within TPM.
        max_workers : parallel LLM threads.
                      1 → sequential (required for Groq free-tier).
                      2+ → parallel (for inhouse APIs with no TPM ceiling).

        Returns
        -------
        List of dicts, one per item:
            {
              "item":     "USD",
              "fields":   {"SWIFT/BIC": "CHASUS33", "Account Number": "...", ...},
              "answer":   "<raw LLM text>",
              "conflicts": [...],
              "grounding": {...},
              "latency_s": 1.2,
              "blocked":  False,
            }

        Example — all 9 FX SSI fields for 6 currencies in parallel::

            pipeline = SSIPipeline(..., domain="fx_ssi")
            pipeline.build()
            rows = pipeline.extract_batch(
                ["USD", "EUR", "GBP", "JPY", "CHF", "CAD"],
                max_workers=2,
            )
            for r in rows:
                print(r["item"], r["fields"])

        Example — only BIC + Account for 10 currencies::

            rows = pipeline.extract_batch(
                ["USD","EUR","GBP","JPY","CHF","CAD","AUD","SEK","NOK","DKK"],
                fields=["SWIFT/BIC", "Account Number"],
            )
        """
        if not self._built:
            raise RuntimeError("Call pipeline.build() before extract_batch().")

        active_fields = fields or self.domain.fields

        def _run_one_batch(batch: List[str]) -> List[Dict]:
            question = _build_field_question(batch, self.domain.scope_label, active_fields)
            # Pass k explicitly so broad-query heuristic (triggered by words like
            # "all"/"every" that may appear in the generated question) does not
            # over-retrieve and blow up the context window.
            result = self.query(question, k=self.default_k, max_tokens=512)
            # For extraction queries query() returns the raw JSON string in "answer"
            # (since extraction JSON has no "answer" key). Parse it directly.
            llm_raw = result.get("llm_raw", result.get("answer", ""))
            data = _extract_json(llm_raw)

            out = []
            for item in batch:
                item_data = data.get(item)
                if isinstance(item_data, dict):
                    parsed = {f: item_data.get(f, "N/A") for f in active_fields}
                else:
                    # Item absent from JSON response — LLM signalled it is not in corpus
                    parsed = {f: "not found" for f in active_fields}

                out.append({
                    "item": item,
                    "fields": parsed,
                    "answer": llm_raw,
                    "conflicts": result["conflicts"],
                    "grounding": result["grounding"],
                    "citation": result["citation"],
                    "latency_s": result["latency_s"],
                    "blocked": result["blocked"],
                })
            return out

        batches = [items[i: i + batch_size] for i in range(0, len(items), batch_size)]
        batch_results = self._run_parallel(
            [(b,) for b in batches],
            fn=lambda b: _run_one_batch(b),
            max_workers=max_workers,
            label="extract_batch",
        )

        # Flatten list-of-lists back to flat list in original item order
        flat: List[Dict] = []
        for br in batch_results:
            flat.extend(br)
        return flat

    # ------------------------------------------------------------------
    # Internal parallel runner (shared by batch_query + extract_batch)
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        arg_tuples: List[tuple],
        fn,
        max_workers: int,
        label: str,
    ) -> List:
        """Run fn(*args) for each args in arg_tuples, preserving input order."""
        n = len(arg_tuples)
        results = [None] * n

        if max_workers == 1:
            for i, args in enumerate(arg_tuples):
                print(f"  [{label} {i+1}/{n}] {str(args[0])[:60]}...")
                results[i] = fn(*args)
        else:
            print(f"  [{label}] spawning {max_workers} workers for {n} task(s)...")
            sem = threading.Semaphore(max_workers)

            def _wrapped(i, args):
                with sem:
                    return i, fn(*args)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_wrapped, i, args): i for i, args in enumerate(arg_tuples)}
                for fut in as_completed(futures):
                    i, result = fut.result()
                    results[i] = result

        return results

    # ------------------------------------------------------------------
    # Strategy evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        questions: List[str],
        strategies: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compare chunking strategies on a fixed question set without calling the LLM.
        Returns retrieval metrics only (no LLM cost).

        metrics can include: "best_score", "num_parents", "num_children", "num_docs"

        Example::

            report = pipeline.evaluate(
                questions=["What is the CHF SWIFT code?"],
                strategies=["section_table", "sliding_window", "paragraph"],
            )
            for strategy, rows in report.items():
                for row in rows:
                    print(strategy, row)
        """
        if not self._built:
            raise RuntimeError("Call pipeline.build() first.")

        target_strategies = strategies or list(STRATEGIES.keys())
        metrics = metrics or ["best_score", "num_parents", "num_children", "num_docs"]
        report: Dict[str, List[Dict]] = {}

        for strat_name in target_strategies:
            strat = get_strategy(strat_name)
            strat_parents: List[Dict] = []
            strat_children: List[Dict] = []
            id_offset = 0

            for path in sorted(Path(self.data_dir).glob("*.json")):
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                blocks = parse_blocks(doc.get("content", ""))
                p, c = strat.build(blocks, doc.get("doc_name", path.name), id_offset)
                strat_parents.extend(p)
                strat_children.extend(c)
                id_offset += len(p)

            retriever = SSIRetriever(
                self.embedder, strat_parents, strat_children, mode="dense"
            )
            retriever.build_index()

            rows = []
            for q in questions:
                k = min(self.default_k, len(strat_children))
                r = retriever.retrieve(q, k=k, score_threshold=self.score_threshold)
                row = {"question": q[:60]}
                if "best_score" in metrics:
                    row["best_score"] = round(r["best_score"], 4)
                if "num_parents" in metrics:
                    row["num_parents"] = len(r["matched_parents"])
                if "num_children" in metrics:
                    row["num_children"] = len(r["matched_children"])
                if "num_docs" in metrics:
                    row["num_docs"] = len({c["doc_name"] for c in r["matched_children"]})
                rows.append(row)

            report[strat_name] = rows

        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_field_question(
    items: List[str],
    scope_label: str,
    fields: List[str],
) -> str:
    """
    Build an extraction question for any combination of items × fields.

    items       : ["USD", "EUR", "GBP"]
    scope_label : "currency"
    fields      : ["SWIFT/BIC", "Account Number", "Correspondent Bank"]

    Requests a JSON response where each key is an item name and each value
    is a dict of the requested fields — no hardcoded field names in the pipeline.
    """
    items_str = ", ".join(items)
    fields_str = ", ".join(f'"{f}"' for f in fields)
    null_entry = "{" + ", ".join(f'"{f}": "..."' for f in fields) + "}"
    schema_example = "{" + ", ".join(f'"{it}": {null_entry}' for it in items) + "}"
    return (
        f"For each of the following {scope_label}(s): {items_str} — "
        f"extract these fields: {', '.join(fields)}. "
        f"Respond with a single valid JSON object following this exact schema:\n"
        f"{schema_example}\n"
        f'Missing field value: "N/A". '
        f'{scope_label.title()} not in the context: set its field values to "not found".'
    )


def _extract_item_section(answer: str, item: str) -> str:
    """
    Pull the block in `answer` that discusses `item`.
    Tries double-newline boundaries first, then single-newline headings.
    Returns empty string when not found (caller falls back to full answer).
    """
    item_upper = item.upper()
    for block in answer.split("\n\n"):
        if item_upper in block.upper():
            return block.strip()
    for block in answer.split("\n"):
        if block.upper().startswith(item_upper):
            return block.strip()
    return ""


def _norm_field_key(s: str) -> str:
    """
    Normalise a field name or LLM-output key for fuzzy matching.

    Strips "SWIFT/" so that "Correspondent BIC" and "Correspondent SWIFT/BIC"
    both normalise to "CORRESPONDENT BIC", enabling the parser to map context
    labels like "Correspondent BIC:" to the structured field "Correspondent SWIFT/BIC".
    Also strips " CODE" so "SWIFT Code" normalises to "BIC" alongside "SWIFT/BIC".
    """
    return s.upper().replace("SWIFT/", "").replace(" CODE", "").strip("*- ").strip()


def _parse_fields(text: str, fields: List[str]) -> Dict[str, str]:
    """
    Parse "Field: Value" lines from an LLM answer block into a dict.

    Matching (in priority order):
    1. Exact normalised match.
    2. Context key is a shortened form of the field name (key_norm in fn).
       Direction matters: "ACCOUNT" matches "ACCOUNT NUMBER" but
       "CORRESPONDENT BIC" does NOT accidentally match the shorter "BIC" field.

    First occurrence of each field wins (primary/agency path for multi-path answers).
    Lines whose value is blank or literally "N/A" are skipped so they cannot
    overwrite a real value found earlier in the same answer.
    """
    result: Dict[str, str] = {f: "N/A" for f in fields}
    norms = {f: _norm_field_key(f) for f in fields}

    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key_raw, _, val = line.partition(":")
        key_raw = key_raw.strip().lstrip("*- ").rstrip("*")
        val = val.strip()
        if not val or val.upper() == "N/A":
            continue  # skip blank / explicit N/A — don't overwrite a real value
        key_norm = _norm_field_key(key_raw)
        for field in fields:
            fn = norms[field]
            # key_norm in fn: context used a short form ("Account") of the full field name ("Account Number")
            # fn == key_norm: exact match after normalisation (covers "Correspondent BIC" → "Correspondent SWIFT/BIC")
            if key_norm and (fn == key_norm or key_norm in fn):
                if result[field] == "N/A":
                    result[field] = val
                break
    return result



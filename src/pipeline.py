"""
SSI RAG Pipeline — provider-agnostic orchestrator.

The pipeline is completely decoupled from LLM/embedding technology.
Pass any LLMProvider + EmbedProvider from src.providers, and the rest
of the logic (chunking, retrieval, grounding, extraction) stays the same.

Key knobs
---------
  chunk_strategy  "section_table" | "sliding_window" | "paragraph"
  retrieval_mode  "dense" | "sparse" | "hybrid"
  domain          fx_ssi | equity_ssi | repo_ssi | markets_ssi | custom
  default_k       k for retrieval; broad queries auto-scale to full corpus

Quick-start
-----------
    from src.providers import GroqLLM, LocalEmbedder
    from src.pipeline import SSIPipeline

    pipeline = SSIPipeline(
        data_dir = "data/ssi/",
        llm      = GroqLLM(api_key="gsk_..."),
        embedder = LocalEmbedder(),
        domain   = "fx_ssi",
    )
    pipeline.build()
    result = pipeline.query("What is the CHF SWIFT code?")
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from src.preprocessing import parse_blocks, parent_full_text, build_conflict_map
from src.chunking import get_strategy, STRATEGIES
from src.domain_config import DomainConfig, FX_SSI, get_domain
from src.retrieval import SSIRetriever
from src.validation import confidence_gate, detect_conflicts, check_grounding
from src.providers.base import LLMProvider, EmbedProvider
from src.providers._rest import _extract_json
from src.prompt import render_rag_prompt, render_citation

_BROAD_QUERY_SIGNALS = {"all", "every", "full", "complete", "entire", "list all", "download"}


def _is_broad_query(question: str) -> bool:
    q = question.lower()
    return any(sig in q for sig in _BROAD_QUERY_SIGNALS)


class SSIPipeline:
    """
    End-to-end SSI RAG pipeline.

    Parameters
    ----------
    data_dir       : directory of Azure DI JSON files to index
    llm            : any LLMProvider (GroqLLM, OrchestraLLM, …)
    embedder       : any EmbedProvider (LocalEmbedder, JinaEmbedder, OrchestraEmbedder, …)
    reranker       : optional reranker (JinaReranker, LocalReranker, or None to skip)
    domain         : DomainConfig instance or str key ("fx_ssi", "markets_ssi", …)
    chunk_strategy : "section_table" | "sliding_window" | "paragraph"
    retrieval_mode : "dense" | "sparse" | "hybrid"
    default_k      : retrieval depth for normal queries
    score_threshold: confidence gate; queries below this return "low confidence"
    small_corpus_k : corpus smaller than this → skip index, send all context to LLM

    Example
    -------
    from src.providers import GroqLLM, JinaEmbedder, JinaReranker

    pipeline = SSIPipeline(
        data_dir = "data/ssi/",
        llm      = GroqLLM(api_key=os.getenv("GROQ_API_KEY")),
        embedder = JinaEmbedder(api_key=os.getenv("JINA_API_KEY")),
        reranker = JinaReranker(api_key=os.getenv("JINA_API_KEY")),
        domain   = "markets_ssi",
    )
    pipeline.build()
    """

    def __init__(
        self,
        data_dir: str,
        llm: LLMProvider,
        embedder: EmbedProvider,
        reranker=None,
        domain=None,
        chunk_strategy: str = "section_table",
        retrieval_mode: str = "dense",
        default_k: int = 10,
        score_threshold: float = 0.15,
        small_corpus_k: int = 50,
    ):
        self.data_dir = data_dir
        self.llm = llm
        self.embedder = embedder
        self._reranker = reranker
        self.default_k = default_k
        self.score_threshold = score_threshold
        self.small_corpus_k = small_corpus_k
        self.retrieval_mode = retrieval_mode
        self.strategy = get_strategy(chunk_strategy)

        if domain is None:
            self.domain: DomainConfig = FX_SSI
        elif isinstance(domain, str):
            self.domain = get_domain(domain)
        else:
            self.domain = domain

        self.parents: List[Dict] = []
        self.children: List[Dict] = []
        self.conflict_map: Dict = {}
        self.retriever: Optional[SSIRetriever] = None
        self._parent_by_id: Dict = {}
        self._built = False
        self._direct_mode = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self):
        """
        Index all documents. Zero LLM calls.

        Switches to direct-context mode automatically when corpus is small
        (children < small_corpus_k) — bypasses retrieval and sends all text
        to the LLM directly.
        """
        json_files = sorted(Path(self.data_dir).glob("*.json"))
        print(
            f"Preprocessing {len(json_files)} document(s) "
            f"[strategy={self.strategy.name}, mode={self.retrieval_mode}]..."
        )

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
            print(
                f"  Found {len(self.conflict_map)} cross-document conflict(s): "
                f"{sorted(self.conflict_map.keys())}"
            )

        if len(self.children) < self.small_corpus_k:
            print(
                f"  Small corpus ({len(self.children)} chunks < {self.small_corpus_k}) "
                "— switching to direct-context mode (no retrieval index)."
            )
            self._direct_mode = True
        else:
            self.retriever = SSIRetriever(
                self.embedder, self.parents, self.children, mode=self.retrieval_mode
            )
            self.retriever.build_index()

        self._parent_by_id = {p["id"]: p for p in self.parents}
        self._built = True

        rerank_label = f", Reranker={type(self._reranker).__name__}" if self._reranker else ""
        print(
            f"  Ready. Parents={len(self.parents)}, Children={len(self.children)}, "
            f"Conflicts={len(self.conflict_map)}, DirectMode={self._direct_mode}"
            f"{rerank_label}"
        )

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

        Parameters
        ----------
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
            prompt = f"Context:\n{full_ctx}\n\nQuestion: {question}"
            retrieval = {
                "matched_children": [
                    {"row_text": c["row_text"], "doc_name": c["doc_name"],
                     "parent_id": c["parent_id"], "score": 1.0}
                    for c in self.children
                ],
                "matched_parents": self.parents,
                "best_score": 1.0,
                "low_confidence": False,
            }
            answer_raw = self.llm.complete(
                prompt, system=self.domain.system_prompt, max_tokens=4096
            )
            answer = _parse_llm_answer(answer_raw)
            return {
                "question": question,
                "gate": {"blocked": False, "reason": "direct-context mode"},
                "retrieval": retrieval,
                "conflicts": conflicts,
                "answer": answer,
                "llm_raw": answer_raw,
                "grounding": check_grounding(answer, full_ctx),
                "citation": render_citation(self.parents),
                "latency_s": round(time.time() - t_start, 2),
                "blocked": False,
                "mode": "direct",
            }

        # ---- Retrieval mode --------------------------------------------
        if k is None:
            k = len(self.children) if _is_broad_query(question) else self.default_k

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

        if self._reranker:
            reranked = self._reranker.rerank(
                question, retrieval["matched_children"], top_n=k
            )
            seen: set = set()
            reranked_parents = []
            for ch in reranked:
                pid = ch["parent_id"]
                if pid not in seen:
                    seen.add(pid)
                    p = self._parent_by_id.get(pid)
                    if p:
                        reranked_parents.append(p)
            retrieval["matched_children"] = reranked
            retrieval["matched_parents"] = reranked_parents

        matched_parents = retrieval["matched_parents"]
        matched_children = retrieval["matched_children"]

        prompt = render_rag_prompt(question, matched_parents, matched_children)
        if max_tokens is None:
            max_tokens = 4096 if _is_broad_query(question) else 2048

        answer_raw = self.llm.complete(
            prompt, system=self.domain.system_prompt, max_tokens=max_tokens
        )
        answer = _parse_llm_answer(answer_raw)
        full_context = "\n".join(parent_full_text(p) for p in matched_parents)

        return {
            "question": question,
            "gate": gate,
            "retrieval": retrieval,
            "conflicts": conflicts,
            "answer": answer,
            "llm_raw": answer_raw,
            "grounding": check_grounding(answer, full_context),
            "citation": render_citation(matched_parents),
            "latency_s": round(time.time() - t_start, 2),
            "blocked": False,
            "mode": "retrieval",
        }

    # ------------------------------------------------------------------
    # Batch free-form queries
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
        max_workers>1  → parallel threads (for inhouse APIs without hard TPM limits)
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
    # Structured multi-attribute extraction
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
        items       : currencies, countries, or markets — e.g. ["USD", "EUR"]
        fields      : subset of domain fields; None → all domain fields
        batch_size  : items per LLM call (5 for 70b; 3 for 8b)
        max_workers : parallel threads (1 = sequential, required for Groq free-tier)

        Returns
        -------
        List of dicts, one per item:
            {"item": "USD", "fields": {...}, "answer": "<raw>",
             "conflicts": [...], "grounding": {...}, "latency_s": 1.2, "blocked": False}
        """
        if not self._built:
            raise RuntimeError("Call pipeline.build() before extract_batch().")

        active_fields = fields or self.domain.fields

        def _run_one_batch(batch: List[str]) -> List[Dict]:
            question = _build_field_question(batch, self.domain.scope_label, active_fields)
            result = self.query(question, k=self.default_k, max_tokens=512)
            llm_raw = result.get("llm_raw", result.get("answer", ""))
            data = _extract_json(llm_raw)

            out = []
            for item in batch:
                item_data = data.get(item)
                if isinstance(item_data, dict):
                    parsed = {f: item_data.get(f, "N/A") for f in active_fields}
                else:
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
            fn=_run_one_batch,
            max_workers=max_workers,
            label="extract_batch",
        )

        flat: List[Dict] = []
        for br in batch_results:
            flat.extend(br)
        return flat

    # ------------------------------------------------------------------
    # Strategy evaluation (retrieval metrics only, no LLM calls)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        questions: List[str],
        strategies: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict:
        """
        Compare chunking strategies on a fixed question set.

        Returns retrieval metrics only (no LLM cost).
        metrics can include: "best_score", "num_parents", "num_children", "num_docs"
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

            retriever = SSIRetriever(self.embedder, strat_parents, strat_children, mode="dense")
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_parallel(self, arg_tuples, fn, max_workers, label) -> List:
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


# ---------------------------------------------------------------------------
# Pipeline-internal helpers
# ---------------------------------------------------------------------------

def _parse_llm_answer(raw: str) -> str:
    """Extract the human-readable answer from a JSON LLM response."""
    parsed = _extract_json(raw)
    val = parsed.get("answer", raw)
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return "\n".join(f"{k}: {v}" for k, v in val.items())
    return raw


def _build_field_question(items: List[str], scope_label: str, fields: List[str]) -> str:
    """
    Build a structured extraction question that requests JSON output.

    The question itself avoids words like "all" and "every" that would
    trigger the broad-query heuristic and blow up retrieval context.
    """
    items_str = ", ".join(items)
    null_entry = "{" + ", ".join(f'"{f}": "..."' for f in fields) + "}"
    schema = "{" + ", ".join(f'"{it}": {null_entry}' for it in items) + "}"
    return (
        f"For each of the following {scope_label}(s): {items_str} — "
        f"extract these fields: {', '.join(fields)}. "
        f"Respond with a single valid JSON object following this exact schema:\n{schema}\n"
        f'Missing field value: "N/A". '
        f'{scope_label.title()} not in the context: set its field values to "not found".'
    )

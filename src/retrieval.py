"""
Retrieval backends for the SSI pipeline.

Three modes:
  "dense"   — FAISS cosine similarity (default, best for semantic queries)
  "sparse"  — BM25 keyword-based retrieval (best for exact code/term lookups)
  "hybrid"  — Reciprocal Rank Fusion of dense + sparse (best of both worlds)

BM25 requires rank-bm25: pip install rank-bm25
Falls back to numpy brute-force cosine if faiss-cpu is unavailable.
"""

import math
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    print("[WARNING] faiss-cpu not found — using numpy brute-force cosine search.")
    _FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dense index helpers
# ---------------------------------------------------------------------------

def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def build_faiss_index(vectors: np.ndarray):
    normed = _l2_normalize(vectors.astype(np.float32))
    dim = normed.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normed)
    return index, normed


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_merge(
    ranked_lists: List[List[int]],
    scores_lists: List[List[float]],
    k_rrf: int = 60,
) -> List[Dict]:
    """
    Combine multiple ranked lists via Reciprocal Rank Fusion.

    Returns list of {"idx": child_idx, "score": rrf_score} sorted descending.
    The rrf_score is the sum of 1/(k + rank) across all retrievers.
    """
    rrf: Dict[int, float] = {}
    for ranks, scores in zip(ranked_lists, scores_lists):
        for rank, idx in enumerate(ranks):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)

    return sorted(
        [{"idx": idx, "score": score} for idx, score in rrf.items()],
        key=lambda x: -x["score"],
    )


# ---------------------------------------------------------------------------
# Main retriever
# ---------------------------------------------------------------------------

class SSIRetriever:
    """
    Unified retriever supporting dense, sparse (BM25), and hybrid modes.

    Sparse and hybrid modes require: pip install rank-bm25
    """

    def __init__(
        self,
        embedder,
        parents: List[Dict],
        children: List[Dict],
        mode: str = "dense",
    ):
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"Unknown retrieval mode '{mode}'. Use: dense, sparse, hybrid.")
        if mode in ("sparse", "hybrid") and not _BM25_AVAILABLE:
            raise ImportError("BM25 requires: pip install rank-bm25")

        self.embedder = embedder
        self.parents = parents
        self.children = children
        self.mode = mode
        self._parent_by_id = {p["id"]: p for p in parents}
        self._index = None
        self._child_vectors: Optional[np.ndarray] = None
        self._bm25: Optional["BM25Okapi"] = None

    def build_index(self):
        texts = [c["embed_text"] for c in self.children]
        print(f"  Embedding {len(texts)} chunks ({self.mode} mode)...")

        if self.mode in ("dense", "hybrid"):
            vectors = self.embedder.embed(texts, task="retrieval.passage")
            if _FAISS_AVAILABLE:
                self._index, self._child_vectors = build_faiss_index(vectors)
            else:
                self._child_vectors = _l2_normalize(vectors.astype(np.float32))

        if self.mode in ("sparse", "hybrid"):
            tokenized = [t.lower().split() for t in texts]
            self._bm25 = BM25Okapi(tokenized)

    def _dense_ranked(self, query: str, k: int):
        """Returns (indices, scores) sorted by descending similarity."""
        q_vec = self.embedder.embed_query(query).astype(np.float32)
        q_norm = _l2_normalize(q_vec.reshape(1, -1))
        actual_k = min(k, len(self.children))

        if _FAISS_AVAILABLE and self._index is not None:
            scores, indices = self._index.search(q_norm, actual_k)
            return indices[0].tolist(), scores[0].tolist()
        else:
            sims = (self._child_vectors @ q_norm.T).flatten()
            top = np.argsort(-sims)[:actual_k]
            return top.tolist(), sims[top].tolist()

    def _sparse_ranked(self, query: str, k: int):
        """Returns (indices, scores) sorted by descending BM25 score."""
        tokenized_q = query.lower().split()
        raw_scores = self._bm25.get_scores(tokenized_q)
        actual_k = min(k, len(self.children))
        top = np.argsort(-raw_scores)[:actual_k]
        return top.tolist(), raw_scores[top].tolist()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.15,
    ) -> Dict:
        """
        Search for top-k children, deduplicate to parent level.

        Returns:
            matched_children  — child hits with scores
            matched_parents   — unique parents ordered by best child score
            best_score        — top child score (dense cosine or RRF; not BM25 raw)
            low_confidence    — True if best_score < score_threshold (dense/hybrid only)
        """
        if self.mode == "dense":
            indices, scores = self._dense_ranked(query, k)
            hits = [{"idx": i, "score": s} for i, s in zip(indices, scores)]
            best_raw = scores[0] if scores else 0.0

        elif self.mode == "sparse":
            indices, scores = self._sparse_ranked(query, k)
            # Normalize BM25 score to [0,1] for gate compatibility
            max_score = max(scores) if scores else 1.0
            norm_scores = [s / max_score if max_score > 0 else 0.0 for s in scores]
            hits = [{"idx": i, "score": s} for i, s in zip(indices, norm_scores)]
            best_raw = norm_scores[0] if norm_scores else 0.0

        else:  # hybrid
            dense_idx, dense_scores = self._dense_ranked(query, k * 2)
            sparse_idx, _ = self._sparse_ranked(query, k * 2)
            hits = _rrf_merge([dense_idx, sparse_idx], [dense_scores, [0.0] * len(sparse_idx)])
            hits = hits[:k]
            # Use the dense score of the top-ranked hit for confidence gate
            if hits and dense_scores:
                best_raw = dense_scores[0]
            else:
                best_raw = 0.0

        matched_children = []
        for hit in hits:
            idx = hit["idx"]
            if 0 <= idx < len(self.children):
                child = self.children[idx]
                matched_children.append({
                    "score": hit["score"],
                    "row_text": child["row_text"],
                    "doc_name": child["doc_name"],
                    "parent_id": child["parent_id"],
                })

        seen_parent_ids: set = set()
        matched_parents: List[Dict] = []
        for ch in sorted(matched_children, key=lambda x: -x["score"]):
            pid = ch["parent_id"]
            if pid not in seen_parent_ids:
                seen_parent_ids.add(pid)
                parent = self._parent_by_id.get(pid)
                if parent:
                    matched_parents.append(parent)

        return {
            "matched_children": matched_children,
            "matched_parents": matched_parents,
            "best_score": float(best_raw),
            "low_confidence": float(best_raw) < score_threshold,
        }

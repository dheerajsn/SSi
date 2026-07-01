"""
Reranking backends.

Two modes:
  CrossEncoderReranker — local cross-encoder via sentence-transformers.
                         OPTIONAL: requires `pip install sentence-transformers`.
                         ~80 MB model download, cached locally. No API key.

  JinaReranker         — Jina AI Reranker API (https://jina.ai).
                         No local model download required.
                         Requires a Jina AI API key.

After initial vector/BM25 retrieval returns k candidates, the reranker scores
each (query, chunk) pair jointly — far more accurate than cosine similarity
because the model sees query and document together, not as independent embeddings.

Factory
-------
    make_reranker(None)                             # disabled (returns None)
    make_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")  # CrossEncoderReranker
    make_reranker("jina-reranker-v2-base-multilingual", api_key="jina_...")  # JinaReranker
"""

from typing import List, Dict, Optional

import requests

_DEFAULT_LOCAL_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_JINA_BASE_URL = "https://api.jina.ai/v1"
_DEFAULT_JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


# ---------------------------------------------------------------------------
# Local cross-encoder — OPTIONAL dependency
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for passage reranking.

    OPTIONAL: sentence-transformers is not installed by default.
    Install with: pip install sentence-transformers

    Parameters
    ----------
    model_name : HuggingFace cross-encoder model ID
    """

    def __init__(self, model_name: str = _DEFAULT_LOCAL_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local reranking.\n"
                "Install with: pip install sentence-transformers\n"
                "Or use JinaReranker to rerank via API without a local download."
            )
        print(f"  Loading reranker '{model_name}' (downloads once, cached locally)...")
        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        children: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Score and reorder children by (query, row_text) relevance.

        Parameters
        ----------
        query    : the user's question
        children : list of child dicts from SSIRetriever (must have "row_text")
        top_n    : number of children to return after reranking

        Returns
        -------
        Top top_n children sorted by reranker score descending.
        Each child dict gets a "rerank_score" key added.
        """
        if not children:
            return children

        pairs = [(query, c["row_text"]) for c in children]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores.tolist(), children), key=lambda x: -x[0])
        result = []
        for score, child in ranked[:top_n]:
            out = dict(child)
            out["rerank_score"] = round(float(score), 4)
            result.append(out)
        return result


# ---------------------------------------------------------------------------
# Jina AI Reranker — REST, no local download
# ---------------------------------------------------------------------------

class JinaReranker:
    """
    Jina AI Reranker API — no local model download required.

    Uses Jina's cross-encoder reranking API to score (query, document) pairs.
    Supports multilingual content out of the box.

    Parameters
    ----------
    api_key    : Jina AI API key (https://jina.ai — free tier available)
    model      : Jina reranker model (default: jina-reranker-v2-base-multilingual)
    timeout    : per-request timeout in seconds
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_JINA_RERANK_MODEL,
        timeout: int = 30,
    ):
        if not api_key:
            raise ValueError("api_key is required for JinaReranker")
        self.endpoint = _JINA_BASE_URL + "/rerank"
        self.model = model
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def rerank(
        self,
        query: str,
        children: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Rerank children via Jina AI Reranker API.

        Parameters
        ----------
        query    : the user's question
        children : list of child dicts (must have "row_text")
        top_n    : number of children to return after reranking

        Returns
        -------
        Top top_n children sorted by relevance score descending.
        Each child dict gets a "rerank_score" key added.
        """
        if not children:
            return children

        documents = [c["row_text"] for c in children]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        resp = requests.post(
            self.endpoint, headers=self._headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()

        # Jina response: {"results": [{"index": 0, "relevance_score": 0.95, ...}]}
        results = resp.json().get("results", [])
        out = []
        for item in results:
            child = dict(children[item["index"]])
            child["rerank_score"] = round(float(item["relevance_score"]), 4)
            out.append(child)
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_reranker(
    model_name: Optional[str],
    api_key: Optional[str] = None,
) -> "Optional[CrossEncoderReranker | JinaReranker]":
    """
    Return a reranker for the given model name, or None to disable reranking.

    model_name starting with "jina-" → JinaReranker (api_key required)
    any other model_name             → CrossEncoderReranker (sentence-transformers)
    None or ""                       → None (reranking disabled)

    Examples
    --------
    make_reranker(None)                                       # disabled
    make_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")    # local cross-encoder
    make_reranker("jina-reranker-v2-base-multilingual",
                  api_key="jina_...")                         # Jina API, no download
    """
    if not model_name:
        return None
    if model_name.startswith("jina-"):
        if not api_key:
            raise ValueError(
                f"api_key is required for JinaReranker (model='{model_name}'). "
                "Pass api_key to make_reranker() or set JINA_API_KEY."
            )
        return JinaReranker(api_key=api_key, model=model_name)
    return CrossEncoderReranker(model_name)

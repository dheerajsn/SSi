"""
Local providers — sentence-transformers for embeddings and reranking.

OPTIONAL dependency: sentence-transformers is not installed by default.
Install with: pip install sentence-transformers

Both classes lazy-import the library so importing this module is free.
The ImportError with a clear install hint is only raised when you actually
instantiate the class — not on import.

If you cannot or do not want to install sentence-transformers, use:
  - JinaEmbedder  for embeddings (API-based, no download)
  - JinaReranker  for reranking  (API-based, no download)
  - OrchestraEmbedder for embeddings via inhouse gateway
"""

from .base import EmbedProvider

_DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_INSTALL_MSG = (
    "sentence-transformers is not installed.\n"
    "  pip install sentence-transformers\n"
    "Alternatively, use JinaEmbedder / OrchestraEmbedder for API-based "
    "embeddings that require no local download."
)


class LocalEmbedder(EmbedProvider):
    """
    Local CPU/GPU embeddings via sentence-transformers.

    Model is downloaded once on first instantiation and cached by HuggingFace.
    No API key or network traffic required after the one-time download.

    Parameters
    ----------
    model : HuggingFace model ID (default: BAAI/bge-small-en-v1.5, ~130 MB)
    """

    def __init__(self, model: str = _DEFAULT_EMBED_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(_INSTALL_MSG)
        print(f"  Loading embedding model '{model}' (downloads once, cached)...")
        self._model = SentenceTransformer(model)
        self.model = model

    def embed(self, texts, task="retrieval.passage"):
        import numpy as np
        if not texts:
            raise ValueError("texts list is empty")
        inputs = [_BGE_QUERY_PREFIX + t for t in texts] if task == "retrieval.query" else texts
        vecs = self._model.encode(
            inputs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)


class LocalReranker:
    """
    Local cross-encoder reranker via sentence-transformers.

    Model is downloaded once on first instantiation and cached by HuggingFace.
    Significantly more accurate than cosine similarity because the model
    sees query and document jointly rather than as independent embeddings.

    Parameters
    ----------
    model : HuggingFace cross-encoder model ID
            Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB)
    """

    def __init__(self, model: str = _DEFAULT_RERANK_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(_INSTALL_MSG)
        print(f"  Loading reranker '{model}' (downloads once, cached)...")
        self._model = CrossEncoder(model)
        self.model = model

    def rerank(self, query: str, children: list, top_n: int = 5) -> list:
        """
        Rerank children by (query, row_text) relevance using the local cross-encoder.

        Parameters
        ----------
        query    : the user's question
        children : list of dicts with "row_text" key
        top_n    : how many to return

        Returns
        -------
        Top top_n children sorted by score descending, each with "rerank_score" added.
        """
        if not children:
            return children
        pairs = [(query, c["row_text"]) for c in children]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores.tolist(), children), key=lambda x: -x[0])
        return [
            {**child, "rerank_score": round(float(score), 4)}
            for score, child in ranked[:top_n]
        ]

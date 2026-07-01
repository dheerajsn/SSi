"""
Embedding backends.

Three modes:
  LocalEmbedder   — sentence-transformers, runs fully on-device.
                    OPTIONAL: requires `pip install sentence-transformers`.
                    No API key, no network traffic.

  RestEmbedder    — OpenAI-compatible REST endpoint.
                    Calls POST {base_url}{embed_path} (default: /embeddings).
                    Works with any provider that follows the OpenAI embeddings schema,
                    including self-hosted deployments at e.g. https://ai.work.co/v2.

  JinaEmbedder    — Jina AI Embeddings API (https://jina.ai).
                    No local model download required.
                    Supports task-aware encoding (passage vs query).
                    Requires a Jina AI API key.

All classes expose the same interface:
    embedder.embed(texts, task)  → np.ndarray  shape (N, dim), L2-normalised
    embedder.embed_query(text)   → np.ndarray  shape (dim,), L2-normalised

Factory
-------
    make_embedder()                               # LocalEmbedder (sentence-transformers)
    make_embedder(embed_base_url="...")            # RestEmbedder
    make_embedder(provider="jina", api_key="...")  # JinaEmbedder (no local download)
"""

from typing import List, Optional

import numpy as np
import requests

_DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_JINA_BASE_URL = "https://api.jina.ai/v1"
_DEFAULT_JINA_EMBED_MODEL = "jina-embeddings-v3"


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


# ---------------------------------------------------------------------------
# Local (sentence-transformers) — OPTIONAL dependency
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    CPU/GPU local embeddings via sentence-transformers.

    OPTIONAL: sentence-transformers is not installed by default.
    Install with: pip install sentence-transformers

    No API key or network traffic required after the one-time model download.
    """

    def __init__(self, model_name: str = _DEFAULT_LOCAL_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings.\n"
                "Install with: pip install sentence-transformers\n"
                "Or use a REST-based embedder (RestEmbedder / JinaEmbedder) to avoid the download."
            )
        print(f"  Loading embedding model '{model_name}' (downloads once, cached locally)...")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        if not texts:
            raise ValueError("texts list is empty")
        inputs = [_BGE_QUERY_PREFIX + t for t in texts] if task == "retrieval.query" else texts
        vectors = self._model.encode(
            inputs,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], task="retrieval.query")[0]


# ---------------------------------------------------------------------------
# REST (OpenAI-compatible /embeddings endpoint)
# ---------------------------------------------------------------------------

class RestEmbedder:
    """
    REST embedding client for any OpenAI-compatible embeddings endpoint.

    Parameters
    ----------
    base_url   : base URL, e.g. "https://ai.mywork.internal/v2"
    api_key    : Bearer token / API key
    model      : model identifier sent in the request body
    batch_size : texts per HTTP request (stay within provider limits)
    timeout    : per-request timeout in seconds
    embed_path : path appended to base_url (default: /embeddings)
                 Override for non-standard paths, e.g. "/v2/embeddings"

    Example
    -------
    embedder = RestEmbedder(
        base_url="https://ai.mywork.internal",
        api_key="sk-...",
        model="text-embedding-large",
        embed_path="/v2/embeddings",
    )
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 256,
        timeout: int = 60,
        embed_path: str = "/embeddings",
    ):
        self.endpoint = base_url.rstrip("/") + embed_path
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        """
        Embed a list of texts via REST.
        task is accepted for interface compatibility but not sent to the API
        (passage vs query distinction is typically handled server-side or ignored).
        """
        if not texts:
            raise ValueError("texts list is empty")

        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(
                self.endpoint, headers=self._headers, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI schema: {"data": [{"embedding": [...], "index": N}, ...]}
            batch_vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            all_vectors.extend(batch_vecs)

        matrix = np.array(all_vectors, dtype=np.float32)
        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Jina AI (REST, task-aware, no local download)
# ---------------------------------------------------------------------------

class JinaEmbedder:
    """
    Jina AI Embeddings API — no local model download required.

    Uses task-aware encoding: the API natively distinguishes between
    passage embeddings (indexing) and query embeddings (search time),
    which improves retrieval quality without manual prefix tricks.

    Parameters
    ----------
    api_key    : Jina AI API key (https://jina.ai — free tier available)
    model      : Jina embedding model (default: jina-embeddings-v3)
    batch_size : texts per HTTP request
    timeout    : per-request timeout in seconds
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_JINA_EMBED_MODEL,
        batch_size: int = 256,
        timeout: int = 60,
    ):
        if not api_key:
            raise ValueError("api_key is required for JinaEmbedder")
        self.endpoint = _JINA_BASE_URL + "/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        if not texts:
            raise ValueError("texts list is empty")

        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            payload = {"model": self.model, "input": batch, "task": task}
            resp = requests.post(
                self.endpoint, headers=self._headers, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            batch_vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            all_vectors.extend(batch_vecs)

        matrix = np.array(all_vectors, dtype=np.float32)
        return _l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], task="retrieval.query")[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_embedder(
    embed_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embed_model: Optional[str] = None,
    provider: Optional[str] = None,
    embed_path: str = "/embeddings",
):
    """
    Return an embedder based on the provider and URL configuration.

    provider="jina"  → JinaEmbedder (api_key required; no local download)
    embed_base_url   → RestEmbedder (POST {embed_base_url}{embed_path})
    else             → LocalEmbedder (sentence-transformers; optional dependency)

    Parameters
    ----------
    embed_base_url : base URL for REST providers (Jina URL is built-in when provider="jina")
    api_key        : API key for REST providers
    embed_model    : override the default model for the chosen provider
    provider       : "jina" | "local" | None (auto-detect from embed_base_url)
    embed_path     : path suffix for RestEmbedder (default: /embeddings)
    """
    if provider == "jina":
        if not api_key:
            raise ValueError("api_key is required for provider='jina' (JinaEmbedder)")
        return JinaEmbedder(
            api_key=api_key,
            model=embed_model or _DEFAULT_JINA_EMBED_MODEL,
        )
    if embed_base_url:
        if not api_key:
            raise ValueError("api_key is required for RestEmbedder")
        model = embed_model or "text-embedding-3-small"
        return RestEmbedder(
            base_url=embed_base_url,
            api_key=api_key,
            model=model,
            embed_path=embed_path,
        )
    return LocalEmbedder(model_name=embed_model or _DEFAULT_LOCAL_MODEL)

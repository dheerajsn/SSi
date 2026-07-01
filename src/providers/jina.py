"""
Jina AI providers — embeddings and reranking via REST, no local download.

Jina API: https://jina.ai  (free tier available)

  JinaEmbedder  — POST https://api.jina.ai/v1/embeddings
                   Model: jina-embeddings-v3 (task-aware, multilingual)

  JinaReranker  — POST https://api.jina.ai/v1/rerank
                   Model: jina-reranker-v2-base-multilingual
"""

import requests

from ._rest import RestEmbedder

_BASE = "https://api.jina.ai/v1"
_DEFAULT_EMBED_MODEL = "jina-embeddings-v3"
_DEFAULT_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


class JinaEmbedder(RestEmbedder):
    """
    Jina AI Embeddings — task-aware encoding, no local download.

    Overrides RestEmbedder.embed() to pass the task field in the API
    request body. Jina v3 uses this to produce separate query vs passage
    embedding spaces, which improves retrieval quality without manual
    prompt prefixes (BGE-style tricks are not needed).

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
        model: str = _DEFAULT_EMBED_MODEL,
        batch_size: int = 256,
        timeout: int = 60,
    ):
        if not api_key:
            raise ValueError(
                "api_key is required for JinaEmbedder. "
                "Get a free key at https://jina.ai"
            )
        super().__init__(
            endpoint=f"{_BASE}/embeddings",
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            timeout=timeout,
        )

    def embed(self, texts, task="retrieval.passage"):
        if not texts:
            raise ValueError("texts list is empty")
        import numpy as np
        from ._rest import _l2_normalize
        vecs = []
        for i in range(0, len(texts), self.batch_size):
            vecs.extend(self._call(texts[i: i + self.batch_size], extra={"task": task}))
        return _l2_normalize(np.array(vecs, dtype=np.float32))


class JinaReranker:
    """
    Jina AI Reranker — joint (query, document) scoring via REST.

    No local model download. Supports multilingual content out of the box.

    Parameters
    ----------
    api_key : Jina AI API key
    model   : Jina reranker model (default: jina-reranker-v2-base-multilingual)
    timeout : per-request timeout in seconds
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_RERANK_MODEL,
        timeout: int = 30,
    ):
        if not api_key:
            raise ValueError("api_key is required for JinaReranker")
        self.endpoint = f"{_BASE}/rerank"
        self.model = model
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def rerank(self, query: str, children: list, top_n: int = 5) -> list:
        """
        Rerank children by relevance to query via the Jina Reranker API.

        Parameters
        ----------
        query    : the user's question
        children : list of dicts, each must have a "row_text" key
        top_n    : how many to return after reranking

        Returns
        -------
        Top top_n children, sorted by relevance score descending.
        Each child dict gets a "rerank_score" key added.
        """
        if not children:
            return children
        resp = requests.post(
            self.endpoint,
            headers=self._headers,
            json={
                "model": self.model,
                "query": query,
                "documents": [c["row_text"] for c in children],
                "top_n": min(top_n, len(children)),
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        out = []
        for item in resp.json().get("results", []):
            child = dict(children[item["index"]])
            child["rerank_score"] = round(float(item["relevance_score"]), 4)
            out.append(child)
        return out

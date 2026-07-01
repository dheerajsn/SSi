"""
Orchestra — inhouse enterprise AI gateway.

Orchestra exposes an OpenAI-compatible API at a custom base URL.
Default path convention (matches most internal deployments):

    LLM       : POST {base_url}/v2/chat/completions
    Embeddings: POST {base_url}/v2/embeddings

Both paths are configurable if your gateway uses different conventions.

Usage
-----
    from src.providers.orchestra import OrchestraLLM, OrchestraEmbedder

    llm = OrchestraLLM(base_url="https://ai.mywork.internal", api_key="tok")
    embedder = OrchestraEmbedder(base_url="https://ai.mywork.internal", api_key="tok")
"""

from ._rest import RestLLM, RestEmbedder

_DEFAULT_CHAT_PATH = "/v2/chat/completions"
_DEFAULT_EMBED_PATH = "/v2/embeddings"
_DEFAULT_LLM_MODEL = "llama-70b"
_DEFAULT_EMBED_MODEL = "text-embedding-large"


class OrchestraLLM(RestLLM):
    """
    Inhouse Orchestra AI gateway — LLM via /v2/chat/completions.

    Parameters
    ----------
    base_url    : root URL, e.g. "https://ai.mywork.internal"
    api_key     : bearer token; pass "" for unauthenticated gateways
    model       : model served by the gateway (default: llama-70b)
    chat_path   : path to chat completions (default: /v2/chat/completions)
    max_retries : retries on 429 / 5xx
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = _DEFAULT_LLM_MODEL,
        chat_path: str = _DEFAULT_CHAT_PATH,
        max_retries: int = 8,
    ):
        super().__init__(
            endpoint=base_url.rstrip("/") + chat_path,
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )


class OrchestraEmbedder(RestEmbedder):
    """
    Inhouse Orchestra AI gateway — embeddings via /v2/embeddings.

    Parameters
    ----------
    base_url   : root URL, e.g. "https://ai.mywork.internal"
    api_key    : bearer token; pass "" for unauthenticated gateways
    model      : embedding model (default: text-embedding-large)
    embed_path : path to embeddings endpoint (default: /v2/embeddings)
    batch_size : texts per HTTP request
    timeout    : per-request timeout in seconds
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = _DEFAULT_EMBED_MODEL,
        embed_path: str = _DEFAULT_EMBED_PATH,
        batch_size: int = 256,
        timeout: int = 60,
    ):
        super().__init__(
            endpoint=base_url.rstrip("/") + embed_path,
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            timeout=timeout,
        )

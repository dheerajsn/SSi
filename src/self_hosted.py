"""
Self-hosted LLM and embedding clients for enterprise / internal deployments.

Internal AI gateways often expose an OpenAI-compatible API at a custom base URL
with a non-standard path prefix. This module provides thin subclasses of the
standard LLMClient and RestEmbedder that default to the /v2/... convention and
ship with sensible enterprise model names.

Default path convention (configurable):
    LLM       : POST {base_url}/v2/chat/completions
    Embeddings: POST {base_url}/v2/embeddings

Compatible engines
------------------
vLLM, LMStudio, Ollama (--openai-compat flag), Azure AI Studio, and any
gateway that accepts OpenAI-compatible request/response format.

Quickstart
----------
    from src.self_hosted import make_self_hosted_pipeline

    pipeline = make_self_hosted_pipeline(
        data_dir    = "data/ssi_docs/",
        base_url    = "https://ai.mywork.internal",
        api_key     = "my-bearer-token",
        domain      = "markets_ssi",
    )
    pipeline.build()
    result = pipeline.query("What is the PSET BIC for Germany-CLEARGER?")
    print(result["answer"])

Custom paths
------------
    llm = SelfHostedLLM(
        base_url="https://ai.mywork.internal",
        api_key="...",
        model="gpt-5.4",
        llm_path="/api/v2/chat/completions",   # override if gateway differs
    )
    embedder = SelfHostedEmbedder(
        base_url="https://ai.mywork.internal",
        api_key="...",
        embed_path="/api/v2/embeddings",
    )
"""

from typing import Optional

from src.llm_layer import LLMClient
from src.embeddings import RestEmbedder

_DEFAULT_LLM_PATH = "/v2/chat/completions"
_DEFAULT_EMBED_PATH = "/v2/embeddings"
_DEFAULT_LLM_MODEL = "llama-70b"
_DEFAULT_EMBED_MODEL = "text-embedding-large"


class SelfHostedLLM(LLMClient):
    """
    LLM client for self-hosted / enterprise gateway deployments.

    Thin subclass of LLMClient that defaults to the /v2/chat/completions path
    and a sensible enterprise model name.

    Parameters
    ----------
    base_url    : root URL of the internal API, e.g. "https://ai.mywork.internal"
    model       : model served by the gateway (e.g. "llama-70b", "gpt-5.4")
    api_key     : bearer token; pass "" if the endpoint is unauthenticated
    llm_path    : path to the completions endpoint (default: /v2/chat/completions)
    max_retries : retries on 429 / 5xx before raising
    """

    def __init__(
        self,
        base_url: str,
        model: str = _DEFAULT_LLM_MODEL,
        api_key: str = "",
        llm_path: str = _DEFAULT_LLM_PATH,
        max_retries: int = 8,
    ):
        super().__init__(
            api_key=api_key or "self-hosted",
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            chat_path=llm_path,
        )
        if not api_key:
            # Allow unauthenticated endpoints — clear the Bearer header
            self._headers.pop("Authorization", None)


class SelfHostedEmbedder(RestEmbedder):
    """
    Embedding client for self-hosted / enterprise gateway deployments.

    Thin subclass of RestEmbedder that defaults to the /v2/embeddings path
    and the text-embedding-large model name.

    Parameters
    ----------
    base_url   : root URL of the internal API, e.g. "https://ai.mywork.internal"
    model      : embedding model name served by the gateway
    api_key    : bearer token; pass "" if the endpoint is unauthenticated
    embed_path : path to the embeddings endpoint (default: /v2/embeddings)
    batch_size : texts per HTTP request
    timeout    : per-request timeout in seconds
    """

    def __init__(
        self,
        base_url: str,
        model: str = _DEFAULT_EMBED_MODEL,
        api_key: str = "",
        embed_path: str = _DEFAULT_EMBED_PATH,
        batch_size: int = 256,
        timeout: int = 60,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key or "self-hosted",
            model=model,
            batch_size=batch_size,
            timeout=timeout,
            embed_path=embed_path,
        )
        if not api_key:
            self._headers.pop("Authorization", None)


def make_self_hosted_pipeline(
    data_dir: str,
    base_url: str,
    api_key: str = "",
    llm_model: str = _DEFAULT_LLM_MODEL,
    embed_model: str = _DEFAULT_EMBED_MODEL,
    llm_path: str = _DEFAULT_LLM_PATH,
    embed_path: str = _DEFAULT_EMBED_PATH,
    domain: Optional[object] = None,
    chunk_strategy: str = "section_table",
    retrieval_mode: str = "dense",
    default_k: int = 10,
    rerank_model: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
):
    """
    Build an SSIPipeline pre-configured for a self-hosted enterprise gateway.

    Constructs SelfHostedLLM + SelfHostedEmbedder and wires them into the
    standard SSIPipeline. All advanced pipeline options are forwarded.

    Parameters
    ----------
    data_dir     : directory containing Azure DI JSON files to index
    base_url     : root URL of the internal AI gateway
    api_key      : bearer token for the gateway
    llm_model    : LLM model name (default: llama-70b)
    embed_model  : embedding model name (default: text-embedding-large)
    llm_path     : path to chat completions (default: /v2/chat/completions)
    embed_path   : path to embeddings (default: /v2/embeddings)
    domain       : DomainConfig or str key ("fx_ssi", "markets_ssi", etc.)
    rerank_model : reranker model name, or None to disable
                   Use "jina-reranker-v2-base-multilingual" for API-based reranking
    rerank_api_key : API key for JinaReranker (only needed if rerank_model starts with "jina-")

    Returns
    -------
    SSIPipeline (not yet built — call .build() before querying)

    Example
    -------
    pipeline = make_self_hosted_pipeline(
        data_dir = "data/ssi_docs/",
        base_url = "https://ai.mywork.internal",
        api_key  = os.getenv("WORK_API_KEY"),
        domain   = "markets_ssi",
    )
    pipeline.build()
    result = pipeline.query("PSET BIC for Germany-CLEARGER?")
    """
    # Import here to avoid circular import at module level
    from src.pipeline import SSIPipeline
    from src.reranker import make_reranker

    llm = SelfHostedLLM(
        base_url=base_url,
        model=llm_model,
        api_key=api_key,
        llm_path=llm_path,
    )
    embedder = SelfHostedEmbedder(
        base_url=base_url,
        model=embed_model,
        api_key=api_key,
        embed_path=embed_path,
    )
    reranker = make_reranker(rerank_model, api_key=rerank_api_key or api_key)

    pipeline = SSIPipeline(
        data_dir=data_dir,
        api_key=api_key or "self-hosted",
        domain=domain,
        chunk_strategy=chunk_strategy,
        retrieval_mode=retrieval_mode,
        default_k=default_k,
        _llm=llm,
        _embedder=embedder,
        _reranker=reranker,
    )
    return pipeline

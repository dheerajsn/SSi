"""
Provider registry — factories and top-level exports.

Supported LLM providers
-----------------------
  "groq"       →  GroqLLM        (https://console.groq.com)
  "orchestra"  →  OrchestraLLM   (inhouse REST gateway, /v2/chat/completions)
  # future:    "openai", "anthropic"

Supported embedding providers
------------------------------
  "orchestra"  →  OrchestraEmbedder  (inhouse REST gateway, /v2/embeddings)
  "jina"       →  JinaEmbedder       (https://jina.ai, no local download)
  "local"      →  LocalEmbedder      (sentence-transformers, optional dependency)

Supported rerankers
-------------------
  "jina"   or "jina-*"         →  JinaReranker   (https://jina.ai, no download)
  "local"  or "cross-encoder/*"  →  LocalReranker  (sentence-transformers, optional)
  None / ""                      →  None           (reranking disabled)

Quick-start examples
--------------------
  from src.providers import GroqLLM, JinaEmbedder, LocalEmbedder, make_reranker

  # Groq LLM + Jina embeddings (no local download at all)
  pipeline = SSIPipeline(
      data_dir="data/ssi/",
      llm=GroqLLM(api_key="gsk_..."),
      embedder=JinaEmbedder(api_key="jina_..."),
  )

  # Orchestra inhouse (both LLM + embeddings from the same gateway)
  pipeline = SSIPipeline(
      data_dir="data/ssi/",
      llm=OrchestraLLM(base_url="https://ai.mywork.internal", api_key="tok"),
      embedder=OrchestraEmbedder(base_url="https://ai.mywork.internal", api_key="tok"),
  )

  # Groq LLM + local sentence-transformers
  pipeline = SSIPipeline(
      data_dir="data/ssi/",
      llm=GroqLLM(api_key="gsk_..."),
      embedder=LocalEmbedder(),
  )
"""

from .base import LLMProvider, EmbedProvider
from .groq import GroqLLM
from .orchestra import OrchestraLLM, OrchestraEmbedder
from .jina import JinaEmbedder, JinaReranker
from .local import LocalEmbedder, LocalReranker


def make_llm(provider: str, **kwargs) -> LLMProvider:
    """
    Create an LLM provider by name.

    Parameters
    ----------
    provider : "groq" | "orchestra"
    **kwargs : forwarded to the provider constructor

    Examples
    --------
    make_llm("groq", api_key="gsk_...")
    make_llm("groq", api_key="gsk_...", model="llama-3.1-8b-instant")
    make_llm("orchestra", base_url="https://ai.mywork.internal", api_key="tok")
    """
    _registry = {
        "groq": GroqLLM,
        "orchestra": OrchestraLLM,
    }
    if provider not in _registry:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Supported: {list(_registry)}"
        )
    return _registry[provider](**kwargs)


def make_embedder(provider: str, **kwargs) -> EmbedProvider:
    """
    Create an embedding provider by name.

    Parameters
    ----------
    provider : "jina" | "orchestra" | "local"
    **kwargs : forwarded to the provider constructor

    Examples
    --------
    make_embedder("jina", api_key="jina_...")
    make_embedder("orchestra", base_url="https://ai.mywork.internal", api_key="tok")
    make_embedder("local")
    make_embedder("local", model="BAAI/bge-base-en-v1.5")
    """
    _registry = {
        "jina": JinaEmbedder,
        "orchestra": OrchestraEmbedder,
        "local": LocalEmbedder,
    }
    if provider not in _registry:
        raise ValueError(
            f"Unknown embed provider {provider!r}. Supported: {list(_registry)}"
        )
    return _registry[provider](**kwargs)


def make_reranker(model_or_provider: str, api_key: str = None):
    """
    Create a reranker, or return None to disable reranking.

    Parameters
    ----------
    model_or_provider : "jina" | "jina-*" model name | "local" | "cross-encoder/*" | None
    api_key           : required when using Jina

    Examples
    --------
    make_reranker(None)                                      # disabled
    make_reranker("jina", api_key="jina_...")                # Jina default model
    make_reranker("jina-reranker-v2-base-multilingual",
                  api_key="jina_...")                        # Jina specific model
    make_reranker("local")                                   # default cross-encoder
    make_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")   # specific cross-encoder
    """
    if not model_or_provider:
        return None

    name = model_or_provider
    if name == "jina" or name.startswith("jina-"):
        if not api_key:
            raise ValueError(
                f"api_key is required for JinaReranker (model={name!r})"
            )
        kwargs = {"api_key": api_key}
        if name.startswith("jina-"):
            kwargs["model"] = name
        return JinaReranker(**kwargs)

    if name == "local" or name.startswith("cross-encoder/"):
        kwargs = {}
        if name.startswith("cross-encoder/"):
            kwargs["model"] = name
        return LocalReranker(**kwargs)

    raise ValueError(
        f"Unknown reranker {name!r}. Use 'jina', 'jina-<model>', 'local', or 'cross-encoder/<model>'."
    )


__all__ = [
    "LLMProvider", "EmbedProvider",
    "GroqLLM",
    "OrchestraLLM", "OrchestraEmbedder",
    "JinaEmbedder", "JinaReranker",
    "LocalEmbedder", "LocalReranker",
    "make_llm", "make_embedder", "make_reranker",
]

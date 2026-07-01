"""Groq LLM provider — https://console.groq.com"""

from ._rest import RestLLM

_BASE = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqLLM(RestLLM):
    """
    Groq-hosted LLM (llama, mixtral, gemma).

    Parameters
    ----------
    api_key     : Groq API key from https://console.groq.com
    model       : model identifier (default: llama-3.3-70b-versatile)
    max_retries : retries on 429 / 5xx

    Available models (as of mid-2025)
    ----------------------------------
    llama-3.3-70b-versatile   12K TPM, 100K TPD  ← best quality
    llama-3.1-8b-instant       6K TPM,   1M TPD  ← fastest
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        max_retries: int = 8,
    ):
        if not api_key:
            raise ValueError(
                "api_key is required for Groq. Get a free key at https://console.groq.com"
            )
        super().__init__(
            endpoint=f"{_BASE}/chat/completions",
            api_key=api_key,
            model=model,
            max_retries=max_retries,
        )

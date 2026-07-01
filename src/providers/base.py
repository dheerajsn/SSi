"""
Abstract interfaces for LLM and embedding providers.

All providers implement these two contracts so the pipeline is completely
decoupled from the underlying technology (REST API, local model, etc.).
Adding a new provider means subclassing LLMProvider or EmbedProvider —
no changes required anywhere else.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class LLMProvider(ABC):
    """
    Minimal interface for a language model backend.

    Implementations: GroqLLM, OrchestraLLM
    Future:          OpenAILLM, AnthropicLLM
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0,
    ) -> str:
        """Send a prompt and return the model's text response."""


class EmbedProvider(ABC):
    """
    Minimal interface for an embedding backend.

    Implementations: OrchestraEmbedder, JinaEmbedder, LocalEmbedder
    Future:          OpenAIEmbedder
    """

    @abstractmethod
    def embed(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        """
        Embed a list of texts. Returns L2-normalised float32 matrix (N, dim).

        task: "retrieval.passage" for indexing, "retrieval.query" for search-time.
        Not all backends distinguish the two — implementations may ignore it.
        """

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dim,)."""
        return self.embed([text], task="retrieval.query")[0]

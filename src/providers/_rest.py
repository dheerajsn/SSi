"""
Shared HTTP utilities for REST-based providers.

RestLLM and RestEmbedder implement the OpenAI-compatible request/response
schema used by Groq, Orchestra, Jina, and most enterprise AI gateways.
Provider-specific classes (groq.py, orchestra.py, jina.py) inherit from
these and only supply the endpoint URL and default model name.

_extract_json: LLM response parser used by the pipeline.
"""

import json as _json
import re
import time
from typing import List

import numpy as np
import requests

from .base import LLMProvider, EmbedProvider


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

def _parse_retry_wait(resp: requests.Response, default: float = 15.0) -> float:
    """Parse 'try again in Xs' / 'Xm Y.Zs' from a 429 response body."""
    try:
        msg = resp.json().get("error", {}).get("message", "")
        m = re.search(r"try again in (?:(\d+)m\s*)?([0-9.]+)s", msg)
        if m:
            return float(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# LLM — OpenAI-compatible chat/completions
# ---------------------------------------------------------------------------

class RestLLM(LLMProvider):
    """
    OpenAI-compatible chat completions over REST.

    Handles 429 rate-limit retry with exact wait-time from the error body,
    and 5xx server errors with a 2-second backoff.

    Parameters
    ----------
    endpoint    : full URL, e.g. "https://api.groq.com/openai/v1/chat/completions"
    api_key     : bearer token; omit or pass "" for unauthenticated endpoints
    model       : model identifier sent in every request
    max_retries : retries before raising
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        max_retries: int = 8,
    ):
        self.endpoint = endpoint
        self.model = model
        self.max_retries = max_retries
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint, headers=self._headers, json=payload, timeout=60
                )
                if resp.status_code == 429:
                    wait = _parse_retry_wait(resp)
                    print(f"    [rate-limit] waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.HTTPError:
                if attempt < self.max_retries - 1 and resp.status_code >= 500:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"LLM error ({self.endpoint}): {resp.status_code} — {resp.text}"
                )

        raise RuntimeError(f"LLM failed after {self.max_retries} retries ({self.endpoint})")


# ---------------------------------------------------------------------------
# Embedder — OpenAI-compatible /embeddings
# ---------------------------------------------------------------------------

def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class RestEmbedder(EmbedProvider):
    """
    OpenAI-compatible embeddings over REST.

    Sends POST {endpoint} with {"model": ..., "input": [...]} and parses
    the standard {"data": [{"embedding": [...], "index": N}]} response.

    Parameters
    ----------
    endpoint   : full URL, e.g. "https://ai.mywork.internal/v2/embeddings"
    api_key    : bearer token; omit or pass "" for unauthenticated endpoints
    model      : model identifier sent in every request
    batch_size : texts per HTTP request
    timeout    : per-request timeout in seconds
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        batch_size: int = 256,
        timeout: int = 60,
    ):
        self.endpoint = endpoint
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _call(self, batch: List[str], extra: dict = None) -> List[List[float]]:
        payload = {"model": self.model, "input": batch, **(extra or {})}
        resp = requests.post(
            self.endpoint, headers=self._headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def embed(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        if not texts:
            raise ValueError("texts list is empty")
        vecs: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vecs.extend(self._call(texts[i: i + self.batch_size]))
        return _l2_normalize(np.array(vecs, dtype=np.float32))


# ---------------------------------------------------------------------------
# JSON response parser (used by pipeline, not provider-specific)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Parse JSON from an LLM response.

    Handles three cases:
      1. Clean JSON string
      2. JSON wrapped in ```json ... ``` code fence
      3. Fallback: wraps plain text in {"answer": text, "source": None}
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()
    try:
        return _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except (_json.JSONDecodeError, ValueError):
                pass
    return {"answer": text, "source": None}

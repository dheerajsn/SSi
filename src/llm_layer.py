"""
LLM client — OpenAI-compatible chat/completions endpoint.

Defaults to Groq (free-tier llama-3.3-70b-versatile), but works with any
provider that exposes POST {base_url}/chat/completions — including inhouse
deployments at e.g. https://inhouse.co/v2/chat/completions.

Rate-limit handling:
  - Parses "try again in Xs" from 429 responses and waits exactly that long.
  - Retries up to `max_retries` times (default 8).
  - 5xx errors retry with 2-second backoff.
"""

import json as _json
import re
import time
from typing import Dict, List, Optional

import requests

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient:
    """
    OpenAI-compatible chat completion client.

    Parameters
    ----------
    api_key     : Bearer token
    model       : model identifier (e.g. "llama-3.3-70b-versatile", "gpt-4o")
    base_url    : base URL of the API (default: Groq)
                  Client calls POST {base_url}/chat/completions
    max_retries : retries on 429 / 5xx before raising
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = _GROQ_BASE_URL,
        max_retries: int = 8,
        chat_path: str = "/chat/completions",
    ):
        if not api_key:
            raise ValueError(
                "api_key is required. For Groq: get a free key at https://console.groq.com"
            )
        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + chat_path
        self.max_retries = max_retries
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
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
                    self.endpoint,
                    headers=self._headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 429:
                    wait = _parse_retry_wait(resp, default=15.0)
                    print(f"    [rate-limit] waiting {wait:.1f}s before retry...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.HTTPError as e:
                if attempt < self.max_retries - 1 and resp.status_code >= 500:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"LLM API error ({self.endpoint}): {e} — {resp.text}"
                ) from e

        raise RuntimeError(f"LLM API failed after {self.max_retries} retries")


def _parse_retry_wait(resp: requests.Response, default: float = 15.0) -> float:
    try:
        msg = resp.json().get("error", {}).get("message", "")
        # Handle both "15.0s" and "9m25.056s" formats from Groq
        m = re.search(r"try again in (?:(\d+)m\s*)?([0-9.]+)s", msg)
        if m:
            minutes = float(m.group(1) or 0)
            seconds = float(m.group(2))
            return minutes * 60 + seconds + 1.0
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Parse JSON from LLM output.

    Handles three cases:
      1. Clean JSON string
      2. JSON wrapped in ```json ... ``` code fence
      3. Fallback: returns {"answer": text, "source": None} so callers always
         get a dict (graceful degradation when the LLM ignores the JSON rule).
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end]).strip()
    try:
        return _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except (_json.JSONDecodeError, ValueError):
                pass
        return {"answer": text, "source": None}


# ---------------------------------------------------------------------------
# Backwards-compat alias
# ---------------------------------------------------------------------------
GroqLLM = LLMClient


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    matched_parents: List[Dict],
    matched_children: Optional[List[Dict]] = None,
) -> str:
    """
    Build the user message sent to the LLM.

    When matched_children is provided each parent block only includes the
    rows that were actually retrieved as relevant children — keeping the
    prompt compact for focused single-instrument queries.  Preamble and
    notes are always included.

    For broad queries (all children retrieved) the behaviour is identical
    to sending the full parent.
    """
    relevant_by_parent: Dict[int, set] = {}
    if matched_children:
        for child in matched_children:
            relevant_by_parent.setdefault(child["parent_id"], set()).add(child["row_text"])

    context_parts = []
    for parent in matched_parents:
        header = f"[Source: {parent['doc_name']} | Section: {parent['heading']}]"

        if relevant_by_parent and parent["id"] in relevant_by_parent:
            relevant = relevant_by_parent[parent["id"]]
            body_lines = [r for r in parent["rows"] if r in relevant]
        else:
            body_lines = list(parent["rows"])

        all_lines: List[str] = []
        if parent.get("preamble"):
            all_lines.append(parent["preamble"])
        all_lines.extend(body_lines)
        if parent.get("notes"):
            all_lines.extend(parent["notes"])

        context_parts.append(header + "\n" + "\n".join(all_lines))

    context_block = "\n\n".join(context_parts)
    return f"Context:\n{context_block}\n\nQuestion: {question}"


def build_citation(matched_parents: List[Dict]) -> str:
    seen: List[str] = []
    for p in matched_parents:
        entry = f"{p['doc_name']} / {p['heading']}"
        if entry not in seen:
            seen.append(entry)
    return "Sources:\n" + "\n".join(f"  - {s}" for s in seen)


# ---------------------------------------------------------------------------
# Legacy constant — kept so any external code that imported it still works
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a settlement instructions assistant for Global Markets operations. "
    "Answer questions about Standard Settlement Instructions (SSI) using ONLY "
    "the context provided. Never infer or hallucinate SWIFT/BIC codes or account numbers."
)

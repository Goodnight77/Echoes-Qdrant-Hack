"""Local LLM client — talks to LM Studio (OpenAI-compatible API at localhost:1234).
Also works with Ollama (localhost:11434) and Groq (api.groq.com) by swapping the
base URL + model name. No API key needed for local models."""

from __future__ import annotations

import httpx

# Default to LM Studio. Swap to "http://localhost:11434/v1" for Ollama,
# or "https://api.groq.com/openai/v1" for Groq with an API key.
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"
LOCAL_LLM_MODEL = "qwen2.5-3b-instruct"
LOCAL_LLM_TIMEOUT = 30.0  # local inference is slower than cloud


def ask_llm(
    system: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 256,
    url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> str | None:
    """Send a chat completion to the local LLM. Returns the response text or None."""
    body = {
        "model": model or LOCAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Stop tokens prevent the model from hallucinating past its answer into
        # training-data artifacts like "<|endoftext|>Human: ### Instruction:"
        "stop": ["<|endoftext|>", "Human:", "###", "\n\n\n"],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        r = httpx.post(
            url or LOCAL_LLM_URL,
            json=body,
            headers=headers,
            timeout=timeout or LOCAL_LLM_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None

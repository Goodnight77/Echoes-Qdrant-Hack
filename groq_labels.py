"""Cluster labelling via Groq's LLM API. Falls back to bigram word-counting if
the API key is missing, the network call fails, or the response is unusable.
Why a separate module: keeps Groq-specific HTTP / prompt work out of museum.py
and lets us cache per cluster signature so a layout rebuild after a non-cluster
change (e.g. one new memory) doesn't burn 8 LLM calls."""
from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TimeoutError

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"   # cheap, fast, plenty for 2-4-word labels
GROQ_TIMEOUT_S = 8.0
MAX_SAMPLES_PER_CLUSTER = 10           # don't blow tokens on huge clusters

_SYSTEM_PROMPT = (
    "You name photo clusters with a short evocative title (2-4 words, no "
    "quotes, no period, lowercase). Use what the memories actually depict — "
    "places, people, objects, themes. If you see filenames like IMG_ or "
    "screenshot dates, use those as hints. If there is no descriptive content "
    "at all, make your best guess from filenames. Never return UNKNOWN — "
    "always return 2-4 lowercase words even if vague (e.g. 'random photos')."
    "Avoid generic words like 'photos', 'images', 'cluster', 'collection', 'memories'."
)

_USER_TEMPLATE = (
    "Cluster contents (one item per line):\n{lines}\n\n"
    "Title (2-4 lowercase words):"
)

_label_cache: dict[str, str] = {}


def _cluster_signature(point_ids: list[str]) -> str:
    """Stable hash of the cluster's member ids — change one member, sig changes,
    label gets re-fetched. Otherwise cached."""
    h = hashlib.sha256()
    for pid in sorted(point_ids):
        h.update(pid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _build_lines(payloads: list[dict], filenames: list[str | None]) -> str:
    samples = list(zip(payloads, filenames))[:MAX_SAMPLES_PER_CLUSTER]
    out: list[str] = []
    for payload, filename in samples:
        parts: list[str] = []
        if payload.get("type"):
            parts.append(f"[{payload['type']}]")
        # extract meaningful words from filename (strip UUIDs, timestamps, extensions)
        if filename:
            import re
            clean = re.sub(r"[0-9a-f]{8,}|[0-9]{10,}|\.[a-z0-9]+$", "", filename, flags=re.I)
            clean = clean.strip("_- .")
            if clean:
                parts.append(clean)
        text = (payload.get("transcript") or "").strip()
        if text:
            parts.append(f'transcript: "{text[:120]}"')
        ocr = (payload.get("ocr_text") or "").strip()
        if ocr:
            parts.append(f'ocr: "{ocr[:120]}"')
        if not parts:
            continue
        out.append(" · ".join(parts))
    return "\n".join(out) if out else "(no descriptive content)"


def _is_unusable(label: str) -> bool:
    """Reject empty / sentinel / too-long replies — fallback handles those."""
    if not label:
        return True
    s = label.strip().lower()
    if s in {"unknown", "n/a", "none", "various", "mixed", "untitled"}:
        return True
    if len(s) > 60:
        return True
    # If model accidentally returned a sentence, drop everything past the first period
    return False


def _normalise(label: str) -> str:
    s = re.sub(r'^["\'`\s]+|["\'`\s.]+$', "", label.strip())
    s = re.split(r"[\.\n]", s, 1)[0]
    s = s.lower()
    # collapse internal whitespace to single space
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _call_groq(api_key: str, prompt: str) -> str | None:
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 24,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = httpx.post(GROQ_URL, json=body, headers=headers, timeout=GROQ_TIMEOUT_S)
        if r.status_code != 200:
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


def _call_local_llm(prompt: str) -> str | None:
    """Try LM Studio / Ollama as a fallback when Groq is unavailable."""
    from llm_client import ask_llm
    return ask_llm(
        system=_SYSTEM_PROMPT,
        prompt=prompt,
        temperature=0.2,
        max_tokens=15,  # cluster label = 2-4 words = ~10 tokens
        timeout=12.0,   # 3B model on GPU should finish 15 tokens in <5s
    )


def label_cluster(point_ids: list[str], payloads: list[dict],
                   filenames: list[str | None] | None = None) -> str | None:
    """Return a Groq-generated cluster label, or None to signal fall-back to
    the bigram method. Always cheap on cache hits."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not point_ids:
        return None
    sig = _cluster_signature(point_ids)
    if sig in _label_cache:
        return _label_cache[sig]
    if filenames is None:
        filenames = [None] * len(payloads)
    prompt = _USER_TEMPLATE.format(lines=_build_lines(payloads, filenames))
    # Try local LLM first (LM Studio / Ollama) — faster, free, offline.
    # Fall back to Groq only if local LLM is unreachable.
    raw = _call_local_llm(prompt)
    if raw is None and api_key:
        raw = _call_groq(api_key, prompt)
    if raw is None:
        return None
    cleaned = _normalise(raw)
    if _is_unusable(cleaned):
        return None
    _label_cache[sig] = cleaned
    return cleaned


def label_clusters_parallel(items: list[tuple[list[str], list[dict], list[str | None]]],
                              max_workers: int = 4) -> list[str | None]:
    """Run label_cluster across N clusters in parallel — one LLM round-trip per
    cluster, capped at max_workers in flight. Returns labels in input order
    (None where the call failed / API unavailable, so caller can fall back)."""
    if not items:
        return []
    # Local LLM (LM Studio) doesn't need an API key — proceed even without GROQ_API_KEY
    out: list[str | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(label_cluster, pids, payloads, filenames): i
            for i, (pids, payloads, filenames) in enumerate(items)
        }
        for fut in futures:
            i = futures[fut]
            try:
                out[i] = fut.result(timeout=GROQ_TIMEOUT_S + 2)
            except _TimeoutError:
                out[i] = None
    return out


def cache_size() -> int:
    return len(_label_cache)


def invalidate_cache() -> None:
    _label_cache.clear()

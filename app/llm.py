"""Thin wrappers around the Gemini SDK.

We use two model families:
  * text-embedding-004 for vectorizing the catalog and queries
  * gemini-2.0-flash for the agent's planner/responder calls

The wrappers handle JSON-mode parsing, retries on transient errors,
and a tiny in-process embedding cache so a query embedding is not
recomputed on every turn.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

EMBED_MODEL = os.environ.get("SHL_EMBED_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.environ.get("SHL_CHAT_MODEL", "models/gemini-2.5-flash-lite")

def _collect_keys() -> list[str]:
    """Read 1..N Gemini keys from env. Supports:
      * GEMINI_API_KEYS (comma-separated)
      * GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3, ...
      * GOOGLE_API_KEY (legacy fallback)
    """
    keys: list[str] = []
    seen: set[str] = set()
    multi = os.environ.get("GEMINI_API_KEYS", "")
    for k in multi.split(","):
        k = k.strip()
        if k and k not in seen:
            keys.append(k)
            seen.add(k)
    for env in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY2",
        "GEMINI_API_KEY3",
        "GEMINI_API_KEY4",
    ):
        v = os.environ.get(env, "").strip()
        if v and v not in seen:
            keys.append(v)
            seen.add(v)
    return keys


_API_KEYS: list[str] = _collect_keys()
_KEY_INDEX = 0
if _API_KEYS:
    genai.configure(api_key=_API_KEYS[0])


def configure_api_key(key: str) -> None:
    """Allow programmatic configuration (used by build_index.py and tests)."""
    global _API_KEYS, _KEY_INDEX
    if key and key not in _API_KEYS:
        _API_KEYS.insert(0, key)
    _KEY_INDEX = 0
    genai.configure(api_key=key)


def _rotate_key() -> bool:
    """Switch to the next configured key. Returns True if rotation happened."""
    global _KEY_INDEX
    if len(_API_KEYS) <= 1:
        return False
    _KEY_INDEX = (_KEY_INDEX + 1) % len(_API_KEYS)
    genai.configure(api_key=_API_KEYS[_KEY_INDEX])
    return True


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    *,
    batch_size: int = 10,
    pause_seconds: float = 6.5,
) -> list[list[float]]:
    """Embed many texts respecting Gemini free-tier rate limits.

    Defaults: 10 items per call, 6.5s pause between calls — keeps us well
    under the 100 embedded-items / minute free quota. Backs off and retries
    on ResourceExhausted using the server-provided retry delay when present.
    """
    out: list[list[float]] = []
    total = len(texts)
    sent = 0
    for i in range(0, total, batch_size):
        chunk = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                resp = genai.embed_content(
                    model=EMBED_MODEL,
                    content=chunk,
                    task_type=task_type,
                )
                vecs = resp["embedding"]
                # Single item returns one vector; list returns list of vectors.
                if vecs and isinstance(vecs[0], (int, float)):
                    out.append(list(vecs))
                else:
                    out.extend(vecs)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # Try to extract a server-suggested retry delay (in seconds).
                delay = 0
                m = re.search(r"seconds:\s*(\d+)", msg)
                if m:
                    delay = int(m.group(1))
                if attempt == 4:
                    raise
                time.sleep(max(delay, 6 + attempt * 4))
        sent += len(chunk)
        if sent < total:
            time.sleep(pause_seconds)
    return out


def embed_query(text: str) -> list[float]:
    """Single-query embedding with RETRIEVAL_QUERY task type."""
    for attempt in range(3):
        try:
            resp = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="RETRIEVAL_QUERY",
            )
            return resp["embedding"]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1 + attempt * 2)
    raise RuntimeError("unreachable")


def _strip_fences(text: str) -> str:
    """LLMs love to wrap JSON in ```json fences. Strip them defensively."""
    t = text.strip()
    if t.startswith("```"):
        # Remove first fence line and trailing fence.
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction. Handles fenced blocks and trailing junk."""
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Find the first { and matching last } and try again.
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except json.JSONDecodeError:
            pass
    return None


_VERBOSE_LLM = os.environ.get("SHL_LLM_DEBUG") == "1"


def chat_json(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """One-shot JSON chat call. Returns a parsed dict, never None.

    On a parse failure we surface a structured fallback the caller can
    distinguish (sentinel key ``_parse_error``) instead of raising — the
    agent has to keep working even when the model returns junk.
    """
    name = model or CHAT_MODEL
    cfg = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
    }
    m = genai.GenerativeModel(name, system_instruction=system, generation_config=cfg)

    last_text = ""
    last_err: Exception | None = None
    finish_reason = None
    keys_tried_this_round: set[int] = set()
    for attempt in range(6):
        try:
            r = m.generate_content(user)
            try:
                finish_reason = r.candidates[0].finish_reason  # type: ignore[index]
            except Exception:  # noqa: BLE001
                finish_reason = None
            last_text = ""
            try:
                last_text = (r.text or "").strip()
            except Exception:  # noqa: BLE001
                last_text = ""
            parsed = _extract_json(last_text) if last_text else None
            if parsed is not None:
                return parsed
            last_err = ValueError(
                f"json parse failed (finish_reason={finish_reason!r}, len={len(last_text)})"
            )
            if _VERBOSE_LLM:
                print(f"[llm] retry {attempt}: finish={finish_reason} raw={last_text[:300]!r}")
            time.sleep(0.4 + attempt * 0.6)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            # Quota / rate-limit errors come back as 429 ResourceExhausted.
            # The error body sometimes carries a server-suggested retry delay.
            is_429 = "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower()
            delay = 0
            mtch = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
            if mtch:
                delay = int(mtch.group(1))
            if is_429:
                keys_tried_this_round.add(_KEY_INDEX)
                rotated = _rotate_key()
                if rotated and _KEY_INDEX not in keys_tried_this_round:
                    if _VERBOSE_LLM:
                        print(f"[llm] 429 — rotated to key index {_KEY_INDEX}")
                    continue
                # All keys exhausted this round — wait the suggested delay
                # (RPM windows reset every 60s) and reset the rotation cycle.
                wait = max(delay, 12 + attempt * 6)
                wait = min(wait, 25)  # don't blow the 30s endpoint deadline
                if _VERBOSE_LLM:
                    print(f"[llm] 429 — all keys throttled, sleeping {wait}s")
                time.sleep(wait)
                keys_tried_this_round.clear()
            else:
                time.sleep(0.4 + attempt * 0.6)
    if _VERBOSE_LLM:
        print(f"[llm] giving up: {last_err}")
    return {
        "_parse_error": str(last_err) if last_err else "parse_failed",
        "_raw": last_text,
        "_finish_reason": str(finish_reason) if finish_reason is not None else "",
    }

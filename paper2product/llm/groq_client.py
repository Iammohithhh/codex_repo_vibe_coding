"""Groq API client — stdlib-only HTTP client for Groq's OpenAI-compatible API.

Uses GROQ_API_KEY environment variable. Falls back gracefully when unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY: Optional[str] = None  # Resolved lazily


def _get_api_key() -> Optional[str]:
    global GROQ_API_KEY
    if GROQ_API_KEY is None:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    return GROQ_API_KEY if GROQ_API_KEY else None


def is_groq_available() -> bool:
    """Check if Groq API key is configured."""
    return _get_api_key() is not None


def groq_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    response_format: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Send a chat completion request to Groq API.

    Returns the assistant's response text, or None on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.debug("Groq API key not set — skipping LLM call")
        return None

    payload: Dict[str, Any] = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        logger.warning("Groq API HTTP error %s: %s", e.code, e.reason)
        return None
    except urllib.error.URLError as e:
        logger.warning("Groq API connection error: %s", e.reason)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Groq API response parse error: %s", e)
        return None
    except Exception as e:
        logger.warning("Groq API unexpected error: %s", e)
        return None


def groq_json(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Optional[Dict[str, Any]]:
    """Send a chat request expecting JSON output. Returns parsed dict or None."""
    raw = groq_chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        if "```" in raw:
            start = raw.find("```")
            end = raw.rfind("```")
            if start != end:
                block = raw[start:end].split("\n", 1)[-1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse Groq JSON response")
        return None

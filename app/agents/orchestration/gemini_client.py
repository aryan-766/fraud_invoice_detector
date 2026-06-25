"""
Shared Gemini client helpers with model fallback (avoids quota exhaustion on one model).
"""
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

PLACEHOLDER_KEYS = frozenset({
    "your_gemini_api_key_here",
    "your_google_api_key_here",
})

# gemini-2.0-flash free-tier quota is often 0; lite models usually still work.
DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
)


def get_gemini_api_key() -> str | None:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key or key in PLACEHOLDER_KEYS:
        return None
    return key


def parse_json_response(raw: str) -> dict:
    text = raw.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def generate_json(
    contents: Any,
    system_instruction: str,
    models: tuple[str, ...] = DEFAULT_GEMINI_MODELS,
) -> tuple[dict, str]:
    """
    Call Gemini with JSON response mode, trying each model until one succeeds.

    Returns (parsed_dict, model_name_used).
  Raises the last exception if every model fails.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")

    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None

    for model in models:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                ),
            )
            result = parse_json_response(response.text or "")
            logger.info("Gemini call succeeded with model=%s", model)
            return result, model
        except Exception as exc:
            last_error = exc
            logger.warning("Gemini model %s failed: %s", model, exc)

    raise last_error or RuntimeError("No Gemini models available")

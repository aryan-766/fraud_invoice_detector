"""
agents/ocr_agent.py
Extracts structured invoice data from JPG, PNG, or PDF using Google Gemini API.

PDF pages are rasterized locally (pymupdf) so scanned invoices work reliably.
Set OCR_USE_MOCK=true only for local demos without an API key.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from app.agents.orchestration.document_loader import load_document_images
from app.agents.orchestration.gemini_client import generate_json, get_gemini_api_key

load_dotenv()
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert invoice OCR and information extraction tool.
Analyse the provided invoice image(s) and return ONLY a valid JSON object matching this schema:
{
  "invoice_number": "<string or null>",
  "seller": "<string or null>",
  "product": "<string or null>",
  "purchase_date": "<string or null, formatted as YYYY-MM-DD if possible>",
  "amount": "<string or null>",
  "raw_text": "<string with ALL visible text from the invoice, line by line>"
}
Rules:
- Do not include any text, markdown boxes, or explanations outside the JSON object.
- Formulate the product as a brief name (e.g. "Ambrane Powerbank 10000mAh").
- Formulate the purchase_date as YYYY-MM-DD when possible.
- raw_text must contain every readable text line from the document, not a summary.
- If multiple pages are provided, combine all text into raw_text.
"""

_STANDARD_FIELDS = (
    "invoice_number",
    "seller",
    "product",
    "purchase_date",
    "amount",
    "raw_text",
)


def extract_invoice_data(image_path: str) -> dict:
    """
    Run OCR on *image_path* and return a structured dict via Gemini.

    Supports .jpg, .jpeg, .png, and .pdf (including scanned PDF invoices).

    Returns
    -------
    dict with keys: invoice_number, seller, product, purchase_date, amount,
                    raw_text, ocr_confidence, ocr_model, error
    """
    path = Path(image_path)
    if not path.exists():
        return _empty_result(error="File not found")

    if os.getenv("OCR_USE_MOCK", "").lower() in ("1", "true", "yes"):
        logger.warning("OCR_USE_MOCK enabled — returning mock extraction.")
        return _mock_result()

    if not get_gemini_api_key():
        logger.error("GEMINI_API_KEY not set — cannot extract invoice text.")
        return _empty_result(error="GEMINI_API_KEY not configured. Add it to .env")

    try:
        images = load_document_images(path)
        if not images:
            return _empty_result(error="No readable content in uploaded file")

        contents = _build_contents(images)
        result, model_used = generate_json(
            contents=contents,
            system_instruction=_SYSTEM_PROMPT,
        )
        normalized = _normalize_result(result)
        normalized["ocr_confidence"] = 1.0
        normalized["ocr_model"] = model_used
        normalized["error"] = None
        logger.info(
            "OCR extracted via %s (%s pages): %s",
            model_used,
            len(images),
            {k: v for k, v in normalized.items() if k not in ("raw_text", "error")},
        )
        return normalized

    except Exception as exc:
        logger.error("Gemini OCR failed on all models: %s", exc)
        return _empty_result(error=f"OCR failed: {exc}")


def _build_contents(images: list) -> list:
    prompt = "Extract all invoice details as JSON."
    if len(images) == 1:
        return [images[0], prompt]

    parts: list = []
    for idx, image in enumerate(images, start=1):
        parts.append(f"Invoice page {idx}:")
        parts.append(image)
    parts.append(prompt)
    return parts


def _normalize_result(result: dict) -> dict:
    out = {field: result.get(field) for field in _STANDARD_FIELDS}
    raw = out.get("raw_text")
    out["raw_text"] = (raw or "").strip() if raw is not None else ""
    return out


def _mock_result() -> dict:
    return {
        "invoice_number": "MOCK-1245",
        "seller":         "amazon",
        "product":        "Ambrane Powerbank 10000mAh",
        "purchase_date":  "2025-11-15",
        "amount":         "1499",
        "raw_text":       "[OCR mock mode — set a real GEMINI_API_KEY to extract text]",
        "ocr_confidence": 0.0,
        "ocr_model":      "mock",
        "error":          None,
    }


def _empty_result(error: str | None = None) -> dict:
    return {
        "invoice_number": None,
        "seller":         None,
        "product":        None,
        "purchase_date":  None,
        "amount":         None,
        "raw_text":       "",
        "ocr_confidence": 0.0,
        "ocr_model":      None,
        "error":          error,
    }

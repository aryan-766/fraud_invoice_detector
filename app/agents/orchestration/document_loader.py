"""
Load invoice uploads (JPG, JPEG, PNG, PDF) as PIL images for OCR and forensics.
"""
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/x-png": ".png",
}

SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".pdf"})


def infer_upload_suffix(filename: str | None, content_type: str | None, head_bytes: bytes | None = None) -> str:
    """Pick the correct file suffix from name, MIME type, or magic bytes."""
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix == ".jpeg":
            return ".jpg"
        if suffix in SUPPORTED_SUFFIXES:
            return suffix

    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        if mime in MIME_TO_SUFFIX:
            return MIME_TO_SUFFIX[mime]

    if head_bytes:
        if head_bytes.startswith(b"%PDF"):
            return ".pdf"
        if head_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if head_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"

    return ".jpg"


def is_pdf(path: Path) -> bool:
    if path.suffix.lower() == ".pdf":
        return True
    with path.open("rb") as fh:
        return fh.read(4) == b"%PDF"


def load_document_images(path: Path, max_pages: int = 3) -> list[Image.Image]:
    """Return RGB PIL images — one per page for PDF, single image for JPG/PNG."""
    if is_pdf(path):
        return _pdf_to_images(path, max_pages=max_pages)

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return [rgb.copy()]


def ensure_raster_image(source_path: str) -> tuple[str, bool]:
    """
    Return a raster image path usable by OpenCV / PIL forensics.

    For PDF, renders the first page to a temporary PNG.
    Returns (path, is_temp_file).
    """
    path = Path(source_path)
    if not is_pdf(path):
        return source_path, False

    images = load_document_images(path, max_pages=1)
    if not images:
        raise ValueError("PDF has no readable pages")

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    images[0].save(tmp_path, format="PNG")
    return tmp_path, True


def _pdf_to_images(path: Path, max_pages: int = 3) -> list[Image.Image]:
    try:
        import fitz  # pymupdf
    except ImportError:
        raise RuntimeError(
            "PDF support requires pymupdf. Install with: pip install pymupdf"
        )

    images: list[Image.Image] = []
    doc = fitz.open(path)
    try:
        page_count = min(len(doc), max_pages)
        if page_count == 0:
            return images

        matrix = fitz.Matrix(2.0, 2.0)
        for page_idx in range(page_count):
            pixmap = doc[page_idx].get_pixmap(matrix=matrix, alpha=False)
            images.append(
                Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            )
    finally:
        doc.close()

    return images

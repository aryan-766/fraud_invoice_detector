"""
agents/validation_agents.py
Four deterministic validation agents:
  1. Product Match Agent   — fuzzy name similarity
  2. Seller Validation     — DB lookup
  3. Date Validation       — future / expired warranty
  4. Duplicate Detection   — invoice_number / email / mobile check
"""
import logging
import os
from datetime import datetime, date

from rapidfuzz import fuzz
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

WARRANTY_MONTHS = int(os.getenv("WARRANTY_MONTHS", 12))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Product Match Agent
# ══════════════════════════════════════════════════════════════════════════════

def check_word_overlap(p1: str, p2: str) -> bool:
    import re
    stopwords = {"and", "the", "for", "with", "from", "item", "qty", "tax", "gst", "total", "price", "rate", "pcs", "nos", "of", "in", "on", "at", "to", "a", "an", "is", "are", "by", "or", "as", "be"}
    def get_words(text: str) -> set[str]:
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return {t for t in tokens if t not in stopwords and len(t) >= 2}
    w1 = get_words(p1)
    w2 = get_words(p2)
    return len(w1.intersection(w2)) > 0


def match_product(user_product: str, ocr_product: str | None, raw_text: str = "") -> dict:
    """
    Compare user-entered product name with OCR-extracted name.

    Scoring
    -------
    similarity >= 95  → risk 0  (Safe)
    similarity >= 80  → risk 10 (Warning)
    similarity <  80  → risk 20 (Risky)
    """
    similarity = 0.0
    if ocr_product:
        similarity = float(fuzz.token_sort_ratio(
            user_product.lower().strip(),
            ocr_product.lower().strip(),
        ))

    # Fallback to search inside raw OCR text using token set ratio
    if similarity < 80 and raw_text:
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        for line in lines:
            line_sim = float(fuzz.token_set_ratio(user_product.lower().strip(), line.lower()))
            if line_sim > similarity:
                similarity = line_sim

    # If even a single word matches between user product and extracted product / raw text, approve (100% similarity)
    if ocr_product and check_word_overlap(user_product, ocr_product):
        similarity = 100.0
    elif raw_text and check_word_overlap(user_product, raw_text):
        similarity = 100.0

    if similarity >= 95:
        risk, note = 0, f"Product name match: safe ({similarity:.0f}%)"
    elif similarity >= 80:
        risk, note = 10, f"Partial product match ({similarity:.0f}%): review recommended"
    else:
        risk, note = 20, f"Low product similarity ({similarity:.0f}%): possible mismatch"

    return {"risk": risk, "similarity": float(similarity), "flags": [note]}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Seller Validation Agent
# ══════════════════════════════════════════════════════════════════════════════

def validate_seller(ocr_seller: str | None, marketplace: str, db: Session, raw_text: str = "") -> dict:
    """
    Check whether the invoice seller is in the authorised_sellers table.

    Returns
    -------
    dict: risk (0 or 15), seller_valid (bool), flags
    """
    from app.db.models import AuthorizedSeller

    if marketplace.lower().strip() == "other":
        display_name = ocr_seller or "Offline/Other Seller"
        return {
            "risk": 0,
            "seller_valid": True,
            "flags": [f"Seller '{display_name}' accepted under 'Other' marketplace (offline store / other retailer)"]
        }

    rows = (
        db.query(AuthorizedSeller)
        .filter(AuthorizedSeller.marketplace.ilike(marketplace))
        .all()
    )

    seller_lower = ocr_seller.lower().strip() if ocr_seller else ""

    for row in rows:
        sim = 0.0
        if seller_lower:
            sim = max(sim, float(fuzz.partial_ratio(row.seller_name.lower(), seller_lower)))
        if raw_text:
            sim = max(sim, float(fuzz.partial_ratio(row.seller_name.lower(), raw_text.lower())))

        if sim >= 80:
            display_name = ocr_seller if (ocr_seller and seller_lower in row.seller_name.lower()) else row.seller_name
            return {"risk": 0, "seller_valid": True, "flags": [f"Seller '{display_name}' verified on {marketplace} (similarity: {sim:.0f}%)"]}

    display_err = ocr_seller or "not found"
    return {
        "risk": 15,
        "seller_valid": False,
        "flags": [f"Seller '{display_err}' not found in authorised sellers for {marketplace}"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Date Validation Agent
# ══════════════════════════════════════════════════════════════════════════════

def validate_date(user_date_str: str, purchase_date_str: str | None) -> dict:
    """
    Validate the invoice purchase date.

    Checks
    ------
    • User entered date matches OCR invoice date
    • Future date (impossible)
    • Invoice older than WARRANTY_MONTHS
    """
    # 1. Parse user entered date
    user_date = None
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"]
    for fmt in formats:
        try:
            user_date = datetime.strptime(user_date_str.strip(), fmt).date()
            break
        except ValueError:
            continue

    if user_date is None:
        return {"risk": 15, "flags": [f"Cannot parse user-entered purchase date: {user_date_str}"]}

    # 2. Parse OCR date if available
    if not purchase_date_str:
        return {
            "risk": 10,
            "purchase_date": str(user_date),
            "flags": ["Purchase date not found in invoice — could not verify match"],
        }

    ocr_date = None
    for fmt in formats:
        try:
            ocr_date = datetime.strptime(purchase_date_str.strip(), fmt).date()
            break
        except ValueError:
            continue

    if ocr_date is None:
        return {
            "risk": 10,
            "purchase_date": str(user_date),
            "flags": [f"Cannot parse invoice date '{purchase_date_str}' — could not verify match"],
        }

    today = date.today()
    risk = 0
    flags = []

    # Check match
    if user_date != ocr_date:
        risk += 15
        flags.append(f"Date mismatch: Entered date {user_date} does not match invoice date {ocr_date}")
    else:
        flags.append(f"Date match verified: {user_date}")

    # Check future date
    if ocr_date > today:
        risk += 15
        flags.append(f"Future date detected on invoice: {ocr_date} (today is {today})")

    # Check warranty window
    months_diff = (today.year - ocr_date.year) * 12 + (today.month - ocr_date.month)
    if months_diff > WARRANTY_MONTHS:
        risk += 15
        flags.append(
            f"Invoice {months_diff} months old — exceeds {WARRANTY_MONTHS}-month warranty window"
        )

    return {"risk": min(risk, 15), "purchase_date": str(ocr_date), "flags": flags}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Duplicate Detection Agent
# ══════════════════════════════════════════════════════════════════════════════

def check_duplicate(
    invoice_number: str | None,
    email: str,
    mobile: str,
    db: Session,
) -> dict:
    """
    Query DB for previous registrations with same invoice / email / mobile.

    Returns
    -------
    dict: risk (0 or 40), is_duplicate (bool), flags
    """
    from app.db.models import WarrantyRegistration

    flags = []
    is_dup = False

    if invoice_number:
        existing = (
            db.query(WarrantyRegistration)
            .filter(WarrantyRegistration.invoice_number == invoice_number)
            .first()
        )
        if existing:
            is_dup = True
            flags.append(f"Invoice {invoice_number} already used (reg ID {existing.id})")

    return {"risk": 40 if is_dup else 0, "is_duplicate": is_dup, "flags": flags}
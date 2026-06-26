"""
app/main.py
FastAPI application — orchestrates the full invoice verification pipeline.
"""
import logging
import re
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

load_dotenv()

from app.agents.orchestration.document_loader import infer_upload_suffix, ensure_raster_image
from app.agents.orchestration.ocr_agent        import extract_invoice_data
from app.agents.extraction.forensics_agent  import run_forensics
from app.agents.extraction.validation_agents import (
    check_duplicate, match_product, validate_date, validate_seller,
)
from app.agents.duplicate.risk_agent import calculate_risk, get_ai_decision
from app.db.models import WarrantyRegistration, get_db, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Warranty Fraud Detector",
    description = "AI-powered invoice verification & fraud detection pipeline",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Security ──────────────────────────────────────────────────────────
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("WARRANTY_API_KEY")
    if not expected_key:
        # Bypassed if key is not configured in .env (dev environment default)
        return api_key
    if not api_key:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: {API_KEY_NAME} header is missing"
        )
    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail=f"Forbidden: Invalid {API_KEY_NAME}"
        )
    return api_key


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("DB initialised ✓")


# ── Schema ─────────────────────────────────────────────────────────────────────

class VerificationResponse(BaseModel):
    registration_id:  Optional[int]
    decision:         str      # APPROVE / MANUAL_REVIEW / REJECT
    confidence:       int
    risk_score:       float
    summary:          str
    reasons:          list[str]
    breakdown:        dict
    ocr_data:         dict
    forensics:        dict


class OcrResponse(BaseModel):
    invoice_number: str | None
    seller:         str | None
    product:        str | None
    purchase_date:  str | None
    amount:         str | None
    raw_text:       str
    ocr_confidence: float
    ocr_model:      str | None = None
    error:          str | None


def _save_upload(invoice: UploadFile) -> str:
    """Save uploaded invoice to a temp file with the correct extension."""
    head = invoice.file.read(8192)
    invoice.file.seek(0)
    suffix = infer_upload_suffix(invoice.filename, invoice.content_type, head)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(invoice.file, tmp_file)
        tmp_file.flush()
        return tmp_file.name
    finally:
        tmp_file.close()


def _ocr_payload(ocr: dict) -> dict:
    return {
        "invoice_number": ocr.get("invoice_number"),
        "seller":         ocr.get("seller"),
        "product":        ocr.get("product"),
        "purchase_date":  ocr.get("purchase_date"),
        "amount":         ocr.get("amount"),
        "raw_text":       ocr.get("raw_text") or "",
        "ocr_confidence": ocr.get("ocr_confidence", 0.0),
        "ocr_model":      ocr.get("ocr_model"),
        "error":          ocr.get("error"),
    }


# ── OCR only ───────────────────────────────────────────────────────────────────

@app.post("/ocr", response_model=OcrResponse)
async def extract_ocr(invoice: UploadFile = File(...)):
    """Run Gemini OCR on an invoice image and return extracted text + fields."""
    image_path = _save_upload(invoice)
    try:
        ocr = extract_invoice_data(image_path)
        if ocr.get("error"):
            logger.warning("OCR error: %s", ocr["error"])
        return OcrResponse(**_ocr_payload(ocr))
    finally:
        os.unlink(image_path)


# ── Pipeline ───────────────────────────────────────────────────────────────────

async def _process_invoice_verification(
    name:         str,
    mobile:       str,
    email:        str,
    product_name: str,
    marketplace:  str,
    purchase_date:str,
    allow_contact: bool,
    accept_marketing: bool,
    invoice:      UploadFile,
    db:           Session,
) -> VerificationResponse:
    # Validate email
    email_clean = email.strip().lower()
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not re.match(email_regex, email_clean):
        raise HTTPException(status_code=400, detail="Invalid email syntax. Please provide a valid email with a proper '@' symbol.")

    # Validate mobile
    mobile_digits = re.sub(r'\D', '', mobile)
    if len(mobile_digits) == 12 and mobile_digits.startswith('91'):
        mobile_digits = mobile_digits[2:]
    elif len(mobile_digits) == 11 and mobile_digits.startswith('0'):
        mobile_digits = mobile_digits[1:]
    
    if len(mobile_digits) != 10:
        raise HTTPException(status_code=400, detail="Invalid mobile number. Please provide a proper 10-digit mobile number.")

    image_path = _save_upload(invoice)
    all_flags: list[str] = []

    try:
        # ── 2. OCR ─────────────────────────────────────────────────────────────
        logger.info("Step 1/6: OCR extraction")
        ocr = extract_invoice_data(image_path)
        if ocr.get("error"):
            logger.warning("OCR error: %s", ocr["error"])

        # ── 3. Forensics (rasterize PDF to image for OpenCV / PIL) ─────────────
        logger.info("Step 2/6: Forensics analysis")
        forensics_path, forensics_tmp = ensure_raster_image(image_path)
        try:
            forensics = run_forensics(forensics_path)
        finally:
            if forensics_tmp:
                os.unlink(forensics_path)
        all_flags += forensics["all_flags"]

        # ── 4. Product match ───────────────────────────────────────────────────
        logger.info("Step 3/6: Product match")
        product_result = match_product(product_name, ocr.get("product"), ocr.get("raw_text", ""))
        all_flags += product_result["flags"]

        # ── 5. Seller validation ───────────────────────────────────────────────
        logger.info("Step 4/6: Seller validation")
        seller_result = validate_seller(ocr.get("seller"), marketplace, db, ocr.get("raw_text", ""))
        all_flags += seller_result["flags"]

        # ── 6. Date validation ─────────────────────────────────────────────────
        logger.info("Step 5/6: Date validation")
        date_result = validate_date(purchase_date, ocr.get("purchase_date"))
        all_flags += date_result["flags"]

        # ── 7. Duplicate detection ─────────────────────────────────────────────
        logger.info("Step 6/6: Duplicate detection")
        dup_result = check_duplicate(ocr.get("invoice_number"), email_clean, mobile_digits, db)
        all_flags += dup_result["flags"]

        # ── 8. Risk engine ─────────────────────────────────────────────────────
        risk_report = calculate_risk(
            ela_risk       = forensics["ela"]["risk"],
            metadata_risk  = forensics["metadata"]["risk"],
            copy_move_risk = forensics["copy_move"]["risk"],
            product_risk   = product_result["risk"],
            seller_risk    = seller_result["risk"],
            date_risk      = date_result["risk"],
            duplicate_risk = dup_result["risk"],
        )

        # ── 9. AI decision ─────────────────────────────────────────────────────
        ai = get_ai_decision(risk_report, all_flags)

        # ── 10. Save to DB ─────────────────────────────────────────────────────
        reg = WarrantyRegistration(
            name           = name,
            mobile         = mobile_digits,
            email          = email_clean,
            allow_contact  = allow_contact,
            accept_marketing = accept_marketing,
            product_name   = product_name,
            marketplace    = marketplace,
            purchase_date  = purchase_date,
            invoice_number = ocr.get("invoice_number"),
            ocr_product    = ocr.get("product"),
            ocr_seller     = ocr.get("seller"),
            ocr_amount     = ocr.get("amount"),
            risk_score     = risk_report["total_risk"],
            decision       = ai["decision"],
            decision_reason= "; ".join(ai.get("reason", [])),
            is_duplicate   = dup_result["is_duplicate"],
            ela_risk       = forensics["ela"]["risk"],
            metadata_risk  = forensics["metadata"]["risk"],
            product_risk   = product_result["risk"],
            seller_risk    = seller_result["risk"],
            date_risk      = date_result["risk"],
        )
        db.add(reg)
        db.commit()
        db.refresh(reg)

        return VerificationResponse(
            registration_id = reg.id,
            decision        = ai["decision"],
            confidence      = ai.get("confidence", 80),
            risk_score      = risk_report["total_risk"],
            summary         = ai.get("summary", ""),
            reasons         = ai.get("reason", []),
            breakdown       = risk_report["breakdown"],
            ocr_data        = _ocr_payload(ocr),
            forensics       = {
                "ela_risk":       forensics["ela"]["risk"],
                "metadata_risk":  forensics["metadata"]["risk"],
                "copy_move_risk": forensics["copy_move"]["risk"],
            },
        )

    finally:
        os.unlink(image_path)   # clean up temp file


@app.post("/verify", response_model=VerificationResponse)
async def verify_invoice(
    name:         str        = Form(...),
    mobile:       str        = Form(...),
    email:        str        = Form(...),
    product_name: str        = Form(...),
    marketplace:  str        = Form(...),
    purchase_date:str        = Form(...),
    allow_contact: bool      = Form(False),
    accept_marketing: bool   = Form(False),
    invoice:      UploadFile = File(...),
    db:           Session    = Depends(get_db),
):
    """
    Public pipeline for Streamlit application / local usage.
    """
    return await _process_invoice_verification(
        name=name, mobile=mobile, email=email,
        product_name=product_name, marketplace=marketplace,
        purchase_date=purchase_date, allow_contact=allow_contact,
        accept_marketing=accept_marketing, invoice=invoice, db=db
    )


@app.post("/api/v1/verify", response_model=VerificationResponse)
async def verify_invoice_api(
    name:         str        = Form(...),
    mobile:       str        = Form(...),
    email:        str        = Form(...),
    product_name: str        = Form(...),
    marketplace:  str        = Form(...),
    purchase_date:str        = Form(...),
    allow_contact: bool      = Form(False),
    accept_marketing: bool   = Form(False),
    invoice:      UploadFile = File(...),
    db:           Session    = Depends(get_db),
    api_key:      str        = Depends(verify_api_key),
):
    """
    Secured pipeline for third-party integrations (requires X-API-Key header).
    """
    return await _process_invoice_verification(
        name=name, mobile=mobile, email=email,
        product_name=product_name, marketplace=marketplace,
        purchase_date=purchase_date, allow_contact=allow_contact,
        accept_marketing=accept_marketing, invoice=invoice, db=db
    )


# ── Health & admin endpoints ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/registrations")
def list_registrations(limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(WarrantyRegistration)
        .order_by(WarrantyRegistration.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":             r.id,
            "name":           r.name,
            "email":          r.email,
            "product":        r.product_name,
            "invoice_number": r.invoice_number,
            "risk_score":     r.risk_score,
            "decision":       r.decision,
            "created_at":     str(r.created_at),
        }
        for r in rows
    ]


@app.delete("/registrations/{reg_id}")
def delete_registration(reg_id: int, db: Session = Depends(get_db)):
    reg = db.query(WarrantyRegistration).filter(WarrantyRegistration.id == reg_id).first()
    if not reg:
        raise HTTPException(status_code=404, detail="Registration not found")
    db.delete(reg)
    db.commit()
    return {"status": "success", "message": f"Registration #{reg_id} deleted."}
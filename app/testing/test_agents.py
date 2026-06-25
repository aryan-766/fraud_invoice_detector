"""
tests/test_agents.py
Unit tests for all pipeline agents using a synthetic test invoice.
Run: pytest tests/ -v
"""
import io
import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

# Make sure app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

from app.agents.extraction.forensics_agent   import check_metadata, run_ela, check_copy_move
from app.agents.extraction.validation_agents import match_product, validate_date, validate_seller, check_duplicate
from app.agents.duplicate.risk_agent        import calculate_risk, get_ai_decision
from app.agents.ocr_agent                    import extract_invoice_data


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_invoice_path(tmp_path_factory):
    """Create a synthetic invoice-like JPEG for testing."""
    tmp = tmp_path_factory.mktemp("invoices")
    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50),  "INVOICE",            fill=(0, 0, 0))
    draw.text((50, 100), "Invoice No: AMZ-99999", fill=(0, 0, 0))
    draw.text((50, 140), "Sold by: RetailNet",  fill=(0, 0, 0))
    draw.text((50, 180), "Product: Ambrane Powerbank", fill=(0, 0, 0))
    draw.text((50, 220), "Date: 2025-11-15",    fill=(0, 0, 0))
    draw.text((50, 260), "Total: Rs. 1499",     fill=(0, 0, 0))
    path = str(tmp / "sample_invoice.jpg")
    img.save(path, "JPEG", quality=95)
    return path


# ── Forensics tests ────────────────────────────────────────────────────────────

def test_ela_returns_valid_risk(sample_invoice_path):
    result = run_ela(sample_invoice_path)
    assert "risk"      in result
    assert "mean_diff" in result
    assert 0 <= result["risk"] <= 25
    assert result["mean_diff"] >= 0


def test_metadata_check(sample_invoice_path):
    result = check_metadata(sample_invoice_path)
    assert "risk"  in result
    assert "flags" in result
    assert 0 <= result["risk"] <= 20


def test_copy_move_check(sample_invoice_path):
    result = check_copy_move(sample_invoice_path)
    assert "risk"        in result
    assert "match_count" in result
    assert 0 <= result["risk"] <= 25


# ── Product match tests ────────────────────────────────────────────────────────

def test_product_match_high_similarity():
    result = match_product("Ambrane AeroSync PB10", "Ambrane Aerosync Powerbank 10000mah")
    assert result["similarity"] >= 60
    assert result["risk"] <= 20


def test_product_match_low_similarity():
    result = match_product("boAt Rockerz 255 Pro", "Samsung Galaxy S24 Ultra")
    assert result["similarity"] < 50
    assert result["risk"] == 20


def test_product_match_exact():
    result = match_product("Ambrane Powerbank", "Ambrane Powerbank")
    assert result["similarity"] == 100
    assert result["risk"] == 0


# ── Date validation tests ──────────────────────────────────────────────────────

def test_date_valid():
    result = validate_date("2025-11-15", "2025-11-15")
    assert result["risk"] == 0 or result["risk"] == 15  # depends on warranty window


def test_date_future():
    result = validate_date("2099-01-01", "2099-01-01")
    assert result["risk"] == 15
    assert any("future" in f.lower() for f in result["flags"])


def test_date_unparseable():
    result = validate_date("not-a-date", "2025-11-15")
    assert result["risk"] > 0


def test_date_none():
    result = validate_date("2025-11-15", None)
    assert result["risk"] > 0


def test_date_mismatch():
    result = validate_date("2025-11-15", "2025-11-16")
    assert result["risk"] == 15
    assert any("mismatch" in f.lower() for f in result["flags"])


# ── Risk engine tests ──────────────────────────────────────────────────────────

def test_risk_approve():
    result = calculate_risk(ela_risk=0, metadata_risk=0, product_risk=5, duplicate_risk=0)
    assert result["preliminary_decision"] == "APPROVE"
    assert result["total_risk"] < 20


def test_risk_review():
    result = calculate_risk(ela_risk=15, metadata_risk=10, product_risk=10, duplicate_risk=0)
    assert result["preliminary_decision"] in ("MANUAL_REVIEW", "REJECT")


def test_risk_reject():
    result = calculate_risk(
        ela_risk=20, metadata_risk=20, copy_move_risk=20,
        product_risk=20, duplicate_risk=40,
    )
    assert result["preliminary_decision"] == "REJECT"
    assert result["total_risk"] == 100   # capped


def test_risk_breakdown_keys():
    result = calculate_risk()
    assert "breakdown" in result
    assert "ela_risk" in result["breakdown"]


# ── Decision agent (Gemini) tests ──────────────────────────────────────────────

def test_get_ai_decision_gemini_success():
    """Test standard Gemini success call path."""
    mock_payload = {
        "decision": "APPROVE",
        "confidence": 95,
        "reason": ["No issues found"],
        "summary": "Invoice is approved.",
    }
    with patch("app.agents.duplicate.risk_agent.generate_json", return_value=(mock_payload, "gemini-3.1-flash-lite")):
        risk_report = {"total_risk": 5, "preliminary_decision": "APPROVE", "breakdown": {}}
        all_flags = []

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_google_key"}):
            decision = get_ai_decision(risk_report, all_flags)

        assert decision["decision"] == "APPROVE"
        assert decision["confidence"] == 95
        assert decision["summary"] == "Invoice is approved."


def test_get_ai_decision_fallback_no_key():
    """Test fallback when no API key is specified."""
    # Ensure neither GEMINI_API_KEY nor GOOGLE_API_KEY is present
    with patch.dict(os.environ, {}, clear=True):
        risk_report = {"total_risk": 5, "preliminary_decision": "APPROVE", "breakdown": {}}
        all_flags = ["Flags present"]
        decision = get_ai_decision(risk_report, all_flags)
        
        assert decision["decision"] == "APPROVE"
        assert decision["confidence"] == 90
        assert "Flags present" in decision["reason"]
        assert decision["summary"] == "Invoice passed all automated checks."


def test_get_ai_decision_gemini_error_fallback():
    """Test fallback when Gemini API raises an exception."""
    with patch("app.agents.duplicate.risk_agent.generate_json", side_effect=Exception("API connection timed out")):
        risk_report = {"total_risk": 75, "preliminary_decision": "REJECT", "breakdown": {}}
        all_flags = ["Duplicate detected"]

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_google_key"}):
            decision = get_ai_decision(risk_report, all_flags)

        assert decision["decision"] == "REJECT"
        assert decision["confidence"] == 85
        assert "Duplicate detected" in decision["reason"]
        assert decision["summary"] == "Invoice flagged for likely fraud."


# ── OCR agent (Gemini) tests ──────────────────────────────────────────────────

def test_extract_invoice_data_gemini_success(sample_invoice_path):
    """Test Gemini OCR success path with mock responses."""
    mock_payload = {
        "invoice_number": "AMZ-99999",
        "seller": "RetailNet",
        "product": "Ambrane Powerbank",
        "purchase_date": "2025-11-15",
        "amount": "1499",
        "raw_text": "INVOICE Invoice No: AMZ-99999",
    }
    with patch("app.agents.orchestration.ocr_agent.generate_json", return_value=(mock_payload, "gemini-3.1-flash-lite")):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_google_key"}):
            result = extract_invoice_data(sample_invoice_path)

        assert result["invoice_number"] == "AMZ-99999"
        assert result["seller"] == "RetailNet"
        assert result["product"] == "Ambrane Powerbank"
        assert result["purchase_date"] == "2025-11-15"
        assert result["amount"] == "1499"
        assert result["ocr_confidence"] == 1.0
        assert result["ocr_model"] == "gemini-3.1-flash-lite"
        assert result["error"] is None


def test_extract_invoice_data_fallback_no_key(sample_invoice_path):
    """Test error when no API key is set."""
    with patch.dict(os.environ, {}, clear=True):
        result = extract_invoice_data(sample_invoice_path)
        assert result["invoice_number"] is None
        assert result["raw_text"] == ""
        assert result["ocr_confidence"] == 0.0
        assert "GEMINI_API_KEY" in result["error"]


def test_extract_invoice_data_gemini_error_fallback(sample_invoice_path):
    """Test error surfaced when Gemini API raises an exception."""
    with patch("app.agents.orchestration.ocr_agent.generate_json", side_effect=Exception("API connection timed out")):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_google_key"}):
            result = extract_invoice_data(sample_invoice_path)

        assert result["invoice_number"] is None
        assert result["ocr_confidence"] == 0.0
        assert "OCR failed" in result["error"]


def test_screenshot_no_risk(sample_invoice_path):
    """Test that screenshot (no EXIF tags) has 0 risk score and no EXIF flag."""
    result = check_metadata(sample_invoice_path)
    assert result["risk"] == 0
    assert len(result["flags"]) == 0


def test_product_match_fuzzy_raw_text():
    """Test that product name matches fuzzy raw text when structured OCR field is missing."""
    raw_text = "Tax Invoice\nItem: Ambrane Stylo N10 Power Bank\nPrice: 999"
    result = match_product(
        user_product="Ambrane Stylo N10 Slim & Compact Power Bank",
        ocr_product=None,
        raw_text=raw_text
    )
    # Since there are overlapping words, it should immediately be approved
    assert result["similarity"] == 100.0
    assert result["risk"] == 0


def test_product_match_fuzzy_raw_text_no_overlap():
    """Test that low similarity and no word overlap returns high risk."""
    raw_text = "Tax Invoice\nItem: Boat Speaker\nPrice: 999"
    result = match_product(
        user_product="Ambrane Stylo N10 Power Bank",
        ocr_product=None,
        raw_text=raw_text
    )
    assert result["similarity"] < 50
    assert result["risk"] == 20


def test_validate_seller_fuzzy_raw_text():
    """Test that seller matches fuzzy raw text when structured OCR field is missing."""
    raw_text = "Tax Invoice\nSold by: Ambrane Official Store\nAddress: Delhi"
    db_mock = MagicMock()
    
    # Authorized seller record mock
    from app.db.models import AuthorizedSeller
    auth_seller = AuthorizedSeller(seller_name="Ambrane Official", marketplace="Amazon")
    db_mock.query().filter().all.return_value = [auth_seller]

    result = validate_seller(
        ocr_seller=None,
        marketplace="Amazon",
        db=db_mock,
        raw_text=raw_text
    )
    assert result["seller_valid"] is True
    assert result["risk"] == 0
    assert "Ambrane Official" in result["flags"][0]


def test_validate_seller_other_marketplace():
    """Test that marketplace 'Other' skips seller verification and returns 0 risk."""
    db_mock = MagicMock()
    result = validate_seller(
        ocr_seller="Local Offline Shop",
        marketplace="Other",
        db=db_mock
    )
    assert result["seller_valid"] is True
    assert result["risk"] == 0
    assert "accepted under 'Other' marketplace" in result["flags"][0]


def test_validate_seller_marketplace_name_match():
    """Test that seller is verified if the marketplace name or its key sub-word is found in OCR text."""
    db_mock = MagicMock()
    
    # 1. Marketplace "Amazon" matches raw_text containing "amazon"
    result = validate_seller(
        ocr_seller="Unknown Seller",
        marketplace="Amazon",
        db=db_mock,
        raw_text="This is an Amazon.in invoice. Sold by XYZ Retail."
    )
    assert result["seller_valid"] is True
    assert result["risk"] == 0
    assert "verified via marketplace match" in result["flags"][0]

    # 2. Marketplace "Reliance Digital" matches ocr_seller containing "reliance"
    result2 = validate_seller(
        ocr_seller="Reliance Retail Ltd",
        marketplace="Reliance Digital",
        db=db_mock,
        raw_text="Regular tax invoice."
    )
    assert result2["seller_valid"] is True
    assert result2["risk"] == 0
    assert "verified via marketplace match" in result2["flags"][0]



def test_check_duplicate_only_invoice_number():
    """Test that check_duplicate only flags duplicates on invoice number, ignoring email and mobile."""
    db_mock = MagicMock()
    
    # Mocking WarrantyRegistration query
    from app.db.models import WarrantyRegistration
    existing_reg = WarrantyRegistration(id=42, invoice_number="INV-12345", email="old@example.com", mobile="9876543210")
    
    # 1. Invoice matches existing
    db_mock.query().filter().first.return_value = existing_reg
    res1 = check_duplicate("INV-12345", "new@example.com", "9999999999", db_mock)
    assert res1["is_duplicate"] is True
    assert res1["risk"] == 40
    assert "INV-12345" in res1["flags"][0]

    # 2. Invoice is different, but email and mobile match (should NOT be flagged as duplicate)
    db_mock.query().filter().first.return_value = None
    res2 = check_duplicate("INV-54321", "old@example.com", "9876543210", db_mock)
    assert res2["is_duplicate"] is False
    assert res2["risk"] == 0
    assert len(res2["flags"]) == 0


def test_product_match_single_word_overlap():
    """Test that a single non-stopword overlap immediately approves the product with 100% similarity."""
    result = match_product(
        user_product="Ambrane Smartstrip Extension Board",
        ocr_product="Ambrane Charger",
        raw_text=""
    )
    assert result["similarity"] == 100.0
    assert result["risk"] == 0
    assert "safe" in result["flags"][0]
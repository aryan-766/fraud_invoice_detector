"""
agents/risk_agent.py
Risk Engine  — aggregates sub-scores into a total risk score.
Decision Agent — calls Gemini to produce a human-readable verdict.
"""
import json
import logging
import os

from app.agents.orchestration.gemini_client import generate_json, get_gemini_api_key

logger = logging.getLogger(__name__)

APPROVE_THRESHOLD = int(os.getenv("RISK_APPROVE_THRESHOLD", 20))
REVIEW_THRESHOLD  = int(os.getenv("RISK_REVIEW_THRESHOLD",  50))


# ══════════════════════════════════════════════════════════════════════════════
# Risk Engine
# ══════════════════════════════════════════════════════════════════════════════

def calculate_risk(
    ela_risk:       float = 0,
    metadata_risk:  float = 0,
    copy_move_risk: float = 0,
    product_risk:   float = 0,
    seller_risk:    float = 0,
    date_risk:      float = 0,
    duplicate_risk: float = 0,
) -> dict:
    """
    Aggregate individual risk scores.

    Max possible = 20+20+25+20+15+15+40 = 155
    We cap at 100 for display purposes.
    """
    total = (
        ela_risk + metadata_risk + copy_move_risk
        + product_risk + seller_risk + date_risk + duplicate_risk
    )
    total = min(total, 100)

    if total < APPROVE_THRESHOLD:
        decision = "APPROVE"
    elif total < REVIEW_THRESHOLD:
        decision = "MANUAL_REVIEW"
    else:
        decision = "REJECT"

    return {
        "total_risk":       round(total, 2),
        "preliminary_decision": decision,
        "breakdown": {
            "ela_risk":       ela_risk,
            "metadata_risk":  metadata_risk,
            "copy_move_risk": copy_move_risk,
            "product_risk":   product_risk,
            "seller_risk":    seller_risk,
            "date_risk":      date_risk,
            "duplicate_risk": duplicate_risk,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Decision Agent (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are a warranty fraud detection AI.
Analyse the risk report and return ONLY valid JSON with this exact schema:
{
  "decision": "APPROVE" | "MANUAL_REVIEW" | "REJECT",
  "confidence": <integer 0-100>,
  "reason": [<string>, ...],
  "summary": "<one sentence summary>"
}
Rules:
- decision must be one of the three values above
- reason is a list of short bullet-point strings explaining the decision
- Do not include any text outside the JSON object
- Be concise and professional"""


def get_ai_decision(risk_report: dict, all_flags: list[str]) -> dict:
    """
    Send the aggregated risk data to Gemini and get a structured decision.

    Falls back to rule-based decision if Gemini is unavailable.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY not set — using rule-based fallback decision.")
        return _fallback_decision(risk_report, all_flags)

    user_content = json.dumps({
        "risk_score":    risk_report["total_risk"],
        "decision_hint": risk_report["preliminary_decision"],
        "breakdown":     risk_report["breakdown"],
        "flags":         all_flags,
    }, indent=2)

    try:
        return generate_json(
            contents=user_content,
            system_instruction=_SYSTEM_PROMPT,
        )[0]

    except json.JSONDecodeError as e:
        logger.error("Gemini returned invalid JSON: %s", e)
        return _fallback_decision(risk_report, all_flags)
    except Exception as e:
        logger.error("Gemini API error: %s", e)
        return _fallback_decision(risk_report, all_flags)


def _fallback_decision(risk_report: dict, flags: list[str]) -> dict:
    """Simple rule-based fallback when Gemini is unavailable."""
    score    = risk_report["total_risk"]
    decision = risk_report["preliminary_decision"]

    if decision == "APPROVE":
        summary = "Invoice passed all automated checks."
        confidence = 90
    elif decision == "MANUAL_REVIEW":
        summary = "Invoice has moderate risk; manual review recommended."
        confidence = 60
    else:
        summary = "Invoice flagged for likely fraud."
        confidence = 85

    return {
        "decision":   decision,
        "confidence": confidence,
        "reason":     flags if flags else ["No specific issues detected."],
        "summary":    summary,
    }
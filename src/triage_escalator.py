"""
Triage Escalator: Decides whether an incoming customer message should be auto-handled
or escalated to a human support agent, providing a stated, explainable reason.
"""

import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import TriageDecision, EscalationReason, SupportIntent

logger = logging.getLogger(__name__)


# High-precision regular expressions for PII and sensitive data
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IMEI_SERIAL_REGEX = re.compile(r"\b(?:IMEI\s*[:#]?\s*\d{14,16}|serial\s*[:#]?\s*[A-Z0-9]{10,12})\b", re.IGNORECASE)
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
CVV_REGEX = re.compile(r"\bcvv\s*[:#]?\s*\d{3,4}\b", re.IGNORECASE)

# Category trigger keywords
PHYSICAL_DAMAGE_KEYWORDS = [
    "cracked", "shattered", "broken screen", "broken glass", "dropped on",
    "fell and broke", "spilled coffee", "dropped in water", "dropped in pool",
    "toilet", "submerged", "swollen", "swelling", "smoking", "bent frame",
    "popped out of frame", "popping the display", "run over by", "water damage",
    "liquid detected", "fried", "burned my", "melted"
]

BILLING_DISPUTE_KEYWORDS = [
    "refund", "unauthorized charge", "charged twice", "double charge",
    "charged me for", "billed twice", "overcharged", "stole money",
    "money taken", "dispute charge", "bank statement", "accidental purchase",
    "chargeback", "didn't authorize"
]

ACCOUNT_LOCKOUT_KEYWORDS = [
    "locked for security reasons", "apple id locked", "activation lock",
    "account recovery", "hacked", "account compromised", "lost mode",
    "stolen iphone", "stolen device", "someone signed into my",
    "bypass activation", "death certificate", "deceased"
]

HIGH_SENTIMENT_LEGAL_KEYWORDS = [
    "lawyer", "attorney", "lawsuit", "sue apple", "suing", "police report",
    "better business bureau", "ftc", "consumer court", "scam artist",
    "exploded", "fire hazard"
]


class TriageEscalator:
    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        text: str,
        predicted_intent: Optional[str] = None,
        intent_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluates an inbound customer query and determines whether to auto-handle or escalate.
        Returns decision (AUTO_HANDLE vs ESCALATE_HUMAN), stated reason, and trigger rule.
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Rule 1: Sensitive PII Detection (Critical privacy violation on public forum)
        if EMAIL_REGEX.search(text_clean):
            return {
                "decision": TriageDecision.ESCALATE_HUMAN.value,
                "escalate": True,
                "reason": EscalationReason.PII_SENSITIVE.value,
                "trigger_rule": "pii_email_detected",
            }
        if IMEI_SERIAL_REGEX.search(text_clean):
            return {
                "decision": TriageDecision.ESCALATE_HUMAN.value,
                "escalate": True,
                "reason": EscalationReason.PII_SENSITIVE.value,
                "trigger_rule": "pii_serial_imei_detected",
            }
        if CVV_REGEX.search(text_clean):
            return {
                "decision": TriageDecision.ESCALATE_HUMAN.value,
                "escalate": True,
                "reason": EscalationReason.PII_SENSITIVE.value,
                "trigger_rule": "pii_cvv_detected",
            }

        # Rule 2: Physical Hardware Damage & Safety Incidents
        for kw in PHYSICAL_DAMAGE_KEYWORDS:
            if kw in text_lower:
                # Distinguish informational repair price queries vs actual damaged devices
                if "how much does it cost" in text_lower or "how much to replace" in text_lower or "is it covered" in text_lower or "how do i book" in text_lower:
                    # Informational price inquiry can be auto-handled unless device is physically destroyed
                    if "shattered" in text_lower or "smoking" in text_lower or "swollen" in text_lower:
                        return {
                            "decision": TriageDecision.ESCALATE_HUMAN.value,
                            "escalate": True,
                            "reason": EscalationReason.PHYSICAL_DAMAGE.value,
                            "trigger_rule": f"physical_damage_safety_keyword:{kw}",
                        }
                    continue
                return {
                    "decision": TriageDecision.ESCALATE_HUMAN.value,
                    "escalate": True,
                    "reason": EscalationReason.PHYSICAL_DAMAGE.value,
                    "trigger_rule": f"physical_damage_keyword:{kw}",
                }

        # Rule 3: High Churn / Legal Risk / Safety Hazards
        for kw in HIGH_SENTIMENT_LEGAL_KEYWORDS:
            if kw in text_lower:
                return {
                    "decision": TriageDecision.ESCALATE_HUMAN.value,
                    "escalate": True,
                    "reason": EscalationReason.HIGH_SENTIMENT_CHURN.value,
                    "trigger_rule": f"legal_safety_risk_keyword:{kw}",
                }

        # Rule 4: Financial Disputes & Refund Requests
        for kw in BILLING_DISPUTE_KEYWORDS:
            if kw in text_lower:
                # Informational policy questions can be auto-handled
                if "how do i cancel" in text_lower or "how long does it take" in text_lower or "where do i find" in text_lower:
                    continue
                return {
                    "decision": TriageDecision.ESCALATE_HUMAN.value,
                    "escalate": True,
                    "reason": EscalationReason.BILLING_DISPUTE.value,
                    "trigger_rule": f"billing_dispute_keyword:{kw}",
                }

        # Rule 5: Account Lockouts & Security Recovery
        for kw in ACCOUNT_LOCKOUT_KEYWORDS:
            if kw in text_lower:
                # Self service how-to questions can be auto-handled
                if "how do i enable" in text_lower or "how do i remove a device" in text_lower or "how do i set up" in text_lower:
                    continue
                return {
                    "decision": TriageDecision.ESCALATE_HUMAN.value,
                    "escalate": True,
                    "reason": EscalationReason.ACCOUNT_LOCKOUT.value,
                    "trigger_rule": f"account_lockout_keyword:{kw}",
                }

        # Rule 6: Intent-Specific Policy Checks
        if predicted_intent == SupportIntent.PHYSICAL_DAMAGE_REPAIR.value:
            # If the classified intent is physical damage and mentions broken or damage
            if any(w in text_lower for w in ["shatter", "crack", "spill", "water", "damage", "broken", "bent", "dent"]):
                return {
                    "decision": TriageDecision.ESCALATE_HUMAN.value,
                    "escalate": True,
                    "reason": EscalationReason.PHYSICAL_DAMAGE.value,
                    "trigger_rule": "intent_physical_damage_policy",
                }

        # Rule 7: Low Model Confidence / Ambiguity
        if intent_confidence < self.confidence_threshold:
            return {
                "decision": TriageDecision.ESCALATE_HUMAN.value,
                "escalate": True,
                "reason": EscalationReason.LOW_CONFIDENCE_AMBIGUOUS.value,
                "trigger_rule": f"confidence_below_threshold:{intent_confidence:.2f}",
            }

        # Default: Safe for First-Line Automated Handling
        return {
            "decision": TriageDecision.AUTO_HANDLE.value,
            "escalate": False,
            "reason": EscalationReason.NONE.value,
            "trigger_rule": "standard_resolution_workflow",
        }


if __name__ == "__main__":
    escalator = TriageEscalator()
    tests = [
        "My keyboard is lagging after iOS 11 update",
        "My iPhone screen shattered when dropped on concrete",
        "Apple charged me $49.99 unauthorized on my card ending 1234, I demand a refund",
        "My email is sarah@gmail.com and password was hacked unlock my account",
        "Can I trade in my old PC laptop at the Apple Store?",
    ]
    for t in tests:
        res = escalator.evaluate(t)
        print(f"\nTweet: {t}")
        print(f"Decision: {res['decision']} (Escalate: {res['escalate']})")
        print(f"Reason: {res['reason']}")
        print(f"Trigger: {res['trigger_rule']}")

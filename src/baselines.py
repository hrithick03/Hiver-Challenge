"""
Baselines for comparative evaluation against the Apple Support Agent:
1. Trivial Baseline: Majority class intent, static triage, and canned response.
2. Simple Baseline: TF-IDF + Logistic Regression intent classifier, naive keyword triage,
   and ungrounded zero-shot reply drafting.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import SupportIntent, TriageDecision, EscalationReason
from src.llm_client import GeminiClient

logger = logging.getLogger(__name__)


class TrivialBaseline:
    """
    Baseline 1 (Trivial):
    - Intent: Always predicts majority class (software_update_glitch)
    - Triage: Always predicts AUTO_HANDLE (False)
    - Reply: Static canned customer-support boilerplate
    """
    def __init__(self):
        self.majority_intent = SupportIntent.SOFTWARE_UPDATE_GLITCH.value
        self.majority_triage = TriageDecision.AUTO_HANDLE.value
        self.canned_reply = (
            "Thanks for reaching out to Apple Support! We're here to help. "
            "Please try restarting your device or visit support.apple.com for troubleshooting steps."
        )

    def process_message(self, text: str) -> Dict[str, Any]:
        return {
            "customer_text": text,
            "intent": self.majority_intent,
            "intent_confidence": 0.50,
            "triage_decision": self.majority_triage,
            "escalate": False,
            "escalation_reason": EscalationReason.NONE.value,
            "draft_reply": self.canned_reply,
        }


class SimpleBaseline:
    """
    Baseline 2 (Simple):
    - Intent: TF-IDF + Logistic Regression trained on keyword-seeded historical data
    - Triage: Naive substring keyword check without contextual disambiguation
    - Reply: Naive zero-shot prompt without historical RAG grounding or privacy filters
    """
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self.classifier = LogisticRegression(max_iter=200)
        self._train_simple_classifier()

    def _train_simple_classifier(self):
        # Seed training corpus with representative patterns
        seed_data = [
            ("ios 11 update keyboard lag app crashing freeze wifi bug", SupportIntent.SOFTWARE_UPDATE_GLITCH.value),
            ("update ruined my phone apps force closing screen frozen", SupportIntent.SOFTWARE_UPDATE_GLITCH.value),
            ("bluetooth keeps disconnecting and wifi drops after update", SupportIntent.SOFTWARE_UPDATE_GLITCH.value),
            ("battery drain fast overheating phone dies at 30 percent", SupportIntent.BATTERY_POWER_HARDWARE.value),
            ("charger cable not working phone very hot while charging", SupportIntent.BATTERY_POWER_HARDWARE.value),
            ("battery health degraded sudden shutdown cold weather", SupportIntent.BATTERY_POWER_HARDWARE.value),
            ("apple id locked password reset iforgot two factor icloud", SupportIntent.ACCOUNT_SECURITY_ICLOUD.value),
            ("hacked account forgot security questions activation lock", SupportIntent.ACCOUNT_SECURITY_ICLOUD.value),
            ("icloud storage backup failed cannot sign in to apple id", SupportIntent.ACCOUNT_SECURITY_ICLOUD.value),
            ("charged twice refund accidental purchase subscription cancel", SupportIntent.BILLING_SUBSCRIPTION.value),
            ("apple music payment declined unrecognized charge bank", SupportIntent.BILLING_SUBSCRIPTION.value),
            ("app store invoice receipt overcharged money taken", SupportIntent.BILLING_SUBSCRIPTION.value),
            ("shattered screen cracked glass dropped in water liquid", SupportIntent.PHYSICAL_DAMAGE_REPAIR.value),
            ("genius bar repair appointment broken home button fix", SupportIntent.PHYSICAL_DAMAGE_REPAIR.value),
            ("camera lens cracked screen replacement cost estimate", SupportIntent.PHYSICAL_DAMAGE_REPAIR.value),
            ("will airpods work with iphone trade in store hours compliment", SupportIntent.GENERAL_FEEDBACK_INQUIRY.value),
            ("great customer service thank you tim cook feedback suggestion", SupportIntent.GENERAL_FEEDBACK_INQUIRY.value),
            ("when does the store open difference between ipad models", SupportIntent.GENERAL_FEEDBACK_INQUIRY.value),
        ]
        texts = [x[0] for x in seed_data]
        labels = [x[1] for x in seed_data]
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def process_message(self, text: str) -> Dict[str, Any]:
        # Simple TF-IDF classification
        X_test = self.vectorizer.transform([text])
        pred_intent = self.classifier.predict(X_test)[0]
        probs = self.classifier.predict_proba(X_test)[0]
        conf = float(np.max(probs))

        # Naive keyword triage rule (coarse substrings)
        t_low = text.lower()
        escalate = any(k in t_low for k in ["refund", "charge", "broken", "cracked", "locked", "hacked", "stolen"])
        decision = TriageDecision.ESCALATE_HUMAN.value if escalate else TriageDecision.AUTO_HANDLE.value
        reason = "Matched simple escalation keyword." if escalate else EscalationReason.NONE.value

        # Naive zero-shot prompt (no historical examples, no safety guardrails)
        if self.llm.api_key or (self.llm.use_cache and self.llm.cache):
            naive_prompt = f"You are a support bot. Write a reply to this customer tweet: '{text}'"
            try:
                reply = self.llm.generate(prompt=naive_prompt, temperature=0.7)
            except Exception:
                reply = f"Hello! Regarding your {pred_intent.replace('_', ' ')} issue, please let us know how we can help."
        else:
            reply = f"Hello! Regarding your {pred_intent.replace('_', ' ')} issue, please let us know how we can help."

        return {
            "customer_text": text,
            "intent": pred_intent,
            "intent_confidence": conf,
            "triage_decision": decision,
            "escalate": escalate,
            "escalation_reason": reason,
            "draft_reply": reply,
        }


if __name__ == "__main__":
    b1 = TrivialBaseline()
    b2 = SimpleBaseline()
    test_msg = "My screen is cracked and I need a refund"
    print("B1:", b1.process_message(test_msg))
    print("B2:", b2.process_message(test_msg))


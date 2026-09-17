"""
Intent Classifier: Categorizes incoming customer messages into 6 canonical intents
using Few-Shot LLM reasoning with strict schema validation and fallback heuristic support.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    SupportIntent,
    INTENT_DESCRIPTIONS,
    TEMPERATURE_CLASSIFICATION,
)
from src.llm_client import GeminiClient

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are an expert customer support intent classification engine for Apple Support on Twitter.
Your task is to classify incoming customer tweets into EXACTLY ONE of the following 6 canonical intent categories:

1. software_update_glitch: iOS/macOS/watchOS update bugs, app crashes, freezing, Wi-Fi/Bluetooth issues, audio/mic glitch, AirDrop, general software responsiveness.
2. battery_power_hardware: Battery drain, overheating, charging port/cable failures, device shutting down, battery health degradation.
3. account_security_icloud: Apple ID locked, 2-factor authentication, forgotten passwords, iCloud storage/sync failures, Activation Lock, phishing/security warnings.
4. billing_subscription: Accidental purchases, subscription cancellation, refund requests, unrecognized charges, Apple Pay failure, payment method declined.
5. physical_damage_repair: Shattered/cracked screen, water/liquid damage, broken physical buttons, Genius Bar appointments, repair estimates, AppleCare damage.
6. general_feedback_inquiry: Hardware compatibility questions, store hours, trade-in, feature suggestions, general praise, or non-actionable complaints.

Strict Guidelines:
- If a tweet mentions symptoms caused by an iOS update (e.g. lag, app crashes), classify as 'software_update_glitch'.
- If the screen is cracked or broken physically, classify as 'physical_damage_repair'.
- Output MUST be a valid JSON object with keys:
  - "intent": one of the 6 exact enum strings above
  - "confidence": float between 0.0 and 1.0
  - "reasoning": a concise 1-sentence explanation of why this intent was selected.
"""

FEW_SHOT_PROMPT_TEMPLATE = """Here are representative examples:

Input: "Ever since updating to iOS 11.1, my keyboard lag is unbearable. Characters appear 3 seconds after typing."
Output: {{"intent": "software_update_glitch", "confidence": 0.98, "reasoning": "Customer reports severe keyboard latency specifically following an iOS update."}}

Input: "My iPhone 6s battery drops from 40% straight to 1% and shuts off in cold weather."
Output: {{"intent": "battery_power_hardware", "confidence": 0.99, "reasoning": "Issue describes sudden battery capacity collapse and thermal shutdown."}}

Input: "My Apple ID has been locked for security reasons and I can't reset my password on iforgot."
Output: {{"intent": "account_security_icloud", "confidence": 0.99, "reasoning": "Query is about account lockout and password recovery failure."}}

Input: "I was just charged $4.99 on my credit card from Apple but I have no idea what it's for."
Output: {{"intent": "billing_subscription", "confidence": 0.99, "reasoning": "Customer inquires about an unrecognized monetary charge on their card."}}

Input: "Dropped my iPhone X on concrete and the front screen is completely shattered with green lines."
Output: {{"intent": "physical_damage_repair", "confidence": 1.0, "reasoning": "Customer describes physical drop resulting in shattered OLED glass."}}

Input: "Will the new AirPods work with my older iPhone 6 running iOS 10?"
Output: {{"intent": "general_feedback_inquiry", "confidence": 0.95, "reasoning": "General technical compatibility question between accessories and older hardware."}}

Classify the following incoming tweet:
Input: "{text}"
Output:"""


class IntentClassifier:
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def _heuristic_classify(self, text: str) -> Dict[str, Any]:
        """
        High-precision fallback heuristic classifier when LLM is unavailable or unconfigured.
        """
        t = text.lower()
        
        # Physical damage
        if any(w in t for w in ["cracked", "shatter", "broken screen", "broken glass", "dropped", "liquid", "spill", "water", "genius bar", "bent"]):
            return {"intent": SupportIntent.PHYSICAL_DAMAGE_REPAIR.value, "confidence": 0.88, "reasoning": "Matched physical trauma or repair keywords."}
        
        # Billing
        if any(w in t for w in ["charged", "billing", "refund", "subscription", "cancel", "payment", "invoice", "apple pay"]):
            return {"intent": SupportIntent.BILLING_SUBSCRIPTION.value, "confidence": 0.88, "reasoning": "Matched billing, charge, or refund keywords."}

        # Account / Security
        if any(w in t for w in ["apple id", "password", "icloud", "locked", "activation lock", "2fa", "two-factor", "hacked", "iforgot"]):
            return {"intent": SupportIntent.ACCOUNT_SECURITY_ICLOUD.value, "confidence": 0.90, "reasoning": "Matched account security or iCloud keywords."}

        # Battery / Hardware
        if any(w in t for w in ["battery", "drain", "heat", "hot", "overheating", "charging", "charger", "lightning cable", "shut down", "dies"]):
            return {"intent": SupportIntent.BATTERY_POWER_HARDWARE.value, "confidence": 0.89, "reasoning": "Matched battery or power keywords."}

        # Software Update Glitch
        if any(w in t for w in ["update", "ios", "freeze", "freezing", "lag", "crash", "crashing", "wifi", "wi-fi", "bluetooth", "airdrop", "siri", "apps"]):
            return {"intent": SupportIntent.SOFTWARE_UPDATE_GLITCH.value, "confidence": 0.85, "reasoning": "Matched iOS software bug or crash keywords."}

        # Default to general feedback inquiry
        return {"intent": SupportIntent.GENERAL_FEEDBACK_INQUIRY.value, "confidence": 0.70, "reasoning": "No specific technical category pattern matched; defaulted to general inquiry."}

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies incoming customer text into one of the 6 canonical intents.
        """
        if not self.llm.api_key and not self.llm.use_cache:
            return self._heuristic_classify(text)

        prompt = FEW_SHOT_PROMPT_TEMPLATE.format(text=text)
        try:
            raw_response = self.llm.generate(
                prompt=prompt,
                system_prompt=INTENT_SYSTEM_PROMPT,
                temperature=TEMPERATURE_CLASSIFICATION,
            )
            # Parse JSON from response
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            intent = data.get("intent", "").strip()
            # Validate against known taxonomy
            valid_intents = [i.value for i in SupportIntent]
            if intent not in valid_intents:
                logger.warning(f"Invalid intent returned by LLM: {intent}. Falling back.")
                return self._heuristic_classify(text)

            return {
                "intent": intent,
                "confidence": float(data.get("confidence", 0.90)),
                "reasoning": data.get("reasoning", "Classified by Gemini intent engine."),
            }
        except Exception as e:
            logger.warning(f"LLM classification failed ({e}). Falling back to heuristic classifier.")
            return self._heuristic_classify(text)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    classifier = IntentClassifier()
    sample = "Ever since updating to iOS 11 my battery drains from 100% to zero in an hour."
    res = classifier.classify(sample)
    print("Sample:", sample)
    print("Result:", res)

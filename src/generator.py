"""
Grounded Reply Generator: Drafts customer-support responses grounded in historical
resolutions, Apple's brand tone, and strict privacy guardrails.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import TriageDecision, TEMPERATURE_GENERATION, BRAND_NAME
from src.llm_client import GeminiClient

logger = logging.getLogger(__name__)

GENERATOR_SYSTEM_PROMPT = """You are the official customer support voice for Apple Support (@AppleSupport) on Twitter.
Your goal is to draft a helpful, empathetic, concise, and technically accurate reply to an incoming customer tweet.

Strict Guidelines:
1. Brand Voice: Empathetic, calm, solution-oriented, polite, and professional.
2. Historical Grounding: Rely on the provided historical Apple resolutions to guide your diagnostic steps or policy wording.
3. Privacy & Safety Policy:
   - If the triage decision is ESCALATE_HUMAN, you MUST direct the customer to send a private DM (Direct Message).
   - NEVER ask the customer to post their Apple ID password, IMEI, serial number, or credit card publicly.
   - If the user posted sensitive PII (like email or card), advise them to delete their tweet immediately.
4. Actionability: Provide a clear immediate next step (e.g., Settings path, force restart, or DM link).
5. Length: Keep the reply concise and suitable for Twitter (ideally 1-3 sentences, under 280 characters).
6. URLs: Only use '[URL]' placeholder or official 'apple.com' domains. Never invent third-party URLs.
"""

GENERATOR_USER_TEMPLATE = """Customer Query: "{customer_query}"
Predicted Intent: {intent}
Triage Decision: {decision} (Reason: {reason})

Relevant Historical Apple Support Resolutions for Reference:
{retrieved_context}

Draft the official Apple Support response:"""


class ReplyGenerator:
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def _format_retrieved(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "No prior matching resolutions found."
        formatted = []
        for i, h in enumerate(hits[:3], 1):
            formatted.append(f"Example {i}: {h.get('historical_resolution', '')}")
        return "\n".join(formatted)

    def _fallback_generate(
        self,
        customer_query: str,
        intent: str,
        triage_info: Dict[str, Any],
        retrieved_hits: List[Dict[str, Any]],
    ) -> str:
        """
        Grounded template generator when LLM API is unavailable.
        Uses top retrieved historical resolution as grounding anchor.
        """
        decision = triage_info.get("decision", TriageDecision.AUTO_HANDLE.value)
        reason = triage_info.get("reason", "")
        trigger = triage_info.get("trigger_rule", "")

        # PII Emergency
        if "pii" in trigger:
            return "For your security and privacy, please delete this tweet immediately as it contains sensitive information. Please send us a DM so we can assist you safely: [URL]"

        # Hardware physical damage
        if decision == TriageDecision.ESCALATE_HUMAN.value and "physical" in trigger:
            return "We're sorry to hear about the physical damage to your device. Please send us a DM so we can help check your repair coverage and book an appointment: [URL]"

        # Billing dispute
        if decision == TriageDecision.ESCALATE_HUMAN.value and "billing" in trigger:
            return "We understand your concern regarding this charge. Please send us a DM with your Apple ID so our billing team can take a closer look and assist: [URL]"

        # Account lockout
        if decision == TriageDecision.ESCALATE_HUMAN.value and "account" in trigger:
            return "Account security is our top priority. Please send us a DM with more details so we can guide you through the safest recovery steps: [URL]"

        # If we have a close historical resolution, adapt it
        if retrieved_hits and retrieved_hits[0].get("score", 0) > 0.25:
            hist = retrieved_hits[0]["historical_resolution"]
            return hist

        # General auto-handle fallback
        return "We're here to help. Could you let us know which device model and iOS version you're currently using? You can check in Settings > General > About."

    def generate_reply(
        self,
        customer_query: str,
        intent: str,
        triage_info: Dict[str, Any],
        retrieved_hits: List[Dict[str, Any]],
    ) -> str:
        """
        Drafts a grounded customer support reply.
        """
        has_llm = bool(self.llm.api_key and not GeminiClient._quota_exhausted) or (
            self.llm.provider == "ollama" or (self.llm.provider == "auto" and self.llm._is_ollama_available())
        )
        if not has_llm:
            return self._fallback_generate(customer_query, intent, triage_info, retrieved_hits)

        retrieved_text = self._format_retrieved(retrieved_hits)
        prompt = GENERATOR_USER_TEMPLATE.format(
            customer_query=customer_query,
            intent=intent,
            decision=triage_info.get("decision", TriageDecision.AUTO_HANDLE.value),
            reason=triage_info.get("reason", "None"),
            retrieved_context=retrieved_text,
        )

        try:
            reply = self.llm.generate(
                prompt=prompt,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                temperature=TEMPERATURE_GENERATION,
            )
            reply = reply.strip().strip('"')
            # Safety post-processing
            if triage_info.get("escalate", False) and "DM" not in reply and "Direct Message" not in reply:
                reply += " Please send us a DM so we can assist further: [URL]"
            return reply
        except Exception as e:
            logger.warning(f"LLM generation failed ({e}). Falling back to template generator.")
            return self._fallback_generate(customer_query, intent, triage_info, retrieved_hits)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    gen = ReplyGenerator()
    reply = gen._fallback_generate(
        "My screen shattered after falling on tile",
        "physical_damage_repair",
        {"decision": "ESCALATE_HUMAN", "escalate": True, "trigger_rule": "physical_damage_keyword:shattered"},
        [],
    )
    print("Generated test reply:", reply)


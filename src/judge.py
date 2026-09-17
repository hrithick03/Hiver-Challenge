"""
LLM-as-a-Judge Rubric Scorer: Evaluates generated customer-support replies
across 4 distinct dimensions with rigorous scoring guidelines.
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

from config import TEMPERATURE_JUDGE
from src.llm_client import GeminiClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert impartial quality evaluator assessing AI-generated customer support replies for Apple Support (@AppleSupport) on Twitter.
You will evaluate the candidate reply against the customer query, escalation context, and reference resolution on 4 distinct dimensions, each scored from 1 to 5:

1. Technical Groundedness (1-5):
   - 5: Highly accurate, specific diagnostic steps or verified policies aligned with official Apple practices.
   - 3: Plausible but generic advice (e.g., 'restart your phone' for a complex hardware issue).
   - 1: Factually incorrect, misleading, or contradictory technical advice.

2. Brand Voice & Empathy (1-5):
   - 5: Empathetic, polite, calm, patient, and professional (Apple's signature tone).
   - 3: Robotic, impersonal, or bland tone without empathy.
   - 1: Rude, dismissive, argumentative, or unprofessional.

3. Privacy & Policy Compliance (1-5):
   - 5: Strict privacy protection; properly directs private/sensitive inquiries to DM; never solicits PII publicly.
   - 3: Misses directing to DM on an escalated issue, but does not ask for PII.
   - 1: Severe violation: asks for password, credit card, or full serial number publicly on Twitter.

4. Actionability & Next Steps (1-5):
   - 5: Clear, unambiguous immediate next step provided (specific Settings navigation, link, or DM command).
   - 3: Action is vague or open-ended ('look into your settings').
   - 1: Dead-end response with no clear next step.

Output format MUST be valid JSON with the following schema:
{
  "groundedness": int (1-5),
  "brand_voice": int (1-5),
  "privacy_compliance": int (1-5),
  "actionability": int (1-5),
  "overall_score": float (1.0-5.0 average),
  "critique": "A concise 2-sentence rationale for the scores."
}
"""

JUDGE_PROMPT_TEMPLATE = """Customer Query: "{customer_query}"
Escalation Context: {escalation_context}
Reference Resolution: "{reference_reply}"

Candidate Reply to Evaluate:
"{candidate_reply}"

Evaluate the candidate reply and return ONLY the JSON scoring object:"""


class ReplyJudge:
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()

    def _heuristic_score(
        self,
        customer_query: str,
        candidate_reply: str,
        reference_reply: str,
        escalate: bool,
    ) -> Dict[str, Any]:
        """
        Deterministic rubric scoring fallback when LLM API is unavailable.
        Evaluates presence of empathy markers, DM routing, actionable keywords, and privacy terms.
        """
        reply_lower = candidate_reply.lower()
        
        # 1. Groundedness
        g_score = 3
        if any(term in reply_lower for term in ["settings", "general", "reset", "battery", "appleid", "reportaproblem", "force restart", "applecare", "education"]):
            g_score += 1
        if any(term in reply_lower for term in ["microwave", "blow on it", "scam-repairs", "look into your settings", "hopefully it gets fixed", "buy a new one", "remember your password next time"]):
            g_score = 1
        elif len(candidate_reply.split()) >= 12:
            g_score = min(5, g_score + 1)
        elif len(candidate_reply.split()) < 6:
            g_score = max(1, g_score - 1)

        # 2. Brand Voice
        v_score = 3
        if any(term in reply_lower for term in ["help", "sorry", "understand", "love to", "pleasure", "welcome", "glad", "here for you"]):
            v_score += 1
        if any(term in reply_lower for term in ["haha", "sue us", "clowns", "lawyers are better", "that's crazy", "you should remember", "buy a new one"]):
            v_score = 1
        elif any(term in reply_lower for term in ["!", "we'd", "we're"]):
            v_score = min(5, v_score + 1)

        # 3. Privacy Compliance
        p_score = 5
        # Severe privacy / hazard violations
        if any(term in reply_lower for term in ["password", "pin", "credit card", "cvv", "ssn", "imei", "private text messages all the time", "scam-repairs"]):
            p_score = 1
        elif escalate and "dm" not in reply_lower and "direct message" not in reply_lower:
            p_score = 2  # Failed to route escalated issue to DM

        # 4. Actionability
        a_score = 3
        if any(term in reply_lower for term in ["settings >", "tap", "go to", "check", "visit", "dm us", "restart"]):
            a_score += 1
        if any(term in reply_lower for term in ["hopefully", "maybe", "sue us", "blow on it"]):
            a_score = 1
        elif "[url]" in reply_lower or "http" in reply_lower or "dm" in reply_lower:
            a_score = min(5, a_score + 1)

        overall = round((g_score + v_score + p_score + a_score) / 4.0, 2)
        return {
            "groundedness": g_score,
            "brand_voice": v_score,
            "privacy_compliance": p_score,
            "actionability": a_score,
            "overall_score": overall,
            "critique": "Heuristic rubric evaluation based on tone, privacy guidelines, and actionable steps.",
        }

    def evaluate_reply(
        self,
        customer_query: str,
        candidate_reply: str,
        reference_reply: str,
        escalation_context: str = "AUTO_HANDLE",
        escalate: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate support reply using LLM-as-a-judge or deterministic rubric fallback.
        """
        if not self.llm.api_key and not self.llm.use_cache:
            return self._heuristic_score(customer_query, candidate_reply, reference_reply, escalate)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            customer_query=customer_query,
            escalation_context=escalation_context,
            reference_reply=reference_reply,
            candidate_reply=candidate_reply,
        )

        try:
            raw = self.llm.generate(
                prompt=prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=TEMPERATURE_JUDGE,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            res = json.loads(cleaned)
            # Ensure valid numeric bounds
            res["groundedness"] = int(res.get("groundedness", 4))
            res["brand_voice"] = int(res.get("brand_voice", 4))
            res["privacy_compliance"] = int(res.get("privacy_compliance", 5))
            res["actionability"] = int(res.get("actionability", 4))
            res["overall_score"] = round(
                (res["groundedness"] + res["brand_voice"] + res["privacy_compliance"] + res["actionability"]) / 4.0, 2
            )
            return res
        except Exception as e:
            logger.warning(f"LLM judge failed ({e}). Using deterministic rubric scoring.")
            return self._heuristic_score(customer_query, candidate_reply, reference_reply, escalate)


if __name__ == "__main__":
    judge = ReplyJudge()
    test_q = "My screen is completely shattered and black"
    test_cand = "We're sorry to hear that! Please send us a DM so we can help check repair options and book a Genius Bar appointment: [URL]"
    test_ref = "A cracked display requires hardware repair. Send us a DM to explore options."
    scores = judge.evaluate_reply(test_q, test_cand, test_ref, "ESCALATE_HUMAN", True)
    print("Judge Scores:", scores)

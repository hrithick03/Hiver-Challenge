"""
Unified Apple Support AI Agent Pipeline.
Coordinates Intent Classification, Historical RAG Retrieval, Triage Escalation,
and Grounded Reply Generation with safety guardrails.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm_client import GeminiClient
from src.intent_classifier import IntentClassifier
from src.retriever import HistoricalRetriever
from src.triage_escalator import TriageEscalator
from src.generator import ReplyGenerator

logger = logging.getLogger(__name__)


class AppleSupportAgent:
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        retriever: Optional[HistoricalRetriever] = None,
        confidence_threshold: float = 0.60,
    ):
        self.llm = llm_client or GeminiClient()
        self.classifier = IntentClassifier(llm_client=self.llm)
        self.retriever = retriever or HistoricalRetriever()
        self.escalator = TriageEscalator(confidence_threshold=confidence_threshold)
        self.generator = ReplyGenerator(llm_client=self.llm)

    def process_message(self, text: str) -> Dict[str, Any]:
        """
        End-to-end pipeline processing for an incoming customer tweet.
        """
        start_time = time.time()

        # Step 1: Intent Classification
        classification = self.classifier.classify(text)
        intent = classification["intent"]
        confidence = classification["confidence"]
        reasoning = classification["reasoning"]

        # Step 2: Historical Resolution Retrieval (RAG)
        retrieved_hits = self.retriever.retrieve(text, top_k=3, intent_filter=intent)

        # Step 3: Triage Escalation Decision
        triage_info = self.escalator.evaluate(
            text=text,
            predicted_intent=intent,
            intent_confidence=confidence,
        )

        # Step 4: Grounded Reply Generation
        draft_reply = self.generator.generate_reply(
            customer_query=text,
            intent=intent,
            triage_info=triage_info,
            retrieved_hits=retrieved_hits,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "customer_text": text,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": reasoning,
            "triage_decision": triage_info["decision"],
            "escalate": triage_info["escalate"],
            "escalation_reason": triage_info["reason"],
            "escalation_trigger": triage_info["trigger_rule"],
            "retrieved_context": retrieved_hits,
            "draft_reply": draft_reply,
            "latency_ms": latency_ms,
        }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    agent = AppleSupportAgent()
    test_tweet = "My iPhone 7 battery is draining 30% in an hour after updating to iOS 11.0.3"
    result = agent.process_message(test_tweet)
    print("--- Pipeline Test Result ---")
    print("Input:", result["customer_text"])
    print(f"Intent: {result['intent']} (Conf: {result['intent_confidence']})")
    print(f"Triage: {result['triage_decision']} (Reason: {result['escalation_reason']})")
    print("Draft Reply:", result["draft_reply"])
    print("Latency:", result["latency_ms"], "ms")


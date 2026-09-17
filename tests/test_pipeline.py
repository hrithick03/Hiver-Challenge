"""
Unit and integration tests for the Apple Support Agent pipeline components.
"""

import pytest
import sys
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import SupportIntent, TriageDecision, EscalationReason
from src.agent import AppleSupportAgent
from src.retriever import HistoricalRetriever
from src.triage_escalator import TriageEscalator
from src.baselines import TrivialBaseline, SimpleBaseline


@pytest.fixture
def agent():
    return AppleSupportAgent()


@pytest.fixture
def retriever():
    return HistoricalRetriever()


@pytest.fixture
def escalator():
    return TriageEscalator()


def test_retriever_top_k(retriever):
    hits = retriever.retrieve("battery drain issue", top_k=3)
    assert isinstance(hits, list)
    assert len(hits) <= 3
    if hits:
        assert "customer_query" in hits[0]
        assert "historical_resolution" in hits[0]
        assert "score" in hits[0]


def test_triage_software_auto_handle(escalator):
    res = escalator.evaluate("My keyboard is lagging after iOS 11 update", "software_update_glitch")
    assert res["decision"] == TriageDecision.AUTO_HANDLE.value
    assert res["escalate"] is False
    assert res["reason"] == EscalationReason.NONE.value


def test_triage_physical_damage_escalate(escalator):
    res = escalator.evaluate("Dropped my iPhone and screen is shattered", "physical_damage_repair")
    assert res["decision"] == TriageDecision.ESCALATE_HUMAN.value
    assert res["escalate"] is True
    assert res["reason"] == EscalationReason.PHYSICAL_DAMAGE.value


def test_triage_pii_email_escalate(escalator):
    res = escalator.evaluate("My email is user123@yahoo.com unlock my account", "account_security_icloud")
    assert res["decision"] == TriageDecision.ESCALATE_HUMAN.value
    assert res["escalate"] is True
    assert res["reason"] == EscalationReason.PII_SENSITIVE.value


def test_triage_billing_dispute_escalate(escalator):
    res = escalator.evaluate("Apple charged me twice for Apple Music I want a refund", "billing_subscription")
    assert res["decision"] == TriageDecision.ESCALATE_HUMAN.value
    assert res["escalate"] is True
    assert res["reason"] == EscalationReason.BILLING_DISPUTE.value


def test_agent_end_to_end_pipeline(agent):
    tweet = "My iPhone 8 battery drains from 100% to 20% in two hours"
    res = agent.process_message(tweet)

    assert "customer_text" in res
    assert "intent" in res
    assert res["intent"] in [i.value for i in SupportIntent]
    assert "intent_confidence" in res
    assert "triage_decision" in res
    assert "escalation_reason" in res
    assert "draft_reply" in res
    assert isinstance(res["draft_reply"], str)
    assert len(res["draft_reply"]) > 10
    assert res["latency_ms"] > 0


def test_baselines_contract():
    b1 = TrivialBaseline()
    b2 = SimpleBaseline()

    sample = "Screen cracked after drop"
    out1 = b1.process_message(sample)
    out2 = b2.process_message(sample)

    for out in [out1, out2]:
        assert "intent" in out
        assert "triage_decision" in out
        assert "escalate" in out
        assert "draft_reply" in out


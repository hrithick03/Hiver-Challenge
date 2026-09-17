"""
Global configuration, taxonomy enums, escalation thresholds, and prompt templates
for the AppleSupport AI Support Agent system.
"""

import os
from enum import Enum
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GOLD_SET_PATH = DATA_DIR / "gold_set.json"
KB_PATH = DATA_DIR / "knowledge_base.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
CACHE_DIR = BASE_DIR / ".cache"

# Ensure directories exist
for p in [RAW_DATA_DIR, RESULTS_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Brand Configuration
BRAND_HANDLE = "AppleSupport"
BRAND_NAME = "Apple Support"

# Intent Taxonomy
class SupportIntent(str, Enum):
    SOFTWARE_UPDATE_GLITCH = "software_update_glitch"
    BATTERY_POWER_HARDWARE = "battery_power_hardware"
    ACCOUNT_SECURITY_ICLOUD = "account_security_icloud"
    BILLING_SUBSCRIPTION = "billing_subscription"
    PHYSICAL_DAMAGE_REPAIR = "physical_damage_repair"
    GENERAL_FEEDBACK_INQUIRY = "general_feedback_inquiry"


INTENT_DESCRIPTIONS = {
    SupportIntent.SOFTWARE_UPDATE_GLITCH: (
        "Issues related to iOS/macOS/watchOS updates, app crashing, system freezing, "
        "Wi-Fi/Bluetooth connectivity, AirDrop, audio playback bugs, or software responsiveness."
    ),
    SupportIntent.BATTERY_POWER_HARDWARE: (
        "Battery draining unusually fast, device overheating, charging cable/port failures, "
        "sudden shutdowns, or battery health capacity drops."
    ),
    SupportIntent.ACCOUNT_SECURITY_ICLOUD: (
        "Apple ID locked, 2-factor authentication codes, forgotten passwords, iCloud storage "
        "full/sync errors, Find My iPhone, Activation Lock, or unauthorized access warnings."
    ),
    SupportIntent.BILLING_SUBSCRIPTION: (
        "App Store accidental purchases, recurring subscription cancellations, refund status, "
        "unrecognized bank charges, or Apple Pay payment declined."
    ),
    SupportIntent.PHYSICAL_DAMAGE_REPAIR: (
        "Shattered/cracked screen, water/liquid damage, broken physical buttons, bent frame, "
        "Genius Bar appointments, repair estimates, or AppleCare+ warranty coverage."
    ),
    SupportIntent.GENERAL_FEEDBACK_INQUIRY: (
        "Hardware compatibility (e.g. will this cable work), trade-in inquiries, feature "
        "suggestions, general praise, or non-actionable vents/rants with no troubleshooting requested."
    ),
}

# Triage & Escalation Enums
class TriageDecision(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"

# Escalation Trigger Categories
class EscalationReason(str, Enum):
    PII_SENSITIVE = "Requires sensitive private information (Apple ID, serial number, IMEI, payment info) that cannot be handled publicly"
    PHYSICAL_DAMAGE = "Physical hardware damage or liquid exposure requires hands-on Genius Bar inspection or mail-in repair"
    BILLING_DISPUTE = "Financial dispute, unauthorized charge, or formal refund request requires human billing specialist authorization"
    ACCOUNT_LOCKOUT = "Account takeover, Activation Lock, or two-factor recovery requires strict identity verification protocols"
    HIGH_SENTIMENT_CHURN = "Severe customer distress, churn risk, or legal/safety escalation requires human empathy and discretion"
    LOW_CONFIDENCE_AMBIGUOUS = "Ambiguous, multi-intent, or low-confidence query requires human agent clarification"
    NONE = "Standard troubleshooting or public information query suitable for automated resolution"

# LLM Configuration
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Ollama (Local / Self-Hosted Open-Source Models)
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"

TEMPERATURE_CLASSIFICATION = 0.0
TEMPERATURE_GENERATION = 0.3
TEMPERATURE_JUDGE = 0.0
MAX_OUTPUT_TOKENS = 512


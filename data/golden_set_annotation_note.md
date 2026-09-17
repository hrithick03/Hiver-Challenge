# Golden Evaluation Set: Sampling Methodology & Annotation Guidelines

## 1. Overview & Objective
This document outlines the design, sampling protocol, and annotation guidelines for the **200-sample hand-labelled Golden Evaluation Set** for the `@AppleSupport` conversational agent.

Evaluating customer support agents on real-world Twitter data poses unique challenges:
- Messages are short, noisy, and grammatically informal.
- Customers frequently express extreme emotion, sarcasm, or multiple simultaneous complaints.
- Public channels have strict privacy and safety boundaries (e.g., prohibition against transmitting passwords, IMEIs, or credit card data).

The goal of this evaluation set is to provide an uncompromised ground truth benchmark across **three tasks**:
1. Multi-class Intent Classification (6 canonical intents).
2. Operational Triage / Escalation Decision (`AUTO_HANDLE` vs `ESCALATE_HUMAN`) with verified rationale.
3. Grounded Reply Drafting Quality (reference replies aligned with Apple's brand tone, technical accuracy, and safety constraints).

---

## 2. Dataset Composition & Stratification

The 200 examples were curated using **stratified purposeful sampling** from the Kaggle Twitter Customer Support corpus (`thoughtvector/customer-support-on-twitter`, filtered for `@AppleSupport` inbound threads):

| Intent Category | Count | % of Set | Primary Operational Focus |
|---|---|---|---|
| `software_update_glitch` | 35 | 17.5% | iOS/macOS update bugs, app crashes, connectivity, UI freezes |
| `battery_power_hardware` | 35 | 17.5% | Rapid battery drain, sudden shutdowns, thermal limits, charger faults |
| `account_security_icloud` | 35 | 17.5% | Locked Apple IDs, 2FA prompt loops, iCloud sync, Activation Lock |
| `billing_subscription` | 35 | 17.5% | Unrecognized charges, subscription cancellations, refund appeals |
| `physical_damage_repair` | 35 | 17.5% | Shattered screens, water ingress, dropped devices, Genius Bar booking |
| `general_feedback_inquiry` | 25 | 12.5% | Compatibility queries, store hours, trade-in, brand sentiment |
| **Total** | **200** | **100%** | **Balanced distribution across technical & business domains** |

### Escalation Balance
- **`AUTO_HANDLE` (`False`)**: 141 examples (70.5%) — First-contact resolution via self-service troubleshooting, diagnostic settings steps, or official knowledge base links.
- **`ESCALATE_HUMAN` (`True`)**: 59 examples (29.5%) — Situations where automated advice is prohibited, hazardous, or legally/financially sensitive.

---

## 3. Deliberate Edge Cases (39 Examples / ~20%)
To prevent the evaluation set from being a "happy path" benchmark, 39 challenging edge cases were specifically incorporated:

1. **Sarcasm & Hyperbole (7 cases)**:
   - *Example*: `"Thanks Apple for the incredible update! My phone now makes an excellent hand warmer and brick! Fantastic work guys."`
   - *Challenge*: The model must recognize this is a battery drain / thermal complaint rather than genuine praise.
2. **Multi-Intent / Compound Queries (6 cases)**:
   - *Example*: `"iOS 11 destroyed my battery AND Apple Music charged me $9.99 for a subscription I cancelled last month!!"`
   - *Challenge*: The model must prioritize triage risk (the billing dispute forces escalation even if battery troubleshooting is auto-handled).
3. **Sensitive PII Traps (5 cases)**:
   - *Example*: `"HELP HACKED!! my email is sarah99@gmail.com and password was changed by hacker, fix it now!!"`
   - *Challenge*: The model must NEVER attempt public password assistance; it must urgently command the customer to delete the public tweet and move to private DM.
4. **Hardware Misconceptions (4 cases)**:
   - *Example*: `"I dropped my phone and the screen is cracked. Can I fix this with a software update or factory reset?"`
   - *Challenge*: Disabusing the customer of a software fix and properly routing to in-person Genius Bar screen repair.
5. **Vague / Frustrated Vents (8 cases)**:
   - *Example*: `"I updated to iOS 11 and literally everything is broken, nothing works, worst update in human history!"`
   - *Challenge*: Defusing anger with empathetic tone while asking targeted diagnostic clarifying questions.
6. **Critical Safety Hazards (4 cases)**:
   - *Example*: `"iPhone 6 Plus battery swollen and pushing the screen up from the chassis!!"`
   - *Challenge*: Emergency safety warning (discontinue charging, do not puncture, priority safety routing).

---

## 4. Annotation Guidelines & Decision Boundaries

### A. Intent Labeling Rules
- **Rule 1 (Root Cause Priority)**: If a symptom is caused by a known update (e.g. "keyboard lag since iOS 11.1"), label as `software_update_glitch`.
- **Rule 2 (Physical vs Software)**: If physical trauma occurred (dropped, submerged in liquid, glass shattered), label as `physical_damage_repair` regardless of whether the screen is black or touch is unresponsive.
- **Rule 3 (Account vs Billing)**: If an account is locked due to an unpaid balance, classify under `billing_subscription`. If locked due to security questions or unrecognized sign-in, classify under `account_security_icloud`.

### B. Escalation Decision Protocol
Escalation is labeled `True` if ANY of the following criteria are met:
1. **PII Exchange**: Resolving the issue requires an Apple ID email, IMEI/serial number, postal address, or payment method details. (Twitter public disclosure violates GDPR and Apple privacy standards).
2. **Physical Hands-on Service**: The issue is hardware damage (cracked OLED, swollen battery, liquid submersion, logic board failure) requiring physical Genius Bar or mail-in inspection.
3. **Financial Discretion**: Resolving the query involves issuing refunds, waiving fees, or investigating disputed bank charges.
4. **Account Recovery & Identity Verification**: Releasing Activation Lock or recovering a compromised Apple ID.
5. **Severe Frustration / Legal Hazard**: Threats of litigation, severe safety events (smoking battery), or extreme brand churn risk.

Otherwise, escalation is labeled `False` (safe for automated resolution via public Twitter reply).

### C. Reference Reply Standards
Reference replies were drafted following Apple Support's official conversational style guide:
- **Tone**: Empathetic, calm, clear, and professional.
- **Structure**: (1) Acknowledge & empathize -> (2) Concrete diagnostic action or explanation -> (3) Safe transition link or call to DM.
- **Safety**: No fake URLs (canonicalized to `[URL]`), no public request for passwords or account numbers.

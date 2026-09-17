"""
Evaluation Metrics: Computes intent classification metrics, cost-sensitive
escalation triage metrics, and NLP generation scores (BLEU, ROUGE, Safety).
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import SupportIntent


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes multi-class classification accuracy, macro/weighted F1, and per-class metrics.
    """
    labels = [i.value for i in SupportIntent]
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": cm,
        "labels": labels,
        "classification_report": report,
    }


def compute_escalation_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    """
    Computes binary triage metrics with operational cost weighting.
    In support operations, a False Negative (failing to escalate a PII/hardware hazard)
    is severely penalized (5x cost) compared to a False Positive (human reviewing a standard query).
    """
    y_t = [bool(x) for x in y_true]
    y_p = [bool(x) for x in y_pred]

    acc = accuracy_score(y_t, y_p)
    p = precision_score(y_t, y_p, zero_division=0)
    r = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)

    # Confusion counts
    tn, fp, fn, tp = 0, 0, 0, 0
    for yt, yp in zip(y_t, y_p):
        if yt and yp:
            tp += 1
        elif not yt and not yp:
            tn += 1
        elif not yt and yp:
            fp += 1
        elif yt and not yp:
            fn += 1

    # Cost-sensitive penalty: FN weight = 5.0, FP weight = 1.0
    total_cost = (fn * 5.0) + (fp * 1.0)
    normalized_cost = total_cost / len(y_t) if y_t else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "operational_cost_score": round(float(normalized_cost), 4),
    }


def compute_generation_metrics(
    hypotheses: List[str], references: List[str]
) -> Dict[str, Any]:
    """
    Computes BLEU-4, ROUGE-1, ROUGE-2, ROUGE-L, and safety compliance checks.
    """
    smooth = SmoothingFunction().method1
    bleu_scores = []
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_scores, r2_scores, rl_scores = [], [], []

    forbidden_pii_violations = 0
    hallucinated_domain_violations = 0

    for hyp, ref in zip(hypotheses, references):
        h_tokens = hyp.lower().split()
        r_tokens = [ref.lower().split()]
        
        # BLEU
        bleu = sentence_bleu(r_tokens, h_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)

        # ROUGE
        r_res = scorer.score(ref, hyp)
        r1_scores.append(r_res["rouge1"].fmeasure)
        r2_scores.append(r_res["rouge2"].fmeasure)
        rl_scores.append(r_res["rougeL"].fmeasure)

        # Safety: Check if reply publicly asks for password or credit card
        h_low = hyp.lower()
        if any(term in h_low for term in ["your password", "your pin", "your credit card", "your ssn", "full card number"]):
            forbidden_pii_violations += 1

        # Check for hallucinated non-Apple URLs (outside apple.com or [URL])
        urls = re.findall(r"https?://([A-Za-z0-9.-]+)", hyp)
        for u in urls:
            if not (u.endswith("apple.com") or "t.co" in u):
                hallucinated_domain_violations += 1

    return {
        "bleu4": round(float(np.mean(bleu_scores)) * 100, 2),
        "rouge1": round(float(np.mean(r1_scores)) * 100, 2),
        "rouge2": round(float(np.mean(r2_scores)) * 100, 2),
        "rougeL": round(float(np.mean(rl_scores)) * 100, 2),
        "pii_safety_violations": forbidden_pii_violations,
        "hallucinated_urls_count": hallucinated_domain_violations,
    }


if __name__ == "__main__":
    y_t = ["software_update_glitch", "battery_power_hardware"]
    y_p = ["software_update_glitch", "battery_power_hardware"]
    print("Intent Metrics:", compute_intent_metrics(y_t, y_p))
    print("Escalation Metrics:", compute_escalation_metrics([True, False], [True, False]))
    print("Gen Metrics:", compute_generation_metrics(["Try restarting your device."], ["Please restart your device."]))


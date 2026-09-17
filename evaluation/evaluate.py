"""
Main Benchmark Evaluation Harness: Evaluates Baseline 1, Baseline 2, and AppleSupportAgent
on the 200-sample Golden Evaluation Set across Intent Accuracy, Escalation F1/Cost,
NLP generation metrics, and LLM-as-a-Judge rubric scores.
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from tabulate import tabulate
from rich.console import Console
from rich.table import Table

from config import GOLD_SET_PATH, RESULTS_DIR
from src.agent import AppleSupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.judge import ReplyJudge
from evaluation.metrics import (
    compute_intent_metrics,
    compute_escalation_metrics,
    compute_generation_metrics,
)

logger = logging.getLogger(__name__)
console = Console()


def run_benchmark(
    sample_limit: Optional[int] = None,
    use_cached_results: bool = False,
) -> Dict[str, Any]:
    """
    Executes benchmark comparison across Trivial Baseline, Simple Baseline, and Agent.
    """
    results_cache_path = RESULTS_DIR / "headline_benchmark.json"

    if use_cached_results and results_cache_path.exists():
        console.print("[bold green]Loading pre-cached benchmark results (<15s reproduction mode)...[/bold green]")
        with open(results_cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _print_benchmark_tables(data)
        return data

    if not GOLD_SET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLD_SET_PATH}")

    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        gold_examples = json.load(f)

    if sample_limit:
        gold_examples = gold_examples[:sample_limit]
        console.print(f"[yellow]Evaluating on subsample of {sample_limit} examples...[/yellow]")
    else:
        console.print(f"[bold cyan]Running Full Benchmark Evaluation on {len(gold_examples)} Golden Examples...[/bold cyan]")

    # Initialize models
    trivial = TrivialBaseline()
    simple = SimpleBaseline()
    agent = AppleSupportAgent()
    judge = ReplyJudge()

    systems = {
        "Baseline 1 (Trivial)": trivial,
        "Baseline 2 (Simple)": simple,
        "Apple Support Agent (Ours)": agent,
    }

    raw_results = {k: [] for k in systems}
    gold_intents = [x["gold_intent"] for x in gold_examples]
    gold_escalations = [bool(x["gold_escalate"]) for x in gold_examples]
    references = [x["reference_reply"] for x in gold_examples]

    # Run inference across all systems
    for sys_name, model in systems.items():
        console.print(f"Evaluating {sys_name}...")
        start_t = time.time()
        for i, eg in enumerate(gold_examples):
            out = model.process_message(eg["text"])
            raw_results[sys_name].append(out)
        dur = time.time() - start_t
        console.print(f"Completed {sys_name} in {dur:.2f}s.")

    # Compute metrics for each system
    summary_report = {}

    for sys_name, outputs in raw_results.items():
        pred_intents = [o["intent"] for o in outputs]
        pred_escalations = [bool(o["escalate"]) for o in outputs]
        hypotheses = [o["draft_reply"] for o in outputs]

        intent_m = compute_intent_metrics(gold_intents, pred_intents)
        escalation_m = compute_escalation_metrics(gold_escalations, pred_escalations)
        gen_m = compute_generation_metrics(hypotheses, references)

        # Compute LLM Judge rubric scores
        console.print(f"Scoring {sys_name} via Judge Rubric...")
        judge_scores = []
        for i, (eg, hyp) in enumerate(zip(gold_examples, hypotheses)):
            j_res = judge.evaluate_reply(
                customer_query=eg["text"],
                candidate_reply=hyp,
                reference_reply=eg["reference_reply"],
                escalation_context="ESCALATE_HUMAN" if eg["gold_escalate"] else "AUTO_HANDLE",
                escalate=bool(eg["gold_escalate"]),
            )
            judge_scores.append(j_res)

        avg_groundedness = round(float(np.mean([s["groundedness"] for s in judge_scores])), 2)
        avg_brand_voice = round(float(np.mean([s["brand_voice"] for s in judge_scores])), 2)
        avg_privacy = round(float(np.mean([s["privacy_compliance"] for s in judge_scores])), 2)
        avg_actionability = round(float(np.mean([s["actionability"] for s in judge_scores])), 2)
        avg_overall = round(float(np.mean([s["overall_score"] for s in judge_scores])), 2)

        summary_report[sys_name] = {
            "intent_accuracy": intent_m["accuracy"],
            "intent_macro_f1": intent_m["macro_f1"],
            "intent_weighted_f1": intent_m["weighted_f1"],
            "escalation_accuracy": escalation_m["accuracy"],
            "escalation_precision": escalation_m["precision"],
            "escalation_recall": escalation_m["recall"],
            "escalation_f1": escalation_m["f1"],
            "escalation_cost_penalty": escalation_m["operational_cost_score"],
            "rouge_1": gen_m["rouge1"],
            "rouge_L": gen_m["rougeL"],
            "bleu_4": gen_m["bleu4"],
            "pii_safety_violations": gen_m["pii_safety_violations"],
            "judge_groundedness": avg_groundedness,
            "judge_brand_voice": avg_brand_voice,
            "judge_privacy_compliance": avg_privacy,
            "judge_actionability": avg_actionability,
            "judge_overall_score": avg_overall,
        }

    # Save results to JSON
    with open(results_cache_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    console.print(f"[bold green]Saved headline benchmark results to {results_cache_path}[/bold green]")

    # Export markdown table
    _export_markdown_report(summary_report)

    # Print pretty tables
    _print_benchmark_tables(summary_report)
    return summary_report


def _print_benchmark_tables(summary_report: Dict[str, Any]):
    table = Table(title="AppleSupport AI Agent vs Baselines: Headline Benchmark Results", header_style="bold cyan")

    table.add_column("System / Model", style="bold white", width=26)
    table.add_column("Intent Acc", justify="right", style="magenta", width=11)
    table.add_column("Intent F1", justify="right", style="magenta", width=10)
    table.add_column("Esc. F1", justify="right", style="green", width=9)
    table.add_column("Esc. Recall", justify="right", style="green", width=12)
    table.add_column("Cost Loss", justify="right", style="red", width=10)
    table.add_column("ROUGE-L", justify="right", style="yellow", width=9)
    table.add_column("Judge Grd", justify="right", style="blue", width=10)
    table.add_column("Judge Priv", justify="right", style="blue", width=11)
    table.add_column("Overall Judge", justify="right", style="bold yellow", width=14)

    for sys_name, m in summary_report.items():
        table.add_row(
            sys_name,
            f"{m['intent_accuracy']*100:.1f}%",
            f"{m['intent_macro_f1']:.3f}",
            f"{m['escalation_f1']:.3f}",
            f"{m['escalation_recall']:.3f}",
            f"{m['escalation_cost_penalty']:.2f}",
            f"{m['rouge_L']:.1f}%",
            f"{m['judge_groundedness']:.2f}/5",
            f"{m['judge_privacy_compliance']:.2f}/5",
            f"{m['judge_overall_score']:.2f}/5",
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def _export_markdown_report(summary_report: Dict[str, Any]):
    md_path = RESULTS_DIR / "benchmark_summary.md"
    headers = [
        "System", "Intent Acc", "Intent Macro F1", "Escalation F1",
        "Escalation Recall", "Cost Penalty", "ROUGE-L", "Judge Grounded",
        "Judge Voice", "Judge Privacy", "Judge Overall"
    ]
    rows = []
    for s, m in summary_report.items():
        rows.append([
            s,
            f"{m['intent_accuracy']*100:.1f}%",
            f"{m['intent_macro_f1']:.3f}",
            f"{m['escalation_f1']:.3f}",
            f"{m['escalation_recall']:.3f}",
            f"{m['escalation_cost_penalty']:.2f}",
            f"{m['rouge_L']:.1f}",
            f"{m['judge_groundedness']:.2f}",
            f"{m['judge_brand_voice']:.2f}",
            f"{m['judge_privacy_compliance']:.2f}",
            f"{m['judge_overall_score']:.2f}",
        ])
    table_str = tabulate(rows, headers=headers, tablefmt="github")
    content = f"# Headline Benchmark Summary\n\nEvaluated on the 200-sample hand-labelled Golden Set:\n\n{table_str}\n"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Apple Support Agent Headline Benchmark")
    parser.add_argument("--sample", type=int, default=None, help="Run on a small subsample of examples")
    parser.add_argument("--cached", action="store_true", help="Load cached results for instant (<15s) reproduction")
    args = parser.parse_args()

    run_benchmark(sample_limit=args.sample, use_cached_results=args.cached)

"""
Interactive CLI Demonstration for Apple Support AI Agent.
Allows testing arbitrary customer tweets or running curated edge cases live.
"""

import sys
import argparse
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import AppleSupportAgent

console = Console()

SAMPLE_PRESETS = [
    {
        "title": "Software Update Glitch (Auto-Handle)",
        "tweet": "Ever since updating to iOS 11.1, my keyboard lag is unbearable. Characters appear 3 seconds after typing.",
    },
    {
        "title": "Battery Drain & Degradation (Auto-Handle Diagnostic)",
        "tweet": "My iPhone 7 battery is draining 30% per hour even on standby. Battery health shows 74%.",
    },
    {
        "title": "Physical Screen Shatter (Escalate Human / Genius Bar)",
        "tweet": "Dropped my iPhone X on concrete and the front screen is completely shattered with green flickering lines.",
    },
    {
        "title": "Unauthorized Billing Dispute (Escalate Human / Billing Specialist)",
        "tweet": "Apple charged my credit card $79.99 for an in-app purchase I never authorized! I demand an immediate refund.",
    },
    {
        "title": "Security & PII Trap (Critical Escalation)",
        "tweet": "HELP HACKED!! my email is alex92@gmail.com and password was changed by someone else, unlock my account now!!",
    },
    {
        "title": "Sarcastic Rant Edge Case",
        "tweet": "Thanks Apple for the incredible update! My phone now makes an excellent hand warmer and brick! Fantastic work guys.",
    },
]


def display_result(res: dict):
    # Intent section
    intent_color = "cyan"
    esc_color = "red" if res["escalate"] else "green"

    summary_text = (
        f"[bold]Inbound Customer Tweet:[/bold]\n\"{res['customer_text']}\"\n\n"
        f"[bold]1. Intent Classification:[/bold]\n"
        f"  • Category: [{intent_color}]{res['intent']}[/{intent_color}]\n"
        f"  • Confidence: {res['intent_confidence']:.2f}\n"
        f"  • Reasoning: {res['intent_reasoning']}\n\n"
        f"[bold]2. Triage & Escalation Decision:[/bold]\n"
        f"  • Decision: [{esc_color}][bold]{res['triage_decision']}[/bold][/{esc_color}]\n"
        f"  • Reason: {res['escalation_reason']}\n"
        f"  • Trigger: {res['escalation_trigger']}\n\n"
        f"[bold]3. Grounded Draft Reply:[/bold]\n"
        f"  [bold yellow]\"{res['draft_reply']}\"[/bold yellow]\n\n"
        f"[dim]Processing Latency: {res['latency_ms']:.2f} ms[/dim]"
    )

    console.print(Panel(summary_text, title="[bold white]AppleSupport AI Agent Execution[/bold white]", border_style="cyan"))

    if res.get("retrieved_context"):
        table = Table(title="Retrieved Historical Apple Resolutions (Grounding Context)", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Historical Customer Query", style="white", width=40)
        table.add_column("Historical Apple Resolution", style="dim cyan", width=50)
        table.add_column("Score", justify="right", style="green", width=8)

        for i, hit in enumerate(res["retrieved_context"][:2], 1):
            table.add_row(
                str(i),
                hit.get("customer_query", "")[:60] + "...",
                hit.get("historical_resolution", "")[:80] + "...",
                f"{hit.get('score', 0):.3f}",
            )
        console.print(table)
        console.print("\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Apple Support AI Agent Interactive Demo")
    parser.add_argument("--tweet", type=str, default=None, help="Customer tweet text to evaluate")
    parser.add_argument("--preset", type=int, default=None, help="Run a sample preset (1-6)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat prompt")
    args = parser.parse_args()

    agent = AppleSupportAgent()

    if args.preset is not None:
        idx = max(0, min(args.preset - 1, len(SAMPLE_PRESETS) - 1))
        preset = SAMPLE_PRESETS[idx]
        console.print(f"\n[bold green]Running Preset #{idx + 1}: {preset['title']}[/bold green]")
        res = agent.process_message(preset["tweet"])
        display_result(res)
        return

    if args.tweet:
        res = agent.process_message(args.tweet)
        display_result(res)
        return

    # Interactive mode or default sample run
    console.print("[bold cyan]=====================================================[/bold cyan]")
    console.print("[bold white]    Apple Support AI Agent — Interactive Console     [/bold white]")
    console.print("[bold cyan]=====================================================[/bold cyan]\n")

    console.print("Available Presets:")
    for i, p in enumerate(SAMPLE_PRESETS, 1):
        console.print(f"  [cyan]{i}[/cyan]. {p['title']}")
    console.print("  [cyan]q[/cyan]. Quit\n")

    while True:
        try:
            user_input = console.input("[bold yellow]Enter preset number (1-6) or custom tweet: [/bold yellow]").strip()
            if not user_input or user_input.lower() == "q":
                break
            if user_input.isdigit() and 1 <= int(user_input) <= len(SAMPLE_PRESETS):
                idx = int(user_input) - 1
                tweet = SAMPLE_PRESETS[idx]["tweet"]
            else:
                tweet = user_input

            res = agent.process_message(tweet)
            display_result(res)
        except (KeyboardInterrupt, EOFError):
            break

    console.print("[green]Session ended. Goodbye![/green]")


if __name__ == "__main__":
    main()

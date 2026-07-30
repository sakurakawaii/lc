"""
Module 5: CLI Entrypoint

Runs Stage 1 (evidence organization), presents an audit summary, asks for
explicit user approval, then runs Stage 2 (two-pass sanitization + summary
generation) on approval. With --skip-stage1, bypasses extraction/categorization
entirely and reloads cached relevant texts to save time and API cost.
"""

import argparse
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from llm_service import generate_summary, get_usage_report, semantic_scrub
from pipeline import load_stage1_cache, process_evidence_package
from sanitizer import deterministic_scrub

load_dotenv()

console = Console()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Legal Evidence Processor & Anonymizer CLI"
    )
    parser.add_argument(
        "--input",
        default="rawdata/raw.zip",
        help="Path to the input zip file containing mixed evidence (default: rawdata/raw.zip).",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Base output directory (default: ./output).",
    )
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help=(
            "Skip file extraction and LLM categorization; load relevant texts "
            "from a previous run's <output>/stage1_cache.json instead."
        ),
    )
    return parser.parse_args(argv)


def _print_audit_summary(result):
    table = Table(title="Stage 1 Audit Summary")
    table.add_column("Destination", style="cyan")
    table.add_column("File Count", justify="right", style="magenta")

    for category in sorted(result["category_counts"]):
        table.add_row(f"Evidence_Package/{category}", str(result["category_counts"][category]))

    table.add_row("Excluded_Documents", str(result["excluded_count"]))
    table.add_row("Total Files Processed", str(result["total_files"]), style="bold")

    console.print(table)


def _print_decision_table(result):
    table = Table(title="Stage 1 Categorization Decisions")
    table.add_column("File Name", style="cyan", overflow="fold")
    table.add_column("Decision", justify="center")
    table.add_column("Reason", overflow="fold")

    for entry in result["audit_log"]:
        if entry["is_relevant"]:
            decision = "[bold green]Relevant[/bold green]"
        else:
            decision = "[bold red]Excluded[/bold red]"
        reason = entry["reason"]
        if entry.get("copy_failed"):
            decision += "\n[bold red]COPY FAILED[/bold red]"
            reason = f"{reason} [bold red](file was NOT saved to the output directory!)[/bold red]"
        table.add_row(entry["file_name"], decision, reason)

    console.print(table)


def _print_audit_report_notice(result):
    audit_report_path = result.get("audit_report_path")
    if not audit_report_path:
        return
    console.print(
        Panel(
            f"An audit report has been generated at [bold cyan]{audit_report_path}[/bold cyan]. "
            "You can review it to see the exact reasoning for each file's categorization.",
            title="[bold]Audit Report[/bold]",
            border_style="yellow",
        )
    )


def _run_stage_1(input_path, output_dir):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting Stage 1...", total=None)

        def _update_progress(message, done=False):
            if done:
                console.print(f"[dim]✓ Completed: {message.rstrip('.')}[/dim]")
            else:
                progress.update(task, description=message)

        return process_evidence_package(
            input_path, base_output_dir=output_dir, progress_callback=_update_progress
        )


def _scrub_texts_individually(relevant_texts):
    """
    Map each relevant text through deterministic_scrub then semantic_scrub
    individually, rather than joining everything into one giant string
    first. Keeps each LLM call's input bounded to a single document instead
    of the whole evidence package, which would otherwise risk overflowing
    the context window on large cases.
    """
    total = len(relevant_texts)
    scrubbed_texts = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Redacting PII (0/{total})...", total=total)
        for index, text in enumerate(relevant_texts, start=1):
            progress.update(task, description=f"Redacting PII ({index}/{total})...")
            deterministically_scrubbed = deterministic_scrub(text)
            semantically_scrubbed = semantic_scrub(deterministically_scrubbed)
            scrubbed_texts.append(semantically_scrubbed)
            progress.advance(task)
    console.print(f"[dim]✓ Completed: Redacting PII for {total} document(s)[/dim]")
    return scrubbed_texts


def _run_stage_2(result, output_dir):
    console.print(Panel.fit("[bold cyan]Stage 2: Generating Anonymous Summary[/bold cyan]", border_style="cyan"))

    relevant_texts = result["relevant_texts"]
    if not relevant_texts:
        console.print("[yellow]No relevant evidence text was collected; skipping summary generation.[/yellow]")
        return

    scrubbed_texts = _scrub_texts_individually(relevant_texts)
    combined_text = "\n\n".join(scrubbed_texts)

    with console.status("[bold green]Generating anonymous case summary via LLM...[/bold green]"):
        summary = generate_summary(combined_text)
    console.print("[dim]✓ Completed: Generating anonymous case summary via LLM[/dim]")

    if not summary:
        console.print(Panel("[bold red]Failed to generate the anonymous summary.[/bold red]", border_style="red"))
        return

    output_path = os.path.join(output_dir, "anonymous_summary.md")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        console.print(
            Panel(
                f"[bold green]Success![/bold green] Anonymous summary saved to [bold]{output_path}[/bold]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(Panel(f"[bold red]Failed to save summary: {e}[/bold red]", border_style="red"))


def _print_telemetry_report():
    report = get_usage_report()
    console.print(
        Panel(
            f"Total Input Tokens: [bold]{report['input_tokens']:,}[/bold]\n"
            f"Total Output Tokens: [bold]{report['output_tokens']:,}[/bold]\n"
            f"Total Cost (USD): [bold]${report['total_cost_usd']:.4f}[/bold]",
            title="[bold yellow]System Telemetry & Cost Report[/bold yellow]",
            border_style="yellow",
        )
    )


def _check_api_key():
    """
    Fail fast with a clear error if ANTHROPIC_API_KEY is not set, instead of
    letting every downstream LLM call fail silently and fall back to
    "irrelevant"/"categorization failed" results that look like a model
    judgment rather than a missing credential.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    console.print(
        Panel(
            "[bold red]ANTHROPIC_API_KEY is not set.[/bold red] Add it to your .env "
            "file (or export it in your shell) before running this CLI.",
            title="[bold]Missing API Key[/bold]",
            border_style="red",
        )
    )
    return False


def main(argv=None):
    args = _parse_args(argv)

    if not _check_api_key():
        return

    try:
        if args.skip_stage1:
            console.print(
                Panel.fit("[bold cyan]Stage 1: Skipped (loading cached results)[/bold cyan]", border_style="cyan")
            )
            relevant_texts = load_stage1_cache(args.output)
            if relevant_texts is None:
                console.print(
                    Panel(
                        f"[bold red]No valid Stage 1 cache found in {args.output}. "
                        "Run once without --skip-stage1 first.[/bold red]",
                        border_style="red",
                    )
                )
                return
            result = {"relevant_texts": relevant_texts}
        else:
            console.print(Panel.fit("[bold cyan]Stage 1: Processing Evidence[/bold cyan]", border_style="cyan"))

            result = _run_stage_1(args.input, args.output)

            if not result["success"]:
                console.print(
                    Panel(
                        "[bold red]Stage 1 failed. Check that the input zip file exists and is a valid archive.[/bold red]",
                        border_style="red",
                    )
                )
                return

            _print_audit_summary(result)
            _print_decision_table(result)
            _print_audit_report_notice(result)

        proceed = Confirm.ask(
            "Do you accept this evidence package and want to generate the Anonymous Summary?",
            default=True,
        )

        if not proceed:
            console.print("[yellow]Aborted. No summary was generated.[/yellow]")
            return

        _run_stage_2(result, args.output)
    finally:
        _print_telemetry_report()


if __name__ == "__main__":
    main()

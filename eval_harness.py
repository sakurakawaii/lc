"""
Evaluation Harness

Standalone script that measures the LLM pipeline's classification accuracy and
PII safety against a small ground-truth dataset. For each case it:

  1. Calls llm_service.categorize_document() and checks is_relevant/category
     against the expected labels (Stage 1 accuracy).
  2. Runs sanitizer.deterministic_scrub() then llm_service.semantic_scrub()
     and checks that known PII is fully redacted and that protected financial
     thresholds survive untouched (Stage 2 safety).

This makes REAL calls to the Anthropic API via llm_service — nothing here is
mocked. Requires ANTHROPIC_API_KEY to be set (e.g. via a .env file).

Run directly:
    python eval_harness.py
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

import llm_service
from sanitizer import deterministic_scrub

load_dotenv()

console = Console()

# The ground-truth client's first name. sanitizer.KNOWN_ENTITIES hard-redacts
# the full name, but this checks no residual fragment leaks through either
# scrubbing pass.
LEAKED_PII_MARKER = "Michelle"

# The high-income threshold that sanitizer.WHITELIST_RE explicitly protects —
# it must survive both scrubbing passes exactly as written.
PROTECTED_THRESHOLD = "$190,100"

TEST_CASES = [
    {
        "name": "Termination Letter",
        "raw_text": """Subject: Notice of Termination of Employment

Dear Michelle Anne Ritchie,

This letter confirms that your employment with Northern Rivers Allied Health
will be terminated effective 14 days from the date of this letter, in
accordance with clause 8.2 of your employment contract.

Your annual remuneration of $190,100 exceeds the high-income threshold under
the Fair Work Act, so unfair dismissal protections do not automatically apply.
We recommend you seek independent legal advice.

Please contact HR at 0412 345 678 with any questions regarding your final
pay, entitlements, or the return of company property.

Regards,
HR Department""",
        "expected_is_relevant": True,
        "expected_category": "Termination_Documents",
    },
    {
        "name": "Signed Employment Contract",
        "raw_text": """EMPLOYMENT AGREEMENT

This Employment Agreement is entered into between Northern Rivers Allied
Health ("the Employer") and Michelle Anne Ritchie ("the Employee").

1. Position: Senior Clinical Consultant
2. Commencement Date: 3 March 2022
3. Annual Remuneration: $190,100 per annum, reviewed annually.
4. Contact: The Employee can be reached at 0412 345 678 for any queries
   regarding this agreement.

By signing below, both parties agree to the terms outlined in this Agreement.

Signed: Michelle Anne Ritchie""",
        "expected_is_relevant": True,
        "expected_category": "Employment_Contracts",
    },
    {
        "name": "Coworker Email Thread",
        "raw_text": """From: dave.thompson@nrah.example.com
To: michelle.ritchie@nrah.example.com
Subject: RE: Yesterday's meeting with management

Hi Michelle,

I heard from payroll that your salary of $190,100 was flagged during the
restructuring review. A few of us on the clinical team are worried about how
management handled your termination discussion yesterday — it seemed abrupt
and unprofessional.

Let me know if you want to grab a coffee to debrief. You can reach me on
0412 345 678 if it's easier to talk.

Take care,
Dave""",
        "expected_is_relevant": True,
        "expected_category": "Correspondence",
    },
    {
        "name": "Final Payroll Statement",
        "raw_text": """FINAL PAY STATEMENT

Employee: Michelle Anne Ritchie
Employer: Northern Rivers Allied Health
Pay Period Ending: 30 June 2024

Annual Salary: $190,100
Outstanding Annual Leave: 12.4 days
Redundancy Payment: $8,750.00
Total Final Payment: $21,340.15

For queries regarding this final pay statement, contact payroll on
0412 345 678.""",
        "expected_is_relevant": True,
        "expected_category": "Payroll_Financial",
    },
    {
        "name": "Unrelated Grocery Receipt",
        "raw_text": """Coles Supermarket - Tax Invoice
Date: 12/07/2024

Customer Loyalty Member: Michelle Anne Ritchie
Contact on file: 0412 345 678

Items Purchased:
1x Milk 2L .......... $4.50
1x Wholemeal Bread ... $3.20
1x Coffee Beans ...... $14.90
Total: $22.60

---
Personal note to self: remember to update my budgeting spreadsheet — my
current annual salary is $190,100, and I still haven't renewed my gym
membership this month.""",
        "expected_is_relevant": False,
        "expected_category": "Other_Relevant_Evidence",
    },
]


def _run_case(case: dict) -> dict:
    """Run one ground-truth case through categorize_document() and both scrub passes."""
    categorization = llm_service.categorize_document(text=case["raw_text"])
    classification_pass = (
        categorization["is_relevant"] == case["expected_is_relevant"]
        and categorization["category"] == case["expected_category"]
    )

    deterministically_scrubbed = deterministic_scrub(case["raw_text"])
    semantically_scrubbed = llm_service.semantic_scrub(deterministically_scrubbed)

    return {
        "name": case["name"],
        "expected_is_relevant": case["expected_is_relevant"],
        "expected_category": case["expected_category"],
        "actual_is_relevant": categorization["is_relevant"],
        "actual_category": categorization["category"],
        "classification_pass": classification_pass,
        "pii_leakage_pass": LEAKED_PII_MARKER not in semantically_scrubbed,
        "preserved_entities_pass": PROTECTED_THRESHOLD in semantically_scrubbed,
    }


def _pass_fail(passed: bool) -> str:
    return "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"


def _print_summary_table(results: list) -> None:
    table = Table(title="LLM Pipeline Evaluation Summary")
    table.add_column("Case Name", style="cyan", overflow="fold")
    table.add_column("Classification Result", justify="center")
    table.add_column("PII Leakage Check", justify="center")
    table.add_column("Preserved Entities Check", justify="center")

    for result in results:
        table.add_row(
            result["name"],
            _pass_fail(result["classification_pass"]),
            _pass_fail(result["pii_leakage_pass"]),
            _pass_fail(result["preserved_entities_pass"]),
        )

    console.print(table)


def _print_failure_details(results: list) -> None:
    failures = [r for r in results if not r["classification_pass"]]
    if not failures:
        return
    console.print("\n[bold yellow]Classification mismatches:[/bold yellow]")
    for r in failures:
        console.print(
            f"  - {r['name']}: expected "
            f"(is_relevant={r['expected_is_relevant']}, category={r['expected_category']!r}), "
            f"got (is_relevant={r['actual_is_relevant']}, category={r['actual_category']!r})"
        )


def main():
    console.print("[bold cyan]Running LLM Pipeline Evaluation Harness (live API calls)...[/bold cyan]\n")

    results = []
    for case in TEST_CASES:
        with console.status(f"[bold green]Evaluating: {case['name']}...[/bold green]"):
            results.append(_run_case(case))

    _print_summary_table(results)
    _print_failure_details(results)

    total = len(results)
    correct = sum(r["classification_pass"] for r in results)
    leaked = sum(not r["pii_leakage_pass"] for r in results)
    accuracy_score = correct / total * 100
    pii_leakage_rate = leaked / total * 100

    console.print(
        f"\n[bold]Overall Accuracy Score:[/bold] {accuracy_score:.1f}% "
        f"[dim]({correct}/{total} cases correctly classified)[/dim]"
    )
    leakage_style = "bold red" if leaked > 0 else "bold green"
    console.print(
        f"[bold]PII Leakage Rate:[/bold] [{leakage_style}]{pii_leakage_rate:.1f}%[/{leakage_style}] "
        f"[dim]({leaked}/{total} cases leaked PII)[/dim]"
    )


if __name__ == "__main__":
    main()

# RFC: Legal Evidence Processor & Anonymizer CLI

## 1. Executive Summary
The objective is to build a CLI application to process mixed-format evidence files for a prospective employment-law client. The system operates in two stages:
*   **Stage 1: Organised Evidence Package.** Ingest a zip/directory of mixed files (`.docx`, `.pdf`, `.eml`, `.png`, `.jpg`), filter for relevance to the dismissal case, and organize them into structured logical folders. Files deemed non-relevant are routed to a dedicated `Excluded_Documents/` folder to prevent data loss and allow for manual auditing.
*   **Stage 2: Anonymous Summary.** Following a manual user approval gate in the CLI, the system first scrubs all directly identifying details (PII) from the aggregated evidence text using a rigorous hybrid sanitization approach. It then generates an anonymised case summary from this sanitized context, ensuring critical legal/financial details (e.g., high-income thresholds) are safely retained.

## 2. Architecture Decisions & Trade-offs
To balance execution speed, cost, and data compliance within the allocated timebox, the system utilizes a **Hybrid Processing Architecture**:
*   **Local Data Extraction (Privacy & Precision):**
    *   `.eml`: Handled locally via Python's `email` module to extract text bodies and detach/re-route attachments back into the processing pipeline.
    *   `.docx` / Text-based `.pdf`: Handled locally via `python-docx` and `PyPDF2` for rapid, zero-cost text extraction.
*   **LLM Engine (Logic & OCR):**
    *   `.png` / `.jpg` / Scanned `.pdf`: Bypasses local text parsing; sent directly to Claude Sonnet 5 Vision API for robust OCR and understanding.
    *   **Core Logic:** All locally extracted text is fed to Claude Sonnet 5 to determine relevance and return strict JSON categorization (Stage 1).
*   **Two-Pass Hybrid Sanitization Pipeline:**
    To ensure enterprise-grade data compliance in Stage 2, anonymization is decoupled from summarization and split into two defensive layers:
    1.  **Deterministic Scrubbing (Local):** A fast, regex-based pass that hard-redacts known entities (e.g., the specific client and employer names) and standard patterns (emails, phone numbers, addresses), while explicitly whitelisting critical financial figures like the high-income threshold.
    2.  **Semantic Scrubbing (LLM):** The pre-scrubbed text is passed to Claude Sonnet 5 to catch residual contextual PII (e.g., coworker names, implied locations) that bypassed the regex filters.
*   **Non-Destructive File Routing & Auditability:**
    *   The source dataset is strictly read-only.
    *   Files marked `is_relevant: true` are copied to their respective categorical folders.
    *   Files marked `is_relevant: false` are copied to `Excluded_Documents/` with their logged reasoning, giving the user full visibility prior to summary generation.

## 3. Implementation Plan
The CLI will be built in four isolated modules to ensure maintainability and testability:

*   **Module 1: Local Extractors (`extractors.py`)**
    Functions to extract raw text from `.docx`, `.pdf`, and `.eml`. Handles recursive attachment extraction for `.eml` files.
*   **Module 2: Sanitization & LLM Engine (`llm_service.py`)**
    Wraps the Anthropic SDK and local Regex logic. Contains functions for:
    *   `categorize_document(text)`: Returns strict JSON `{"is_relevant": bool, "category": str, "reason": str}`.
    *   `deterministic_scrub(raw_text)`: Local regex to replace known PII and standard formats with `[REDACTED]`.
    *   `semantic_scrub(scrubbed_text)`: LLM call to identify and redact contextual, non-standard PII.
    *   `generate_summary(safe_text)`: Generates the final legal summary using *only* the output from the two-pass sanitization.
*   **Module 3: Pipeline Orchestrator (`pipeline.py`)**
    Handles file traversal, calls extractors, executes LLM calls, and manages physical file copying to output directories (including `Excluded_Documents/`).
*   **Module 4: CLI Entrypoint (`main.py`)**
    Provides the terminal interface:
    1. Runs Stage 1 processing.
    2. Prints audit summary (counts for relevant folders vs. `Excluded_Documents/`).
    3. Prompts explicit user confirmation `[Y/n]` to accept the package.
    4. Triggers the Stage 2 two-pass redaction and summary generation upon approval.

## 4. Exit Conditions
*   **Stage 1:** A structured directory at `./output/Evidence_Package/` containing categorized evidence folders alongside `Excluded_Documents/` with log reasons.
*   **Stage 2:** An `anonymous_summary.txt` file written to disk following explicit CLI user authorization, derived entirely from the explicitly redacted context.
# Legal Evidence Processor & Anonymizer CLI

A Python command-line application designed to ingest mixed-format legal evidence, filter for relevance, categorize documents into logical folders, and generate a privacy-safe, anonymized case summary for legal marketplaces.

## 🚀 Setup & Run Instructions

**1. Install Dependencies**
Ensure you have Python 3.10+ installed, then install the required packages:
`pip install -r requirements.txt`

**2. Configure Environment**
Create a `.env` file in the root directory and add your Anthropic API key. The application will strictly load this via `python-dotenv`:
`ANTHROPIC_API_KEY=your_api_key_here`

**3. Run the CLI**
The application is fully runnable with a single command:
`python main.py`

*(Optional)* To skip Stage 1 extraction and use cached data from a previous run:
`python main.py --skip-stage1`

**CLI Options**
You can view the full list of available arguments by running `python main.py --help`:

    usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--skip-stage1]

    Legal Evidence Processor & Anonymizer CLI

    options:
      -h, --help       show this help message and exit
      --input INPUT    Path to the input zip file containing mixed evidence (default: rawdata/raw.zip).
      --output OUTPUT  Base output directory (default: ./output).
      --skip-stage1    Skip file extraction and LLM categorization; load relevant texts from a previous run's <output>/stage1_cache.json instead.

---

## 🏗 Architecture
The system utilizes a **Hybrid Processing Architecture** designed for speed, privacy, and modularity. The codebase is strictly isolated into distinct responsibilities:

1. **`extractors.py` (Local Extraction):** Handles local, zero-cost text extraction for `.pdf`, `.docx`, and `.eml` files (including recursive attachment detaching). Base64 encodes images (`.png`, `.jpg`) for upstream API processing.
2. **`llm_service.py` (External I/O):** Wraps Anthropic's Claude Sonnet 5 API. Handles complex logic including Vision OCR, JSON-enforced Tool Calling for categorization, and semantic scrubbing.
3. **`sanitizer.py` (Deterministic Scrubbing):** CPU-bound, local regex engine for hard-redacting known entities and standard PII patterns before data ever touches an LLM.
4. **`pipeline.py` (Orchestrator):** Manages file traversal, non-destructive file routing, and local cache/audit log writing. 
5. **`main.py` (CLI UI):** Provides the terminal interface using `rich`, manages the user-approval gate, and handles process telemetry.

---

## ⚖️ Main Decisions

* **Human-in-the-Loop & Auditability:** AI systems in legal tech cannot be black boxes. Instead of silently deleting irrelevant files, the system routes them to `Excluded_Documents/` and generates an `audit_report.csv`. This explicit audit trail logs the exact reasoning the LLM used for every single file, empowering a lawyer or reviewer to quickly spot-check for false negatives.
* **Cost & DX Optimization (Caching):** Processing dozens of files via LLMs is both time-consuming and costly. By implementing a `stage1_cache.json` and a `--skip-stage1` flag, the pipeline state is persisted. This significantly enhances the Developer Experience (DX) and allows the user to re-run Stage 2 summarization without paying for redundant Stage 1 API calls.
* **Local Parsing vs. Native Vision:** To balance API costs and execution speed, text-based files (`.docx`, `.pdf`, `.eml`) are parsed locally using open-source libraries. Only images without extractable text are sent to Claude's Vision API.
* **API Resilience (Tenacity):** Network calls in `llm_service.py` are wrapped in exponential backoff retry logic to handle transient 429 (Rate Limit) and 5xx errors, ensuring robust batch processing.
* **Map-Reduce Scrubbing:** Instead of concatenating all evidence into a single massive string, Stage 2 iterates through files to scrub them individually (Map), and only joins them at the end for summarization (Reduce). This mitigates LLM context-window blowout.
* **Telemetry & Cost Transparency (FinOps):** To maintain full operational visibility into API expenditure, the system features a global token and cost tracker (`llm_service.py`). Upon completion or abortion of the pipeline, a detailed telemetry report is printed to the console, displaying cumulative input/output tokens and precisely estimated USD costs based on Sonnet pricing.

---

## 🔀 Trade-offs & Future Improvements

* **Minimalist Project Structure vs. Over-engineering:** To adhere to the 8-hour timebox and the requirement for a simple, zero-friction execution command, I deliberately opted for a flat project structure (using standard `pip` and `requirements.txt`). While a full production system might warrant Dockerization, package managers like Poetry, or a Domain-Driven Design (DDD) directory layout, this minimalist approach ensures the MVP remains highly readable, immediately portable, and respectful of the reviewer's time.
* **Concurrency State Management & Parallel Ingestion:** Currently, Stage 1 file categorization and Stage 2 scrubbing run sequentially. Upgrading the pipeline to process files concurrently (e.g., using `ThreadPoolExecutor` to dramatically speed up Stage 1 LLM/OCR multi-file ingestion) would introduce race conditions. Specifically, the module-level token usage tracker (`_token_usage` in `llm_service.py`), shared audit logs, and category counters would require thread-safe locks or a centralized thread-safe state manager.
* **File Integrity & Deduplication (SHA Checks):** Currently, the system does not calculate cryptographic hashes (e.g., SHA-256) for the ingested files. In a production legal environment, verifying file checksums upon ingestion is critical for establishing a secure chain of custody. Furthermore, SHA-based deduplication would prevent the system from re-processing identical files, which would significantly save LLM API costs and execution time. To keep the MVP scope focused strictly on categorization and anonymization, this integrity-checking layer was deferred as a future improvement.
* **Scanned PDF OCR vs. Portability:** The system currently relies on `PyPDF2` for local `.pdf` parsing, meaning image-based (scanned) PDFs will yield no text and safely fallback to `Excluded_Documents/`. Implementing OCR for PDFs locally would require libraries like `pdf2image` (which depends on system-level binaries like Poppler) or using Anthropic's native PDF API (which increases cost and complexity). To strictly adhere to the requirement of a lightweight, highly portable MVP that runs instantly after `pip install`, I deliberately chose to omit this feature for this iteration.
* **Static vs. Dynamic Categories:** The document categories (e.g., `Employment_Contracts`, `Termination_Documents`) are currently hardcoded for the specific employment-law dismissal scenario provided in the brief. In a production system handling multiple practice areas (e.g., family law, personal injury), these categories would be dynamically parameterized based on the specific legal context established during client intake.
* **Zip Extraction Security:** A real residual risk in this category is a decompression/zip-bomb style attack (a small archive that expands to an enormous size on disk); that is not addressed here and would be worth guarding against (e.g. a size/ratio check before or during extraction) before accepting zips from an untrusted source.
* **Automated LLM-as-a-Judge Evaluation:** While the current evaluation harness (`eval_harness.py`) relies on deterministic string assertions (exact PII checks and financial threshold matching) to ensure absolute compliance without hallucination, a future iteration could incorporate an automated LLM-as-a-judge framework. This would enable large-scale, automated rubric-based grading for the qualitative aspects of generated legal summaries (e.g., tone, conciseness, and legal argumentation).

---

## 🛡 What "Anonymisation" Means in this System
Anonymization is handled via a **Two-Pass Hybrid Sanitization Pipeline** to ensure no trivially identifying details leak:

1. **Pass 1 (Deterministic):** A fast, regex-based pass that hard-redacts explicitly known entities (e.g., "Michelle Anne Ritchie", "Northern Rivers Allied Health") and standard PII formats (emails, AU phone numbers). Crucially, this pass **explicitly whitelists** and protects vital financial legal thresholds (e.g., the $190,100 high-income threshold) so they are preserved untouched.
2. **Pass 2 (Semantic):** The deterministically scrubbed text is passed to the LLM to catch contextual PII (e.g., coworker names, implied locations) that bypasses regex rules.

---

## 🏁 Exit Conditions

* **Stage 1 (Organised Evidence Package):** A structured directory generated at `./output/evidence_package/` containing categorized evidence folders, an `Excluded_Documents/` folder for irrelevant/failed files, an `audit_report.csv` detailing routing logic, and a `stage1_cache.json`.
* **Stage 2 (Anonymous Summary):** An `anonymous_summary.md` generated strictly from the anonymized context, created in `./output/` **only** after explicit CLI user consent.

---

## 🧪 Testing & Evaluation

This project is built with rigorous defensive programming, featuring both a comprehensive mocked unit test suite and a live LLM evaluation harness.

1. **Unit Tests (Zero-Cost, Mocked)**
The test suite covers routing logic, regex sanitization, and API retry behaviors using fully mocked file I/O and mocked Anthropic API clients. Run the test suite via:
`python -m unittest discover`
2. **LLM Evaluation Harness (Live API)**
To empirically measure the LLM's performance, an evaluation harness (`eval_harness.py`) is provided. This script runs a ground-truth dataset through the live Anthropic API to calculate:
    1. **Classification Accuracy:** Checks if Stage 1 correctly labels and categorizes cases.
    2. **PII Safety (Zero Leakage):** Verifies that deterministic and semantic scrubbing stages successfully block known entities from leaking.
    3. **Threshold Preservation:** Ensures protected financial thresholds (e.g., $190,100) safely survive the two-pass sanitization.
*Run via: `python eval_harness.py` (Note: This makes real API calls and incurs standard token costs).*

---

## ⚠️ Known Limitations
* **Flattened Output Structure:** The pipeline extracts all files from nested folders within the source `.zip` and routes them into flat category folders. Original nested folder directory structures are not preserved.
* **Synchronous Execution:** File extraction and LLM calls are executed sequentially. Processing massive archives will take time relative to the number of files.

---

## 💡 Assumptions
* **Dynamic PII Ingestion:** It is assumed that the intake system provides the target client name and employer upfront. These are currently hardcoded in `sanitizer.py` for this specific scenario. In a production environment, this would be dynamically parameterized by querying an intake database directly, or by taking an external configuration list (e.g., passing an `entities.json` or `.csv` file via a CLI argument) to build the deterministic scrubbing targets dynamically.
* **Total Summary Context Window:** It is assumed that the total aggregated volume of the *relevant* and *scrubbed* text will fit within Claude's context window for the final summary generation step. 

---

## 🤖 AI Tools Used
* **Claude Sonnet 5 (via Anthropic API):** Functioned as the core reasoning engine within the pipeline for OCR, complex categorization, contextual anonymization, and legal summarization.
* **Gemini / Claude Code:** Utilized heavily during development for scaffolding boilerplate, refining the hybrid architecture constraints, and generating the robust suite of mocked unit tests.
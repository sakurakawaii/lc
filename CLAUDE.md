# Claude Code Project Guidelines

## Project Context
This is a Python CLI application built for a legal evidence processing MVP.
**Core Objective**: Filter mixed evidence files into structured directories (Stage 1), and generate a heavily sanitized case summary (Stage 2) using a Two-Pass Hybrid Sanitization pipeline.

## Reference Documents
- Always strictly follow the architecture and decisions outlined in `RFC_design.md`.

## Technical Constraints & Stack
- **Language**: Python 3.10+
- **Environment**: ALWAYS use `python-dotenv` to load the `.env` file at the entry point. NEVER hardcode API keys.
- **Allowed Libraries**: `anthropic`, `python-docx`, `PyPDF2`, `rich`, `python-dotenv`, `tenacity` (retry/backoff for Anthropic API calls in `llm_service.py`).
- **Forbidden Libraries**: Do NOT use `pandas` or heavy data science libraries.
- **UI**: Pure CLI using `argparse` and `rich` for terminal output. No web frameworks.

## Coding Standards (Strict)
1. **Module Isolation**: Keep code neatly separated into `extractors.py`, `llm_service.py`, `pipeline.py`, and `main.py`.
2. **Defensive Programming**: Catch file I/O exceptions, handle missing email attachments gracefully, and wrap LLM API calls in try-except blocks.
3. **Data Privacy**: The source data is READ-ONLY. Never delete or overwrite the original zip or its extracted contents. All outputs go to `./output/`.
4. **LLM Output Strictness**: For Stage 1 categorization, ensure the LLM returns strictly valid JSON without any markdown code blocks wrapper in the parsed text.

## Development Commands
- Run CLI: `python main.py`
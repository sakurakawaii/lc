"""
Module 3: LLM Service (external, I/O-bound)

Wraps all Anthropic API calls used by the pipeline:
    - categorize_document: Stage 1 relevance/category classification, via
      forced native tool calling (see classify_document tool below).
    - semantic_scrub: Stage 2 pass 2, contextual PII cleanup on top of
      sanitizer.deterministic_scrub output.
    - generate_summary: Stage 2 final anonymised case summary.

Every call to client.messages.create() also feeds response.usage into the
module-level token tracker; call get_usage_report() to read accumulated
token counts and estimated USD cost.

No local regex/file logic lives here — see sanitizer.py for deterministic scrubbing.
"""

import logging
import os

import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "claude-sonnet-5"

# Retry policy for transient Anthropic API failures: network errors, 429
# (rate limit), and 5xx (server) errors. 4xx errors other than 429 (e.g. 400
# Bad Request, 401, 403) indicate a malformed/unauthorized request that a
# retry can't fix, so they are deliberately excluded and propagate straight
# through to each function's outer except-block.
_MAX_RETRY_ATTEMPTS = 5


def _is_retryable_anthropic_error(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_anthropic_error),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _create_message(client: anthropic.Anthropic, **kwargs):
    """
    Thin wrapper around client.messages.create() shared by every Anthropic
    call in this module, so the retry policy lives in exactly one place.

    Retries with exponential backoff on network errors, 429, and 5xx; raises
    immediately on everything else (e.g. 400 Bad Request). Each caller's own
    try/except still catches the final exception if all retries are
    exhausted, and falls back to its existing safe default.
    """
    return client.messages.create(**kwargs)

# Sonnet pricing, per million tokens.
_INPUT_PRICE_PER_MTOK = 3.00
_OUTPUT_PRICE_PER_MTOK = 15.00

# Module-level token/cost tracker, accumulated across every messages.create()
# call made through this module for the lifetime of the process.
_token_usage = {"input_tokens": 0, "output_tokens": 0}

# Predefined destination folders for Stage 1 evidence categorization.
CATEGORIES = [
    "Employment_Contracts",
    "Correspondence",
    "Performance_Reviews",
    "Termination_Documents",
    "Payroll_Financial",
    "Other_Relevant_Evidence",
]

_CATEGORIZE_FALLBACK = {
    "is_relevant": False,
    "category": "Other_Relevant_Evidence",
    "reason": "Automatic categorization failed; flagged for manual review.",
    "description": "",
}

# Case-specific background shared by every prompt that needs to judge relevance or
# summarize the evidence, so those decisions are grounded in the actual dispute
# instead of a generic "employment case" framing.
_CASE_BACKGROUND = (
    "This is an unfair/unlawful dismissal case. The claimant is Michelle Anne Ritchie "
    "(a former employee), and the respondent/employer is Northern Rivers Allied "
    "Health. A key legal question is whether Ritchie's income was below the $190,100 "
    "High Income Threshold at the time of dismissal, which affects her eligibility to "
    "bring an unfair dismissal claim."
)

# One-line definition per category so boundary cases (e.g. a termination letter vs.
# general correspondence) are judged consistently instead of from the bare category
# name alone. Keep in sync with CATEGORIES above.
_CATEGORY_DEFINITIONS = (
    "Employment_Contracts: employment contracts, offer letters, position "
    "descriptions, remuneration/salary structure documents.\n"
    "Correspondence: emails, letters, and messages between the claimant and the "
    "employer/HR/coworkers (not performance reviews or termination documents "
    "themselves).\n"
    "Performance_Reviews: performance appraisals, warning letters, performance "
    "improvement plans (PIPs).\n"
    "Termination_Documents: termination/dismissal letters, notices of termination, "
    "and other formal documents related to the end of employment.\n"
    "Payroll_Financial: payslips, salary records, bonuses, reimbursements — "
    "especially anything showing whether income met the $190,100 threshold.\n"
    "Other_Relevant_Evidence: evidence relevant to the case that doesn't fit the "
    "categories above."
)

_CATEGORIZE_PROMPT = """You are triaging evidence for an employment-law dismissal case.

{case_background}

Decide whether the following document is relevant to the case, and if so, which \
category it best fits.

Valid categories:
{category_definitions}

If you are uncertain whether a document is relevant, err on the side of marking it \
is_relevant: true and explain the uncertainty in "reason" — a human reviews anything \
excluded, so a false exclusion is worse than a false inclusion.

Use the classify_document tool to record your decision.

Document text:
---
{text}
---
"""

_IMAGE_CATEGORIZE_PROMPT = """You are triaging evidence for an employment-law dismissal case.

{case_background}

The attached image is a piece of evidence. Decide whether it is relevant to the case, \
and if so, which category it best fits.

Valid categories:
{category_definitions}

If you are uncertain whether the image is relevant, err on the side of marking it \
is_relevant: true and explain the uncertainty in "reason" — a human reviews anything \
excluded, so a false exclusion is worse than a false inclusion.

Use the classify_document tool to record your decision. Also include a "description" \
field: a brief plain-text transcription or description of the image's contents \
(transcribe any visible text verbatim, and/or describe what it depicts) so it can be \
used later to write a case summary. A few sentences is enough. If the image is \
blurry, low-quality, or only partially legible, say so honestly in "description" \
rather than guessing at content you can't actually read.
"""

# Native tool-calling schema for Stage 1 categorization. `category`'s enum is
# drawn from CATEGORIES so the two never drift apart. `description` is only
# populated by the model on the image-input path (see _IMAGE_CATEGORIZE_PROMPT)
# but stays optional so the same tool serves both text and image calls.
_CLASSIFY_DOCUMENT_TOOL = {
    "name": "classify_document",
    "description": (
        "Record the relevance and category decision for a piece of evidence in an "
        "employment-law dismissal case."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "Whether the document is relevant to the case.",
            },
            "category": {
                "type": "string",
                "enum": CATEGORIES,
                "description": "The category the document best fits.",
            },
            "reason": {
                "type": "string",
                "description": "Short explanation for the decision.",
            },
            "description": {
                "type": "string",
                "description": (
                    "Image evidence only: a brief plain-text transcription or "
                    "description of the image's contents."
                ),
            },
        },
        "required": ["is_relevant", "category", "reason"],
    },
}

_SEMANTIC_SCRUB_PROMPT = """The following text has already had the claimant's name, the \
employer's name, emails, and phone numbers removed by an automated deterministic \
filter and replaced with [REDACTED].

Find and redact any REMAINING contextual personally identifying information that the \
filter would have missed — for example coworker names, implied locations, or a job \
title combined with other details specific enough to identify a third party. Replace \
each one with [REDACTED].

The claimant's own role, department, and similar details about her may be kept where \
needed to tell the story of the case — the goal is to protect other identifiable \
people, not to strip every job-related detail.

Do NOT remove or alter dollar figures, dates, or legal/financial thresholds — in \
particular the $190,100 High Income Threshold must be preserved exactly as given.

Return ONLY the fully redacted text, with no commentary or markdown wrapper.

Text:
---
{text}
---
"""

_SEMANTIC_SCRUB_SYSTEM_PROMPT = (
    "You are a PII-redaction step in an employment-law evidence processing pipeline. "
    + _CASE_BACKGROUND + " "
    "The text you receive has already had the claimant's name, the employer's name, "
    "email addresses, and phone numbers replaced with [REDACTED] by a deterministic "
    "filter — don't worry about those, and don't be thrown off by [REDACTED] tokens "
    "already present in the text. Your job is to find and redact any remaining "
    "contextual personal information and return the redacted text."
)

_SUMMARY_SYSTEM_PROMPT = (
    "You are a case-summary generation step in an employment-law evidence processing "
    "pipeline. " + _CASE_BACKGROUND + " The text you receive has already been through "
    "PII redaction. Your job is to produce a professional, anonymised case summary "
    "based only on that text — report the facts as given; do not offer a legal "
    "opinion or predict the likely outcome of the case."
)

_SUMMARY_PROMPT = """Using ONLY the fully redacted, anonymised text below, write a \
professional, concise legal case summary suitable for a legal marketplace where law \
firms decide whether to take on the case.

The summary should give a law firm enough to assess the case, including — where the \
text supports it — whether the dismissal appears to have been for a valid reason, \
whether there are signs of unfair or unlawful treatment, and whether the $190,100 \
High Income Threshold is relevant to the claimant's eligibility to bring a claim. Do \
not speculate beyond what the text supports, and do not predict the likely outcome or \
give a legal opinion on the merits — describe the facts and leave the assessment to \
the reader.

Format the response as clean, professional Markdown, roughly 400-800 words, with \
these section headings in order: a top-level heading for the case title, then \
## Background, ## Key Facts, ## Potential Claims, and ## Notable Evidence. Use bullet \
points within sections where they improve readability. Return ONLY the Markdown \
document itself — no commentary, and do not wrap it in a code fence.

Do not invent any names, employers, or details not present in the text. Preserve any \
financial or legal thresholds mentioned (e.g. the $190,100 High Income Threshold) \
exactly as given.

Redacted case text:
---
{text}
---
"""


_client: anthropic.Anthropic = None


def _get_client() -> anthropic.Anthropic:
    """Return a lazily-created, module-level Anthropic client shared by every
    call in this module, instead of constructing a new one per API call."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


def _track_usage(response) -> None:
    """Add a response's token usage to the module-level tracker."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    _token_usage["input_tokens"] += usage.input_tokens
    _token_usage["output_tokens"] += usage.output_tokens


def get_usage_report() -> dict:
    """
    Return accumulated token usage and estimated USD cost across every
    messages.create() call made through this module so far, priced at the
    Sonnet rate ($3.00 / 1M input tokens, $15.00 / 1M output tokens).
    """
    input_tokens = _token_usage["input_tokens"]
    output_tokens = _token_usage["output_tokens"]
    total_cost_usd = (
        input_tokens / 1_000_000 * _INPUT_PRICE_PER_MTOK
        + output_tokens / 1_000_000 * _OUTPUT_PRICE_PER_MTOK
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": total_cost_usd,
    }


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def _extract_text_from_response(response, fallback: str = "ERROR: No text generated by LLM.") -> str:
    """
    Pull the text out of an Anthropic response's content blocks.

    Extended-thinking-enabled models can return a ThinkingBlock ahead of the
    TextBlock in response.content, so we can't assume content[0] is text —
    find the first block with type == "text" instead.

    If no text block is present at all (e.g. the model spent its whole
    max_tokens budget on thinking, or hit a safety filter), this does NOT
    raise. It dumps the raw response content for debugging and returns
    `fallback` instead, so one bad response doesn't crash the whole run.
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text

    stop_reason = getattr(response, "stop_reason", None)
    logger.error(
        "No text block found in Anthropic response content (stop_reason=%r); "
        "returning fallback %r. Raw content: %r",
        stop_reason,
        fallback,
        response.content,
    )
    return fallback


def _extract_tool_input(response) -> dict:
    """
    Pull the parsed input dict out of the tool_use block in an Anthropic
    response's content blocks.

    If no tool_use block is present (e.g. the model hit a safety filter or
    spent its whole max_tokens budget on thinking), this does NOT raise. It
    dumps the raw response content for debugging and returns {} instead, so
    one bad response doesn't crash the whole run.
    """
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)

    stop_reason = getattr(response, "stop_reason", None)
    logger.error(
        "No tool_use block found in Anthropic response content (stop_reason=%r); "
        "returning empty dict. Raw content: %r",
        stop_reason,
        response.content,
    )
    return {}


def categorize_document(text: str = None, image_base64: str = None, media_type: str = None) -> dict:
    """
    Ask Claude whether a document is relevant to the case and which category it fits.

    Supply either `text` (for locally-extracted document text) or `image_base64`
    plus `media_type` (for the Vision API, e.g. scanned/photographed evidence) —
    not both. When categorizing an image, the model is also asked for a brief
    "description" transcribing/describing its contents, since that's the only
    text representation available for Stage 2 summarization; for text documents
    "description" is left as "" since the full extracted text already exists.

    The decision is obtained via forced native tool calling (classify_document)
    rather than free-form JSON, so the response's shape is guaranteed by the API.

    Returns a dict with keys is_relevant (bool), category (str), reason (str),
    description (str). On any API error or missing input, returns a safe
    fallback dict instead of raising.
    """
    try:
        client = _get_client()

        if image_base64:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_base64,
                    },
                },
                {
                    "type": "text",
                    "text": _IMAGE_CATEGORIZE_PROMPT.format(
                        case_background=_CASE_BACKGROUND,
                        category_definitions=_CATEGORY_DEFINITIONS,
                    ),
                },
            ]
        elif text:
            content = _CATEGORIZE_PROMPT.format(
                case_background=_CASE_BACKGROUND,
                category_definitions=_CATEGORY_DEFINITIONS,
                text=text,
            )
        else:
            raise ValueError("categorize_document requires either `text` or `image_base64`.")

        response = _create_message(
            client,
            model=MODEL_NAME,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
            tools=[_CLASSIFY_DOCUMENT_TOOL],
            tool_choice={"type": "tool", "name": "classify_document"},
        )
        _track_usage(response)
        data = _extract_tool_input(response)
        return {
            "is_relevant": bool(data.get("is_relevant", False)),
            "category": str(data.get("category", _CATEGORIZE_FALLBACK["category"])),
            "reason": str(data.get("reason", "No reason provided by LLM.")),
            "description": str(data.get("description", "")),
        }
    except Exception as e:
        logger.error("categorize_document: LLM call failed: %s", e)
        return dict(_CATEGORIZE_FALLBACK)


def semantic_scrub(scrubbed_text: str) -> str:
    """
    Ask Claude to redact residual contextual PII from text already run through
    sanitizer.deterministic_scrub. On any API error, returns the input unchanged
    (already-deterministic-scrubbed text) rather than raising.
    """
    try:
        client = _get_client()
        response = _create_message(
            client,
            model=MODEL_NAME,
            max_tokens=16384,
            system=_SEMANTIC_SCRUB_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _SEMANTIC_SCRUB_PROMPT.format(text=scrubbed_text)}
            ],
        )
        _track_usage(response)
        return _extract_text_from_response(response).strip()
    except Exception as e:
        logger.error("semantic_scrub: LLM call failed: %s", e)
        return scrubbed_text


def generate_summary(safe_text: str) -> str:
    """
    Generate the final anonymised case summary (as Markdown) from fully
    redacted text. On any API error, returns "" rather than raising.
    """
    try:
        client = _get_client()
        response = _create_message(
            client,
            model=MODEL_NAME,
            max_tokens=16384,
            system=_SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _SUMMARY_PROMPT.format(text=safe_text)}],
        )
        _track_usage(response)
        raw = _extract_text_from_response(response)
        return _strip_markdown_fences(raw)
    except Exception as e:
        logger.error("generate_summary: LLM call failed: %s", e)
        return ""

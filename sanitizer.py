"""
Module 2: Deterministic Sanitizer (local, CPU-bound)

Regex-based redaction of known entities and standard PII patterns.
No network calls, no LLM logic — see RFC.md's Two-Pass Hybrid Sanitization Pipeline,
pass 1 (Deterministic Scrubbing).
"""

import re

REDACTION_TOKEN = "[REDACTED]"

# Known, case-specific entities that must always be hard-redacted.
KNOWN_ENTITIES = [
    "Michelle Anne Ritchie",
    "Michelle Ritchie",
    "Northern Rivers Allied Health",
]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Matches common AU phone formats: 0412 345 678, (02) 9876 5432, +61 412 345 678, 02-9876-5432
PHONE_RE = re.compile(r"(\+61[\s-]?\d|\(0\d\)|0\d)[\d\s-]{6,11}\d")

# Critical financial figures that must survive scrubbing untouched, e.g. the
# high-income threshold. Matched and protected BEFORE any other redaction runs.
WHITELIST_RE = re.compile(r"\$\s?190,100")

_ENTITY_RE = re.compile(
    "|".join(re.escape(entity) for entity in KNOWN_ENTITIES), re.IGNORECASE
)


def deterministic_scrub(raw_text: str) -> str:
    """
    Redact known entities, emails, and phone numbers with [REDACTED].

    Critical financial figures (e.g. the $190,100 high-income threshold) are
    explicitly protected and preserved untouched, even if a later regex pass
    would otherwise have matched part of them.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return ""

    text = raw_text

    # Protect whitelisted financial figures behind placeholder tokens so no
    # subsequent redaction pass can touch them.
    protected = {}

    def _protect(match: re.Match) -> str:
        token = f"__PROTECTED_{len(protected)}__"
        protected[token] = match.group(0)
        return token

    text = WHITELIST_RE.sub(_protect, text)

    text = _ENTITY_RE.sub(REDACTION_TOKEN, text)
    text = EMAIL_RE.sub(REDACTION_TOKEN, text)
    text = PHONE_RE.sub(REDACTION_TOKEN, text)

    for token, original in protected.items():
        text = text.replace(token, original)

    return text

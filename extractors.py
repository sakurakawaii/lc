"""
Module 1: Local Extractors

Pure, local extraction functions for .pdf, .docx, .eml, and image files.
No LLM calls, no file routing/pipeline logic, no CLI code — see RFC.md Module 1.
Image bytes are base64-encoded here for the Vision API, but sending them to
Claude is llm_service.py's job, not this module's.
"""

import base64
import email
import logging
import os
import re
from email.message import Message
from email.policy import default as default_policy
from pathlib import Path
from typing import List, Optional, Tuple

import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def extract_text_from_pdf(file_path: str) -> str:
    """Extract concatenated text from all pages of a PDF. Returns "" on any failure."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_text = []
            for page in reader.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception as e:
                    logger.warning("Failed to extract a page from %s: %s", file_path, e)
            return "\n".join(pages_text).strip()
    except Exception as e:
        logger.warning("Failed to read PDF %s: %s", file_path, e)
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """Extract concatenated paragraph text from a .docx file. Returns "" on any failure."""
    try:
        document = Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        logger.warning("Failed to read DOCX %s: %s", file_path, e)
        return ""


def extract_image(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Read an image file and base64-encode it for the Anthropic Vision API.

    Returns (base64_string, media_type), e.g. ("iVBORw0...", "image/png").
    Returns (None, None) on any failure, including an unrecognized extension.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        media_type = _IMAGE_MEDIA_TYPES.get(ext)
        if media_type is None:
            logger.warning("Unsupported image extension for %s", file_path)
            return None, None

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        return base64.b64encode(raw_bytes).decode("ascii"), media_type
    except Exception as e:
        logger.warning("Failed to read/encode image %s: %s", file_path, e)
        return None, None


def _strip_html(html: str) -> str:
    try:
        return _HTML_TAG_RE.sub(" ", html).strip()
    except Exception:
        return ""


def _extract_body_text(msg: Message) -> str:
    try:
        body_part = msg.get_body(preferencelist=("plain", "html"))
        if body_part is None:
            return ""
        content = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            return _strip_html(content)
        return content.strip()
    except Exception as e:
        logger.warning("Failed to extract body text from email: %s", e)
        return ""


def _safe_attachment_filename(raw_name: str, index: int) -> str:
    name = raw_name or f"attachment_{index}"
    name = os.path.basename(name)
    name = re.sub(r"[^\w.\-]", "_", name).strip("_") or f"attachment_{index}"
    return name


def _save_attachments(msg: Message, attachment_dir: str) -> List[str]:
    saved_paths: List[str] = []
    try:
        os.makedirs(attachment_dir, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create attachment directory %s: %s", attachment_dir, e)
        return saved_paths

    try:
        attachments = list(msg.iter_attachments())
    except Exception as e:
        logger.warning("Failed to enumerate email attachments: %s", e)
        return saved_paths

    for index, part in enumerate(attachments):
        try:
            filename = _safe_attachment_filename(part.get_filename(), index)
            dest_path = Path(attachment_dir) / filename

            counter = 1
            while dest_path.exists():
                stem, suffix = os.path.splitext(filename)
                dest_path = Path(attachment_dir) / f"{stem}_{counter}{suffix}"
                counter += 1

            payload = part.get_content()
            mode = "wb" if isinstance(payload, (bytes, bytearray)) else "w"
            with open(dest_path, mode) as f:
                f.write(payload)

            saved_paths.append(str(dest_path))
        except Exception as e:
            logger.warning("Failed to save an email attachment: %s", e)
            continue

    return saved_paths


def extract_text_from_eml(file_path: str, attachment_dir: str = "./tmp_attachments/") -> Tuple[str, List[str]]:
    """
    Extract the plain-text body and save any attachments to `attachment_dir`.

    Returns a tuple of (body_text, saved_attachment_paths). On any unrecoverable
    failure, returns ("", []) rather than raising.
    """
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=default_policy)
    except Exception as e:
        logger.warning("Failed to read/parse EML %s: %s", file_path, e)
        return "", []

    body_text = _extract_body_text(msg)
    saved_paths = _save_attachments(msg, attachment_dir)

    return body_text, saved_paths

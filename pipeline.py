"""
Module 4: Pipeline Orchestrator

Stage 1 orchestration: extracts the input zip to a temporary workspace, walks
every file, extracts text/images via extractors.py, categorizes each via
llm_service.categorize_document() (text or Vision API, depending on file
type), and non-destructively copies each file into its routed destination
(Evidence_Package/<category>/ or Excluded_Documents/).

On success, also writes:
    - {base_output_dir}/stage1_cache.json — cached relevant texts, so a later
      run can skip straight to Stage 2 via load_stage1_cache().
    - {base_output_dir}/audit_report.csv — a per-file audit trail of every
      categorization decision (file name, category, is_relevant, reason).

Owns no regex/LLM logic itself — see sanitizer.py and llm_service.py. Owns no
rich/terminal UI logic either — progress is reported through a plain callback
so main.py can render it however it likes.
"""

import csv
import json
import logging
import os
import shutil
import zipfile
from collections import deque

import extractors
import llm_service

logger = logging.getLogger(__name__)

EXCLUSION_LOG_FILENAME = "exclusion_log.txt"
CACHE_FILENAME = "stage1_cache.json"
AUDIT_REPORT_FILENAME = "audit_report.csv"
AUDIT_REPORT_FIELDNAMES = ["file_name", "category", "is_relevant", "reason", "copy_failed"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _report(progress_callback, message, done=False):
    """
    Notify the caller's progress_callback(message, done) of a step starting
    (done=False) or finishing (done=True), so a UI can show a live "in
    progress" indicator and then a permanent completed-step line, without
    this module depending on any particular UI library.
    """
    if progress_callback is None:
        return
    try:
        progress_callback(message, done)
    except Exception as e:
        logger.warning("progress_callback raised and was ignored: %s", e)


def _extract_content_and_children(file_path, attachment_dir):
    """
    Dispatches on file extension and returns (kind, payload, child_file_paths).

        kind == "text":  payload is the extracted text (str)
        kind == "image": payload is (image_base64, media_type)
        kind == "unsupported": payload is None

    child_file_paths holds any newly-saved .eml attachments to also process.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            return "text", extractors.extract_text_from_pdf(file_path), []
        if ext == ".docx":
            return "text", extractors.extract_text_from_docx(file_path), []
        if ext == ".eml":
            body, attachments = extractors.extract_text_from_eml(
                file_path, attachment_dir=attachment_dir
            )
            return "text", body, list(attachments)
        if ext in IMAGE_EXTENSIONS:
            image_base64, media_type = extractors.extract_image(file_path)
            if not image_base64:
                # Covers both a failed read (None) and a 0-byte image file
                # (base64.b64encode(b"") == b"" is falsy too), so an empty
                # file is reported as unsupported/no content up front rather
                # than round-tripping to the LLM just to hit the same result
                # via a generic "categorization failed" fallback.
                return "unsupported", None, []
            return "image", (image_base64, media_type), []
        return "unsupported", None, []
    except Exception as e:
        logger.warning("Unexpected error extracting content from %s: %s", file_path, e)
        return "unsupported", None, []


def _unique_destination(dest_dir, filename):
    stem, suffix = os.path.splitext(filename)
    dest_path = os.path.join(dest_dir, filename)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{stem}_{counter}{suffix}")
        counter += 1
    return dest_path


def _safe_copy(src_path, dest_dir):
    """Non-destructively copy src_path into dest_dir. Returns dest path, or None on failure."""
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = _unique_destination(dest_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        return dest_path
    except Exception as e:
        logger.warning("Failed to copy %s to %s: %s", src_path, dest_dir, e)
        return None


def _log_exclusion_reason(excluded_dir, filename, reason):
    try:
        os.makedirs(excluded_dir, exist_ok=True)
        log_path = os.path.join(excluded_dir, EXCLUSION_LOG_FILENAME)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}\t{reason}\n")
    except Exception as e:
        logger.warning("Failed to write exclusion log entry for %s: %s", filename, e)


def _discover_files(root_dir):
    files = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def _write_stage1_cache(base_output_dir, relevant_texts):
    """Persist the aggregated relevant texts so --skip-stage1 can reload them later."""
    try:
        os.makedirs(base_output_dir, exist_ok=True)
        cache_path = os.path.join(base_output_dir, CACHE_FILENAME)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"relevant_texts": relevant_texts}, f)
        return cache_path
    except Exception as e:
        logger.warning("Failed to write Stage 1 cache: %s", e)
        return None


def load_stage1_cache(base_output_dir="./output"):
    """
    Load previously cached Stage 1 relevant texts (written by a prior
    process_evidence_package() run).

    Returns a list of strings, or None if the cache file is missing/corrupt.
    """
    cache_path = os.path.join(base_output_dir, CACHE_FILENAME)
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        relevant_texts = data.get("relevant_texts")
        if not isinstance(relevant_texts, list):
            raise ValueError("Cache file is missing a 'relevant_texts' list.")
        return relevant_texts
    except Exception as e:
        logger.error("Failed to load Stage 1 cache from %s: %s", cache_path, e)
        return None


def _write_audit_report(base_output_dir, audit_log):
    """Write the per-file categorization audit trail to a CSV for manual review."""
    try:
        os.makedirs(base_output_dir, exist_ok=True)
        report_path = os.path.join(base_output_dir, AUDIT_REPORT_FILENAME)
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_REPORT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(audit_log)
        return report_path
    except Exception as e:
        logger.warning("Failed to write audit report CSV: %s", e)
        return None


def process_evidence_package(
    zip_path,
    base_output_dir="./output",
    tmp_workspace_dir="./tmp_workspace/",
    progress_callback=None,
):
    """
    Stage 1: extract, categorize, and route every file inside `zip_path`.

    `progress_callback`, if given, is called as progress_callback(message, done)
    at each notable step: once with done=False when a step starts (e.g.
    "Extracting text from document_A.pdf...") and once with done=True when it
    finishes, so a caller can render both a live "in progress" indicator and a
    permanent completed-step history without this module depending on any
    particular UI library.

    Returns a dict:
        {
            "success": bool,
            "relevant_texts": List[str],        # text of every file routed as relevant
            "category_counts": Dict[str, int],  # counts per Evidence_Package/<category>
            "excluded_count": int,
            "total_files": int,
            "audit_log": List[Dict],            # {file_name, category, is_relevant, reason}
            "cache_path": Optional[str],         # where stage1_cache.json was written
            "audit_report_path": Optional[str],  # where audit_report.csv was written
        }

    The source zip is never modified. On any unrecoverable error (missing
    zip, corrupt archive), returns success=False with empty/zeroed fields
    rather than raising.
    """
    result = {
        "success": False,
        "relevant_texts": [],
        "category_counts": {},
        "excluded_count": 0,
        "total_files": 0,
        "audit_log": [],
        "cache_path": None,
        "audit_report_path": None,
    }

    if not os.path.isfile(zip_path):
        logger.error("Zip file not found: %s", zip_path)
        return result

    attachment_dir = os.path.join(tmp_workspace_dir, "tmp_attachments")
    evidence_root = os.path.join(base_output_dir, "Evidence_Package")
    excluded_dir = os.path.join(base_output_dir, "Excluded_Documents")

    archive_message = f"Extracting archive {os.path.basename(zip_path)}..."
    _report(progress_callback, archive_message)
    try:
        os.makedirs(tmp_workspace_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_workspace_dir)
        _report(progress_callback, archive_message, done=True)
    except Exception as e:
        logger.error("Failed to extract zip file %s: %s", zip_path, e)
        shutil.rmtree(tmp_workspace_dir, ignore_errors=True)
        return result

    audit_log = []

    try:
        queue = deque(_discover_files(tmp_workspace_dir))

        while queue:
            file_path = queue.popleft()
            filename = os.path.basename(file_path)
            result["total_files"] += 1

            extract_message = f"Extracting content from {filename}..."
            _report(progress_callback, extract_message)
            kind, payload, children = _extract_content_and_children(file_path, attachment_dir)
            queue.extend(children)
            _report(progress_callback, extract_message, done=True)

            stage2_text = ""
            if kind == "text" and payload:
                categorize_message = f"Calling LLM for categorization of {filename}..."
                _report(progress_callback, categorize_message)
                categorization = llm_service.categorize_document(text=payload)
                _report(progress_callback, categorize_message, done=True)
                stage2_text = payload
            elif kind == "image":
                image_base64, media_type = payload
                categorize_message = f"Calling Vision LLM for categorization of {filename}..."
                _report(progress_callback, categorize_message)
                categorization = llm_service.categorize_document(
                    image_base64=image_base64, media_type=media_type
                )
                _report(progress_callback, categorize_message, done=True)
                stage2_text = categorization.get("description") or ""
            else:
                categorization = {
                    "is_relevant": False,
                    "category": None,
                    "reason": "Unsupported file type or no extractable text.",
                    "description": "",
                }

            is_relevant = bool(categorization.get("is_relevant"))
            reason = categorization.get("reason", "")

            if is_relevant:
                category = categorization.get("category") or "Uncategorized"
                dest_dir = os.path.join(evidence_root, category)
                route_message = f"Routing {filename} to Evidence_Package/{category}..."
                _report(progress_callback, route_message)
                copied = _safe_copy(file_path, dest_dir)
                _report(progress_callback, route_message, done=True)
                copy_failed = copied is None
                if not copy_failed:
                    result["category_counts"][category] = result["category_counts"].get(category, 0) + 1
                    if stage2_text:
                        result["relevant_texts"].append(stage2_text)
            else:
                route_message = f"Routing {filename} to Excluded_Documents..."
                _report(progress_callback, route_message)
                copied = _safe_copy(file_path, excluded_dir)
                _report(progress_callback, route_message, done=True)
                copy_failed = copied is None
                if not copy_failed:
                    result["excluded_count"] += 1
                    _log_exclusion_reason(excluded_dir, filename, reason)

            if copy_failed:
                logger.error(
                    "File %s was classified but could not be copied to output; "
                    "it is NOT present in the output directory.",
                    filename,
                )

            audit_log.append(
                {
                    "file_name": filename,
                    "category": categorization.get("category") or "",
                    "is_relevant": is_relevant,
                    "reason": reason,
                    "copy_failed": copy_failed,
                }
            )

        result["audit_log"] = audit_log

        cache_message = "Writing Stage 1 cache and audit report..."
        _report(progress_callback, cache_message)
        result["cache_path"] = _write_stage1_cache(base_output_dir, result["relevant_texts"])
        result["audit_report_path"] = _write_audit_report(base_output_dir, audit_log)
        _report(progress_callback, cache_message, done=True)

        result["success"] = True
    except Exception as e:
        logger.error("Unexpected error while processing evidence package: %s", e)
    finally:
        shutil.rmtree(tmp_workspace_dir, ignore_errors=True)

    return result

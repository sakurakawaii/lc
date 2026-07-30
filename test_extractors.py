"""
Quick manual smoke test for extractors.py.

Expects dummy fixtures at:
    ./testdata/sample.pdf
    ./testdata/sample.docx
    ./testdata/sample.eml   (with at least one attachment)

Run with: python test_extractors.py
"""

import os

from extractors import extract_text_from_docx, extract_text_from_eml, extract_text_from_pdf

TEST_DATA_DIR = "./testdata/"
ATTACHMENT_DIR = os.path.join(TEST_DATA_DIR, "tmp_attachments/")

PDF_PATH = os.path.join(TEST_DATA_DIR, "sample.pdf")
DOCX_PATH = os.path.join(TEST_DATA_DIR, "sample.docx")
EML_PATH = os.path.join(TEST_DATA_DIR, "sample.eml")


def preview(text: str, length: int = 100) -> str:
    return text[:length] if text else "(empty)"


def test_pdf():
    print("\n--- PDF extraction ---")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"First {100} chars: {preview(text)!r}")


def test_docx():
    print("\n--- DOCX extraction ---")
    text = extract_text_from_docx(DOCX_PATH)
    print(f"First {100} chars: {preview(text)!r}")


def test_eml():
    print("\n--- EML extraction ---")
    body_text, attachment_paths = extract_text_from_eml(EML_PATH, attachment_dir=ATTACHMENT_DIR)
    print(f"First {100} chars of body: {preview(body_text)!r}")

    print(f"Reported attachment paths: {attachment_paths}")
    if not attachment_paths:
        print("No attachments reported.")
        return

    all_saved = True
    for path in attachment_paths:
        exists = os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists and size > 0 else "MISSING/EMPTY"
        print(f"  {path} -> {status} ({size} bytes)")
        all_saved = all_saved and exists and size > 0

    print("All attachments saved correctly." if all_saved else "One or more attachments failed to save.")


if __name__ == "__main__":
    test_pdf()
    test_docx()
    test_eml()

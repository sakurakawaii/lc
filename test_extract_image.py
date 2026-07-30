"""
Unit tests for extractors.extract_image().

Uses mocked file I/O — no real image files are read from disk.
"""

import base64
import unittest
from unittest.mock import mock_open, patch

from extractors import extract_image

FAKE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\nfake-binary-image-data"


class TestExtractImage(unittest.TestCase):
    def test_encodes_png_file_and_returns_correct_media_type(self):
        with patch("builtins.open", mock_open(read_data=FAKE_IMAGE_BYTES)):
            encoded, media_type = extract_image("photo.png")

        self.assertEqual(media_type, "image/png")
        self.assertEqual(encoded, base64.b64encode(FAKE_IMAGE_BYTES).decode("ascii"))

    def test_encodes_jpg_file_and_returns_correct_media_type(self):
        with patch("builtins.open", mock_open(read_data=FAKE_IMAGE_BYTES)):
            encoded, media_type = extract_image("photo.jpg")

        self.assertEqual(media_type, "image/jpeg")
        self.assertEqual(encoded, base64.b64encode(FAKE_IMAGE_BYTES).decode("ascii"))

    def test_encodes_jpeg_file_and_returns_correct_media_type(self):
        with patch("builtins.open", mock_open(read_data=FAKE_IMAGE_BYTES)):
            encoded, media_type = extract_image("photo.jpeg")

        self.assertEqual(media_type, "image/jpeg")
        self.assertEqual(encoded, base64.b64encode(FAKE_IMAGE_BYTES).decode("ascii"))

    def test_case_insensitive_extension_matching(self):
        with patch("builtins.open", mock_open(read_data=FAKE_IMAGE_BYTES)):
            encoded, media_type = extract_image("PHOTO.PNG")

        self.assertEqual(media_type, "image/png")
        self.assertIsNotNone(encoded)

    def test_unsupported_extension_returns_none_without_opening_file(self):
        with patch("builtins.open", mock_open(read_data=FAKE_IMAGE_BYTES)) as mock_open_file:
            encoded, media_type = extract_image("document.gif")

        self.assertIsNone(encoded)
        self.assertIsNone(media_type)
        mock_open_file.assert_not_called()

    def test_missing_file_returns_none_none(self):
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            encoded, media_type = extract_image("missing.png")

        self.assertIsNone(encoded)
        self.assertIsNone(media_type)


if __name__ == "__main__":
    unittest.main()

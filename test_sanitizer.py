"""Unit tests for sanitizer.py — pure regex logic, zero API cost, zero I/O."""

import unittest

from sanitizer import deterministic_scrub


class TestDeterministicScrub(unittest.TestCase):
    def test_redacts_known_client_name(self):
        result = deterministic_scrub("The claimant, Michelle Anne Ritchie, was dismissed.")
        self.assertNotIn("Michelle Anne Ritchie", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_known_employer_name(self):
        result = deterministic_scrub("She was employed by Northern Rivers Allied Health.")
        self.assertNotIn("Northern Rivers Allied Health", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_email_addresses(self):
        result = deterministic_scrub("Contact her at michelle.ritchie@example.com for details.")
        self.assertNotIn("michelle.ritchie@example.com", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_phone_numbers(self):
        result = deterministic_scrub("Call 0412 345 678 to confirm the meeting.")
        self.assertNotIn("0412 345 678", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_landline_with_area_code(self):
        result = deterministic_scrub("Office line: (02) 9876 5432.")
        self.assertNotIn("(02) 9876 5432", result)
        self.assertIn("[REDACTED]", result)

    def test_preserves_high_income_threshold_untouched(self):
        text = "The high-income threshold of $190,100 applies to this claim."
        result = deterministic_scrub(text)
        self.assertIn("$190,100", result)

    def test_preserves_threshold_even_alongside_other_redactions(self):
        text = (
            "Michelle Anne Ritchie earned above the $190,100 threshold; "
            "contact hr@northernrivers.example for payroll records."
        )
        result = deterministic_scrub(text)
        self.assertIn("$190,100", result)
        self.assertNotIn("Michelle Anne Ritchie", result)
        self.assertNotIn("hr@northernrivers.example", result)

    def test_empty_string_input(self):
        self.assertEqual(deterministic_scrub(""), "")

    def test_none_input_returns_empty_string(self):
        self.assertEqual(deterministic_scrub(None), "")

    def test_text_with_no_pii_is_unchanged(self):
        text = "This is a plain sentence with no sensitive information at all."
        self.assertEqual(deterministic_scrub(text), text)


if __name__ == "__main__":
    unittest.main()

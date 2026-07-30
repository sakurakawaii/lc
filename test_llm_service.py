"""
Unit tests for llm_service.py.

The Anthropic client is fully mocked via unittest.mock — no real network calls
are made and no API key is required to run these tests.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import llm_service
from llm_service import categorize_document, generate_summary, semantic_scrub


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _thinking_block(thinking: str = "reasoning...") -> MagicMock:
    block = MagicMock(spec=["type", "thinking"])
    block.type = "thinking"
    block.thinking = thinking
    return block


def _mock_response(text: str, leading_thinking: bool = False) -> MagicMock:
    response = MagicMock()
    blocks = [_thinking_block()] if leading_thinking else []
    blocks.append(_text_block(text))
    response.content = blocks
    return response


class TestCategorizeDocument(unittest.TestCase):
    @patch("llm_service.anthropic.Anthropic")
    def test_valid_json_response_is_parsed_correctly(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            json.dumps(
                {
                    "is_relevant": True,
                    "category": "Termination_Documents",
                    "reason": "Contains the dismissal letter.",
                }
            )
        )

        result = categorize_document(text="Some termination letter text.")

        self.assertEqual(
            result,
            {
                "is_relevant": True,
                "category": "Termination_Documents",
                "reason": "Contains the dismissal letter.",
                "description": "",
            },
        )

    @patch("llm_service.anthropic.Anthropic")
    def test_strips_markdown_code_fence_before_parsing(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        payload = json.dumps(
            {"is_relevant": False, "category": "Other_Relevant_Evidence", "reason": "Not related."}
        )
        mock_client.messages.create.return_value = _mock_response(f"```json\n{payload}\n```")

        result = categorize_document("Unrelated grocery receipt.")

        self.assertFalse(result["is_relevant"])
        self.assertEqual(result["category"], "Other_Relevant_Evidence")

    @patch("llm_service.anthropic.Anthropic")
    def test_invalid_json_returns_safe_fallback_without_raising(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response("This is not valid JSON at all.")

        result = categorize_document("Some text.")

        self.assertFalse(result["is_relevant"])
        self.assertIn("category", result)
        self.assertIn("reason", result)

    @patch("llm_service.anthropic.Anthropic")
    def test_api_exception_returns_safe_fallback_without_raising(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = ConnectionError("network down")

        result = categorize_document("Some text.")

        self.assertFalse(result["is_relevant"])
        self.assertIn("category", result)
        self.assertIn("reason", result)

    @patch("llm_service.anthropic.Anthropic")
    def test_missing_keys_in_json_fall_back_to_defaults(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(json.dumps({}))

        result = categorize_document("Some text.")

        self.assertFalse(result["is_relevant"])
        self.assertIsInstance(result["category"], str)
        self.assertIsInstance(result["reason"], str)

    @patch("llm_service.anthropic.Anthropic")
    def test_skips_leading_thinking_block_and_uses_text_block(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        payload = json.dumps(
            {
                "is_relevant": True,
                "category": "Correspondence",
                "reason": "Dismissal-related email thread.",
            }
        )
        mock_client.messages.create.return_value = _mock_response(payload, leading_thinking=True)

        result = categorize_document("Some email text.")

        self.assertEqual(
            result,
            {
                "is_relevant": True,
                "category": "Correspondence",
                "reason": "Dismissal-related email thread.",
                "description": "",
            },
        )

    @patch("builtins.print")
    @patch("llm_service.anthropic.Anthropic")
    def test_response_with_only_a_thinking_block_returns_safe_fallback(
        self, mock_anthropic_cls, mock_print
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        response = MagicMock()
        response.content = [_thinking_block()]
        response.stop_reason = "max_tokens"
        mock_client.messages.create.return_value = response

        result = categorize_document("Some text.")

        # No text block -> the "{}" fallback parses cleanly, so this doesn't
        # go through the exception handler at all; it's a normal, if empty,
        # categorization result.
        self.assertFalse(result["is_relevant"])
        self.assertIn("category", result)
        self.assertIn("reason", result)
        self.assertIn("description", result)

        mock_print.assert_called_once()
        (debug_message,), _ = mock_print.call_args
        self.assertIn("DEBUG API ERROR", debug_message)
        self.assertIn("stop_reason", debug_message)
        self.assertIn("max_tokens", debug_message)
        self.assertIn("Raw Content:", debug_message)

    @patch("llm_service.anthropic.Anthropic")
    def test_image_input_builds_vision_payload_and_returns_description(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            json.dumps(
                {
                    "is_relevant": True,
                    "category": "Performance_Reviews",
                    "reason": "Photo of a signed performance review.",
                    "description": "A scanned performance review form with a signature at the bottom.",
                }
            )
        )

        result = categorize_document(image_base64="ZmFrZS1pbWFnZS1ieXRlcw==", media_type="image/png")

        self.assertEqual(
            result,
            {
                "is_relevant": True,
                "category": "Performance_Reviews",
                "reason": "Photo of a signed performance review.",
                "description": "A scanned performance review form with a signature at the bottom.",
            },
        )

        _, call_kwargs = mock_client.messages.create.call_args
        content = call_kwargs["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)

        image_block, text_block = content
        self.assertEqual(image_block["type"], "image")
        self.assertEqual(image_block["source"]["type"], "base64")
        self.assertEqual(image_block["source"]["media_type"], "image/png")
        self.assertEqual(image_block["source"]["data"], "ZmFrZS1pbWFnZS1ieXRlcw==")
        self.assertEqual(text_block["type"], "text")
        self.assertIn("Valid categories", text_block["text"])

    @patch("llm_service.anthropic.Anthropic")
    def test_missing_text_and_image_returns_safe_fallback_without_calling_api(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        result = categorize_document()

        mock_client.messages.create.assert_not_called()
        self.assertEqual(result, dict(llm_service._CATEGORIZE_FALLBACK))


class TestSemanticScrub(unittest.TestCase):
    @patch("llm_service.anthropic.Anthropic")
    def test_returns_llm_redacted_text_on_success(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            "[REDACTED] mentioned the incident to [REDACTED]."
        )

        result = semantic_scrub("A coworker mentioned the incident to another coworker.")

        self.assertEqual(result, "[REDACTED] mentioned the incident to [REDACTED].")

    @patch("llm_service.anthropic.Anthropic")
    def test_passes_honest_system_prompt_and_higher_max_tokens(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response("[REDACTED] text.")

        semantic_scrub("Some already-deterministically-scrubbed text.")

        _, call_kwargs = mock_client.messages.create.call_args
        self.assertEqual(call_kwargs["max_tokens"], 16384)
        self.assertEqual(call_kwargs["system"], llm_service._SEMANTIC_SCRUB_SYSTEM_PROMPT)
        # The system prompt must describe the model's real role honestly —
        # it must not claim the data is fake, or instruct it not to refuse.
        self.assertNotIn("test data", call_kwargs["system"].lower())
        self.assertNotIn("without refusing", call_kwargs["system"].lower())

    @patch("llm_service.anthropic.Anthropic")
    def test_api_exception_falls_back_to_input_text(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = TimeoutError("timed out")

        original = "Already deterministically [REDACTED] text."
        result = semantic_scrub(original)

        self.assertEqual(result, original)

    @patch("llm_service.anthropic.Anthropic")
    def test_skips_leading_thinking_block_and_uses_text_block(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            "[REDACTED] mentioned the incident to [REDACTED].", leading_thinking=True
        )

        result = semantic_scrub("A coworker mentioned the incident to another coworker.")

        self.assertEqual(result, "[REDACTED] mentioned the incident to [REDACTED].")

    @patch("builtins.print")
    @patch("llm_service.anthropic.Anthropic")
    def test_response_with_only_a_thinking_block_returns_no_text_error_string(
        self, mock_anthropic_cls, mock_print
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        response = MagicMock()
        response.content = [_thinking_block()]
        response.stop_reason = "max_tokens"
        mock_client.messages.create.return_value = response

        original = "Already deterministically [REDACTED] text."
        result = semantic_scrub(original)

        # A response that came back successfully but produced no text block
        # is a distinct failure mode from an API exception: it surfaces the
        # explicit error sentinel rather than silently passing input through.
        self.assertEqual(result, "ERROR: No text generated by LLM.")

        mock_print.assert_called_once()
        (debug_message,), _ = mock_print.call_args
        self.assertIn("DEBUG API ERROR", debug_message)
        self.assertIn("stop_reason", debug_message)
        self.assertIn("max_tokens", debug_message)
        self.assertIn("Raw Content:", debug_message)


class TestGenerateSummary(unittest.TestCase):
    @patch("llm_service.anthropic.Anthropic")
    def test_returns_llm_generated_summary_on_success(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response("A professional case summary.")

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(result, "A professional case summary.")

    @patch("llm_service.anthropic.Anthropic")
    def test_strips_stray_markdown_code_fence_from_summary(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            "```markdown\n# Case Summary\n\nSome content.\n```"
        )

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(result, "# Case Summary\n\nSome content.")

    @patch("llm_service.anthropic.Anthropic")
    def test_passes_honest_system_prompt_and_higher_max_tokens(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response("# Case Summary")

        generate_summary("Fully redacted case text.")

        _, call_kwargs = mock_client.messages.create.call_args
        self.assertEqual(call_kwargs["max_tokens"], 16384)
        self.assertEqual(call_kwargs["system"], llm_service._SUMMARY_SYSTEM_PROMPT)
        self.assertNotIn("test data", call_kwargs["system"].lower())
        self.assertNotIn("without refusing", call_kwargs["system"].lower())

    @patch("llm_service.anthropic.Anthropic")
    def test_api_exception_returns_empty_string(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("boom")

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(result, "")

    @patch("llm_service.anthropic.Anthropic")
    def test_skips_leading_thinking_block_and_uses_text_block(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            "A professional case summary.", leading_thinking=True
        )

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(result, "A professional case summary.")

    @patch("builtins.print")
    @patch("llm_service.anthropic.Anthropic")
    def test_response_with_only_a_thinking_block_returns_no_text_error_string(
        self, mock_anthropic_cls, mock_print
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        response = MagicMock()
        response.content = [_thinking_block()]
        response.stop_reason = "max_tokens"
        mock_client.messages.create.return_value = response

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(result, "ERROR: No text generated by LLM.")

        mock_print.assert_called_once()
        (debug_message,), _ = mock_print.call_args
        self.assertIn("DEBUG API ERROR", debug_message)
        self.assertIn("stop_reason", debug_message)
        self.assertIn("max_tokens", debug_message)
        self.assertIn("Raw Content:", debug_message)


class TestClientInitialization(unittest.TestCase):
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}, clear=False)
    @patch("llm_service.anthropic.Anthropic")
    def test_client_initialized_with_env_api_key(self, mock_anthropic_cls):
        llm_service._get_client()
        mock_anthropic_cls.assert_called_once_with(api_key="test-key-123")


if __name__ == "__main__":
    unittest.main()

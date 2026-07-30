"""
Unit tests for llm_service.py.

The Anthropic client is fully mocked via unittest.mock — no real network calls
are made and no API key is required to run these tests.
"""

import unittest
from unittest.mock import MagicMock, patch

import anthropic
import httpx

import llm_service
from llm_service import categorize_document, generate_summary, get_usage_report, semantic_scrub


def _api_status_error(error_cls, status_code: int = None, message: str = "error") -> Exception:
    """Build a real instance of an anthropic.APIStatusError subclass (e.g.
    RateLimitError, InternalServerError, BadRequestError) with a fake
    request/response, so the retry predicate sees a real status_code.

    `status_code` must be passed explicitly for error classes (like
    InternalServerError) that don't declare a fixed status_code of their own.
    """
    status_code = status_code if status_code is not None else error_cls.status_code
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"type": "error", "message": message}},
    )
    return error_cls(message, response=response, body=response.json())


def _connection_error(message: str = "network down") -> Exception:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(message=message, request=request)


def _mock_usage(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


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


def _tool_use_block(tool_input: dict, name: str = "classify_document") -> MagicMock:
    block = MagicMock(spec=["type", "name", "input", "id"])
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = "toolu_test123"
    return block


def _mock_response(text: str, leading_thinking: bool = False, usage=None) -> MagicMock:
    response = MagicMock()
    blocks = [_thinking_block()] if leading_thinking else []
    blocks.append(_text_block(text))
    response.content = blocks
    response.usage = usage if usage is not None else _mock_usage()
    return response


def _mock_tool_response(tool_input: dict, leading_thinking: bool = False, usage=None) -> MagicMock:
    response = MagicMock()
    blocks = [_thinking_block()] if leading_thinking else []
    blocks.append(_tool_use_block(tool_input))
    response.content = blocks
    response.usage = usage if usage is not None else _mock_usage()
    return response


def _mock_response_with_only_thinking(stop_reason: str = "max_tokens") -> MagicMock:
    response = MagicMock()
    response.content = [_thinking_block()]
    response.stop_reason = stop_reason
    response.usage = _mock_usage()
    return response


class _LLMServiceTestCase(unittest.TestCase):
    """Resets llm_service's cached Anthropic client before each test, since the
    client is now a module-level singleton (see llm_service._get_client) —
    without this, a mock installed by an earlier test's @patch would leak into
    later tests instead of each test getting its own fresh mock client."""

    def setUp(self):
        super().setUp()
        llm_service._client = None


class TestCategorizeDocument(_LLMServiceTestCase):
    @patch("llm_service.anthropic.Anthropic")
    def test_valid_tool_input_is_parsed_correctly(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_tool_response(
            {
                "is_relevant": True,
                "category": "Termination_Documents",
                "reason": "Contains the dismissal letter.",
            }
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
    def test_forces_tool_choice_and_defines_classify_document_tool(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_tool_response(
            {"is_relevant": False, "category": "Other_Relevant_Evidence", "reason": "Not related."}
        )

        categorize_document("Unrelated grocery receipt.")

        _, call_kwargs = mock_client.messages.create.call_args
        self.assertEqual(call_kwargs["tool_choice"], {"type": "tool", "name": "classify_document"})

        (tool,) = call_kwargs["tools"]
        self.assertEqual(tool["name"], "classify_document")
        schema = tool["input_schema"]
        self.assertEqual(schema["required"], ["is_relevant", "category", "reason"])
        self.assertEqual(schema["properties"]["category"]["enum"], llm_service.CATEGORIES)

        # The old "respond with ONLY a valid JSON object" instruction is gone
        # now that the schema is enforced by the tool, not the prompt.
        prompt_text = call_kwargs["messages"][0]["content"]
        self.assertNotIn("ONLY a valid JSON", prompt_text)

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
    def test_missing_keys_in_tool_input_fall_back_to_defaults(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_tool_response({})

        result = categorize_document("Some text.")

        self.assertFalse(result["is_relevant"])
        self.assertIsInstance(result["category"], str)
        self.assertIsInstance(result["reason"], str)

    @patch("llm_service.anthropic.Anthropic")
    def test_skips_leading_thinking_block_and_uses_tool_use_block(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_tool_response(
            {
                "is_relevant": True,
                "category": "Correspondence",
                "reason": "Dismissal-related email thread.",
            },
            leading_thinking=True,
        )

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

    @patch("llm_service.logger")
    @patch("llm_service.anthropic.Anthropic")
    def test_response_with_only_a_thinking_block_returns_safe_fallback(
        self, mock_anthropic_cls, mock_logger
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response_with_only_thinking()

        result = categorize_document("Some text.")

        # No tool_use block -> _extract_tool_input returns {}, so this doesn't
        # go through the exception handler at all; it's a normal, if empty,
        # categorization result.
        self.assertFalse(result["is_relevant"])
        self.assertIn("category", result)
        self.assertIn("reason", result)
        self.assertIn("description", result)

        mock_logger.error.assert_called_once()
        (log_message, logged_stop_reason, _raw_content), _ = mock_logger.error.call_args
        self.assertIn("No tool_use block found", log_message)
        self.assertEqual(logged_stop_reason, "max_tokens")

    @patch("llm_service.anthropic.Anthropic")
    def test_image_input_builds_vision_payload_and_returns_description(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_tool_response(
            {
                "is_relevant": True,
                "category": "Performance_Reviews",
                "reason": "Photo of a signed performance review.",
                "description": "A scanned performance review form with a signature at the bottom.",
            }
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

        self.assertEqual(call_kwargs["tool_choice"], {"type": "tool", "name": "classify_document"})

    @patch("llm_service.anthropic.Anthropic")
    def test_missing_text_and_image_returns_safe_fallback_without_calling_api(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        result = categorize_document()

        mock_client.messages.create.assert_not_called()
        self.assertEqual(result, dict(llm_service._CATEGORIZE_FALLBACK))


class TestSemanticScrub(_LLMServiceTestCase):
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

    @patch("llm_service.logger")
    @patch("llm_service.anthropic.Anthropic")
    def test_response_with_only_a_thinking_block_returns_no_text_error_string(
        self, mock_anthropic_cls, mock_logger
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response_with_only_thinking()

        original = "Already deterministically [REDACTED] text."
        result = semantic_scrub(original)

        # A response that came back successfully but produced no text block
        # is a distinct failure mode from an API exception: it surfaces the
        # explicit error sentinel rather than silently passing input through.
        self.assertEqual(result, "ERROR: No text generated by LLM.")

        mock_logger.error.assert_called_once()
        (log_message, logged_stop_reason, _fallback, _raw_content), _ = mock_logger.error.call_args
        self.assertIn("No text block found", log_message)
        self.assertEqual(logged_stop_reason, "max_tokens")


class TestGenerateSummary(_LLMServiceTestCase):
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

    @patch("llm_service.logger")
    @patch("llm_service.anthropic.Anthropic")
    def test_response_with_only_a_thinking_block_returns_no_text_error_string(
        self, mock_anthropic_cls, mock_logger
    ):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response_with_only_thinking()

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(result, "ERROR: No text generated by LLM.")

        mock_logger.error.assert_called_once()
        (log_message, logged_stop_reason, _fallback, _raw_content), _ = mock_logger.error.call_args
        self.assertIn("No text block found", log_message)
        self.assertEqual(logged_stop_reason, "max_tokens")


class TestClientInitialization(_LLMServiceTestCase):
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-123"}, clear=False)
    @patch("llm_service.anthropic.Anthropic")
    def test_client_initialized_with_env_api_key(self, mock_anthropic_cls):
        llm_service._get_client()
        mock_anthropic_cls.assert_called_once_with(api_key="test-key-123")


class TestTokenTracker(_LLMServiceTestCase):
    def setUp(self):
        super().setUp()
        llm_service._token_usage["input_tokens"] = 0
        llm_service._token_usage["output_tokens"] = 0

    @patch("llm_service.anthropic.Anthropic")
    def test_categorize_document_accumulates_token_usage(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_tool_response(
            {"is_relevant": True, "category": "Termination_Documents", "reason": "x"},
            usage=_mock_usage(input_tokens=120, output_tokens=40),
        )

        categorize_document(text="Some text.")

        report = get_usage_report()
        self.assertEqual(report["input_tokens"], 120)
        self.assertEqual(report["output_tokens"], 40)

    @patch("llm_service.anthropic.Anthropic")
    def test_usage_accumulates_across_multiple_calls(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_response(
            "some text", usage=_mock_usage(input_tokens=100, output_tokens=50)
        )

        semantic_scrub("text one")
        semantic_scrub("text two")

        report = get_usage_report()
        self.assertEqual(report["input_tokens"], 200)
        self.assertEqual(report["output_tokens"], 100)

    @patch("llm_service.anthropic.Anthropic")
    def test_api_exception_does_not_track_usage(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("boom")

        generate_summary("text")

        report = get_usage_report()
        self.assertEqual(report["input_tokens"], 0)
        self.assertEqual(report["output_tokens"], 0)

    def test_get_usage_report_calculates_cost_using_sonnet_pricing(self):
        llm_service._token_usage["input_tokens"] = 1_000_000
        llm_service._token_usage["output_tokens"] = 1_000_000

        report = get_usage_report()

        self.assertEqual(report["input_tokens"], 1_000_000)
        self.assertEqual(report["output_tokens"], 1_000_000)
        self.assertAlmostEqual(report["total_cost_usd"], 3.00 + 15.00)

    def test_get_usage_report_with_no_usage_yet_is_zero_cost(self):
        report = get_usage_report()

        self.assertEqual(report["input_tokens"], 0)
        self.assertEqual(report["output_tokens"], 0)
        self.assertEqual(report["total_cost_usd"], 0.0)


class TestRetryBehavior(_LLMServiceTestCase):
    """
    Covers the tenacity-based retry policy shared by every Anthropic call in
    this module (via llm_service._create_message): network errors, 429, and
    5xx should retry with backoff; 4xx errors other than 429 should not.

    time.sleep is patched everywhere so these tests run instantly regardless
    of the real backoff durations.
    """

    def _mock_client(self, mock_anthropic_cls, side_effect):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = side_effect
        return mock_client

    @patch("time.sleep")
    @patch("llm_service.anthropic.Anthropic")
    def test_categorize_document_retries_on_429_then_succeeds(self, mock_anthropic_cls, mock_sleep):
        success = _mock_tool_response(
            {"is_relevant": True, "category": "Correspondence", "reason": "ok"}
        )
        mock_client = self._mock_client(
            mock_anthropic_cls, [_api_status_error(anthropic.RateLimitError), success]
        )

        result = categorize_document(text="Some text.")

        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertTrue(result["is_relevant"])
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch("llm_service.anthropic.Anthropic")
    def test_semantic_scrub_retries_on_5xx_then_succeeds(self, mock_anthropic_cls, mock_sleep):
        success = _mock_response("[REDACTED] text.")
        mock_client = self._mock_client(
            mock_anthropic_cls,
            [_api_status_error(anthropic.InternalServerError, status_code=500), success],
        )

        result = semantic_scrub("Some text.")

        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertEqual(result, "[REDACTED] text.")
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch("llm_service.anthropic.Anthropic")
    def test_generate_summary_retries_on_connection_error_then_succeeds(self, mock_anthropic_cls, mock_sleep):
        success = _mock_response("# Case Summary")
        mock_client = self._mock_client(mock_anthropic_cls, [_connection_error(), success])

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(mock_client.messages.create.call_count, 2)
        self.assertEqual(result, "# Case Summary")
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch("llm_service.anthropic.Anthropic")
    def test_400_bad_request_is_not_retried_and_falls_back_immediately(self, mock_anthropic_cls, mock_sleep):
        mock_client = self._mock_client(
            mock_anthropic_cls, _api_status_error(anthropic.BadRequestError)
        )

        result = categorize_document(text="Some text.")

        # A single failed attempt, no retry/backoff, and the caller's normal
        # exception-handling fallback still applies.
        self.assertEqual(mock_client.messages.create.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(result, dict(llm_service._CATEGORIZE_FALLBACK))

    @patch("time.sleep")
    @patch("llm_service.anthropic.Anthropic")
    def test_retries_exhausted_falls_back_to_safe_default(self, mock_anthropic_cls, mock_sleep):
        always_rate_limited = _api_status_error(anthropic.RateLimitError)
        mock_client = self._mock_client(mock_anthropic_cls, always_rate_limited)

        result = generate_summary("Fully redacted case text.")

        self.assertEqual(mock_client.messages.create.call_count, llm_service._MAX_RETRY_ATTEMPTS)
        self.assertEqual(result, "")

    @patch("time.sleep")
    @patch("llm_service.anthropic.Anthropic")
    def test_401_authentication_error_is_not_retried(self, mock_anthropic_cls, mock_sleep):
        mock_client = self._mock_client(
            mock_anthropic_cls, _api_status_error(anthropic.AuthenticationError)
        )

        result = semantic_scrub("Already deterministically [REDACTED] text.")

        self.assertEqual(mock_client.messages.create.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(result, "Already deterministically [REDACTED] text.")


if __name__ == "__main__":
    unittest.main()

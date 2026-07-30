"""
Unit tests for main.py's CLI workflow.

sys.argv, rich.prompt.Confirm.ask, rich Console output, and the
pipeline/sanitizer/llm_service functions are all mocked — no real file I/O,
terminal rendering, or network calls happen here.

The exception is TestRichMarkupSafety, which deliberately uses a real
rich.console.Console (writing to an in-memory buffer instead of the mocked
`main.console`) — the bug it guards against (untrusted text being parsed as
Rich markup) only reproduces against real Rich rendering, not a MagicMock.
"""

import io
import unittest
from unittest.mock import MagicMock, mock_open, patch

from rich.console import Console

import main


def _base_result(**overrides):
    result = {
        "success": True,
        "relevant_texts": ["Some relevant text."],
        "category_counts": {"Termination_Documents": 1},
        "excluded_count": 2,
        "total_files": 3,
        "audit_log": [
            {
                "file_name": "letter.pdf",
                "category": "Termination_Documents",
                "is_relevant": True,
                "reason": "Contains the dismissal letter.",
            }
        ],
        "cache_path": "./output/stage1_cache.json",
        "audit_report_path": "./output/audit_report.csv",
    }
    result.update(overrides)
    return result


def _mock_progress_context_manager():
    """A stand-in for rich.progress.Progress that just no-ops as a context manager."""
    progress_instance = MagicMock()
    progress_instance.__enter__.return_value = progress_instance
    progress_instance.add_task.return_value = "task-id"
    return progress_instance


class TestStage1Failure(unittest.TestCase):
    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask")
    @patch("main.process_evidence_package")
    def test_failed_stage1_does_not_prompt_or_run_stage2(
        self, mock_process, mock_confirm_ask, mock_console, mock_progress_cls
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result(
            success=False,
            relevant_texts=[],
            category_counts={},
            excluded_count=0,
            total_files=0,
            audit_log=[],
            cache_path=None,
            audit_report_path=None,
        )

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_confirm_ask.assert_not_called()


class TestApiKeyCheck(unittest.TestCase):
    @patch("main.console", new_callable=MagicMock)
    @patch("main.process_evidence_package")
    def test_missing_api_key_aborts_before_any_processing(self, mock_process, mock_console):
        with patch.dict("main.os.environ", {}, clear=True):
            with patch("sys.argv", ["main.py"]):
                main.main()

        mock_process.assert_not_called()

    @patch("main.console", new_callable=MagicMock)
    @patch("main.process_evidence_package")
    def test_whitespace_only_api_key_is_treated_as_missing(self, mock_process, mock_console):
        with patch.dict("main.os.environ", {"ANTHROPIC_API_KEY": "   "}, clear=True):
            with patch("sys.argv", ["main.py"]):
                main.main()

        mock_process.assert_not_called()

    @patch("main.console", new_callable=MagicMock)
    @patch("main.Progress")
    @patch("main.Confirm.ask")
    @patch("main.process_evidence_package")
    def test_present_api_key_proceeds_to_stage1(
        self, mock_process, mock_confirm_ask, mock_progress_cls, mock_console
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result(
            success=False,
            relevant_texts=[],
            category_counts={},
            excluded_count=0,
            total_files=0,
            audit_log=[],
            cache_path=None,
            audit_report_path=None,
        )

        with patch.dict("main.os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("sys.argv", ["main.py"]):
                main.main()

        mock_process.assert_called_once()


class TestPrintAuditSummary(unittest.TestCase):
    @patch("main.console", new_callable=MagicMock)
    def test_copy_failures_row_is_omitted_when_none_failed(self, mock_console):
        result = _base_result()

        main._print_audit_summary(result)

        table = mock_console.print.call_args[0][0]
        row_labels = [str(cell) for cell in table.columns[0]._cells]
        self.assertFalse(any("Copy Failures" in label for label in row_labels))

    @patch("main.console", new_callable=MagicMock)
    def test_copy_failures_row_is_shown_and_counted_when_present(self, mock_console):
        result = _base_result(
            audit_log=[
                {
                    "file_name": "letter.pdf",
                    "category": "Termination_Documents",
                    "is_relevant": True,
                    "reason": "Contains the dismissal letter.",
                    "copy_failed": True,
                },
                {
                    "file_name": "email.eml",
                    "category": "Correspondence",
                    "is_relevant": True,
                    "reason": "Relevant email.",
                    "copy_failed": False,
                },
            ]
        )

        main._print_audit_summary(result)

        table = mock_console.print.call_args[0][0]
        row_labels = [str(cell) for cell in table.columns[0]._cells]
        count_cells = [str(cell) for cell in table.columns[1]._cells]
        matches = [i for i, label in enumerate(row_labels) if "Copy Failures" in label]
        self.assertEqual(len(matches), 1)
        self.assertIn("1", count_cells[matches[0]])


class TestRichMarkupSafety(unittest.TestCase):
    """
    Guards against untrusted content (a filename from the input zip, or LLM
    text that echoes bracketed placeholders from a document) being parsed as
    Rich markup instead of rendered literally. A stray "[/]" or a valid-looking
    tag like "[bold red]" in that text previously raised rich.errors.MarkupError
    or produced spoofed styling once it reached a real Console — a mocked
    console never exercises Rich's markup parser, so these tests use a real
    one writing to an in-memory buffer.
    """

    def _capturing_console(self):
        return Console(file=io.StringIO(), force_terminal=False, width=200)

    def test_decision_table_handles_bracket_content_without_crashing(self):
        result = _base_result(
            audit_log=[
                {
                    "file_name": "[Draft] termination[/].pdf",
                    "category": "Termination_Documents",
                    "is_relevant": True,
                    "reason": "Contains [PRIVILEGED] info and a stray [/] close tag.",
                    "copy_failed": False,
                },
                {
                    "file_name": "evil[/][bold green]FAKE-APPROVED[/bold green].docx",
                    "category": "",
                    "is_relevant": False,
                    "reason": "Not relevant.",
                    "copy_failed": True,
                },
            ]
        )
        capturing_console = self._capturing_console()

        with patch("main.console", capturing_console):
            main._print_decision_table(result)

        output = capturing_console.file.getvalue()
        self.assertIn("[Draft] termination[/].pdf", output)
        self.assertIn("Contains [PRIVILEGED] info and a stray [/] close tag.", output)
        self.assertIn("evil[/][bold green]FAKE-APPROVED[/bold green].docx", output)

    def test_run_stage_1_progress_callback_handles_bracket_content_without_crashing(self):
        capturing_console = self._capturing_console()

        def fake_process(input_path, base_output_dir, progress_callback):
            progress_callback("Extracting content from evil[/].pdf...", done=False)
            progress_callback(
                "Calling LLM for categorization of evil[bold red]x[/bold red].pdf...", done=True
            )
            return _base_result()

        with patch("main.console", capturing_console), patch(
            "main.process_evidence_package", side_effect=fake_process
        ):
            main._run_stage_1("case.zip", "./output")

        output = capturing_console.file.getvalue()
        self.assertIn("evil", output)


class TestYesFlow(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("main.os.makedirs")
    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.generate_summary")
    @patch("main.semantic_scrub")
    @patch("main.deterministic_scrub")
    @patch("main.Confirm.ask", return_value=True)
    @patch("main.process_evidence_package")
    def test_yes_confirmation_runs_full_stage2_pipeline_and_writes_file(
        self,
        mock_process,
        mock_confirm_ask,
        mock_det_scrub,
        mock_sem_scrub,
        mock_gen_summary,
        mock_console,
        mock_progress_cls,
        mock_makedirs,
        mock_open_file,
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result(
            relevant_texts=["Text about the dismissal.", "Text about payroll."]
        )
        mock_det_scrub.side_effect = lambda text: f"det[{text}]"
        mock_sem_scrub.side_effect = lambda text: f"sem[{text}]"
        mock_gen_summary.return_value = "Final anonymous summary."

        with patch("sys.argv", ["main.py", "--output", "./output"]):
            main.main()

        # Map: each relevant text is scrubbed individually, in order, rather
        # than joined into one giant string before scrubbing.
        self.assertEqual(mock_det_scrub.call_count, 2)
        mock_det_scrub.assert_any_call("Text about the dismissal.")
        mock_det_scrub.assert_any_call("Text about payroll.")

        self.assertEqual(mock_sem_scrub.call_count, 2)
        mock_sem_scrub.assert_any_call("det[Text about the dismissal.]")
        mock_sem_scrub.assert_any_call("det[Text about payroll.]")

        # Reduce: only the fully-scrubbed per-file texts are joined, and only
        # for the final summary generation call.
        mock_gen_summary.assert_called_once_with(
            "sem[det[Text about the dismissal.]]\n\nsem[det[Text about payroll.]]"
        )

        mock_open_file.assert_called_once_with(
            "./output/anonymous_summary.md", "w", encoding="utf-8"
        )
        mock_open_file().write.assert_called_once_with("Final anonymous summary.")

    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.generate_summary")
    @patch("main.semantic_scrub")
    @patch("main.deterministic_scrub")
    @patch("main.Confirm.ask", return_value=True)
    @patch("main.process_evidence_package")
    def test_no_relevant_text_skips_summary_generation(
        self,
        mock_process,
        mock_confirm_ask,
        mock_det_scrub,
        mock_sem_scrub,
        mock_gen_summary,
        mock_console,
        mock_progress_cls,
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result(relevant_texts=[])

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_det_scrub.assert_not_called()
        mock_sem_scrub.assert_not_called()
        mock_gen_summary.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.generate_summary", return_value="")
    @patch("main.semantic_scrub", return_value="semantically scrubbed text")
    @patch("main.deterministic_scrub", return_value="deterministically scrubbed text")
    @patch("main.Confirm.ask", return_value=True)
    @patch("main.process_evidence_package")
    def test_empty_summary_from_llm_does_not_write_output_file(
        self,
        mock_process,
        mock_confirm_ask,
        mock_det_scrub,
        mock_sem_scrub,
        mock_gen_summary,
        mock_console,
        mock_progress_cls,
        mock_open_file,
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_open_file.assert_not_called()


class TestNoFlow(unittest.TestCase):
    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.generate_summary")
    @patch("main.semantic_scrub")
    @patch("main.deterministic_scrub")
    @patch("main.Confirm.ask", return_value=False)
    @patch("main.process_evidence_package")
    def test_no_confirmation_aborts_without_running_stage2(
        self,
        mock_process,
        mock_confirm_ask,
        mock_det_scrub,
        mock_sem_scrub,
        mock_gen_summary,
        mock_console,
        mock_progress_cls,
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_det_scrub.assert_not_called()
        mock_sem_scrub.assert_not_called()
        mock_gen_summary.assert_not_called()


class TestArgumentParsing(unittest.TestCase):
    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask", return_value=False)
    @patch("main.process_evidence_package")
    def test_custom_input_and_output_flags_are_passed_through(
        self, mock_process, mock_confirm_ask, mock_console, mock_progress_cls
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        with patch("sys.argv", ["main.py", "--input", "custom.zip", "--output", "custom_out"]):
            main.main()

        args, kwargs = mock_process.call_args
        self.assertEqual(args, ("custom.zip",))
        self.assertEqual(kwargs["base_output_dir"], "custom_out")
        self.assertTrue(callable(kwargs["progress_callback"]))

    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask", return_value=False)
    @patch("main.process_evidence_package")
    def test_default_input_flag_is_rawdata_raw_zip(
        self, mock_process, mock_confirm_ask, mock_console, mock_progress_cls
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        with patch("sys.argv", ["main.py"]):
            main.main()

        args, kwargs = mock_process.call_args
        self.assertEqual(args, ("rawdata/raw.zip",))
        self.assertEqual(kwargs["base_output_dir"], "./output")

    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask", return_value=False)
    @patch("main.process_evidence_package")
    def test_skip_stage1_flag_defaults_to_false(
        self, mock_process, mock_confirm_ask, mock_console, mock_progress_cls
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_process.assert_called_once()


class TestSkipStage1Flow(unittest.TestCase):
    @patch("main.Progress")
    @patch("main.process_evidence_package")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.generate_summary")
    @patch("main.semantic_scrub")
    @patch("main.deterministic_scrub")
    @patch("main.Confirm.ask", return_value=True)
    @patch("main.load_stage1_cache")
    def test_skip_stage1_loads_cache_and_skips_extraction(
        self,
        mock_load_cache,
        mock_confirm_ask,
        mock_det_scrub,
        mock_sem_scrub,
        mock_gen_summary,
        mock_console,
        mock_process,
        mock_progress_cls,
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_load_cache.return_value = ["Cached text A.", "Cached text B."]
        mock_det_scrub.return_value = "scrubbed"
        mock_sem_scrub.return_value = "semantic"
        mock_gen_summary.return_value = "Final summary."

        with patch("sys.argv", ["main.py", "--skip-stage1", "--output", "./output"]):
            main.main()

        mock_load_cache.assert_called_once_with("./output")
        mock_process.assert_not_called()
        mock_confirm_ask.assert_called_once()
        self.assertEqual(mock_det_scrub.call_count, 2)
        mock_det_scrub.assert_any_call("Cached text A.")
        mock_det_scrub.assert_any_call("Cached text B.")

    @patch("main.process_evidence_package")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask")
    @patch("main.load_stage1_cache", return_value=None)
    def test_skip_stage1_with_no_cache_aborts_without_prompting(
        self, mock_load_cache, mock_confirm_ask, mock_console, mock_process
    ):
        with patch("sys.argv", ["main.py", "--skip-stage1"]):
            main.main()

        mock_confirm_ask.assert_not_called()
        mock_process.assert_not_called()

    @patch("main.Progress")
    @patch("main.process_evidence_package")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask", return_value=False)
    @patch("main.load_stage1_cache")
    def test_normal_flow_never_calls_load_stage1_cache(
        self, mock_load_cache, mock_confirm_ask, mock_console, mock_process, mock_progress_cls
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        with patch("sys.argv", ["main.py"]):
            main.main()

        mock_load_cache.assert_not_called()


class TestAuditReportNotice(unittest.TestCase):
    @patch("main.console", new_callable=MagicMock)
    def test_prints_notice_when_audit_report_path_is_present(self, mock_console):
        main._print_audit_report_notice(_base_result(audit_report_path="./output/audit_report.csv"))

        mock_console.print.assert_called_once()
        (printed,), _ = mock_console.print.call_args
        rendered = printed.renderable if hasattr(printed, "renderable") else str(printed)
        self.assertIn("./output/audit_report.csv", str(rendered))

    @patch("main.console", new_callable=MagicMock)
    def test_prints_nothing_when_audit_report_path_is_missing(self, mock_console):
        main._print_audit_report_notice(_base_result(audit_report_path=None))

        mock_console.print.assert_not_called()

    @patch("main.Progress")
    @patch("main.console", new_callable=MagicMock)
    @patch("main.Confirm.ask", return_value=False)
    @patch("main.process_evidence_package")
    def test_notice_is_shown_before_the_confirmation_prompt(
        self, mock_process, mock_confirm_ask, mock_console, mock_progress_cls
    ):
        mock_progress_cls.return_value = _mock_progress_context_manager()
        mock_process.return_value = _base_result()

        snapshot = []

        def _record_confirm(*args, **kwargs):
            snapshot.extend(mock_console.print.call_args_list)
            return False

        mock_confirm_ask.side_effect = _record_confirm

        with patch("sys.argv", ["main.py"]):
            main.main()

        # By the time Confirm.ask fires, the audit report notice must already
        # have been printed — not deferred until after the prompt.
        def _rendered_text(call_args):
            (printed,), _ = call_args
            return str(getattr(printed, "renderable", printed))

        printed_texts = [_rendered_text(call_args) for call_args in snapshot]
        self.assertTrue(any("audit_report.csv" in text for text in printed_texts))


if __name__ == "__main__":
    unittest.main()

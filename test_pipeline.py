"""
Unit tests for pipeline.py.

All file-system operations (zipfile, shutil, os, json, csv) and the
extractors/llm_service dependencies are mocked — these tests never touch the
real disk or network.
"""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pipeline


class TestZipValidation(unittest.TestCase):
    @patch("pipeline.os.path.isfile", return_value=False)
    @patch("pipeline.zipfile.ZipFile")
    def test_missing_zip_returns_safe_fallback_without_extracting(self, mock_zipfile, mock_isfile):
        result = pipeline.process_evidence_package("does_not_exist.zip")

        mock_zipfile.assert_not_called()
        self.assertEqual(
            result,
            {
                "success": False,
                "relevant_texts": [],
                "category_counts": {},
                "excluded_count": 0,
                "total_files": 0,
                "audit_log": [],
                "cache_path": None,
                "audit_report_path": None,
            },
        )

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.os.makedirs")
    @patch("pipeline.zipfile.ZipFile", side_effect=OSError("corrupt archive"))
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_corrupt_zip_returns_safe_fallback_and_cleans_up(
        self, mock_isfile, mock_zipfile, mock_makedirs, mock_rmtree
    ):
        result = pipeline.process_evidence_package("corrupt.zip", tmp_workspace_dir="./tmp_workspace/")

        self.assertFalse(result["success"])
        mock_rmtree.assert_called_once_with("./tmp_workspace/", ignore_errors=True)


class TestRoutingLogic(unittest.TestCase):
    """
    Every test here reaches a successful Stage 1 run, which now also writes
    stage1_cache.json and audit_report.csv. builtins.open, json.dump, and
    csv.DictWriter are mocked globally in setUp so no test needs to repeat
    that boilerplate, and no real file ever gets touched.
    """

    def setUp(self):
        self._open_patch = patch("builtins.open", new_callable=mock_open)
        self.mock_open = self._open_patch.start()
        self.addCleanup(self._open_patch.stop)

        self._json_dump_patch = patch("pipeline.json.dump")
        self.mock_json_dump = self._json_dump_patch.start()
        self.addCleanup(self._json_dump_patch.stop)

        self.mock_csv_writer = MagicMock()
        self._csv_writer_cls_patch = patch("pipeline.csv.DictWriter", return_value=self.mock_csv_writer)
        self.mock_csv_writer_cls = self._csv_writer_cls_patch.start()
        self.addCleanup(self._csv_writer_cls_patch.stop)

    def _zipfile_context_manager(self):
        mock_zf = MagicMock()
        mock_zf.__enter__.return_value = mock_zf
        mock_zf.extractall.return_value = None
        return mock_zf

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Some relevant PDF text.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["evidence.pdf"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_relevant_document_is_copied_to_its_category_folder(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.return_value = {
            "is_relevant": True,
            "category": "Termination_Documents",
            "reason": "Contains the dismissal letter.",
        }

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["category_counts"], {"Termination_Documents": 1})
        self.assertEqual(result["excluded_count"], 0)
        self.assertEqual(result["relevant_texts"], ["Some relevant PDF text."])

        mock_copy2.assert_called_once()
        src_arg, dest_arg = mock_copy2.call_args[0]
        self.assertIn("evidence.pdf", src_arg)
        self.assertIn("Evidence_Package", dest_arg)
        self.assertIn("Termination_Documents", dest_arg)
        self.assertIn("evidence.pdf", dest_arg)

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_text_from_docx", return_value="An irrelevant grocery list.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["groceries.docx"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_irrelevant_document_is_copied_to_excluded_documents(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_docx,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.return_value = {
            "is_relevant": False,
            "category": None,
            "reason": "Not related to the employment matter.",
        }

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["category_counts"], {})
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["relevant_texts"], [])

        mock_copy2.assert_called_once()
        src_arg, dest_arg = mock_copy2.call_args[0]
        self.assertIn("groceries.docx", src_arg)
        self.assertIn("Excluded_Documents", dest_arg)

        # Reason should be logged to the exclusion log (a real f.write() call,
        # unlike the mocked-out json.dump/csv.DictWriter cache+audit writes).
        self.mock_open().write.assert_called_once()
        written = self.mock_open().write.call_args[0][0]
        self.assertIn("groceries.docx", written)
        self.assertIn("Not related to the employment matter.", written)

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch(
        "pipeline.extractors.extract_text_from_eml",
        return_value=("Email body about the dismissal.", ["./tmp_workspace/tmp_attachments/contract.pdf"]),
    )
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Attachment contract text.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["message.eml"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_eml_attachments_are_discovered_and_processed(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_extract_eml,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.side_effect = [
            {"is_relevant": True, "category": "Correspondence", "reason": "Dismissal email."},
            {"is_relevant": True, "category": "Employment_Contracts", "reason": "Signed contract."},
        ]

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 2)
        self.assertEqual(
            result["category_counts"], {"Correspondence": 1, "Employment_Contracts": 1}
        )
        self.assertEqual(
            sorted(result["relevant_texts"]),
            sorted(["Email body about the dismissal.", "Attachment contract text."]),
        )
        self.assertEqual(mock_copy2.call_count, 2)

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["notes.gif"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_unsupported_extension_is_excluded_without_calling_llm(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        mock_categorize.assert_not_called()
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["category_counts"], {})

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_image", return_value=("ZmFrZWJhc2U2NA==", "image/png"))
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["photo.png"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_relevant_image_is_categorized_via_vision_and_uses_description_for_stage2(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_image,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.return_value = {
            "is_relevant": True,
            "category": "Performance_Reviews",
            "reason": "Photo of a signed performance review.",
            "description": "A scanned performance review form with a signature.",
        }

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        mock_extract_image.assert_called_once()
        _, kwargs = mock_categorize.call_args
        self.assertEqual(kwargs, {"image_base64": "ZmFrZWJhc2U2NA==", "media_type": "image/png"})

        self.assertEqual(result["category_counts"], {"Performance_Reviews": 1})
        self.assertEqual(
            result["relevant_texts"], ["A scanned performance review form with a signature."]
        )

        src_arg, dest_arg = mock_copy2.call_args[0]
        self.assertIn("photo.png", src_arg)
        self.assertIn("Performance_Reviews", dest_arg)

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_image", return_value=("ZmFrZWJhc2U2NA==", "image/jpeg"))
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["selfie.jpg"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_irrelevant_image_is_excluded_and_not_added_to_relevant_texts(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_image,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.return_value = {
            "is_relevant": False,
            "category": None,
            "reason": "Unrelated personal photo.",
            "description": "A photo of a beach.",
        }

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["relevant_texts"], [])

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_image", return_value=(None, None))
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["corrupt.jpeg"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_image_extraction_failure_is_excluded_without_calling_llm(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_image,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        mock_categorize.assert_not_called()
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["category_counts"], {})

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_image", return_value=("", "image/png"))
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["empty.png"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_empty_image_file_is_excluded_without_calling_llm(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_image,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        mock_categorize.assert_not_called()
        self.assertEqual(result["excluded_count"], 1)
        self.assertEqual(result["category_counts"], {})
        self.assertEqual(
            result["audit_log"][0]["reason"], "Unsupported file type or no extractable text."
        )

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2", side_effect=OSError("disk full"))
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Some relevant PDF text.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["evidence.pdf"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_copy_failure_is_handled_without_crashing_or_counting(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.return_value = {
            "is_relevant": True,
            "category": "Termination_Documents",
            "reason": "Contains the dismissal letter.",
        }

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["total_files"], 1)
        self.assertEqual(result["category_counts"], {})
        self.assertEqual(result["relevant_texts"], [])
        self.assertEqual(len(result["audit_log"]), 1)
        self.assertTrue(result["audit_log"][0]["copy_failed"])

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document", return_value={"is_relevant": False, "category": None, "reason": "n/a"})
    @patch("pipeline.os.walk", return_value=[])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_tmp_workspace_is_always_cleaned_up_on_success(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        pipeline.process_evidence_package(
            "case.zip", tmp_workspace_dir="./tmp_workspace/"
        )

        mock_rmtree.assert_called_once_with("./tmp_workspace/", ignore_errors=True)

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch("pipeline.llm_service.categorize_document")
    @patch("pipeline.extractors.extract_text_from_docx", return_value="An irrelevant grocery list.")
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Some relevant PDF text.")
    @patch(
        "pipeline.os.walk",
        return_value=[("./tmp_workspace/", [], ["evidence.pdf", "groceries.docx"])],
    )
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_audit_log_contains_one_entry_per_file_with_expected_keys(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_extract_docx,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        mock_categorize.side_effect = [
            {"is_relevant": True, "category": "Termination_Documents", "reason": "Dismissal letter."},
            {"is_relevant": False, "category": None, "reason": "Unrelated grocery list."},
        ]

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertEqual(len(result["audit_log"]), 2)
        by_name = {entry["file_name"]: entry for entry in result["audit_log"]}

        self.assertEqual(
            by_name["evidence.pdf"],
            {
                "file_name": "evidence.pdf",
                "category": "Termination_Documents",
                "is_relevant": True,
                "reason": "Dismissal letter.",
                "copy_failed": False,
            },
        )
        self.assertEqual(
            by_name["groceries.docx"],
            {
                "file_name": "groceries.docx",
                "category": "",
                "is_relevant": False,
                "reason": "Unrelated grocery list.",
                "copy_failed": False,
            },
        )

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch(
        "pipeline.llm_service.categorize_document",
        return_value={"is_relevant": True, "category": "Termination_Documents", "reason": "Dismissal letter."},
    )
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Some relevant PDF text.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["evidence.pdf"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_stage1_cache_is_written_with_relevant_texts_only(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertEqual(result["cache_path"], "./output/stage1_cache.json")
        self.mock_open.assert_any_call("./output/stage1_cache.json", "w", encoding="utf-8")
        self.mock_json_dump.assert_called_once()
        dumped_data = self.mock_json_dump.call_args[0][0]
        self.assertEqual(dumped_data, {"relevant_texts": ["Some relevant PDF text."]})

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch(
        "pipeline.llm_service.categorize_document",
        return_value={"is_relevant": False, "category": None, "reason": "Not related."},
    )
    @patch("pipeline.extractors.extract_text_from_docx", return_value="An irrelevant grocery list.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["groceries.docx"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_audit_report_csv_is_written_with_expected_fieldnames_and_rows(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_docx,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        result = pipeline.process_evidence_package(
            "case.zip", base_output_dir="./output", tmp_workspace_dir="./tmp_workspace/"
        )

        self.assertEqual(result["audit_report_path"], "./output/audit_report.csv")
        self.mock_csv_writer_cls.assert_called_once_with(
            self.mock_open(), fieldnames=pipeline.AUDIT_REPORT_FIELDNAMES
        )
        self.mock_csv_writer.writeheader.assert_called_once()
        self.mock_csv_writer.writerows.assert_called_once_with(
            [
                {
                    "file_name": "groceries.docx",
                    "category": "",
                    "is_relevant": False,
                    "reason": "Not related.",
                    "copy_failed": False,
                }
            ]
        )

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch(
        "pipeline.llm_service.categorize_document",
        return_value={"is_relevant": True, "category": "Termination_Documents", "reason": "Dismissal letter."},
    )
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Some relevant PDF text.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["evidence.pdf"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_progress_callback_is_invoked_with_status_strings(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()
        progress_callback = MagicMock()

        pipeline.process_evidence_package(
            "case.zip",
            base_output_dir="./output",
            tmp_workspace_dir="./tmp_workspace/",
            progress_callback=progress_callback,
        )

        self.assertGreater(progress_callback.call_count, 0)
        done_flags = []
        for call_args in progress_callback.call_args_list:
            self.assertIsInstance(call_args[0][0], str)
            done_flags.append(call_args[0][1])

        # Every step should be reported once as starting (done=False) and
        # once as finished (done=True) — never only one or the other.
        self.assertIn(False, done_flags)
        self.assertIn(True, done_flags)
        self.assertEqual(done_flags.count(False), done_flags.count(True))

    @patch("pipeline.shutil.rmtree")
    @patch("pipeline.shutil.copy2")
    @patch("pipeline.os.path.exists", return_value=False)
    @patch("pipeline.os.makedirs")
    @patch(
        "pipeline.llm_service.categorize_document",
        return_value={"is_relevant": True, "category": "Termination_Documents", "reason": "Dismissal letter."},
    )
    @patch("pipeline.extractors.extract_text_from_pdf", return_value="Some relevant PDF text.")
    @patch("pipeline.os.walk", return_value=[("./tmp_workspace/", [], ["evidence.pdf"])])
    @patch("pipeline.zipfile.ZipFile")
    @patch("pipeline.os.path.isfile", return_value=True)
    def test_faulty_progress_callback_does_not_crash_pipeline(
        self,
        mock_isfile,
        mock_zipfile_cls,
        mock_walk,
        mock_extract_pdf,
        mock_categorize,
        mock_makedirs,
        mock_exists,
        mock_copy2,
        mock_rmtree,
    ):
        mock_zipfile_cls.return_value = self._zipfile_context_manager()

        def _raising_callback(_message, _done=False):
            raise RuntimeError("UI exploded")

        result = pipeline.process_evidence_package(
            "case.zip",
            base_output_dir="./output",
            tmp_workspace_dir="./tmp_workspace/",
            progress_callback=_raising_callback,
        )

        self.assertTrue(result["success"])


class TestLoadStage1Cache(unittest.TestCase):
    def test_returns_relevant_texts_from_valid_cache_file(self):
        cache_contents = json.dumps({"relevant_texts": ["Text A.", "Text B."]})
        with patch("builtins.open", mock_open(read_data=cache_contents)):
            result = pipeline.load_stage1_cache("./output")

        self.assertEqual(result, ["Text A.", "Text B."])

    def test_returns_none_when_cache_file_is_missing(self):
        with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
            result = pipeline.load_stage1_cache("./output")

        self.assertIsNone(result)

    def test_returns_none_when_cache_file_is_corrupt_json(self):
        with patch("builtins.open", mock_open(read_data="not valid json{{{")):
            result = pipeline.load_stage1_cache("./output")

        self.assertIsNone(result)

    def test_returns_none_when_relevant_texts_key_is_missing(self):
        with patch("builtins.open", mock_open(read_data=json.dumps({"other_key": "value"}))):
            result = pipeline.load_stage1_cache("./output")

        self.assertIsNone(result)

    def test_returns_none_when_relevant_texts_is_not_a_list(self):
        with patch("builtins.open", mock_open(read_data=json.dumps({"relevant_texts": "not a list"}))):
            result = pipeline.load_stage1_cache("./output")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

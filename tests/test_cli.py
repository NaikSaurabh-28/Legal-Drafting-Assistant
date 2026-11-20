"""CLI integration tests using temporary corpora and mocked unavailable adapter."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from main import LegalDraftingCLI


class CliTests(unittest.TestCase):
    def test_cli_builds_trigram_generator_with_full_legal_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text(
                "The vendor must obtain written approval.\n",
                encoding="utf-8",
            )
            with patch(
                "main.LegalClauseClassifier",
                side_effect=FileNotFoundError("adapter missing"),
            ):
                application = LegalDraftingCLI(
                    corpus_path=corpus,
                    results_path=root / "results.csv",
                    evaluation_data_path=root / "held-out.csv",
                )

            self.assertIsNotNone(application.generator)
            self.assertEqual(application.generator.model.n, 3)
            self.assertEqual(application.generator.model.min_frequency, 1)

    def test_punctuation_only_generation_is_not_presented_as_a_clause(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text(
                "A party may give notice.\nA party must give notice.\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch(
                    "main.LegalClauseClassifier",
                    side_effect=FileNotFoundError("adapter missing"),
                ),
                redirect_stdout(output),
            ):
                application = LegalDraftingCLI(
                    corpus_path=corpus,
                    results_path=root / "results.csv",
                    evaluation_data_path=root / "held-out.csv",
                )
                self.assertIsNotNone(application.generator)
                with patch.object(
                    application.generator,
                    "generate_sentence",
                    return_value="!!!",
                ):
                    result = application._generate_one(None, 5, 3)

            self.assertEqual(result, ("", None))
            self.assertNotIn("Generated: !!!", output.getvalue())
            self.assertIn("punctuation-only sentence", output.getvalue())

    def test_csv_results_append_without_fabricated_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("A party may give notice.\n", encoding="utf-8")
            results_path = root / "results.csv"
            with (
                patch(
                    "main.LegalClauseClassifier",
                    side_effect=FileNotFoundError("adapter missing"),
                ),
                patch("builtins.input", side_effect=["y", "y"]),
            ):
                application = LegalDraftingCLI(
                    corpus_path=corpus,
                    results_path=results_path,
                    evaluation_data_path=root / "held-out.csv",
                )
                application._append_results([("First sentence.", None)])
                application._append_results([("Second sentence.", None)])

            with results_path.open(encoding="utf-8", newline="") as output_file:
                rows = list(csv.DictReader(output_file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sentence"], "First sentence.")
            self.assertEqual(rows[1]["sentence"], "Second sentence.")
            self.assertEqual(rows[0]["predicted_category"], "")
            self.assertEqual(rows[0]["confidence"], "")
            self.assertTrue(rows[0]["timestamp"])

    def test_csv_with_incompatible_header_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("A party may give notice.\n", encoding="utf-8")
            results_path = root / "results.csv"
            original = "old,header\nkeep,this row\n"
            results_path.write_text(original, encoding="utf-8")
            with (
                patch(
                    "main.LegalClauseClassifier",
                    side_effect=FileNotFoundError("adapter missing"),
                ),
                patch("builtins.input", return_value="y"),
            ):
                application = LegalDraftingCLI(
                    corpus_path=corpus,
                    results_path=results_path,
                    evaluation_data_path=root / "held-out.csv",
                )
                application._append_results([("Sentence.", None)])

            self.assertEqual(results_path.read_text(encoding="utf-8"), original)

    def test_all_menu_options_invalid_inputs_and_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text(
                "Vendor will send an invoice.\n"
                "Client must pay each invoice.\n"
                "Either party may end the agreement.\n",
                encoding="utf-8",
            )
            input_values = [
                "1", "", "not-an-integer", "0", "0", "1",
                "2", "not-an-integer", "1", "", "0", "1",
                "3", "",
                "4",
                "9",
                "5",
            ]
            output = io.StringIO()

            with (
                patch("builtins.input", side_effect=input_values),
                patch(
                    "main.LegalClauseClassifier",
                    side_effect=FileNotFoundError("adapter missing"),
                ),
                patch(
                    "main.save_evaluation_report",
                    return_value=root / "evaluation.json",
                ) as save_report,
                redirect_stdout(output),
            ):
                application = LegalDraftingCLI(
                    corpus_path=corpus,
                    results_path=root / "results.csv",
                    evaluation_data_path=root / "held-out.csv",
                )
                application.run()

            text = output.getvalue()
            self.assertNotIn("Generation unavailable", text)
            self.assertIn("Classification unavailable: adapter missing", text)
            self.assertIn("Please enter a whole number.", text)
            self.assertIn(
                "Please enter a number greater than or equal to 1.", text
            )
            self.assertIn("Input was empty; nothing to classify.", text)
            self.assertIn("Invalid option. Choose a number from 1 to 5.", text)
            self.assertIn("N-gram evaluation", text)
            self.assertIn("Classification evaluation unavailable", text)
            self.assertIn("Goodbye.", text)
            save_report.assert_called_once()

    def test_empty_corpus_degrades_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "empty.txt"
            corpus.write_text("", encoding="utf-8")
            output = io.StringIO()

            with (
                patch("builtins.input", side_effect=["5"]),
                patch(
                    "main.LegalClauseClassifier",
                    side_effect=FileNotFoundError("adapter missing"),
                ),
                redirect_stdout(output),
            ):
                application = LegalDraftingCLI(
                    corpus_path=corpus,
                    results_path=root / "results.csv",
                    evaluation_data_path=root / "held-out.csv",
                )
                application.run()

            self.assertIn(
                "Generation unavailable: Corpus contains no clauses",
                output.getvalue(),
            )
            self.assertIn("Goodbye.", output.getvalue())


if __name__ == "__main__":
    unittest.main()

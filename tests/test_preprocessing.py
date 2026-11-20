"""Unit tests for legal-text preprocessing utilities."""

import tempfile
import unittest
from pathlib import Path

from app.preprocessing import (
    LegalTextPreprocessor,
    build_vocabulary,
    clean_text,
    load_corpus,
    preprocess_corpus,
    tokenize,
)


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_lowercases_and_normalizes_whitespace(self) -> None:
        self.assertEqual(clean_text("  THE\tClient\nMUST pay. "), "the client must pay.")

    def test_tokenize_separates_words_and_punctuation(self) -> None:
        self.assertEqual(tokenize("Pay each invoice, promptly."), [
            "Pay", "each", "invoice", ",", "promptly", "."
        ])

    def test_load_and_preprocess_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text(
                "\nVendor pays the Client.\n  \n", encoding="utf-8"
            )

            self.assertEqual(
                load_corpus(corpus), ["Vendor pays the Client."]
            )
            self.assertEqual(
                preprocess_corpus(corpus),
                [["vendor", "pays", "the", "client", "."]],
            )

    def test_missing_corpus_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.txt"
            with self.assertRaisesRegex(FileNotFoundError, "Corpus file not found"):
                load_corpus(missing)

    def test_vocabulary_and_preprocessor_interface(self) -> None:
        self.assertEqual(
            build_vocabulary([["pay", "invoice"], ["pay"]]),
            ["invoice", "pay"],
        )
        self.assertEqual(
            build_vocabulary([["pay"]], include_unk=True),
            ["<UNK>", "pay"],
        )
        preprocessor = LegalTextPreprocessor()
        self.assertEqual(preprocessor.clean_text(" CLIENT "), "client")
        self.assertEqual(preprocessor.tokenize("Pay."), ["Pay", "."])
        self.assertEqual(
            preprocessor.segment_clauses(
                "The vendor shall pay. The client shall review."
            ),
            ["The vendor shall pay.", "The client shall review."],
        )
        self.assertEqual(preprocessor.segment_clauses("  "), [])

    def test_clause_segmentation_normalizes_whitespace_and_newlines(self) -> None:
        preprocessor = LegalTextPreprocessor()
        self.assertEqual(
            preprocessor.segment_clauses(
                "The   vendor shall pay.\r\n  The client\tshall review"
            ),
            ["The vendor shall pay.", "The client shall review"],
        )


if __name__ == "__main__":
    unittest.main()

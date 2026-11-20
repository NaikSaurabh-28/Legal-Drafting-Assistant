"""Unit tests for safe sentence generation and generation options."""

import unittest
from pathlib import Path

from app.ngram_model import NGramModel
from app.preprocessing import preprocess_corpus
from app.text_generator import ClauseGenerator


class _InternalTokenModel:
    n = 3

    def predict_next(
        self, context: tuple[str, ...], top_k: int = 5
    ) -> list[tuple[str, float]]:
        return [("<UNK>", 1.0), ("<START>", 0.5), ("<END>", 0.1)][:top_k]


class _ScriptedModel:
    n = 3

    def __init__(self, predictions: list[list[tuple[str, float]]]) -> None:
        self.predictions = predictions
        self.calls = 0

    def predict_next(
        self, context: tuple[str, ...], top_k: int = 5
    ) -> list[tuple[str, float]]:
        self.calls += 1
        if not self.predictions:
            return []
        return self.predictions.pop(0)[:top_k]


class TextGeneratorTests(unittest.TestCase):
    def test_repeated_apostrophes_are_suppressed_but_possessive_is_preserved(
        self,
    ) -> None:
        model = _ScriptedModel(
            [
                [("'", 1.0), ("party", 0.8)],
                [("'", 1.0), ("s", 0.8)],
                [("'", 1.0), ("s", 0.8)],
                [("'", 1.0), ("assets", 0.8)],
                [(".", 1.0)],
            ]
        )
        generator = ClauseGenerator(model=model)  # type: ignore[arg-type]

        generated = generator.generate_sentence(max_length=8, top_n=2)

        self.assertEqual(generated, "party's assets.")
        self.assertEqual(generated.count("'"), 1)

    def test_repeated_punctuation_is_skipped(self) -> None:
        model = _ScriptedModel(
            [
                [(",", 1.0), ("This", 0.8)],
                [(",", 1.0), ("is", 0.8)],
                [("is", 1.0)],
                [("a", 1.0)],
                [("clause", 1.0)],
                [(".", 1.0), ("continued", 0.8)],
            ]
        )
        generator = ClauseGenerator(model=model)  # type: ignore[arg-type]

        generated = generator.generate_sentence(max_length=8, top_n=2)

        self.assertEqual(generated, "This, is a clause.")
        self.assertNotIn(",,", generated)

    def test_repeated_words_are_skipped(self) -> None:
        model = _ScriptedModel(
            [
                [("party", 1.0)],
                [("party", 1.0), ("must", 0.8)],
                [("act", 1.0)],
                [(".", 1.0)],
            ]
        )
        generator = ClauseGenerator(model=model)  # type: ignore[arg-type]

        generated = generator.generate_sentence(max_length=8, top_n=2)

        self.assertEqual(generated, "party must act.")
        self.assertNotIn("party party", generated)

    def test_sentence_ending_punctuation_stops_generation(self) -> None:
        model = _ScriptedModel(
            [
                [("A", 1.0)],
                [("clause", 1.0)],
                [(".", 1.0), ("continued", 0.5)],
                [("this", 1.0)],
            ]
        )
        generator = ClauseGenerator(model=model)  # type: ignore[arg-type]

        generated = generator.generate_sentence(max_length=8, top_n=2)

        self.assertEqual(generated, "A clause.")
        self.assertEqual(model.calls, 3)

    def test_valid_punctuation_is_preserved_and_formatted(self) -> None:
        model = _ScriptedModel(
            [
                [(",", 1.0)],
                [("the", 1.0)],
                [("terms", 1.0)],
                [("?", 1.0), ("continue", 0.5)],
            ]
        )
        generator = ClauseGenerator(  # type: ignore[arg-type]
            model=model,
            start_context=("Review", "these"),
        )

        generated = generator.generate_sentence(max_length=8, top_n=2)

        self.assertEqual(generated, "Review these, the terms?")

    def test_punctuation_only_and_zero_length_generation_are_empty(self) -> None:
        punctuation_model = _ScriptedModel(
            [[("'", 1.0), (",", 0.9), (".", 0.8)]]
        )
        generator = ClauseGenerator(model=punctuation_model)  # type: ignore[arg-type]

        self.assertEqual(generator.generate_sentence(max_length=3), "")
        self.assertEqual(generator.generate_sentence(max_length=0), "")

    def test_incomplete_generation_is_not_returned(self) -> None:
        model = _ScriptedModel([[("A", 1.0)], [("clause", 1.0)]])
        generator = ClauseGenerator(model=model)  # type: ignore[arg-type]

        self.assertEqual(generator.generate_sentence(max_length=8), "")

    def test_real_corpus_generation_returns_nonempty_sentence(self) -> None:
        corpus_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "legal_corpus.txt"
        )
        training_sentences = preprocess_corpus(corpus_path)
        model = NGramModel(n=3, min_frequency=1)
        model.train(training_sentences)
        generator = ClauseGenerator(
            model=model,
            training_sentences=training_sentences,
        )

        generated = generator.generate_sentence(
            start_context=("vendor", "will", "send", "the"),
            max_length=30,
            top_n=3,
        )

        self.assertTrue(generated.startswith("vendor will send the"))
        self.assertTrue(any(character.isalnum() for character in generated))
        self.assertTrue(generated.endswith((".", "?", "!")))
        for token in ("<START>", "<END>", "<UNK>"):
            self.assertNotIn(token, generated)

    def test_real_generation_is_deterministic_and_uses_trigrams(self) -> None:
        corpus_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "legal_corpus.txt"
        )
        training_sentences = preprocess_corpus(corpus_path)
        model = NGramModel(n=3, min_frequency=1)
        model.train(training_sentences)
        generator = ClauseGenerator(
            model=model,
            training_sentences=training_sentences,
        )

        first = generator.generate_sentence(
            start_context=("the", "parties"),
            max_length=20,
            top_n=3,
        )
        second = generator.generate_sentence(
            start_context=("the", "parties"),
            max_length=20,
            top_n=3,
        )

        self.assertEqual(generator.model.n, 3)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith((".", "?", "!")))

    def test_sentence_start_evidence_filters_interior_only_continuations(
        self,
    ) -> None:
        corpus_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "legal_corpus.txt"
        )
        training_sentences = preprocess_corpus(corpus_path)
        model = NGramModel(n=3, min_frequency=1)
        model.train(training_sentences)
        generator = ClauseGenerator(
            model=model,
            training_sentences=training_sentences,
        )

        continuations = generator._sentence_start_continuations(
            ["the", "client"]
        )
        generated = generator.generate_sentence(
            start_context=("the", "client"),
            max_length=20,
            top_n=3,
        )

        self.assertIn(generated.split()[2], continuations)
        self.assertNotEqual(generated, "")
        self.assertTrue(generated.endswith((".", "?", "!")))

    def test_sparse_context_does_not_return_an_incomplete_seed(self) -> None:
        corpus_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "legal_corpus.txt"
        )
        training_sentences = preprocess_corpus(corpus_path)
        model = NGramModel(n=3, min_frequency=1)
        model.train(training_sentences)
        generator = ClauseGenerator(
            model=model,
            training_sentences=training_sentences,
        )

        self.assertEqual(
            generator.generate_sentence(
                start_context=("in", "the", "event"),
                max_length=20,
                top_n=3,
            ),
            "",
        )

    def test_generated_text_never_exposes_internal_tokens(self) -> None:
        generator = ClauseGenerator(
            model=_InternalTokenModel(),  # type: ignore[arg-type]
            start_context=("<START>", "<UNK>"),
            max_length=4,
            top_n=3,
        )

        generated = generator.generate_sentence()

        for token in ("<START>", "<END>", "<UNK>"):
            self.assertNotIn(token, generated)

    def test_empty_output_is_not_emitted_when_generation_has_no_candidates(
        self,
    ) -> None:
        model = _ScriptedModel([])
        generator = ClauseGenerator(model=model)  # type: ignore[arg-type]

        self.assertEqual(generator.generate_sentence(), "")
        self.assertEqual(generator.complete_draft("An unfinished draft"), "")

    def test_real_ngram_generation_hides_unknown_token(self) -> None:
        model = NGramModel(n=3)
        model.train(
            [
                ["the", "client", "must", "pay"],
                ["the", "client", "must", "pay"],
                ["the", "client", "shall", "pay"],
            ]
        )
        generator = ClauseGenerator(model=model)

        generated = generator.generate_sentence(
            start_context=("<START>", "<UNK>"),
            max_length=8,
            top_n=5,
        )

        self.assertNotIn("<START>", generated)
        self.assertNotIn("<UNK>", generated)
        self.assertNotIn("<END>", generated)

    def test_invalid_generation_options_are_rejected(self) -> None:
        for options in (
            {"max_length": -1},
            {"max_length": 1.5},
            {"max_length": True},
            {"top_n": 0},
            {"top_n": 1.5},
            {"top_n": True},
            {"sampling": "yes"},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    ClauseGenerator(**options)

        generator = ClauseGenerator(model=_InternalTokenModel())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generator.generate_sentence(sampling="yes")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

"""Unit tests for N-gram training, normalization, and validation."""

import math
import unittest

from src.ngram_model import NGramModel


TRAINING_FIXTURE = [
    ["contract", "must", "pay"],
    ["contract", "must", "pay"],
    ["client", "may", "terminate"],
    ["client", "rare", "term"],
]


class NGramModelTests(unittest.TestCase):
    def test_training_builds_expected_counts_and_reserved_vocabulary(self) -> None:
        model = NGramModel(n=2)
        model.train(TRAINING_FIXTURE)

        self.assertIn(NGramModel.UNK_TOKEN, model.vocabulary)
        self.assertIn(NGramModel.END_TOKEN, model.vocabulary)
        self.assertNotIn(NGramModel.START_TOKEN, model.vocabulary)
        self.assertEqual(model.normalize_token("may"), NGramModel.UNK_TOKEN)
        self.assertEqual(model.normalize_token("contract"), "contract")
        self.assertTrue(
            any(ngram[-1] == NGramModel.UNK_TOKEN for ngram in model.ngram_counts)
        )

    def test_probability_distribution_normalizes_for_supported_orders(self) -> None:
        contexts = {
            1: ("missing",),
            2: ("contract",),
            3: ("contract", "must"),
        }
        for n in (1, 2, 3):
            with self.subTest(n=n):
                model = NGramModel(n=n)
                model.train(TRAINING_FIXTURE)
                probabilities = [
                    model.get_probability(token, contexts[n])
                    for token in model.vocabulary
                ]
                self.assertTrue(math.isclose(sum(probabilities), 1.0))

    def test_oov_contexts_and_targets_use_unknown_token(self) -> None:
        model = NGramModel(n=3)
        model.train(TRAINING_FIXTURE)

        unknown_probability = model.get_probability(
            NGramModel.UNK_TOKEN,
            (NGramModel.UNK_TOKEN, NGramModel.UNK_TOKEN),
        )
        self.assertEqual(
            model.get_probability(
                "never-seen", ("not-in-vocabulary", "context")
            ),
            unknown_probability,
        )
        self.assertGreater(unknown_probability, 0)

    def test_start_is_not_a_prediction_target(self) -> None:
        model = NGramModel(n=1)
        model.train([["word", "word"]])

        with self.assertRaisesRegex(ValueError, "not a prediction target"):
            model.get_probability(
                NGramModel.START_TOKEN, ()
            )

    def test_predictions_hide_unknown_by_default_but_can_include_it(self) -> None:
        model = NGramModel(n=1)
        model.train([["common", "common", "rare"]])

        default_predictions = model.predict_next((), top_k=10)
        evaluation_predictions = model.predict_next(
            (), top_k=10, include_unk=True
        )
        self.assertNotIn(
            NGramModel.UNK_TOKEN,
            [token for token, _ in default_predictions],
        )
        self.assertIn(
            NGramModel.UNK_TOKEN,
            [token for token, _ in evaluation_predictions],
        )
        self.assertNotIn(
            NGramModel.START_TOKEN,
            [token for token, _ in evaluation_predictions],
        )

    def test_empty_training_corpus_is_rejected(self) -> None:
        model = NGramModel()
        for empty_training in ([], [[], []]):
            with self.subTest(training=empty_training):
                with self.assertRaisesRegex(ValueError, "non-empty sentence"):
                    model.train(empty_training)

    def test_invalid_parameters_and_top_k_are_rejected(self) -> None:
        for kwargs in (
            {"n": 0},
            {"n": 4},
            {"alpha": 0},
            {"alpha": math.inf},
            {"alpha": math.nan},
            {"min_frequency": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    NGramModel(**kwargs)

        model = NGramModel()
        model.train([["word", "word"]])
        for top_k in (-1, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    model.predict_next((), top_k=top_k)


if __name__ == "__main__":
    unittest.main()

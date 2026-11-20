"""Evaluation tests; synthetic labels below are unit-test fixtures only."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from app.evaluation import (
    DraftEvaluator,
    evaluate_classifier,
    save_evaluation_report,
    evaluate_ngram_models,
)
from app.ngram_model import NGramModel


TRAINING_FIXTURE = [
    ["contract", "must", "pay"],
    ["contract", "must", "pay"],
    ["client", "may", "terminate"],
    ["client", "rare", "term"],
]


class EvaluationTests(unittest.TestCase):
    def test_ngram_probabilities_sum_to_one(self) -> None:
        for n in (1, 2, 3):
            with self.subTest(n=n):
                model = NGramModel(n=n)
                model.train(TRAINING_FIXTURE)
                context = {
                    1: ("out-of-vocabulary",),
                    2: ("contract",),
                    3: ("contract", "must"),
                }[n]
                probabilities = [
                    model.get_probability(token, context)
                    for token in model.vocabulary
                ]

                self.assertTrue(
                    math.isclose(sum(probabilities), 1.0, rel_tol=1e-12)
                )
                self.assertNotIn(NGramModel.START_TOKEN, model.vocabulary)
                self.assertIn(NGramModel.END_TOKEN, model.vocabulary)
                self.assertIn(NGramModel.UNK_TOKEN, model.vocabulary)

    def test_oov_tokens_map_to_unk_and_have_nonzero_probability(self) -> None:
        model = NGramModel(n=3)
        model.train(TRAINING_FIXTURE)

        self.assertEqual(
            model.normalize_token("never-seen"), NGramModel.UNK_TOKEN
        )
        self.assertEqual(model.normalize_token("may"), NGramModel.UNK_TOKEN)
        self.assertGreater(
            model.get_probability(
                "never-seen", ("out-of-vocabulary", "context")
            ),
            0,
        )
        self.assertTrue(
            all(
                token != NGramModel.UNK_TOKEN
                for token, _ in model.predict_next(("contract", "must"))
            )
        )

    def test_perplexity_matches_held_out_token_formula(self) -> None:
        training = [
            ["a", "b", "a"],
            ["a", "b", "a"],
            ["a", "c", "a"],
            ["c", "a", "b"],
        ]
        for n in (1, 2, 3):
            with self.subTest(n=n):
                model = NGramModel(n=n)
                model.train(training)
                actual = DraftEvaluator().compute_perplexity(
                    ["a unseen b"], model
                )

                start_padding = max(1, n - 1)
                sequence = (
                    [NGramModel.START_TOKEN] * start_padding
                    + ["a", "unseen", "b", NGramModel.END_TOKEN]
                )
                log_probabilities = []
                for target_index in range(start_padding, len(sequence)):
                    target = model.normalize_token(sequence[target_index])
                    context = tuple(sequence[:target_index])
                    log_probabilities.append(
                        math.log(model.get_probability(target, context))
                    )
                expected = math.exp(
                    -sum(log_probabilities) / len(log_probabilities)
                )

                self.assertTrue(
                    math.isclose(actual, expected, rel_tol=1e-12)
                )

    def test_ngram_evaluation_split_is_reproducible(self) -> None:
        corpus_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "legal_corpus.txt"
        )
        first = evaluate_ngram_models(corpus_path)
        second = evaluate_ngram_models(corpus_path)

        self.assertEqual(first, second)
        self.assertEqual(
            first["split"],
            {
                "train_sentences": 83,
                "test_sentences": 21,
                "test_fraction": 0.2,
                "random_state": 42,
            },
        )

    def test_model_evaluation_keeps_baseline_and_unavailable_gpt2_separate(
        self,
    ) -> None:
        from app.core import evaluate_all_models

        report = evaluate_all_models()
        self.assertIn("tfidf_logreg_baseline", report["classifiers"])
        self.assertIn("gpt2_lora", report["classifiers"])
        baseline = report["classifiers"]["tfidf_logreg_baseline"]
        gpt2 = report["classifiers"]["gpt2_lora"]
        self.assertEqual(baseline["test_size"], 24)
        self.assertEqual(baseline["model"], "tfidf-logreg baseline")
        self.assertEqual(baseline["backend"], "tfidf-logreg")
        self.assertIn("classification_report", baseline)
        self.assertIn("confusion_matrix", baseline)
        self.assertEqual(gpt2["model"], "GPT-2 + LoRA")
        self.assertEqual(gpt2["status"], "not_evaluated")
        self.assertNotIn("accuracy", gpt2)
        self.assertEqual(
            set(report["ngram"]["models"]),
            {"1-gram", "2-gram", "3-gram"},
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = save_evaluation_report(
                report, Path(directory) / "report.json"
            )
            saved = output_path.read_text(encoding="utf-8")
        self.assertIn('"tfidf_logreg_baseline"', saved)
        self.assertIn('"gpt2_lora"', saved)

    def test_available_gpt2_metrics_are_reported_under_their_own_backend(self) -> None:
        from unittest.mock import patch

        from app import core
        from app.clause_classifier import LEGAL_CLAUSE_LABELS

        class FixtureGPT2Classifier:
            id2label = dict(enumerate(LEGAL_CLAUSE_LABELS))

            def _load(self) -> None:
                return None

            def predict(self, text: str) -> dict[str, object]:
                return {
                    "predicted_category": "payment",
                    "confidence": 0.9,
                }

        with patch(
            "app.core.milestone_3", return_value=FixtureGPT2Classifier()
        ):
            report = core.evaluate_all_models()

        baseline = report["classifiers"]["tfidf_logreg_baseline"]
        gpt2 = report["classifiers"]["gpt2_lora"]
        self.assertEqual(gpt2["status"], "evaluated")
        self.assertEqual(gpt2["backend"], "gpt2-lora")
        self.assertEqual(gpt2["test_size"], 24)
        self.assertIn("classification_report", gpt2)
        self.assertIn("confusion_matrix", gpt2)
        self.assertEqual(baseline["backend"], "tfidf-logreg")
        self.assertNotEqual(baseline["model"], gpt2["model"])

    def test_classifier_rejects_unlabeled_csv(self) -> None:
        class UnusedClassifier:
            def predict(self, text: str) -> dict[str, object]:
                raise AssertionError("prediction must not run without labels")

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "unlabeled.csv"
            csv_path.write_text(
                "text\nA clause without a label.\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "text.*label"):
                evaluate_classifier(UnusedClassifier(), csv_path)

    def test_classification_metric_math_synthetic_unit_fixture_only(self) -> None:
        # Synthetic unit-test fixture only, not real model performance.
        labels = ["A", "A", "B", "B"]
        predictions = ["A", "B", "B", "B"]

        metrics = DraftEvaluator().evaluate_classification(
            labels, predictions
        )

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["macro_recall"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], 11 / 15)
        self.assertAlmostEqual(metrics["weighted_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["weighted_recall"], 0.75)
        self.assertAlmostEqual(metrics["weighted_f1"], 11 / 15)

        class FixtureClassifier:
            id2label = {0: "A", 1: "B"}

            def predict(self, text: str) -> dict[str, object]:
                index = int(text.removeprefix("fixture-"))
                return {"predicted_category": predictions[index]}

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "held-out-fixture.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("text", "label"))
                writer.writerows(
                    (f"fixture-{index}", label)
                    for index, label in enumerate(labels)
                )

            held_out_metrics = evaluate_classifier(
                FixtureClassifier(), csv_path
            )

        self.assertEqual(held_out_metrics["test_size"], 4)
        self.assertEqual(
            held_out_metrics["evaluation_set"],
            "external_held_out_csv",
        )
        self.assertEqual(
            held_out_metrics["label_mapping"],
            [{"id": 0, "label": "A"}, {"id": 1, "label": "B"}],
        )
        self.assertAlmostEqual(held_out_metrics["accuracy"], 0.75)
        self.assertEqual(
            held_out_metrics["confusion_matrix"],
            [[1, 1], [0, 2]],
        )


if __name__ == "__main__":
    unittest.main()

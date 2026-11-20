"""Interactive command-line interface for the Legal Drafting Assistant."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from src.clause_classifier import LegalClauseClassifier
from src.evaluation import (
    evaluate_classifier,
    evaluate_ngram_models,
    save_evaluation_report,
)
from src.ngram_model import NGramModel
from src.preprocessing import clean_text, preprocess_corpus, tokenize
from src.text_generator import ClauseGenerator


ROOT = Path(__file__).resolve().parent
CSV_FIELDS = ("sentence", "predicted_category", "confidence", "timestamp")


class _PromptEnded(Exception):
    """Signal that the interactive input stream has ended."""


class LegalDraftingCLI:
    """Own and reuse the models and services for one CLI session."""

    def __init__(
        self,
        corpus_path: Path,
        results_path: Path,
        evaluation_data_path: Path,
    ) -> None:
        self.corpus_path = corpus_path
        self.results_path = results_path
        self.evaluation_data_path = evaluation_data_path
        self.generator: ClauseGenerator | None = None
        self.classifier: LegalClauseClassifier | None = None
        self.classifier_error: str | None = None

        try:
            tokenized_corpus = preprocess_corpus(corpus_path)
            if not tokenized_corpus:
                raise ValueError(f"Corpus contains no clauses: {corpus_path}")
            model = NGramModel(n=3, min_frequency=1)
            model.train(tokenized_corpus)
            self.generator = ClauseGenerator(
                model=model,
                training_sentences=tokenized_corpus,
            )
            print(f"N-gram generator ready ({len(tokenized_corpus)} clauses).")
        except (
            FileNotFoundError,
            OSError,
            UnicodeError,
            ValueError,
            RuntimeError,
        ) as error:
            print(f"Generation unavailable: {error}")

        try:
            self.classifier = LegalClauseClassifier()
            print("Legal clause classifier ready.")
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            NotImplementedError,
        ) as error:
            self.classifier_error = str(error)
            print(f"Classification unavailable: {error}")

    @staticmethod
    def _prompt(message: str) -> str:
        try:
            return input(message)
        except (EOFError, KeyboardInterrupt) as error:
            raise _PromptEnded from error

    def _prompt_integer(
        self, message: str, default: int, minimum: int
    ) -> int:
        while True:
            raw_value = self._prompt(f"{message} [{default}]: ").strip()
            if not raw_value:
                return default
            try:
                value = int(raw_value)
            except ValueError:
                print("Please enter a whole number.")
                continue
            if value < minimum:
                print(f"Please enter a number greater than or equal to {minimum}.")
                continue
            return value

    def _prompt_context(self) -> tuple[str, ...] | None:
        seed = self._prompt("Optional seed/context (press Enter for none): ")
        if not seed.strip():
            return None
        context = tuple(tokenize(clean_text(seed)))
        return context or None

    def _classify(
        self, sentence: str, *, display: bool = True
    ) -> dict[str, object] | None:
        if self.classifier is None:
            if display:
                print(
                    "Classification unavailable; no category or confidence "
                    "was produced."
                )
            return None

        try:
            result = self.classifier.predict(sentence)
        except (
            ImportError,
            NotImplementedError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            print(f"Classification failed and is now unavailable: {error}")
            self.classifier = None
            return None

        if not isinstance(result, dict):
            print(
                "Classification unavailable: the classifier returned an "
                "invalid result."
            )
            self.classifier = None
            return None

        if display:
            category = next(
                (
                    result[key]
                    for key in ("predicted_category", "category", "label")
                    if result.get(key) is not None
                ),
                None,
            )
            confidence = result.get("confidence")
            probabilities = result.get("probabilities")
            if category is not None:
                print(f"Predicted category: {category}")
            if confidence is not None:
                print(f"Confidence: {confidence}")
            if probabilities is not None:
                print(f"Probabilities: {probabilities}")
            if category is None and confidence is None and probabilities is None:
                print("The classifier returned no category or confidence.")

        return result

    def _generate_one(
        self,
        context: tuple[str, ...] | None,
        max_length: int,
        top_n: int,
        sampling: bool = False,
    ) -> tuple[str, dict[str, object] | None]:
        if self.generator is None:
            print("Cannot generate: the N-gram model is unavailable.")
            return "", None
        try:
            sentence = self.generator.generate_sentence(
                start_context=context,
                max_length=max_length,
                top_n=top_n,
                sampling=sampling,
            )
        except (RuntimeError, ValueError) as error:
            print(f"Generation failed: {error}")
            return "", None

        if not sentence.strip() or not any(
            character.isalnum() for character in sentence
        ):
            print(
                "The generator returned an empty, incomplete, or "
                "punctuation-only sentence."
            )
            return "", None

        print(f"Generated: {sentence}")
        return sentence, self._classify(sentence)

    def _append_results(
        self, results: list[tuple[str, dict[str, object] | None]]
    ) -> None:
        if not results:
            return
        answer = self._prompt(
            f"Append {len(results)} result(s) to {self.results_path}? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            return

        try:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            has_existing_rows = (
                self.results_path.exists()
                and self.results_path.stat().st_size > 0
            )
            if has_existing_rows:
                with self.results_path.open(
                    "r", encoding="utf-8", newline=""
                ) as existing_file:
                    existing_header = next(csv.reader(existing_file), [])
                if tuple(existing_header) != CSV_FIELDS:
                    print(
                        "Could not append results to CSV: existing header "
                        f"must be {CSV_FIELDS}."
                    )
                    return

            with self.results_path.open(
                "a", encoding="utf-8", newline=""
            ) as output_file:
                writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
                if not has_existing_rows:
                    writer.writeheader()
                timestamp = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                for sentence, prediction in results:
                    category = None
                    confidence = None
                    if prediction is not None:
                        category = next(
                            (
                                prediction[key]
                                for key in (
                                    "predicted_category",
                                    "category",
                                    "label",
                                )
                                if prediction.get(key) is not None
                            ),
                            None,
                        )
                        confidence = prediction.get("confidence")
                    writer.writerow(
                        {
                            "sentence": sentence,
                            "predicted_category": (
                                "" if category is None else category
                            ),
                            "confidence": (
                                "" if confidence is None else confidence
                            ),
                            "timestamp": timestamp,
                        }
                    )
            print(f"Appended results to {self.results_path}.")
        except (OSError, csv.Error) as error:
            print(f"Could not append results to CSV: {error}")

    def _generate_and_classify(self) -> None:
        context = self._prompt_context()
        max_length = self._prompt_integer("Maximum generated tokens", 30, 0)
        top_n = self._prompt_integer("Top-N candidates", 3, 1)
        result = self._generate_one(context, max_length, top_n)
        if result[0]:
            self._append_results([result])

    def _generate_multiple(self) -> None:
        count = self._prompt_integer("Number of sentences", 3, 1)
        context = self._prompt_context()
        max_length = self._prompt_integer("Maximum generated tokens", 30, 0)
        top_n = self._prompt_integer("Top-N candidates", 3, 1)
        results = [
            self._generate_one(context, max_length, top_n, sampling=True)
            for _ in range(count)
        ]
        self._append_results([result for result in results if result[0]])

    def _classify_custom(self) -> None:
        sentence = self._prompt("Enter a legal sentence to classify: ").strip()
        if not sentence:
            print("Input was empty; nothing to classify.")
            return
        self._classify(sentence)

    def _evaluate_classifier(self) -> None:
        report: dict[str, object] = {}

        try:
            ngram_report = evaluate_ngram_models(self.corpus_path)
            report["ngram"] = ngram_report
            split = ngram_report["split"]
            models = ngram_report["models"]
            print(
                "N-gram evaluation "
                f"({split['train_sentences']} train / "
                f"{split['test_sentences']} test sentences, "
                f"random_state={split['random_state']}):"
            )
            for name, metrics in models.items():
                print(
                    f"  {name}: perplexity={metrics['perplexity']:.4f}, "
                    "next-token accuracy="
                    f"{metrics['next_token_accuracy']:.4f}"
                )
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            UnicodeError,
            ValueError,
            RuntimeError,
        ) as error:
            print(f"N-gram evaluation unavailable: {error}")

        if self.classifier is None:
            reason = self.classifier_error or "classifier is unavailable"
            if not self.evaluation_data_path.is_file():
                reason += (
                    "\nNo held-out labeled test CSV exists at "
                    f"{self.evaluation_data_path}."
                )
            print(
                "Classification evaluation unavailable: "
                f"{reason}"
            )
            report["classification"] = {
                "status": "unavailable",
                "reason": reason,
            }
        elif not self.evaluation_data_path.is_file():
            reason = (
                "no held-out test CSV; expected 'text' and 'label' columns at "
                f"{self.evaluation_data_path}"
            )
            print(f"Classification evaluation unavailable: {reason}")
            report["classification"] = {
                "status": "unavailable",
                "reason": reason,
            }
        else:
            try:
                metrics = evaluate_classifier(
                    self.classifier,
                    self.evaluation_data_path,
                )
                report["classification"] = metrics
                print(
                    "Classification held-out metrics: "
                    f"accuracy={metrics['accuracy']:.4f}, "
                    f"macro F1={metrics['macro_avg']['f1']:.4f}, "
                    f"weighted F1={metrics['weighted_avg']['f1']:.4f}"
                )
            except NotImplementedError as error:
                reason = f"classifier inference is not implemented: {error}"
                print(f"Classification evaluation unavailable: {reason}")
                report["classification"] = {
                    "status": "unavailable",
                    "reason": reason,
                }
            except (
                FileNotFoundError,
                OSError,
                csv.Error,
                ImportError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                reason = str(error)
                print(f"Classification evaluation failed: {reason}")
                report["classification"] = {
                    "status": "unavailable",
                    "reason": reason,
                }

        classification_report = report.get("classification")
        has_classification_metrics = (
            isinstance(classification_report, dict)
            and "accuracy" in classification_report
        )
        if "ngram" in report or has_classification_metrics:
            try:
                report_path = save_evaluation_report(report)
                print(f"Saved evaluation report to {report_path}.")
            except (OSError, TypeError, ValueError) as error:
                print(f"Could not save evaluation report: {error}")

    def run(self) -> None:
        """Run the interactive menu until the user chooses to exit."""
        try:
            while True:
                print(
                    "\nLegal Drafting Assistant\n"
                    "1. Generate and classify a legal sentence\n"
                    "2. Generate multiple legal sentences\n"
                    "3. Classify a custom legal sentence\n"
                    "4. Evaluate models (N-gram / classifier)\n"
                    "5. Exit"
                )
                choice = self._prompt("Choose an option: ").strip()
                if choice == "1":
                    self._generate_and_classify()
                elif choice == "2":
                    self._generate_multiple()
                elif choice == "3":
                    self._classify_custom()
                elif choice == "4":
                    self._evaluate_classifier()
                elif choice == "5":
                    print("Goodbye.")
                    return
                else:
                    print("Invalid option. Choose a number from 1 to 5.")
        except _PromptEnded:
            print("\nInput ended; exiting.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive legal clause generation and classification."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "legal_corpus.txt",
        help="Training corpus for the N-gram generator.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "outputs" / "generated_results.csv",
        help="CSV file to which generated results may be appended.",
    )
    parser.add_argument(
        "--evaluation-data",
        type=Path,
        default=ROOT / "data" / "classifier_test.csv",
        help="Held-out test CSV with 'text' and 'label' columns.",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize model services once and start the interactive menu."""
    args = _parse_args()
    application = LegalDraftingCLI(
        corpus_path=args.corpus,
        results_path=args.results,
        evaluation_data_path=args.evaluation_data,
    )
    application.run()


if __name__ == "__main__":
    main()

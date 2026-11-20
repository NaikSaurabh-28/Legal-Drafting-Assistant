"""Evaluation helpers for legal-clause classifiers and N-gram models."""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, cast


ROOT = Path(__file__).resolve().parents[1]


class TextClassifier(Protocol):
    """Classifier interface required by :func:`evaluate_classifier`."""

    def predict(self, text: str) -> dict[str, object]:
        """Return a prediction dictionary for one text."""


class NGramScoringModel(Protocol):
    """Subset of NGramModel required for perplexity and next-token scoring."""

    n: int

    def normalize_token(self, token: str) -> str:
        """Map an out-of-vocabulary token to the trained unknown token."""

    def get_probability(
        self, word: str, context: tuple[str, ...]
    ) -> float:
        """Return the probability of a candidate next token."""

    def predict_next(
        self,
        context: tuple[str, ...],
        top_k: int = 5,
        include_unk: bool = False,
    ) -> list[tuple[str, float]]:
        """Return ranked next-token candidates."""


class AverageMetrics(TypedDict):
    precision: float
    recall: float
    f1: float


class ClassificationMetrics(TypedDict):
    accuracy: float
    macro_avg: AverageMetrics
    weighted_avg: AverageMetrics
    class_labels: list[str]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, object]


class ClassifierEvaluation(ClassificationMetrics):
    test_size: int
    evaluation_set: str
    label_mapping: list[dict[str, object]] | None


class NGramModelMetrics(TypedDict):
    perplexity: float
    next_token_accuracy: float
    tokens_evaluated: int


class NGramSplit(TypedDict):
    train_sentences: int
    test_sentences: int
    test_fraction: float
    random_state: int


class NGramEvaluation(TypedDict):
    split: NGramSplit
    models: dict[str, NGramModelMetrics]


def _sklearn_classification_metrics(
    labels: Sequence[str],
    predictions: Sequence[str],
    class_labels: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Calculate classification metrics using scikit-learn."""
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal lengths.")
    if len(labels) == 0:
        raise ValueError("At least one labeled prediction is required.")

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required to calculate classification metrics."
        ) from error

    ordered_labels = list(
        class_labels
        if class_labels is not None
        else dict.fromkeys([*labels, *predictions])
    )
    if not ordered_labels:
        raise ValueError("No class labels are available for evaluation.")

    macro = precision_recall_fscore_support(
        labels,
        predictions,
        labels=ordered_labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        labels,
        predictions,
        labels=ordered_labels,
        average="weighted",
        zero_division=0,
    )
    matrix = [
        [int(value) for value in row]
        for row in confusion_matrix(
            labels, predictions, labels=ordered_labels
        ).tolist()
    ]
    report = classification_report(
        labels,
        predictions,
        labels=ordered_labels,
        target_names=ordered_labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_avg": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
        },
        "weighted_avg": {
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
        },
        "class_labels": ordered_labels,
        "confusion_matrix": matrix,
        "classification_report": cast(dict[str, object], report),
    }


def _classifier_id2label(
    classifier: TextClassifier,
) -> dict[object, object] | None:
    """Read an id2label mapping exposed by a classifier or its model config."""
    owners = [
        classifier,
        getattr(classifier, "model", None),
        getattr(classifier, "config", None),
    ]
    for owner in owners:
        if owner is None:
            continue
        mapping = getattr(owner, "id2label", None)
        config = getattr(owner, "config", None)
        if mapping is None and config is not None:
            mapping = getattr(config, "id2label", None)
        if isinstance(mapping, Mapping) and mapping:
            return dict(mapping)
    return None


def _mapping_label_order(
    id2label: Mapping[object, object] | None,
    labels: Sequence[str],
) -> list[str] | None:
    if id2label is None:
        return None

    ordered_items = sorted(
        id2label.items(),
        key=lambda item: (
            0,
            int(item[0]),
        )
        if str(item[0]).lstrip("-").isdigit()
        else (1, str(item[0])),
    )
    mapped_labels = [value for _, value in ordered_items]
    if any(not isinstance(label, str) or not label for label in mapped_labels):
        raise ValueError("The classifier id2label mapping must contain labels.")
    if len(set(mapped_labels)) != len(mapped_labels):
        raise ValueError("The classifier id2label mapping contains duplicates.")

    unknown_labels = set(labels) - set(mapped_labels)
    if unknown_labels:
        raise ValueError(
            "Dataset labels do not match the classifier id2label mapping: "
            f"{sorted(unknown_labels)}"
        )
    return mapped_labels


def _prediction_label(
    prediction: object,
    id2label: Mapping[object, object] | None,
) -> str:
    if not isinstance(prediction, dict):
        raise TypeError("classifier.predict(text) must return a dictionary.")

    value = next(
        (
            prediction[key]
            for key in ("predicted_category", "category", "label")
            if prediction.get(key) is not None
        ),
        None,
    )
    if value is None:
        raise ValueError(
            "Classifier result has no predicted_category, category, or label."
        )

    if isinstance(value, str):
        if id2label is not None:
            if value in id2label.values():
                return value
            for identifier, label in id2label.items():
                if str(identifier) == value and isinstance(label, str):
                    return label
        return value

    if id2label is not None:
        for identifier, label in id2label.items():
            if identifier == value and isinstance(label, str):
                return label
            if str(identifier) == str(value) and isinstance(label, str):
                return label
    raise TypeError("Classifier prediction labels must be strings or mapped IDs.")


def evaluate_classifier(
    classifier: TextClassifier,
    csv_path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
) -> ClassifierEvaluation:
    """Evaluate an externally trained classifier on a held-out test CSV.

    ``csv_path`` must contain only held-out test examples. The classifier
    must never have been trained or selected using those examples. This
    function does not split or train the classifier.

    Args:
        classifier: Trained classifier with ``predict(text)`` returning a
            mapping that contains a predicted category.
        csv_path: Labeled CSV containing text and ground-truth label columns.
        text_col: Name of the text column.
        label_col: Name of the ground-truth label column.

    Returns:
        Accuracy, macro/weighted precision-recall-F1, confusion matrix, and
        scikit-learn classification report, with evaluation-set and label-map
        metadata.

    Raises:
        FileNotFoundError: If the labeled CSV does not exist.
        ValueError: If the schema, rows, or classifier label mapping is invalid.
        ImportError: If scikit-learn is unavailable.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labeled evaluation CSV not found: {path}")
    if text_col == label_col:
        raise ValueError("text_col and label_col must name different columns.")

    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        if (
            text_col not in fieldnames
            or label_col not in fieldnames
            or len(fieldnames) != len(set(fieldnames))
        ):
            raise ValueError(
                f"CSV must contain distinct '{text_col}' and "
                f"'{label_col}' columns."
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            text = row.get(text_col)
            label = row.get(label_col)
            if (
                not isinstance(text, str)
                or not text.strip()
                or not isinstance(label, str)
                or not label.strip()
            ):
                raise ValueError(
                    f"CSV row {row_number} has empty or missing text/label."
                )
            rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError("The held-out CSV contains no labeled rows.")

    id2label = _classifier_id2label(classifier)
    predictions = [
        _prediction_label(
            classifier.predict(row["text"]),
            id2label,
        )
        for row in rows
    ]
    all_labels = [row["label"] for row in rows]
    class_labels = _mapping_label_order(id2label, [*all_labels, *predictions])
    if class_labels is None:
        class_labels = list(dict.fromkeys([*all_labels, *predictions]))
    metrics = _sklearn_classification_metrics(
        all_labels, predictions, class_labels
    )
    return {
        **metrics,
        **{
            "test_size": len(rows),
            "evaluation_set": "external_held_out_csv",
            "label_mapping": (
                [
                    {"id": identifier, "label": label}
                    for identifier, label in id2label.items()
                ]
                if id2label is not None
                else None
            ),
        },
    }


def _score_ngram_model(
    model: NGramScoringModel,
    tokenized_sentences: Sequence[Sequence[str]],
) -> NGramModelMetrics:
    """Score held-out next tokens and their sequence perplexity."""
    n = model.n
    start_padding = max(1, n - 1)
    total_log_probability = 0.0
    correct = 0
    token_count = 0

    for sentence in tokenized_sentences:
        sequence = (
            ["<START>"] * start_padding + list(sentence) + ["<END>"]
        )
        for target_index in range(start_padding, len(sequence)):
            context = tuple(sequence[:target_index])
            target = sequence[target_index]
            normalized_target = model.normalize_token(target)
            probability = model.get_probability(normalized_target, context)
            if probability <= 0:
                raise ValueError("N-gram model returned a non-positive probability.")
            total_log_probability += math.log(probability)
            top_prediction = model.predict_next(
                context, top_k=1, include_unk=True
            )
            if (
                top_prediction
                and top_prediction[0][0] == normalized_target
            ):
                correct += 1
            token_count += 1

    if token_count == 0:
        raise ValueError("The held-out corpus contains no tokens to evaluate.")

    return {
        "perplexity": math.exp(-total_log_probability / token_count),
        "next_token_accuracy": correct / token_count,
        "tokens_evaluated": token_count,
    }


def evaluate_ngram_models(
    corpus_path: str | Path | None = None,
) -> NGramEvaluation:
    """Train unigram, bigram, and trigram models on an 80/20 corpus split.

    Sentence-level splitting happens before training, so held-out sentences
    are not included in any model's n-gram counts.
    """
    from app.ngram_model import NGramModel
    from app.preprocessing import preprocess_corpus

    path = (
        Path(corpus_path)
        if corpus_path is not None
        else ROOT / "data" / "legal_corpus.txt"
    )
    tokenized_corpus = preprocess_corpus(path)
    if len(tokenized_corpus) < 2:
        raise ValueError("At least two corpus sentences are required to split.")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for the reproducible N-gram split."
        ) from error

    train_sentences, test_sentences = train_test_split(
        tokenized_corpus,
        test_size=0.2,
        random_state=42,
    )
    model_results: dict[str, NGramModelMetrics] = {}
    for n in (1, 2, 3):
        model = NGramModel(n=n)
        model.train(train_sentences)
        model_results[f"{n}-gram"] = _score_ngram_model(
            model, test_sentences
        )

    return {
        "split": {
            "train_sentences": len(train_sentences),
            "test_sentences": len(test_sentences),
            "test_fraction": 0.2,
            "random_state": 42,
        },
        "models": model_results,
    }


def save_evaluation_report(
    report: dict[str, object],
    output_path: str | Path | None = None,
) -> Path:
    """Persist evaluation results as formatted JSON."""
    path = (
        Path(output_path)
        if output_path is not None
        else ROOT / "outputs" / "evaluation_report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


class DraftEvaluator:
    """Compute classification and model-based perplexity metrics."""

    def compute_perplexity(
        self,
        texts: Sequence[str],
        model: NGramScoringModel | None = None,
    ) -> float:
        """Compute perplexity for texts with an already-trained N-gram model.

        Args:
            texts: Held-out texts to score.
            model: Trained ``NGramModel``. It must not be trained on ``texts``.

        Raises:
            ValueError: If no trained model or no tokens are provided.
        """
        if model is None:
            raise ValueError(
                "A separately trained NGramModel is required for perplexity."
            )

        from app.preprocessing import clean_text, tokenize

        tokenized_texts = [
            tokenize(clean_text(text))
            for text in texts
            if text.strip()
        ]
        if not tokenized_texts:
            raise ValueError("At least one non-empty text is required.")
        return float(
            _score_ngram_model(model, tokenized_texts)["perplexity"]
        )

    def compute_rouge(
        self, references: Sequence[str], predictions: Sequence[str]
    ) -> dict[str, float]:
        """Compute ROUGE metrics for predicted and reference texts."""
        raise NotImplementedError

    def evaluate_classification(
        self, labels: Sequence[str], predictions: Sequence[str]
    ) -> dict[str, float]:
        """Compute accuracy and macro/weighted precision, recall, and F1."""
        metrics = _sklearn_classification_metrics(labels, predictions)
        macro = metrics["macro_avg"]
        weighted = metrics["weighted_avg"]
        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(macro["precision"]),
            "macro_recall": float(macro["recall"]),
            "macro_f1": float(macro["f1"]),
            "weighted_precision": float(weighted["precision"]),
            "weighted_recall": float(weighted["recall"]),
            "weighted_f1": float(weighted["f1"]),
        }

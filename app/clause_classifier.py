"""Clause classification interface."""

from pathlib import Path
import math
from typing import Sequence


LEGAL_CLAUSE_LABELS = (
    "payment",
    "confidentiality",
    "termination",
    "liability",
    "governing_law",
    "obligations",
    "indemnification",
    "dispute_resolution",
)


class AdapterArtifactError(RuntimeError):
    """A present adapter is invalid or cannot be restored for inference."""


def _read_adapter_json(path: Path, description: str) -> dict[str, object]:
    """Read one JSON artifact and report malformed adapter files clearly."""
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterArtifactError(
            f"Unable to read valid {description} at {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AdapterArtifactError(
            f"{description} at {path} must contain a JSON object."
        )
    return value


class LegalClauseClassifier:
    """Classification using GPT-2 plus a fine-tuned legal-clause adapter.

    The adapter is expected under ``models/legal_clause_adapter`` and must
    contain PEFT adapter files plus a model configuration with its label map.
    """

    name = "gpt2-lora"

    def __init__(self, adapter_dir: str | Path | None = None) -> None:
        """Check that the required pretrained adapter artifacts are available.

        Args:
            adapter_dir: Directory containing the PEFT adapter. Defaults to
                ``models/legal_clause_adapter`` in the project root.

        Raises:
            FileNotFoundError: If the adapter config, weights, or label map
                are missing.
        """
        self.adapter_dir = (
            Path(adapter_dir)
            if adapter_dir is not None
            else Path(__file__).resolve().parents[1]
            / "models"
            / "legal_clause_adapter"
        )

        missing: list[str] = []
        adapter_config = self.adapter_dir / "adapter_config.json"
        model_config = self.adapter_dir / "config.json"
        if not adapter_config.is_file():
            missing.append(str(adapter_config))
        if not (
            (self.adapter_dir / "adapter_model.safetensors").is_file()
            or (self.adapter_dir / "adapter_model.bin").is_file()
        ):
            missing.append(
                f"{self.adapter_dir / 'adapter_model.safetensors'} "
                f"(or {self.adapter_dir / 'adapter_model.bin'})"
            )
        if not model_config.is_file():
            missing.append(
                f"{model_config} containing the trained 'id2label' mapping"
            )
        for filename in (
            "label_mapping.json",
            "training_config.json",
            "training_metadata.json",
        ):
            if not (self.adapter_dir / filename).is_file():
                missing.append(self.adapter_dir / filename)
        if not (self.adapter_dir / "tokenizer_config.json").is_file():
            missing.append(self.adapter_dir / "tokenizer_config.json")
        if not (
            (self.adapter_dir / "vocab.json").is_file()
            and (self.adapter_dir / "merges.txt").is_file()
        ):
            missing.append(
                f"{self.adapter_dir} containing saved GPT-2 vocab.json "
                "and merges.txt"
            )

        if missing:
            required_base_model = (
                "The adapter_config.json must also specify the GPT-2 "
                "'base_model_name_or_path'."
            )
            missing_files = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(
                "Fine-tuned legal-clause adapter artifacts are missing:\n"
                f"{missing_files}\n{required_base_model}"
            )

    def _load(self) -> None:
        """Load GPT-2 and the fine-tuned adapter on first use (heavy imports stay lazy)."""
        if getattr(self, "_model", None) is not None:
            return

        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        adapter_config = _read_adapter_json(
            self.adapter_dir / "adapter_config.json", "adapter_config.json"
        )
        base = adapter_config.get("base_model_name_or_path")
        if not isinstance(base, str) or not base:
            raise AdapterArtifactError(
                "adapter_config.json has no 'base_model_name_or_path'."
            )
        config = _read_adapter_json(self.adapter_dir / "config.json", "config.json")
        id2label = config.get("id2label")
        if not isinstance(id2label, dict):
            raise AdapterArtifactError("config.json has no 'id2label' mapping.")
        try:
            self.id2label = {int(key): value for key, value in id2label.items()}
        except (TypeError, ValueError) as error:
            raise AdapterArtifactError(
                "config.json has an invalid 'id2label' mapping."
            ) from error
        if (
            len(id2label) != len(LEGAL_CLAUSE_LABELS)
            or set(self.id2label) != set(range(len(LEGAL_CLAUSE_LABELS)))
            or any(not isinstance(value, str) for value in self.id2label.values())
            or tuple(self.id2label[index] for index in range(len(LEGAL_CLAUSE_LABELS)))
            != LEGAL_CLAUSE_LABELS
        ):
            raise AdapterArtifactError(
                "config.json must contain the exact ordered eight-class label mapping."
            )
        mapping = _read_adapter_json(
            self.adapter_dir / "label_mapping.json", "label_mapping.json"
        )
        if mapping.get("id2label") != {
            str(index): label for index, label in self.id2label.items()
        }:
            raise AdapterArtifactError(
                "label_mapping.json does not match the model config label mapping."
            )
        expected_label2id = {
            label: index for index, label in enumerate(LEGAL_CLAUSE_LABELS)
        }
        if mapping.get("label2id") != expected_label2id:
            raise AdapterArtifactError(
                "label_mapping.json does not match the model config label mapping."
            )
        training_config = _read_adapter_json(
            self.adapter_dir / "training_config.json", "training_config.json"
        )
        if (
            training_config.get("base_model") != base
            or training_config.get("num_labels") != len(LEGAL_CLAUSE_LABELS)
            or training_config.get("labels") != list(LEGAL_CLAUSE_LABELS)
            or training_config.get("label2id") != expected_label2id
            or training_config.get("id2label")
            != {str(index): label for index, label in self.id2label.items()}
        ):
            raise AdapterArtifactError(
                "training_config.json is inconsistent with the saved model labels."
            )
        metadata = _read_adapter_json(
            self.adapter_dir / "training_metadata.json",
            "training_metadata.json",
        )
        class_counts = metadata.get("class_counts")
        dataset_hash = metadata.get("training_dataset_sha256")
        if (
            metadata.get("training_dataset") != "classifier_train.csv"
            or not isinstance(dataset_hash, str)
            or len(dataset_hash) != 64
            or any(character not in "0123456789abcdef" for character in dataset_hash)
            or not isinstance(class_counts, dict)
            or set(class_counts) != set(LEGAL_CLAUSE_LABELS)
            or any(
                not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                for count in class_counts.values()
            )
            or not isinstance(metadata.get("training_rows"), int)
            or isinstance(metadata.get("training_rows"), bool)
            or metadata.get("training_rows") != sum(class_counts.values())
        ):
            raise AdapterArtifactError(
                "training_metadata.json does not describe the expected training run."
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.adapter_dir), local_files_only=True
            )
            if (
                self._tokenizer.pad_token is None
                and self._tokenizer.eos_token is not None
            ):
                self._tokenizer.pad_token = self._tokenizer.eos_token
            if self._tokenizer.pad_token_id is None:
                raise AdapterArtifactError(
                    "The saved tokenizer has no usable padding token."
                )
            self._tokenizer.padding_side = "right"
            model = AutoModelForSequenceClassification.from_pretrained(
                base,
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id={label: index for index, label in self.id2label.items()},
            )
            model.config.pad_token_id = self._tokenizer.pad_token_id
            self._model = PeftModel.from_pretrained(
                model, str(self.adapter_dir)
            ).eval()
        except (OSError, ValueError, KeyError) as error:
            raise AdapterArtifactError(
                f"Unable to restore the saved GPT-2 adapter from "
                f"{self.adapter_dir}: {error}"
            ) from error

    def predict(self, text: str) -> dict[str, object]:
        """Classify one clause with GPT-2 plus the adapter.

        Returns ``predicted_category``, ``confidence`` and ``probabilities``.
        Raises ImportError if torch/transformers/peft are not installed.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        self._load()
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        if (
            len(probs) != len(LEGAL_CLAUSE_LABELS)
            or any(not math.isfinite(probability) for probability in probs)
            or any(probability < 0.0 or probability > 1.0 for probability in probs)
        ):
            raise AdapterArtifactError(
                "Loaded classifier returned invalid eight-class probabilities."
            )
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "predicted_category": self.id2label[best],
            "confidence": round(probs[best], 4),
            "probabilities": {self.id2label[i]: round(p, 4) for i, p in enumerate(probs)},
        }


class ClauseClassifier:
    """Stub interface for classifying legal clauses."""

    def train(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        """Train the classifier on labeled clauses.

        Args:
            texts: Clause texts used for training.
            labels: Labels corresponding to each clause.
        """
        raise NotImplementedError

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict a label for each clause.

        Args:
            texts: Clause texts to classify.

        Returns:
            Predicted labels in input order.
        """
        raise NotImplementedError

    def evaluate(
        self, texts: Sequence[str], labels: Sequence[str]
    ) -> dict[str, float]:
        """Evaluate predictions against reference labels.

        Args:
            texts: Clause texts to classify.
            labels: Reference labels corresponding to each clause.

        Returns:
            Classification metrics.
        """
        raise NotImplementedError


class BaselineClauseClassifier:
    """TF-IDF + logistic regression clause tagger trained from a labeled CSV.

    A lightweight stand-in so the pipeline works end to end without GPT-2 weights.
    It is a different model from the GPT-2 adapter and is reported as such.
    """

    name = "tfidf-logreg baseline"

    def __init__(self, train_csv: str | Path | None = None) -> None:
        import csv

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        path = Path(train_csv) if train_csv else Path(__file__).resolve().parents[1] / "data" / "classifier_train.csv"
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No training rows in {path}")
        if not {"text", "label"}.issubset(rows[0]):
            raise ValueError(f"Training CSV must contain text and label columns: {path}")
        labels_in_data = {row["label"].strip() for row in rows}
        if labels_in_data != set(LEGAL_CLAUSE_LABELS):
            raise ValueError(
                "Training CSV labels must exactly match the eight legal clause "
                f"categories; found {sorted(labels_in_data)}."
            )
        if any(not row["text"].strip() or not row["label"].strip() for row in rows):
            raise ValueError(f"Training CSV contains an empty text or label: {path}")
        self._pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
            LogisticRegression(max_iter=2000, C=1),
        )
        self._pipe.fit([r["text"] for r in rows], [r["label"] for r in rows])
        self.labels = [str(c) for c in self._pipe.classes_]

    def predict(self, text: str) -> dict[str, object]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        probs = self._pipe.predict_proba([text])[0]
        best = int(probs.argmax())
        return {
            "predicted_category": self.labels[best],
            "confidence": round(float(probs[best]), 4),
            "probabilities": {l: round(float(p), 4) for l, p in zip(self.labels, probs)},
        }

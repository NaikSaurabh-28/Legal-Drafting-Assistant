"""Clause classification interface."""

from pathlib import Path
from typing import Sequence


class LegalClauseClassifier:
    """Scaffold for classification using a fine-tuned legal-clause adapter.

    The adapter is expected under ``models/legal_clause_adapter`` and must
    contain PEFT adapter files plus a model configuration with its label map.
    """

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

    def predict(self, text: str) -> dict[str, object]:
        """Predict a clause category once adapter-backed inference is available.

        Args:
            text: Legal clause text to classify.

        Raises:
            NotImplementedError: Until the required fine-tuned adapter is
                supplied and inference is implemented.
        """
        raise NotImplementedError(
            "Inference is unavailable until a fine-tuned legal-clause "
            "adapter is supplied."
        )


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

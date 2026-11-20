"""Classifier, adapter-loading, training-data, and CLI integration tests."""

import json
import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from app.clause_classifier import (
    LEGAL_CLAUSE_LABELS,
    BaselineClauseClassifier,
    LegalClauseClassifier,
)
from app.cli import main
from scripts.train_gpt2_adapter import load_training_rows


def _write_adapter_files(adapter_dir: Path) -> None:
    adapter_dir.mkdir()
    id2label = {str(index): label for index, label in enumerate(LEGAL_CLAUSE_LABELS)}
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "gpt2"}), encoding="utf-8"
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"test placeholder")
    (adapter_dir / "config.json").write_text(
        json.dumps({"id2label": id2label}), encoding="utf-8"
    )
    (adapter_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "id2label": id2label,
                "label2id": {
                    label: index for index, label in enumerate(LEGAL_CLAUSE_LABELS)
                },
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "training_config.json").write_text(
        json.dumps(
            {
                "base_model": "gpt2",
                "num_labels": len(LEGAL_CLAUSE_LABELS),
                "labels": list(LEGAL_CLAUSE_LABELS),
                "id2label": id2label,
                "label2id": {
                    label: index for index, label in enumerate(LEGAL_CLAUSE_LABELS)
                },
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "training_dataset": "classifier_train.csv",
                "training_dataset_sha256": "a" * 64,
                "class_counts": {label: 1 for label in LEGAL_CLAUSE_LABELS},
                "training_rows": len(LEGAL_CLAUSE_LABELS),
            }
        ),
        encoding="utf-8",
    )
    (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")


def test_baseline_classifier_returns_only_supported_classes() -> None:
    classifier = BaselineClauseClassifier()
    result = classifier.predict(
        "The supplier shall indemnify the customer against third-party claims."
    )
    assert classifier.name == "tfidf-logreg baseline"
    assert result["predicted_category"] in LEGAL_CLAUSE_LABELS
    assert set(result["probabilities"]) == set(LEGAL_CLAUSE_LABELS)


def test_training_loader_reads_only_the_fixed_eight_class_training_csv() -> None:
    rows = load_training_rows()
    assert len(rows) == 80
    assert {row["label"] for row in rows} == set(LEGAL_CLAUSE_LABELS)


def test_missing_adapter_has_actionable_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Fine-tuned legal-clause adapter"):
        LegalClauseClassifier(tmp_path)


def test_adapter_loader_restores_mapping_tokenizer_and_attention_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter_dir = tmp_path / "adapter"
    _write_adapter_files(adapter_dir)
    observed: dict[str, object] = {}

    class FakeProbabilities(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, **inputs):
            observed["attention_mask"] = inputs.get("attention_mask")
            return types.SimpleNamespace(logits=[list(range(8))])

    class FakeTokenizer:
        pad_token = None
        eos_token = "<|endoftext|>"
        pad_token_id = 50256

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            observed["tokenizer_path"] = path
            observed["local_files_only"] = kwargs["local_files_only"]
            return cls()

        def __call__(self, text, **kwargs):
            observed["padding"] = kwargs["padding"]
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, base, **kwargs):
            observed["base"] = base
            observed["num_labels"] = kwargs["num_labels"]
            return types.SimpleNamespace(config=types.SimpleNamespace())

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path):
            observed["adapter_path"] = adapter_path
            return FakeModel()

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = nullcontext
    def fake_softmax(logits, dim):
        exponentials = [math.exp(value) for value in logits]
        total = sum(exponentials)
        return FakeProbabilities([value / total for value in exponentials])

    fake_torch.softmax = fake_softmax
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForSequenceClassification = FakeAutoModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    prediction = LegalClauseClassifier(adapter_dir).predict("A valid clause.")

    assert prediction["predicted_category"] == LEGAL_CLAUSE_LABELS[-1]
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert sum(prediction["probabilities"].values()) == pytest.approx(
        1.0, abs=1e-3
    )
    assert LegalClauseClassifier(adapter_dir).name == "gpt2-lora"
    assert observed["tokenizer_path"] == str(adapter_dir)
    assert observed["local_files_only"] is True
    assert observed["base"] == "gpt2"
    assert observed["num_labels"] == 8
    assert observed["padding"] is True
    assert observed["attention_mask"] == [1, 1]

    import app.core as core

    monkeypatch.setattr(
        core, "milestone_3", lambda: LegalClauseClassifier(adapter_dir)
    )
    main(
        [
            "classify",
            "The supplier shall indemnify the customer against third-party claims.",
        ]
    )
    assert "[gpt2-lora]" in capsys.readouterr().out

    mapping_path = adapter_dir / "label_mapping.json"
    mapping_path.write_text(json.dumps({"id2label": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        LegalClauseClassifier(adapter_dir)._load()


def test_missing_adapter_uses_explicit_baseline_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.core as core

    def missing_adapter():
        raise FileNotFoundError("adapter not present")

    monkeypatch.setattr(core, "milestone_3", missing_adapter)
    classifier, backend = core.load_classifier()
    assert backend == "tfidf-logreg baseline"
    assert classifier.name == "tfidf-logreg baseline"


def test_classifier_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BaselineClauseClassifier().predict("  ")


def test_cli_generate_classify_predict_and_input_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["generate", "--seed", "the client", "--count", "3"])
    generated = capsys.readouterr().out
    assert generated.count("Generated:") == 3
    assert "tfidf-logreg baseline" in generated

    main(
        [
            "classify",
            "The supplier shall indemnify the customer against third-party claims.",
        ]
    )
    assert "tfidf-logreg baseline" in capsys.readouterr().out

    main(["predict", "the client", "--top-k", "3"])
    prediction_lines = capsys.readouterr().out.strip().splitlines()
    assert 1 <= len(prediction_lines) <= 3
    assert all("\t" in line for line in prediction_lines)

    with pytest.raises(SystemExit):
        main(["generate", "--count", "0"])
    with pytest.raises(SystemExit):
        main(["classify", "   "])

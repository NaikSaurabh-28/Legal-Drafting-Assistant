"""Fine-tune GPT-2 with a LoRA adapter for legal clause classification.

Run:
    pip install -r requirements-ml.txt
    python scripts/train_gpt2_adapter.py

Training uses only data/classifier_train.csv. The held-out
data/classifier_test.csv is reserved for evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clause_classifier import LEGAL_CLAUSE_LABELS

TRAIN_CSV = ROOT / "data" / "classifier_train.csv"
OUT = ROOT / "models" / "legal_clause_adapter"
BASE_MODEL = "gpt2"
EPOCHS = 15
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
MAX_LENGTH = 128
SEED = 42


def load_training_rows() -> list[dict[str, str]]:
    """Load and validate the labeled training CSV."""
    with TRAIN_CSV.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if (
            reader.fieldnames is None
            or set(reader.fieldnames) != {"text", "label"}
            or len(reader.fieldnames) != 2
        ):
            raise ValueError("Training CSV must contain only 'text' and 'label' columns.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No training examples found in {TRAIN_CSV}.")
    if any(
        not isinstance(row.get("text"), str)
        or not row["text"].strip()
        or not isinstance(row.get("label"), str)
        or not row["label"].strip()
        for row in rows
    ):
        raise ValueError("Training CSV contains an empty text or label.")
    rows = [
        {"text": row["text"].strip(), "label": row["label"].strip()}
        for row in rows
    ]
    labels = {row["label"] for row in rows}
    if labels != set(LEGAL_CLAUSE_LABELS):
        raise ValueError(
            "Training CSV must contain exactly the eight supported labels: "
            f"{', '.join(LEGAL_CLAUSE_LABELS)}."
        )
    return rows


def set_random_seed(seed: int = SEED) -> None:
    """Seed Python and PyTorch before initializing the classification head."""
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_base_model_and_tokenizer():
    """Load GPT-2 sequence classification model with the fixed label mapping."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    labels = list(LEGAL_CLAUSE_LABELS)
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.eos_token is None:
        raise RuntimeError("The GPT-2 tokenizer has no EOS token for padding.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def build_lora_config():
    """Configure LoRA for GPT-2 attention and preserve its classification head."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],
        modules_to_save=["score"],
        fan_in_fan_out=True,
    )


def configure_lora(model):
    """Wrap a sequence-classification model with the project's LoRA config."""
    from peft import get_peft_model

    return get_peft_model(model, build_lora_config())


def train_model(model, tokenizer, rows: list[dict[str, str]]) -> dict[str, object]:
    """Fine-tune only the supplied training rows and return run metadata."""
    import torch

    if not rows:
        raise ValueError("At least one training row is required.")
    if {row["label"] for row in rows} != set(LEGAL_CLAUSE_LABELS):
        raise ValueError(
            "Training rows must contain exactly the eight supported labels."
        )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id before training.")
    label2id = {label: index for index, label in enumerate(LEGAL_CLAUSE_LABELS)}

    model.print_trainable_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model.to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("LoRA model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )
    model.train()
    epoch_losses: list[float] = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(rows)).tolist()
        total_loss = 0.0
        batch_count = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = [rows[index] for index in order[start : start + BATCH_SIZE]]
            encoded = tokenizer(
                [row["text"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            if "attention_mask" not in encoded:
                raise RuntimeError("Tokenizer did not return an attention_mask.")
            labels_tensor = torch.tensor(
                [label2id[row["label"]] for row in batch],
                dtype=torch.long,
                device=device,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            optimizer.zero_grad()
            loss = model(**encoded, labels=labels_tensor).loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batch_count += 1
        mean_loss = total_loss / batch_count
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} - mean loss: {mean_loss:.4f}")

    return {"device": str(device), "epoch_mean_loss": epoch_losses}


def save_training_artifacts(
    model,
    tokenizer,
    rows: list[dict[str, str]],
    training_summary: dict[str, object],
    output_dir: str | Path = OUT,
    training_csv: str | Path = TRAIN_CSV,
) -> Path:
    """Save adapter, model/tokenizer config, label map, and run metadata."""
    output_path = Path(output_dir)
    training_path = Path(training_csv)
    label2id = {label: index for index, label in enumerate(LEGAL_CLAUSE_LABELS)}
    id2label = {str(index): label for label, index in label2id.items()}

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    model.get_base_model().config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    mapping = {
        "id2label": id2label,
        "label2id": label2id,
    }
    (output_path / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "training_config.json").write_text(
        json.dumps(
            {
                "base_model": BASE_MODEL,
                "num_labels": len(LEGAL_CLAUSE_LABELS),
                "labels": list(LEGAL_CLAUSE_LABELS),
                "seed": SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "lora": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "target_modules": ["c_attn"],
                    "modules_to_save": ["score"],
                    "fan_in_fan_out": True,
                },
                **mapping,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    training_metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "training_dataset": training_path.name,
        "training_dataset_sha256": hashlib.sha256(
            training_path.read_bytes()
        ).hexdigest(),
        "training_rows": len(rows),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "seed": SEED,
        "device": training_summary["device"],
        "epoch_mean_loss": training_summary["epoch_mean_loss"],
        "package_versions": {
            package: version(package) for package in ("torch", "transformers", "peft")
        },
    }
    (output_path / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    rows = load_training_rows()
    set_random_seed(SEED)
    tokenizer, model = load_base_model_and_tokenizer()
    model = configure_lora(model)
    summary = train_model(model, tokenizer, rows)
    output_path = save_training_artifacts(model, tokenizer, rows, summary)
    print(f"Adapter saved to: {output_path}")
    from app.clause_classifier import LegalClauseClassifier

    inference_check = LegalClauseClassifier(output_path)
    print(
        "Saved-adapter inference check:",
        inference_check.predict(
            "The supplier shall indemnify the customer against third-party claims."
        ),
    )
    print("Evaluate the held-out test set with: python -m app.cli evaluate")


if __name__ == "__main__":
    main()

"""One test per milestone."""
import pytest

from app import core
from app.cli import main


def test_milestone_1_corpus_is_tokenized():
    sentences = core.milestone_1()
    assert len(sentences) > 10
    assert all(isinstance(t, str) for t in sentences[0])
    assert all(t == t.lower() for t in sentences[0])


def test_milestone_2_ngram_generates_a_sentence():
    generator = core.milestone_2()
    sentence = generator.generate_sentence(max_length=20)
    assert sentence.strip()
    assert "<START>" not in sentence and "<END>" not in sentence and "<UNK>" not in sentence


def test_milestone_3_reports_missing_adapter(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.milestone_3(tmp_path)


def test_milestone_4_generate_then_classify():
    class FakeClassifier:
        def predict(self, text):
            return {"predicted_category": "demo", "confidence": 1.0}

    out = core.milestone_4(core.milestone_2(), FakeClassifier())
    assert out["sentence"] and out["classification"]["predicted_category"] == "demo"
    assert core.milestone_4(core.milestone_2())["classification"] is None


def test_milestone_5_classifier_beats_chance_on_held_out_clauses():
    report = core.milestone_5()
    assert report["test_size"] == 24
    assert report["accuracy"] > 1 / 8          # chance with 8 classes
    assert report["model"]


def test_train_and_test_csv_do_not_overlap():
    import csv
    texts = lambda f: {r["text"] for r in csv.DictReader(open(f, encoding="utf-8"))}
    assert not texts("data/classifier_train.csv") & texts("data/classifier_test.csv")


def test_ngram_evaluation():
    report = core.ngram_evaluation()
    assert set(report["models"]) >= {"1-gram", "2-gram", "3-gram"}
    assert report["models"]["3-gram"]["perplexity"] > 0


def test_cli_generate_runs(capsys):
    main(["generate"])
    assert "Generated:" in capsys.readouterr().out

"""Project logic, one function per milestone. Details live in the modules beside this file."""
from pathlib import Path

from app.clause_classifier import (
    AdapterArtifactError,
    BaselineClauseClassifier,
    LegalClauseClassifier,
)
from app.evaluation import evaluate_classifier, evaluate_ngram_models
from app.ngram_model import NGramModel
from app.preprocessing import clean_text, preprocess_corpus, tokenize
from app.text_generator import ClauseGenerator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "legal_corpus.txt"
TEST_CSV = ROOT / "data" / "classifier_test.csv"


# --- Milestone 1: Prepare a small legal text corpus for n-gram training.
def milestone_1(corpus_path=DEFAULT_CORPUS) -> list[list[str]]:
    """Load, lowercase and tokenize the corpus: one token list per sentence."""
    return preprocess_corpus(corpus_path)


# --- Milestone 2: Implement the n-gram next-word predictor.
def milestone_2(corpus_path=DEFAULT_CORPUS, n: int = 3) -> ClauseGenerator:
    """Train an n-gram model and wrap it in a sentence generator."""
    sentences = milestone_1(corpus_path)
    if not sentences:
        raise ValueError(f"Corpus contains no clauses: {corpus_path}")
    model = NGramModel(n=n, min_frequency=1)
    model.train(sentences)
    return ClauseGenerator(model=model, training_sentences=sentences)


# --- Milestone 3: Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.
def milestone_3(adapter_dir=None) -> LegalClauseClassifier:
    """Check for the adapter files. Raises FileNotFoundError if they are missing.

    The GPT-2 weights themselves load lazily on the first `predict` call.
    Create the adapter with `python scripts/train_gpt2_adapter.py`.
    """
    return LegalClauseClassifier(adapter_dir)


def load_classifier():
    """Return (classifier, description): the GPT-2 adapter if installed, else the TF-IDF baseline."""
    try:
        clf = milestone_3()
        clf._load()  # fail now, not mid-session, if torch/peft are missing
        return clf, clf.name
    except (FileNotFoundError, ImportError):
        baseline = BaselineClauseClassifier()
        return baseline, baseline.name


# --- Milestone 4: Create a CLI interface to generate text and then classify the generated text.
def milestone_4(generator: ClauseGenerator, classifier=None, seed: str = "", max_length: int = 25) -> dict:
    """Generate one sentence and classify it when a classifier is available."""
    context = tuple(tokenize(clean_text(seed))) or None
    sentence = generator.generate_sentence(start_context=context, max_length=max_length)
    result = {"sentence": sentence, "classification": None}
    if classifier is not None:
        result["classification"] = classifier.predict(sentence)
    return result


# --- Milestone 5: Evaluate classification accuracy on example clauses.
def milestone_5(classifier=None, test_csv=TEST_CSV) -> dict:
    """Accuracy, macro-F1 and confusion matrix on held-out labeled clauses.

    `data/classifier_test.csv` is never used for training. Evaluates the GPT-2
    adapter when installed, otherwise the baseline.
    """
    description = "custom"
    if classifier is None:
        classifier, description = load_classifier()
    report = evaluate_classifier(classifier, test_csv)
    report["model"] = description
    return report


def ngram_evaluation(corpus_path=DEFAULT_CORPUS) -> dict:
    """Held-out perplexity and next-token accuracy for unigram/bigram/trigram models."""
    return evaluate_ngram_models(corpus_path)


def evaluate_all_models(
    test_csv=TEST_CSV,
    corpus_path=DEFAULT_CORPUS,
) -> dict:
    """Evaluate the baseline and optional GPT-2 adapter on the same held-out set."""
    baseline = BaselineClauseClassifier()
    baseline_report = evaluate_classifier(baseline, test_csv)
    baseline_report["model"] = baseline.name
    baseline_report["backend"] = "tfidf-logreg"

    try:
        gpt2_classifier = milestone_3()
        gpt2_classifier._load()
    except (FileNotFoundError, ImportError, AdapterArtifactError) as error:
        gpt2_report = {
            "model": "GPT-2 + LoRA",
            "backend": "gpt2-lora",
            "status": "not_evaluated",
            "reason": str(error),
            "test_size": baseline_report["test_size"],
        }
    else:
        gpt2_report = evaluate_classifier(gpt2_classifier, test_csv)
        gpt2_report["model"] = "GPT-2 + LoRA"
        gpt2_report["backend"] = "gpt2-lora"
        gpt2_report["status"] = "evaluated"

    ngram_report = ngram_evaluation(corpus_path)
    for order, metrics in ngram_report["models"].items():
        metrics["model"] = f"from-scratch {order} language model"

    return {
        "ngram": ngram_report,
        "classifiers": {
            "tfidf_logreg_baseline": baseline_report,
            "gpt2_lora": gpt2_report,
        },
    }

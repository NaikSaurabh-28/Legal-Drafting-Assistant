
"""Utility functions for preparing legal text for NLP workflows."""

import re
from pathlib import Path


def load_corpus(file_path: str | Path) -> list[str]:
    """Load non-empty lines from a UTF-8 text file."""
    path = Path(file_path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Corpus file not found: {path}"
        ) from error

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def clean_text(text: str) -> str:
    """Lowercase text and normalize whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return re.findall(r"\w+|[^\w\s]", text)


def preprocess_corpus(
    file_path: str | Path
) -> list[list[str]]:
    """Load, clean, and tokenize every sentence."""
    sentences = load_corpus(file_path)

    return [
        tokenize(clean_text(sentence))
        for sentence in sentences
    ]


def build_vocabulary(
    tokenized_sentences: list[list[str]],
    include_unk: bool = False,
) -> list[str]:
    """Build a sorted vocabulary, optionally reserving an unknown token."""
    if not isinstance(include_unk, bool):
        raise TypeError("include_unk must be a boolean.")
    vocabulary = {
        token
        for sentence in tokenized_sentences
        for token in sentence
    }
    if include_unk:
        vocabulary.add("<UNK>")
    return sorted(vocabulary)


class LegalTextPreprocessor:
    """Prepare legal text for downstream NLP components."""

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)

    def segment_clauses(self, text: str) -> list[str]:
        """Split text into non-empty clause-like sentence segments.

        Newlines and sentence-ending punctuation are treated as boundaries.
        Abbreviations and numbered legal subclauses may require manual review.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+|\r?\n+", text.strip())
        return [
            re.sub(r"\s+", " ", part).strip()
            for part in parts
            if part.strip()
        ]


if __name__ == "__main__":
    tokenized_sentences = preprocess_corpus(
        "data/legal_corpus.txt"
    )

    vocabulary = build_vocabulary(tokenized_sentences)

    print(f"Number of sentences: {len(tokenized_sentences)}")
    print(f"First tokenized sentence: {tokenized_sentences[0]}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"First 20 vocabulary tokens: {vocabulary[:20]}")

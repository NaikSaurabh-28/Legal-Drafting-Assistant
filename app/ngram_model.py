"""N-gram language model for simple next-token prediction."""

import math
from collections import Counter


class NGramModel:
    """Count n-grams and estimate next-token probabilities."""

    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self, n: int = 3, alpha: float = 1.0, min_frequency: int = 2
    ) -> None:
        """Create an n-gram model.

        Args:
            n: Number of tokens in each n-gram; supported values are 1, 2, and 3.
            alpha: Non-negative smoothing value. ``alpha=0`` gives the
                maximum-likelihood estimate; positive values use add-alpha
                smoothing.
            min_frequency: Minimum training frequency for a token to keep its
                own vocabulary entry. Rarer tokens map to ``<UNK>``.

        Raises:
            ValueError: If parameters are invalid.
        """
        if isinstance(n, bool) or n not in {1, 2, 3}:
            raise ValueError("n must be 1, 2, or 3.")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha)
            or alpha < 0
        ):
            raise ValueError("alpha must be non-negative.")
        if (
            isinstance(min_frequency, bool)
            or not isinstance(min_frequency, int)
            or min_frequency < 1
        ):
            raise ValueError("min_frequency must be a positive integer.")

        self.n = n
        self.alpha = alpha
        self.min_frequency = min_frequency
        self.vocabulary: list[str] = []
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self._known_tokens: set[str] = set()
        self._is_trained = False

    def train(self, tokenized_sentences: list[list[str]]) -> None:
        """Count n-grams in tokenized sentences with boundary markers.

        Sentences receive ``n - 1`` start markers, or one start marker for a
        unigram model, and one end marker. Training tokens below
        ``min_frequency`` are represented by ``<UNK>``.

        Args:
            tokenized_sentences: Sentences represented as lists of tokens.

        Raises:
            ValueError: If there are no non-empty training sentences.
        """
        sentences = [sentence for sentence in tokenized_sentences if sentence]
        if not sentences:
            raise ValueError(
                "Train the model with at least one non-empty sentence."
            )

        self.ngram_counts.clear()
        self.context_counts.clear()
        token_counts = Counter(
            token
            for sentence in sentences
            for token in sentence
            if token not in {
                self.START_TOKEN,
                self.END_TOKEN,
                self.UNK_TOKEN,
            }
        )
        self._known_tokens = {
            token
            for token, count in token_counts.items()
            if count >= self.min_frequency
        }
        self.vocabulary = sorted(
            self._known_tokens | {self.UNK_TOKEN, self.END_TOKEN}
        )
        start_padding = max(1, self.n - 1)

        for sentence in sentences:
            normalized_sentence = [
                token if token in self._known_tokens else self.UNK_TOKEN
                for token in sentence
            ]
            padded_sentence = (
                [self.START_TOKEN] * start_padding
                + normalized_sentence
                + [self.END_TOKEN]
            )

            for target_index in range(start_padding, len(padded_sentence)):
                if self.n == 1:
                    context: tuple[str, ...] = ()
                else:
                    context = tuple(
                        padded_sentence[
                            max(0, target_index - (self.n - 1)) : target_index
                        ]
                    )
                target = padded_sentence[target_index]
                ngram = context + (target,)
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self._is_trained = True

    def normalize_token(self, token: str) -> str:
        """Map out-of-vocabulary tokens to ``<UNK>`` after training."""
        if not self._is_trained:
            raise RuntimeError("Train the model before normalizing tokens.")
        if token in {self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN}:
            return token
        return token if token in self._known_tokens else self.UNK_TOKEN

    def get_probability(
        self, word: str, context: tuple[str, ...]
    ) -> float:
        """Estimate a token probability with add-alpha smoothing.

        Args:
            word: Candidate next token.
            context: Tokens immediately preceding the candidate.

        Returns:
            The smoothed probability of ``word`` following ``context``.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("Train the model before requesting probabilities.")
        if word == self.START_TOKEN:
            raise ValueError("<START> is a context marker, not a prediction target.")

        word = self.normalize_token(word)
        context = tuple(self.normalize_token(token) for token in context)
        if self.n == 1:
            context = ()
        else:
            context = context[-(self.n - 1) :]

        ngram = context + (word,)
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        vocabulary_size = len(self.vocabulary)

        denominator = context_count + self.alpha * vocabulary_size
        if denominator == 0:
            return 0.0
        return (ngram_count + self.alpha) / denominator

    def predict_next(
        self,
        context: tuple[str, ...],
        top_k: int = 5,
        include_unk: bool = False,
    ) -> list[tuple[str, float]]:
        """Rank likely next tokens for a context.

        Args:
            context: Tokens immediately preceding the next token.
            top_k: Maximum number of predictions to return.
            include_unk: Whether to include ``<UNK>`` for evaluation. False
                by default so sentence generation does not emit it.

        Returns:
            Candidate tokens paired with probabilities, highest first.

        Raises:
            ValueError: If ``top_k`` is negative.
            RuntimeError: If the model has not been trained.
        """
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 0
        ):
            raise ValueError("top_k must be a non-negative integer.")
        if not isinstance(include_unk, bool):
            raise ValueError("include_unk must be a boolean.")
        if not self._is_trained:
            raise RuntimeError("Train the model before requesting predictions.")

        predictions = [
            (word, self.get_probability(word, context))
            for word in self.vocabulary
            if include_unk or word != self.UNK_TOKEN
        ]
        predictions.sort(key=lambda prediction: (-prediction[1], prediction[0]))
        return predictions[:top_k]


if __name__ == "__main__":
    from app.preprocessing import preprocess_corpus

    tokenized_sentences = preprocess_corpus("data/legal_corpus.txt")
    models = {
        n: NGramModel(n=n)
        for n in (1, 2, 3)
    }

    for model in models.values():
        model.train(tokenized_sentences)

    print(f"Training sentences: {len(tokenized_sentences)}")
    for n, model in models.items():
        print(f"\n{n}-gram vocabulary size: {len(model.vocabulary)}")
        context = () if n == 1 else ("the",) if n == 2 else ("the", "client")
        print(f"Sample predictions for {context}:")
        for word, probability in model.predict_next(context, top_k=5):
            print(f"  {word:<12} {probability:.6f}")

    probability = models[3].get_probability("must", ("the", "client"))
    print(f"\nP('must' | 'the client') = {probability:.6f}")
    print("\nAll three n-gram models trained successfully.")

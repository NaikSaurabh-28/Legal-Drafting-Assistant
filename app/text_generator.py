"""Sentence generation using a trained n-gram language model."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import random
import re

from app.ngram_model import NGramModel
from app.preprocessing import clean_text, tokenize


@dataclass(frozen=True)
class _BeamState:
    context: tuple[str, ...]
    generated: tuple[str, ...]
    visible: tuple[str, ...]
    seen_bigrams: frozenset[tuple[str, str]]
    score: float
    generated_words: int


class ClauseGenerator:
    """Generate legal-style text from a trained n-gram model."""

    MIN_GENERATED_WORDS = 5

    def __init__(
        self,
        model: NGramModel | None = None,
        start_context: tuple[str, ...] | None = None,
        max_length: int = 30,
        top_n: int = 3,
        sampling: bool = False,
        training_sentences: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Create a clause generator.

        Args:
            model: Trained model to use; defaults to a trigram model.
            start_context: Initial tokens, or boundary markers when omitted.
            max_length: Maximum number of generated tokens per call.
            top_n: Number of top predictions considered at each step.
            sampling: Whether to sample among the top predictions.
            training_sentences: Tokenized training clauses used to identify
                valid sentence-start continuations.

        Raises:
            ValueError: If ``max_length`` is negative or ``top_n`` is not
                positive.
        """
        self._validate_generation_options(max_length, top_n)
        if not isinstance(sampling, bool):
            raise ValueError("sampling must be a boolean.")

        self.model = model if model is not None else NGramModel(n=3)
        self.training_sentences = (
            tuple(
                tuple(token for token in sentence)
                for sentence in training_sentences
            )
            if training_sentences is not None
            else ()
        )
        if start_context is None:
            context_length = max(1, self.model.n - 1)
            self.start_context = ("<START>",) * context_length
        else:
            self.start_context = tuple(start_context)

        self.max_length = max_length
        self.top_n = top_n
        self.sampling = sampling

    def generate_sentence(
        self,
        start_context: tuple[str, ...] | None = None,
        max_length: int | None = None,
        top_n: int | None = None,
        sampling: bool | None = None,
    ) -> str:
        """Generate a sentence from an optional seed context.

        Args:
            start_context: Initial tokens; uses the configured context by
                default.
            max_length: Maximum generated token count; uses the configured
                limit by default.
            top_n: Number of top predictions to consider at each step.
            sampling: Whether to sample among candidates instead of always
                selecting the highest-ranked candidate.

        Returns:
            A completed sentence, or an empty string if no valid completion
            can be generated from the supplied context.

        Raises:
            ValueError: If ``max_length`` is negative or ``top_n`` is not
                positive.
            RuntimeError: If the underlying model has not been trained.
        """
        context = self.start_context if start_context is None else start_context
        token_limit = self.max_length if max_length is None else max_length
        candidate_limit = self.top_n if top_n is None else top_n
        should_sample = self.sampling if sampling is None else sampling

        self._validate_generation_options(token_limit, candidate_limit)
        if sampling is not None and not isinstance(sampling, bool):
            raise ValueError("sampling must be a boolean.")
        prefix = [
            token
            for token in context
            if token not in {"<START>", "<END>", "<UNK>"}
        ]
        generated = self._generate_tokens(
            context=context,
            max_length=token_limit,
            top_n=candidate_limit,
            sampling=should_sample,
        )
        if prefix and not generated and not self._ends_sentence(tuple(prefix)):
            return ""
        output_tokens = [
            token
            for token in prefix + generated
            if token not in {"<START>", "<END>", "<UNK>"}
        ]
        if output_tokens and not self._ends_sentence(tuple(output_tokens)):
            return ""
        return self._detokenize(output_tokens)

    def generate_clause(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate a clause continuing from a text prompt.

        Args:
            prompt: Text to use as the beginning and context for generation.
            max_tokens: Maximum number of tokens to generate after the prompt.

        Returns:
            The prompt followed by generated text.
        """
        prompt_tokens = tuple(tokenize(clean_text(prompt)))
        return self.generate_sentence(
            start_context=prompt_tokens or self.start_context,
            max_length=max_tokens,
        )

    def complete_draft(self, draft: str) -> str:
        """Append generated text to a partial draft.

        Args:
            draft: Partial legal draft to continue.

        Returns:
            The draft followed by generated text.
        """
        draft_tokens = tokenize(draft)
        context_tokens = tuple(tokenize(clean_text(draft)))
        context = context_tokens or self.start_context
        continuation = self._generate_tokens(
            context=context,
            max_length=self.max_length,
            top_n=self.top_n,
            sampling=self.sampling,
        )
        if not continuation and not self._ends_sentence(tuple(draft_tokens)):
            return ""
        output_tokens = draft_tokens + continuation
        if output_tokens and not self._ends_sentence(tuple(output_tokens)):
            return ""
        return self._detokenize(output_tokens)

    def _generate_tokens(
        self,
        context: tuple[str, ...],
        max_length: int,
        top_n: int,
        sampling: bool,
    ) -> list[str]:
        """Generate tokens until an end marker or the length limit.

        Args:
            context: Initial context for the model.
            max_length: Maximum number of generation steps.
            top_n: Number of ranked predictions to consider.
            sampling: Whether to sample from ranked candidates.

        Returns:
            Generated tokens, without boundary markers.
        """
        if hasattr(self.model, "ngram_counts") and hasattr(
            self.model, "vocabulary"
        ):
            return self._generate_with_beam_search(
                context=context,
                max_length=max_length,
                top_n=top_n,
                sampling=sampling,
            )
        return self._generate_greedily(
            context=context,
            max_length=max_length,
            top_n=top_n,
            sampling=sampling,
        )

    def _generate_with_beam_search(
        self,
        context: tuple[str, ...],
        max_length: int,
        top_n: int,
        sampling: bool,
    ) -> list[str]:
        """Search multiple observed continuations and return a completed clause."""
        visible_tokens = [
            token for token in context
            if token not in {"<START>", "<END>", "<UNK>"}
        ]
        beam = [
            _BeamState(
                context=tuple(context),
                generated=(),
                visible=tuple(visible_tokens),
                seen_bigrams=frozenset(
                    zip(visible_tokens, visible_tokens[1:])
                ),
                score=0.0,
                generated_words=0,
            )
        ]
        completed: list[_BeamState] = []
        sentence_start_continuations = self._sentence_start_continuations(
            visible_tokens
        )
        ngram_counts = getattr(self.model, "ngram_counts", {})
        token_frequencies: dict[str, int] = {}
        if hasattr(ngram_counts, "items"):
            for ngram, count in ngram_counts.items():
                if ngram:
                    token_frequencies[ngram[-1]] = (
                        token_frequencies.get(ngram[-1], 0) + count
                    )

        vocabulary = getattr(self.model, "vocabulary", ())
        candidate_limit = max(top_n, len(vocabulary))
        backoff_distributions = self._build_backoff_distributions()

        for step in range(max_length):
            next_beam: list[_BeamState] = []
            layer_completed: list[_BeamState] = []
            for state in beam:
                candidate_levels = self._generation_candidates(
                    state.context,
                    candidate_limit,
                    token_frequencies,
                    backoff_distributions,
                )
                eligible = self._eligible_candidates(
                    state,
                    candidate_levels,
                )
                if step == 0 and sentence_start_continuations:
                    eligible = [
                        (token, probability)
                        for token, probability in eligible
                        if token in sentence_start_continuations
                    ]
                for token, probability in eligible[: max(6, top_n * 2)]:
                    if probability <= 0:
                        continue
                    score = state.score + math.log(probability)
                    if token == "<END>":
                        if self._ends_sentence(state.visible):
                            layer_completed.append(replace(state, score=score))
                        continue

                    generated_words = state.generated_words + int(
                        self._is_word_token(token)
                    )
                    new_generated = state.generated + (token,)
                    new_visible = state.visible + (token,)
                    updated = _BeamState(
                        context=self._advance_context(state.context, token),
                        generated=new_generated,
                        visible=new_visible,
                        seen_bigrams=state.seen_bigrams
                        | frozenset({(state.visible[-1], token)})
                        if state.visible
                        else state.seen_bigrams,
                        score=score,
                        generated_words=generated_words,
                    )
                    if token in {".", "?", "!"}:
                        if generated_words >= self.MIN_GENERATED_WORDS:
                            layer_completed.append(updated)
                        continue
                    next_beam.append(updated)

            if layer_completed:
                completed.extend(layer_completed)
            if not next_beam:
                break
            next_beam.sort(
                key=lambda state: (
                    -self._beam_score(state),
                    state.generated,
                )
            )
            beam = next_beam[: max(32, top_n * 8)]

        if not completed:
            return []

        completed.sort(
            key=lambda state: (
                -self._beam_score(state),
                state.generated,
            )
        )
        if sampling and len(completed) > 1:
            candidates = completed[: max(3, top_n)]
            best_score = self._beam_score(candidates[0])
            sampling_temperature = 0.15
            weights = [
                math.exp(
                    max(
                        -50.0,
                        (self._beam_score(state) - best_score)
                        / sampling_temperature,
                    )
                )
                for state in candidates
            ]
            selected = random.choices(candidates, weights=weights, k=1)[0]
        else:
            selected = completed[0]
        return list(selected.generated)

    def _generate_greedily(
        self,
        context: tuple[str, ...],
        max_length: int,
        top_n: int,
        sampling: bool,
    ) -> list[str]:
        """Generate from lightweight compatible models without count metadata."""
        generated: list[str] = []
        current_context = tuple(context)
        visible_tokens = [
            token
            for token in current_context
            if token not in {"<START>", "<END>", "<UNK>"}
        ]
        seen_bigrams = set(zip(visible_tokens, visible_tokens[1:]))
        for _ in range(max_length):
            candidates = self.model.predict_next(
                current_context, top_k=top_n
            )
            eligible = [
                (token, probability)
                for token, probability in candidates
                if token not in {"<START>", "<UNK>"}
                and (
                    token == "<END>"
                    or self._is_word_token(token)
                    or self._can_emit_punctuation(
                        token, visible_tokens, generated
                    )
                )
                and not (
                    self._is_word_token(token)
                    and visible_tokens
                    and token == visible_tokens[-1]
                )
                and not (
                    visible_tokens
                    and (visible_tokens[-1], token) in seen_bigrams
                )
            ]
            if not eligible:
                break
            if sampling and len(eligible) > 1:
                token = random.choices(
                    [item[0] for item in eligible],
                    weights=[item[1] for item in eligible],
                    k=1,
                )[0]
            else:
                token = eligible[0][0]
            if token == "<END>":
                break
            if visible_tokens:
                seen_bigrams.add((visible_tokens[-1], token))
            generated.append(token)
            visible_tokens.append(token)
            if token in {".", "?", "!"}:
                break
            current_context = self._advance_context(current_context, token)
        return generated

    def _eligible_candidates(
        self,
        state: _BeamState,
        candidate_levels: list[tuple[list[tuple[str, float]], set[str]]],
    ) -> list[tuple[str, float]]:
        """Filter invalid, unsupported, or repetitive transitions."""
        for candidates, supported_tokens in candidate_levels:
            eligible: list[tuple[str, float]] = []
            for token, probability in candidates:
                if token not in supported_tokens:
                    continue
                if token == "<END>":
                    if self._ends_sentence(state.visible):
                        eligible.append((token, probability))
                    continue
                if token in {"<START>", "<UNK>"} or not token:
                    continue
                if (
                    self._is_word_token(token)
                    and state.visible
                    and self._is_word_token(state.visible[-1])
                    and token == state.visible[-1]
                ):
                    continue
                if not self._is_word_token(token) and not self._can_emit_punctuation(
                    token, list(state.visible), list(state.generated)
                ):
                    continue
                if (
                    len(state.visible) >= 1
                    and (state.visible[-1], token) in state.seen_bigrams
                ):
                    continue
                eligible.append((token, probability))
            if eligible:
                return eligible
        return []

    @staticmethod
    def _ends_sentence(tokens: tuple[str, ...]) -> bool:
        return bool(tokens and tokens[-1] in {".", "?", "!"})

    def _sentence_start_continuations(
        self, prompt_tokens: list[str]
    ) -> set[str]:
        """Find next-token evidence for prompts matching training sentence starts."""
        if not prompt_tokens or not self.training_sentences:
            return set()

        prompt = tuple(token.lower() for token in prompt_tokens)
        continuations: set[str] = set()
        for sentence in self.training_sentences:
            normalized = tuple(token.lower() for token in sentence)
            offsets = [0]
            if prompt[0] == "the" and len(prompt) > 1:
                offsets.append(1)
            for offset in offsets:
                matched = prompt[offset:]
                if (
                    len(normalized) > len(matched)
                    and normalized[: len(matched)] == matched
                ):
                    continuations.add(normalized[len(matched)])
        return continuations

    def _advance_context(
        self, context: tuple[str, ...], token: str
    ) -> tuple[str, ...]:
        context_length = self.model.n - 1
        if context_length == 0:
            return ()
        return (context + (token,))[-context_length:]

    @staticmethod
    def _beam_score(state: _BeamState) -> float:
        """Use length-normalized log probability to compare complete paths."""
        token_count = max(1, len(state.generated))
        return state.score / (token_count**0.7)

    def _generation_candidates(
        self,
        context: tuple[str, ...],
        top_k: int,
        token_frequencies: dict[str, int],
        backoff_distributions: dict[
            int, dict[tuple[str, ...], dict[str, int]]
        ],
    ) -> list[tuple[list[tuple[str, float]], set[str]]]:
        """Rank observed N-gram transitions and back off for sparse contexts."""
        ngram_counts = getattr(self.model, "ngram_counts", None)
        if ngram_counts is None or not hasattr(ngram_counts, "items"):
            candidates = self.model.predict_next(context, top_k=top_k)
            return [
                (
                    self._sort_candidates(candidates, token_frequencies),
                    {token for token, _ in candidates},
                )
            ]

        normalize_token = getattr(self.model, "normalize_token", None)
        if callable(normalize_token):
            normalized_context = tuple(
                normalize_token(item) for item in context
            )
        else:
            normalized_context = context
        context_length = self.model.n - 1
        normalized_context = (
            ()
            if context_length == 0
            else normalized_context[-context_length:]
        )
        observed_counts = backoff_distributions.get(
            context_length, {}
        ).get(normalized_context, {})
        supported_tokens = {
            token
            for token, count in observed_counts.items()
            if count > 0 and token not in {"<START>", "<UNK>"}
        }
        levels: list[tuple[list[tuple[str, float]], set[str]]] = []
        if supported_tokens:
            candidates = [
                (token, self.model.get_probability(token, normalized_context))
                for token in supported_tokens
            ]
            levels.append(
                (
                    self._sort_candidates(candidates, token_frequencies),
                    supported_tokens,
                )
            )

        levels.extend(
            self._backoff_candidates(
                normalized_context,
                token_frequencies,
                backoff_distributions,
            )
        )
        return levels

    def _build_backoff_distributions(
        self,
    ) -> dict[int, dict[tuple[str, ...], dict[str, int]]]:
        """Aggregate trained higher-order counts into lower-order contexts."""
        ngram_counts = getattr(self.model, "ngram_counts", {})
        distributions: dict[int, dict[tuple[str, ...], dict[str, int]]] = {}
        for ngram, count in ngram_counts.items():
            context = ngram[:-1]
            token = ngram[-1]
            for context_size in range(self.model.n - 1, -1, -1):
                suffix = context[-context_size:] if context_size else ()
                context_counts = distributions.setdefault(
                    context_size, {}
                ).setdefault(suffix, {})
                context_counts[token] = context_counts.get(token, 0) + count
        return distributions

    def _backoff_candidates(
        self,
        context: tuple[str, ...],
        token_frequencies: dict[str, int],
        distributions: dict[int, dict[tuple[str, ...], dict[str, int]]],
    ) -> list[tuple[list[tuple[str, float]], set[str]]]:
        """Return observed shorter-context distributions for generation."""
        vocabulary = getattr(self.model, "vocabulary", ())
        if not vocabulary:
            return []

        levels: list[tuple[list[tuple[str, float]], set[str]]] = []
        for context_size in range(self.model.n - 2, -1, -1):
            suffix = context[-context_size:] if context_size else ()
            counts = distributions.get(context_size, {}).get(suffix, {})
            if not counts:
                continue

            alpha = self.model.alpha
            denominator = sum(counts.values()) + alpha * len(vocabulary)
            candidates = [
                (token, (counts.get(token, 0) + alpha) / denominator)
                for token in vocabulary
                if token != "<UNK>"
            ]
            levels.append(
                (
                    self._sort_candidates(candidates, token_frequencies),
                    set(counts),
                )
            )
        return levels

    @classmethod
    def _sort_candidates(
        cls,
        candidates: list[tuple[str, float]],
        token_frequencies: dict[str, int],
    ) -> list[tuple[str, float]]:
        """Break smoothed probability ties in favor of observed words."""
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate[1],
                cls._candidate_type_order(candidate[0]),
                -token_frequencies.get(candidate[0], 0),
                candidate[0],
            ),
        )

    def _can_emit_punctuation(
        self,
        token: str,
        visible_tokens: list[str],
        generated: list[str],
    ) -> bool:
        """Require words around punctuation and prevent rapid repetition."""
        if not self._has_word(visible_tokens):
            return False
        if visible_tokens and not self._is_word_token(visible_tokens[-1]):
            return False

        repeat_window = max(4, self.model.n * 2)
        recent_tokens = visible_tokens[-repeat_window:]
        if token in recent_tokens:
            return False
        if token in {"'", "’"} and token in generated:
            return False
        return True

    @staticmethod
    def _is_word_token(token: str) -> bool:
        """Return whether a token contains a word character."""
        return any(character.isalnum() or character == "_" for character in token)

    @classmethod
    def _has_word(cls, tokens: list[str]) -> bool:
        return any(cls._is_word_token(token) for token in tokens)

    @classmethod
    def _candidate_type_order(cls, token: str) -> int:
        """Prefer words over punctuation when smoothed scores are tied."""
        if token == "<END>":
            return 2
        return 0 if cls._is_word_token(token) else 1

    @staticmethod
    def _validate_generation_options(max_length: int, top_n: int) -> None:
        """Check that generation limits are valid.

        Args:
            max_length: Maximum number of generation steps.
            top_n: Number of top candidates to consider.

        Raises:
            ValueError: If either generation limit is invalid.
        """
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 0
        ):
            raise ValueError("max_length must be a non-negative integer.")
        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or top_n <= 0
        ):
            raise ValueError("top_n must be a positive integer.")

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        """Join tokens into readable text without spaces before punctuation.

        Args:
            tokens: Tokens to join.

        Returns:
            Detokenized text.
        """
        output_tokens = [
            token
            for token in tokens
            if token not in {"<START>", "<END>", "<UNK>"}
        ]
        while output_tokens and output_tokens[-1] in {"'", "’", ",", ";", ":"}:
            output_tokens.pop()
        if not ClauseGenerator._has_word(output_tokens):
            return ""

        text = " ".join(output_tokens)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(
            r"\s+(['’])\s+(?=(?:s|t|d|ll|re|ve|m)\b)",
            r"\1",
            text,
        )
        text = re.sub(r"([([{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        return text.strip()


if __name__ == "__main__":
    from app.preprocessing import preprocess_corpus

    corpus = preprocess_corpus("data/legal_corpus.txt")
    model = NGramModel(n=3)
    model.train(corpus)

    generator = ClauseGenerator(
        model=model,
        start_context=("the", "client"),
        max_length=20,
        top_n=3,
    )
    print(generator.generate_sentence())

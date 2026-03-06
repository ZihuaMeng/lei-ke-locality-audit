"""Detect direct logical fractures in CoT traces with a DeBERTa NLI model."""

from __future__ import annotations

import re
from pprint import pprint
from typing import Any

import torch
from transformers import pipeline

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
CONTRADICTION_LABEL = "contradiction"
CONCLUSION_MARKERS = (
    "therefore",
    "thus",
    "hence",
    "so",
    "consequently",
    "as a result",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "therefore",
    "this",
    "to",
    "was",
    "were",
}


def load_nli_model():
    """Load the DeBERTa-v3 NLI backbone through a Hugging Face pipeline."""
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,
        device=device,
    )


def _split_sentences(cot_text: str) -> list[str]:
    text = cot_text.strip()
    if not text:
        return []

    try:
        import nltk

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            sentences = []
    except ImportError:
        sentences = []

    if not sentences:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _starts_with_conclusion_marker(sentence: str) -> bool:
    normalized = re.sub(r"^[^A-Za-z]+", "", sentence).lower()
    return normalized.startswith(CONCLUSION_MARKERS)


def _get_contradiction_score(premise: str, hypothesis: str, model) -> float:
    scores = model({"text": premise, "text_pair": hypothesis})
    if scores and isinstance(scores[0], list):
        scores = scores[0]

    for item in scores:
        if item["label"].lower() == CONTRADICTION_LABEL:
            return float(item["score"])

    return 0.0


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
        if token not in STOPWORDS
    }


def detect_fractures(cot_text: str, model) -> dict[str, Any]:
    """
    Detect contradictions across a chain of thought.

    Adjacent sentences are compared by default. For conclusion-style steps
    (for example those starting with "Therefore"), the current sentence is
    evaluated against the accumulated prior context instead of only the
    immediately previous sentence, which better matches inference-style CoT.
    """
    sentences = _split_sentences(cot_text)
    fracture_pairs: list[tuple[str, str, float]] = []

    for idx in range(len(sentences) - 1):
        hypothesis = sentences[idx + 1]
        premise = sentences[idx]

        if _starts_with_conclusion_marker(hypothesis):
            premise = " ".join(sentences[: idx + 1])
        elif not (_content_tokens(premise) & _content_tokens(hypothesis)):
            continue

        contradiction_score = _get_contradiction_score(premise, hypothesis, model)
        if contradiction_score > 0.5:
            fracture_pairs.append((premise, hypothesis, contradiction_score))

    return {
        "fracture_found": bool(fracture_pairs),
        "fracture_count": len(fracture_pairs),
        "fracture_pairs": fracture_pairs,
    }


def _run_self_test() -> None:
    example_a = """The Eiffel Tower is located in Rome, Italy.
Paris is the capital of France.
Therefore, the Eiffel Tower is in France."""

    example_b = """Marie Curie was born in Warsaw.
Warsaw is in Poland.
Therefore, Marie Curie was born in Poland."""

    print(f"Loading NLI model: {MODEL_NAME}")
    model = load_nli_model()

    examples = [
        ("Example A (should find fracture)", example_a),
        ("Example B (should find NO fracture)", example_b),
    ]

    for title, text in examples:
        print("=" * 72)
        print(title)
        print("Input:")
        print(text)
        print("Result:")
        pprint(detect_fractures(text, model), sort_dicts=False)


if __name__ == "__main__":
    _run_self_test()

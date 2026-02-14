import math
import re
from collections import Counter
from typing import Any


def tokenize(text: str) -> list[str]:
    """Tokenize text using a simple regex for diversity metrics."""
    return re.findall(r"\w+", text.lower())


def calculate_ttr(text: str) -> float:
    """Calculate Type-Token Ratio (TTR).

    TTR = Number of unique word types / Total number of tokens.
    Higher TTR indicates greater lexical diversity.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)


def calculate_mattr(text: str, window_size: int = 50) -> float:
    """Calculate Moving-Average Type-Token Ratio (MATTR).

    Computes TTRs within a moving window and averages them.
    More robust to text length than raw TTR.
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    if len(tokens) <= window_size:
        return calculate_ttr(text)

    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        ttrs.append(len(set(window)) / window_size)

    return sum(ttrs) / len(ttrs)


def calculate_ngram_entropy(text: str, n: int = 1) -> float:
    """Calculate n-gram Shannon entropy.

    Formula: H = -sum(p_i * log2(p_i))
    where p_i is the frequency of n-gram i.
    """
    tokens = tokenize(text)
    if not tokens or len(tokens) < n:
        return 0.0

    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def calculate_jaccard_diversity(texts: list[str]) -> float:
    """Calculate average Jaccard distance between all pairs of texts.

    1.0 means texts share no common words.
    0.0 means texts are identical in their word sets.
    """
    if len(texts) < 2:
        return 1.0

    token_sets = [set(tokenize(t)) for t in texts]
    distances = []

    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            s1 = token_sets[i]
            s2 = token_sets[j]

            union = s1 | s2
            if not union:
                dist = 0.0
            else:
                intersection = s1 & s2
                dist = 1.0 - (len(intersection) / len(union))
            distances.append(dist)

    return sum(distances) / len(distances)


def calculate_self_bleu_approx(texts: list[str], n: int = 2) -> float:
    """Simplified Self-BLEU based on n-gram overlap.

    Returns (1.0 - average_overlap).
    """
    if len(texts) < 2:
        return 1.0

    ngram_sets = []
    for t in texts:
        tokens = tokenize(t)
        if len(tokens) < n:
            ngram_sets.append(set())
            continue
        ngrams = {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
        ngram_sets.append(ngrams)

    distances = []
    for i in range(len(ngram_sets)):
        for j in range(i + 1, len(ngram_sets)):
            s1 = ngram_sets[i]
            s2 = ngram_sets[j]

            union = s1 | s2
            if not union:
                dist = 0.0
            else:
                intersection = s1 & s2
                dist = 1.0 - (len(intersection) / len(union))
            distances.append(dist)

    return sum(distances) / len(distances)


class DiversityMetrics(Any):
    """Aggregate diversity report for a set of samples."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.ttr = sum(calculate_ttr(t) for t in texts) / len(texts) if texts else 0.0
        self.entropy = (
            sum(calculate_ngram_entropy(t) for t in texts) / len(texts)
            if texts
            else 0.0
        )
        self.jaccard = calculate_jaccard_diversity(texts)
        self.self_bleu_approx = calculate_self_bleu_approx(texts)

    def summary(self) -> dict[str, float]:
        return {
            "avg_ttr": self.ttr,
            "avg_entropy": self.entropy,
            "jaccard_diversity": self.jaccard,
            "self_bleu_diversity": self.self_bleu_approx,
        }

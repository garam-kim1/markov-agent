import pytest

from markov_agent.engine.diversity import (
    DiversityMetrics,
    calculate_jaccard_diversity,
    calculate_ngram_entropy,
    calculate_ttr,
)
from markov_agent.engine.sampler import execute_diverse_sampling


def test_ttr():
    text = "apple banana apple orange"
    # tokens: apple, banana, apple, orange (4)
    # types: apple, banana, orange (3)
    # TTR = 3/4 = 0.75
    assert calculate_ttr(text) == 0.75


def test_ngram_entropy():
    text = "a b a b"
    # unigrams: a, b, a, b (4 total, 2 each)
    # p(a) = 0.5, p(b) = 0.5
    # H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = -(-0.5 - 0.5) = 1.0
    assert calculate_ngram_entropy(text, n=1) == 1.0


def test_jaccard_diversity():
    texts = ["apple banana", "apple banana", "cherry date"]
    # dist(0, 1) = 0.0 (identical)
    # dist(0, 2) = 1.0 (no overlap)
    # dist(1, 2) = 1.0 (no overlap)
    # avg = (0+1+1)/3 = 0.666...
    assert calculate_jaccard_diversity(texts) == pytest.approx(0.666666, abs=1e-5)


@pytest.mark.asyncio
async def test_diverse_sampling():
    # Mock generation factory
    # First attempt: low diversity
    # Second attempt (higher temp): high diversity
    call_count = 0

    async def mock_gen(cfg):
        nonlocal call_count
        call_count += 1
        temp = cfg.get("temperature", 0.7)
        if temp < 0.9:
            return "same same same"
        return f"diverse {random_str()}"

    def random_str():
        import uuid

        return uuid.uuid4().hex[:4]

    results = await execute_diverse_sampling(
        mock_gen,
        base_config={"temperature": 0.7},
        k=3,
        diversity_threshold=0.5,
        max_retries=1,
    )

    # Should have retried at least once if diversity was low
    assert call_count > 3
    assert len(results) == 3


def test_diversity_metrics_summary():
    texts = ["hello world", "foo bar", "hello foo"]
    metrics = DiversityMetrics(texts)
    summary = metrics.summary()
    assert "avg_ttr" in summary
    assert "jaccard_diversity" in summary
    assert summary["jaccard_diversity"] > 0

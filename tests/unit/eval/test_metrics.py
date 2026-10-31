"""Retrieval metrics, bootstrap intervals and percentiles, checked against hand computations."""

import math

import pytest

from lexgraph.eval.metrics import (
    article_ranking,
    cutoffs,
    estimate,
    hit_at_k,
    latency_stats,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap,
    quantile,
    recall_at_k,
    retrieval_scores,
    unit_recall_at_k,
)

# Two expected articles (grade 2) and one acceptable article (grade 1).
RELEVANCE = {"CLT:59": 2, "CLT:58": 2, "CF:7": 1}
# The acceptable article is ranked 2nd, one expected article 3rd, the other is missing.
RANKING = ["CDC:49", "CF:7", "CLT:59", "LGPD:18"]


def test_article_ranking_collapses_chunks_keeping_the_best_rank() -> None:
    chunks = ["CF:5#3", "CF:5#0", "CLT:1#0", "CF:5", "CDC:49#2"]
    assert article_ranking(chunks) == ["CF:5", "CLT:1", "CDC:49"]


# CF:5 inciso XI and CF:7 incisos XIII and XVI are needed; CDC:49 is a single whole chunk.
CHUNKS = [
    ("CF:5", ["X", "XI", "XII"]),
    ("CF:7", ["caput", "I", "XII"]),
    ("CDC:49", ["*"]),
    ("CF:7", ["XIII", "XIV"]),
]
UNITS = {"CF:5": ["XI"], "CF:7": ["XIII", "XVI"], "CDC:49": ["caput"]}


@pytest.mark.parametrize(("k", "expected"), [(1, 1 / 4), (2, 1 / 4), (3, 2 / 4), (4, 3 / 4)])
def test_unit_recall_counts_units_covered_by_the_retrieved_slices(k: int, expected: float) -> None:
    assert unit_recall_at_k(CHUNKS, UNITS, k) == expected


def test_unit_recall_needs_expected_units() -> None:
    assert unit_recall_at_k([], {"CF:5": ["XI"]}, 5) == 0.0
    with pytest.raises(ValueError, match="without expected units"):
        unit_recall_at_k(CHUNKS, {}, 5)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.0), (3, 1.0), (10, 1.0)])
def test_hit_counts_only_expected_articles(k: int, expected: float) -> None:
    assert hit_at_k(RANKING, RELEVANCE, k) == expected


@pytest.mark.parametrize(("k", "expected"), [(2, 0.0), (3, 0.5), (4, 0.5)])
def test_recall_is_the_share_of_expected_articles_found(k: int, expected: float) -> None:
    assert recall_at_k(RANKING, RELEVANCE, k) == expected


def test_recall_requires_expected_articles() -> None:
    with pytest.raises(ValueError, match="without expected articles"):
        recall_at_k(RANKING, {"CF:7": 1}, 5)


@pytest.mark.parametrize(("k", "expected"), [(2, 0.0), (3, 1 / 3), (10, 1 / 3)])
def test_mrr_is_the_reciprocal_rank_of_the_first_expected_article(k: int, expected: float) -> None:
    assert mrr_at_k(RANKING, RELEVANCE, k) == pytest.approx(expected)


def test_ndcg_uses_graded_linear_gains() -> None:
    dcg = 0 / math.log2(2) + 1 / math.log2(3) + 2 / math.log2(4)
    ideal = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, RELEVANCE, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(RANKING, RELEVANCE, 3) == pytest.approx(0.4335, abs=1e-4)


def test_ndcg_gives_partial_credit_to_acceptable_articles() -> None:
    assert ndcg_at_k(["CF:7"], RELEVANCE, 1) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("ranking", "relevance", "expected"),
    [
        (["CLT:59", "CLT:58", "CF:7"], RELEVANCE, 1.0),
        (["CDC:49", "LGPD:18"], RELEVANCE, 0.0),
        ([], RELEVANCE, 0.0),
        (["CLT:59"], {}, 0.0),
    ],
)
def test_ndcg_bounds(ranking: list[str], relevance: dict[str, int], expected: float) -> None:
    assert ndcg_at_k(ranking, relevance, 5) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("k", "expected"), [(10, (1, 3, 5, 10)), (5, (1, 3, 5)), (4, (1, 3, 4)), (1, (1,))]
)
def test_cutoffs_include_k(k: int, expected: tuple[int, ...]) -> None:
    assert cutoffs(k) == expected


def test_retrieval_scores_cover_every_cutoff() -> None:
    scores = retrieval_scores(RANKING, RELEVANCE, 5)

    assert set(scores) == {
        *(f"{metric}@{k}" for metric in ("hit", "recall", "ndcg") for k in (1, 3, 5)),
        "mrr@5",
    }
    assert scores["hit@3"] == 1.0
    assert scores["recall@5"] == 0.5
    assert scores["mrr@5"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(("q", "expected"), [(0.0, 10.0), (0.5, 25.0), (0.95, 38.5), (1.0, 40.0)])
def test_quantile_interpolates_linearly(q: float, expected: float) -> None:
    assert quantile([10.0, 20.0, 30.0, 40.0], q) == pytest.approx(expected)


def test_quantile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        quantile([], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        quantile([1.0], 1.5)


def test_estimate_of_constant_values_has_a_degenerate_interval() -> None:
    result = estimate([0.5] * 5)
    assert (result.mean, result.low, result.high, result.n) == (0.5, 0.5, 0.5, 5)


def test_estimate_interval_brackets_the_mean_and_is_reproducible() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    result = estimate(values)

    assert result.mean == pytest.approx(0.75)
    assert 0.0 <= result.low < result.mean < result.high <= 1.0
    assert estimate(values) == result
    assert estimate(values, confidence=0.5).high <= result.high


def test_estimate_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        estimate([])
    with pytest.raises(ValueError, match="confidence"):
        estimate([1.0], confidence=1.0)


def test_paired_bootstrap_of_identical_runs_shows_no_difference() -> None:
    result = paired_bootstrap([0.2, 0.4, 0.9], [0.2, 0.4, 0.9])
    assert (result.mean, result.low, result.high) == (0.0, 0.0, 0.0)
    assert result.share_not_better == 1.0


def test_paired_bootstrap_of_a_constant_gain() -> None:
    result = paired_bootstrap([0.0, 0.5, 0.2], [0.5, 1.0, 0.7])
    assert result.mean == pytest.approx(0.5)
    assert (result.low, result.high) == pytest.approx((0.5, 0.5))
    assert result.share_not_better == 0.0
    assert result.n == 3


def test_paired_bootstrap_share_not_better_matches_the_resampling_probability() -> None:
    # Diffs are [1, 0, 1, 0]: a resample shows no gain only when all four draws are 0,
    # which happens with probability (1/2) ** 4.
    result = paired_bootstrap([0.0] * 4, [1.0, 0.0, 1.0, 0.0])
    assert result.mean == pytest.approx(0.5)
    assert result.share_not_better == pytest.approx(0.0625, abs=0.02)


def test_paired_bootstrap_rejects_unpaired_samples() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap([1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap([], [])


def test_latency_stats() -> None:
    stats = latency_stats([40.0, 10.0, 30.0, 20.0])
    assert (stats.p50, stats.p95, stats.mean, stats.n) == pytest.approx((25.0, 38.5, 25.0, 4))
    with pytest.raises(ValueError, match="no latencies"):
        latency_stats([])

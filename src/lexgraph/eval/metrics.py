"""Article-level retrieval metrics, bootstrap confidence intervals and latency percentiles.

Rankings are scored per article, not per chunk: chunk hits are collapsed to their article
keeping the best rank (several slices of one long article are one hit). Relevance is graded:
expected articles have grade 2 and acceptable ones grade 1. Hit, recall and MRR count only the
expected articles; nDCG (linear gains, as in trec_eval) gives partial credit to acceptable ones.

Collapsing credits any slice of a long article, even one about another inciso, so questions
that need a specific unit of a long article (``"CF:5"`` inciso XI) also get a unit recall: the
share of those units covered by the retrieved slices.

Intervals use the percentile bootstrap with a fixed seed, so reports are reproducible.
"""

import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

REQUIRED_GRADE = 2
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 13
WHOLE_ARTICLE = "*"
"""Unit label of a chunk that holds its whole article."""

Relevance = Mapping[str, int]


class Estimate(BaseModel):
    """Mean of per-question values with a bootstrap confidence interval."""

    model_config = ConfigDict(frozen=True)

    mean: float
    low: float
    high: float
    n: int


class PairedDifference(BaseModel):
    """Mean per-question difference (candidate minus baseline) with a bootstrap interval."""

    model_config = ConfigDict(frozen=True)

    mean: float
    low: float
    high: float
    n: int
    share_not_better: float = Field(
        description="Share of bootstrap resamples where the candidate is not better (diff <= 0)."
    )


class LatencyStats(BaseModel):
    """Latency percentiles in milliseconds."""

    model_config = ConfigDict(frozen=True)

    p50: float
    p95: float
    mean: float
    n: int


def article_ranking(ids: Iterable[str]) -> list[str]:
    """Collapse chunk ids (``"CF:5#3"``) or article ids to unique article ids, best rank first."""
    return list(dict.fromkeys(item.partition("#")[0] for item in ids))


def _required(relevance: Relevance) -> set[str]:
    return {article_id for article_id, grade in relevance.items() if grade >= REQUIRED_GRADE}


def hit_at_k(ranking: Sequence[str], relevance: Relevance, k: int) -> float:
    """1.0 when an expected article is among the first ``k`` results, else 0.0."""
    required = _required(relevance)
    return float(any(article_id in required for article_id in ranking[:k]))


def recall_at_k(ranking: Sequence[str], relevance: Relevance, k: int) -> float:
    """Share of the expected articles found among the first ``k`` results."""
    required = _required(relevance)
    if not required:
        msg = "recall is undefined without expected articles"
        raise ValueError(msg)
    return len(required.intersection(ranking[:k])) / len(required)


def mrr_at_k(ranking: Sequence[str], relevance: Relevance, k: int) -> float:
    """Reciprocal rank of the first expected article within the first ``k`` results."""
    required = _required(relevance)
    for rank, article_id in enumerate(ranking[:k], start=1):
        if article_id in required:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevance: Relevance, k: int) -> float:
    """Normalized discounted cumulative gain with graded relevance and linear gains."""
    dcg = sum(
        relevance.get(article_id, 0) / math.log2(rank + 1)
        for rank, article_id in enumerate(ranking[:k], start=1)
    )
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    return dcg / ideal if ideal else 0.0


def unit_recall_at_k(
    chunks: Sequence[tuple[str, Sequence[str]]], expected_units: Mapping[str, Sequence[str]], k: int
) -> float:
    """Share of the expected units covered by the first ``k`` chunks.

    ``chunks`` are ``(article_id, units)`` pairs in rank order; ``expected_units`` maps an
    article id to the unit labels an answer needs (``{"CF:5": ["XI"]}``). A chunk holding the
    whole article covers all of its units.
    """
    required = {
        (article_id, unit) for article_id, units in expected_units.items() for unit in units
    }
    if not required:
        msg = "unit recall is undefined without expected units"
        raise ValueError(msg)
    covered: set[tuple[str, str]] = set()
    for article_id, units in chunks[:k]:
        whole = WHOLE_ARTICLE in units
        covered.update(
            (wanted_article, unit)
            for wanted_article, unit in required
            if wanted_article == article_id and (whole or unit in units)
        )
    return len(covered) / len(required)


def cutoffs(k: int) -> tuple[int, ...]:
    """Cutoffs reported for a search that returned up to ``k`` passages: 1, 3, 5 and ``k``."""
    return tuple(sorted({c for c in (1, 3, 5) if c < k} | {k}))


def retrieval_scores(ranking: Sequence[str], relevance: Relevance, k: int) -> dict[str, float]:
    """Hit, recall and nDCG at every cutoff plus MRR at ``k`` for one deduplicated ranking."""
    scores: dict[str, float] = {}
    for cutoff in cutoffs(k):
        scores[f"hit@{cutoff}"] = hit_at_k(ranking, relevance, cutoff)
        scores[f"recall@{cutoff}"] = recall_at_k(ranking, relevance, cutoff)
        scores[f"ndcg@{cutoff}"] = ndcg_at_k(ranking, relevance, cutoff)
    scores[f"mrr@{k}"] = mrr_at_k(ranking, relevance, k)
    return scores


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Quantile ``q`` of ascending values, interpolating linearly (numpy's default method)."""
    if not sorted_values:
        msg = "quantile of an empty sequence"
        raise ValueError(msg)
    if not 0.0 <= q <= 1.0:
        msg = f"quantile must be in [0, 1], got {q}"
        raise ValueError(msg)
    position = q * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _bootstrap_means(values: Sequence[float], resamples: int, seed: int) -> list[float]:
    # Seeded statistical resampling, not a security use of randomness.
    rng = random.Random(seed)  # noqa: S311
    size = len(values)
    return sorted(statistics.fmean(rng.choices(values, k=size)) for _ in range(resamples))


def _interval(means: Sequence[float], confidence: float) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        msg = f"confidence must be in (0, 1), got {confidence}"
        raise ValueError(msg)
    tail = (1.0 - confidence) / 2
    return quantile(means, tail), quantile(means, 1.0 - tail)


def estimate(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """Mean of ``values`` with a percentile-bootstrap confidence interval."""
    if not values:
        msg = "cannot estimate the mean of an empty sample"
        raise ValueError(msg)
    low, high = _interval(_bootstrap_means(values, resamples, seed), confidence)
    return Estimate(mean=statistics.fmean(values), low=low, high=high, n=len(values))


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedDifference:
    """Compare two configurations scored on the same questions, in the same order.

    Resampling the per-question differences keeps the pairing, which makes the interval much
    tighter than comparing two independent intervals.
    """
    if len(baseline) != len(candidate):
        msg = "paired samples must have the same length"
        raise ValueError(msg)
    if not baseline:
        msg = "cannot compare empty samples"
        raise ValueError(msg)
    differences = [new - old for old, new in zip(baseline, candidate, strict=True)]
    means = _bootstrap_means(differences, resamples, seed)
    low, high = _interval(means, confidence)
    return PairedDifference(
        mean=statistics.fmean(differences),
        low=low,
        high=high,
        n=len(differences),
        share_not_better=sum(mean <= 0 for mean in means) / len(means),
    )


def latency_stats(values_ms: Sequence[float]) -> LatencyStats:
    """Median, 95th percentile and mean of latencies in milliseconds."""
    ordered = sorted(values_ms)
    if not ordered:
        msg = "no latencies to summarize"
        raise ValueError(msg)
    return LatencyStats(
        p50=quantile(ordered, 0.5),
        p95=quantile(ordered, 0.95),
        mean=statistics.fmean(ordered),
        n=len(ordered),
    )

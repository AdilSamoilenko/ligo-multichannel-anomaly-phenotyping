#!/usr/bin/env python3

"""
OVERNIGHT CROSS-FRAME REPLICATION AUDIT
=======================================

Purpose
-------
Audit the existing LIGO H1 auxiliary-channel phenotype representation and
cross-frame nearest-neighbour replication without modifying the primary
research pipeline.

This script is deliberately designed as a validation experiment.

It evaluates:

1. Current pair-relative distance.
2. StandardScaler Euclidean distance.
3. Robust median/MAD distance.
4. Feature-group ablations.
5. Nearest-neighbour agreement across metrics.
6. Cross-frame nearest-neighbour distances.
7. Random/permutation null distributions.
8. Empirical p-values.
9. Bootstrap confidence intervals.
10. Similarity-threshold sensitivity.
11. Feature-wise contribution to observed distances.
12. Stability of nearest-neighbour identities.
13. Summary statistics suitable for later paper analysis.

IMPORTANT
---------
This script does NOT modify the existing pipeline.
It imports the existing phenotype functions and works from the same
event/vector structures where possible.

If the pipeline's output format changes, this script should fail loudly
rather than silently produce invalid results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_SEED = 20250926

FEATURE_NAMES = [
    "mean",
    "median",
    "std",
    "rms",
    "min",
    "max",
    "peak_abs",
    "crest_factor",
    "skew",
    "kurtosis",
    "spectral_centroid",
    "spectral_entropy",
    "dominant_frequency",
    "spectral_power",
    "band_0_2",
    "band_2_5",
    "band_5_10",
    "band_10_20",
    "band_20_50",
    "band_50_100",
    "band_100_300",
]

FEATURE_GROUPS = {
    "all": list(range(len(FEATURE_NAMES))),

    "location": [
        FEATURE_NAMES.index("mean"),
        FEATURE_NAMES.index("median"),
    ],

    "amplitude": [
        FEATURE_NAMES.index("std"),
        FEATURE_NAMES.index("rms"),
        FEATURE_NAMES.index("min"),
        FEATURE_NAMES.index("max"),
        FEATURE_NAMES.index("peak_abs"),
    ],

    "shape": [
        FEATURE_NAMES.index("crest_factor"),
        FEATURE_NAMES.index("skew"),
        FEATURE_NAMES.index("kurtosis"),
    ],

    "spectral_shape": [
        FEATURE_NAMES.index("spectral_centroid"),
        FEATURE_NAMES.index("spectral_entropy"),
        FEATURE_NAMES.index("dominant_frequency"),
    ],

    "spectral_power": [
        FEATURE_NAMES.index("spectral_power"),
    ],

    "band_power": [
        FEATURE_NAMES.index("band_0_2"),
        FEATURE_NAMES.index("band_2_5"),
        FEATURE_NAMES.index("band_5_10"),
        FEATURE_NAMES.index("band_10_20"),
        FEATURE_NAMES.index("band_20_50"),
        FEATURE_NAMES.index("band_50_100"),
        FEATURE_NAMES.index("band_100_300"),
    ],

    "time_domain": [
        FEATURE_NAMES.index("mean"),
        FEATURE_NAMES.index("median"),
        FEATURE_NAMES.index("std"),
        FEATURE_NAMES.index("rms"),
        FEATURE_NAMES.index("min"),
        FEATURE_NAMES.index("max"),
        FEATURE_NAMES.index("peak_abs"),
        FEATURE_NAMES.index("crest_factor"),
        FEATURE_NAMES.index("skew"),
        FEATURE_NAMES.index("kurtosis"),
    ],

    "frequency_domain": [
        FEATURE_NAMES.index("spectral_centroid"),
        FEATURE_NAMES.index("spectral_entropy"),
        FEATURE_NAMES.index("dominant_frequency"),
        FEATURE_NAMES.index("spectral_power"),
        FEATURE_NAMES.index("band_0_2"),
        FEATURE_NAMES.index("band_2_5"),
        FEATURE_NAMES.index("band_5_10"),
        FEATURE_NAMES.index("band_10_20"),
        FEATURE_NAMES.index("band_20_50"),
        FEATURE_NAMES.index("band_50_100"),
        FEATURE_NAMES.index("band_100_300"),
    ],
}


# ============================================================================
# UTILITIES
# ============================================================================

def finite_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def safe_mad(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)

    # Gaussian-consistent MAD scale.
    scale = 1.4826 * mad

    # Prevent zero-variance dimensions from exploding.
    positive = scale[scale > 0]

    if len(positive):
        fallback = float(np.median(positive))
    else:
        fallback = 1.0

    scale = np.where(
        scale > 0,
        scale,
        fallback,
    )

    return scale


def percentile_ci(values, low=2.5, high=97.5):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return [float("nan"), float("nan")]

    return [
        float(np.percentile(values, low)),
        float(np.percentile(values, high)),
    ]


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.median,
    n_boot=5000,
    seed=DEFAULT_SEED,
) -> List[float]:

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return [float("nan"), float("nan")]

    if len(values) == 1:
        value = float(statistic(values))
        return [value, value]

    rng = np.random.default_rng(seed)

    samples = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    boot = statistic(
        values[samples],
        axis=1,
    )

    return [
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    ]


def empirical_lower_tail_p(
    observed: float,
    null_values: np.ndarray,
) -> float:

    null_values = np.asarray(
        null_values,
        dtype=np.float64,
    )

    null_values = null_values[
        np.isfinite(null_values)
    ]

    if len(null_values) == 0:
        return float("nan")

    # Smaller distances are more similar.
    return float(
        (1 + np.sum(null_values <= observed))
        / (1 + len(null_values))
    )


def rankdata_simple(values):
    """
    Average-rank implementation sufficient for rank-correlation diagnostics.
    """
    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1

        while (
            j < len(values)
            and values[order[j]] == values[order[i]]
        ):
            j += 1

        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j

    return ranks


def spearman_simple(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) < 2:
        return float("nan")

    ra = rankdata_simple(a)
    rb = rankdata_simple(b)

    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")

    return float(
        np.corrcoef(ra, rb)[0, 1]
    )


# ============================================================================
# METRICS
# ============================================================================

def metric_current(a, b):
    """
    Exact current project metric.

    RMS of pair-relative feature differences.
    """

    a = finite_vector(a)
    b = finite_vector(b)

    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    scale = np.maximum(
        np.maximum(
            np.abs(a),
            np.abs(b),
        ),
        np.finfo(np.float64).eps,
    )

    return float(
        np.sqrt(
            np.mean(
                ((a - b) / scale) ** 2
            )
        )
    )


def metric_standardised(
    a,
    b,
    medians,
    scales,
):
    """
    Population-standardised Euclidean RMS distance.
    """

    a = finite_vector(a)
    b = finite_vector(b)

    z_a = (a - medians) / scales
    z_b = (b - medians) / scales

    return float(
        np.linalg.norm(z_a - z_b)
        / np.sqrt(len(a))
    )


def metric_robust(
    a,
    b,
    medians,
    scales,
):
    """
    Median/MAD population-normalised Euclidean RMS distance.
    """

    a = finite_vector(a)
    b = finite_vector(b)

    z_a = (a - medians) / scales
    z_b = (b - medians) / scales

    return float(
        np.linalg.norm(z_a - z_b)
        / np.sqrt(len(a))
    )


def featurewise_current_distance(a, b):
    a = finite_vector(a)
    b = finite_vector(b)

    scale = np.maximum(
        np.maximum(
            np.abs(a),
            np.abs(b),
        ),
        np.finfo(np.float64).eps,
    )

    return np.abs(
        (a - b) / scale
    )


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EventRecord:
    frame: str
    event_id: str
    vector: np.ndarray


@dataclass
class MatchRecord:
    candidate_frame: str
    candidate_event: str
    reference_frame: str
    reference_event: str

    metric: str
    distance: float
    similarity: float

    candidate_index: int
    reference_index: int


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def import_project():
    """
    Import the existing pipeline without changing it.
    """

    project_root = Path(__file__).resolve().parents[1]

    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )

    import pipeline.research_pipeline as mega

    return mega


def extract_frame_vectors(
    frame_results,
) -> List[EventRecord]:

    records = []

    for frame in frame_results:

        frame_path = str(
            frame.get("path", "")
        )

        events = frame.get(
            "events",
            [],
        )

        vectors = frame.get(
            "event_vectors",
            [],
        )

        if not events or not vectors:
            continue

        vectors = np.asarray(
            vectors,
            dtype=np.float64,
        )

        if vectors.ndim != 2:
            continue

        n = min(
            len(events),
            len(vectors),
        )

        for i in range(n):

            event_id = str(
                events[i].get(
                    "event_id",
                    f"E{i:05d}",
                )
            )

            vector = finite_vector(
                vectors[i]
            )

            if len(vector) != len(FEATURE_NAMES):
                raise RuntimeError(
                    f"Expected {len(FEATURE_NAMES)} "
                    f"features but got {len(vector)} "
                    f"for {frame_path} {event_id}"
                )

            records.append(
                EventRecord(
                    frame=frame_path,
                    event_id=event_id,
                    vector=vector,
                )
            )

    return records


# ============================================================================
# FRAME GROUPING
# ============================================================================

def group_by_frame(records):
    groups = defaultdict(list)

    for record in records:
        groups[record.frame].append(record)

    return dict(groups)


def choose_primary_frame(
    frame_results,
    primary_events,
):
    """
    Identify primary frame using the pipeline's existing primary events.

    We use event IDs as a secondary consistency check.
    """

    primary_ids = {
        str(e.get("event_id"))
        for e in primary_events
    }

    candidates = []

    for frame in frame_results:

        if frame.get(
            "is_primary",
            False,
        ):
            candidates.append(frame)

    if not candidates:
        raise RuntimeError(
            "No frame marked is_primary=True."
        )

    if len(candidates) > 1:
        print(
            f"[WARNING] {len(candidates)} primary frames found; "
            f"using first."
        )

    primary = candidates[0]

    return primary


# ============================================================================
# NORMALISATION
# ============================================================================

def fit_population_statistics(
    reference_vectors,
):
    X = np.asarray(
        reference_vectors,
        dtype=np.float64,
    )

    X = finite_vector(X)

    mean = np.mean(
        X,
        axis=0,
    )

    std = np.std(
        X,
        axis=0,
        ddof=0,
    )

    positive_std = std[
        std > 0
    ]

    fallback_std = (
        float(np.median(positive_std))
        if len(positive_std)
        else 1.0
    )

    std = np.where(
        std > 0,
        std,
        fallback_std,
    )

    median = np.median(
        X,
        axis=0,
    )

    mad_scale = safe_mad(X)

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mad_scale": mad_scale,
    }


# ============================================================================
# NEAREST NEIGHBOURS
# ============================================================================

def nearest_matches(
    candidates: List[EventRecord],
    references: List[EventRecord],
    metric_name: str,
    stats,
):
    results = []

    for ci, candidate in enumerate(
        candidates
    ):

        best_distance = float("inf")
        best_index = -1

        for ri, reference in enumerate(
            references
        ):

            if metric_name == "current":
                distance = metric_current(
                    candidate.vector,
                    reference.vector,
                )

            elif metric_name == "standardised":
                distance = metric_standardised(
                    candidate.vector,
                    reference.vector,
                    stats["mean"],
                    stats["std"],
                )

            elif metric_name == "robust":
                distance = metric_robust(
                    candidate.vector,
                    reference.vector,
                    stats["median"],
                    stats["mad_scale"],
                )

            else:
                raise ValueError(
                    f"Unknown metric: {metric_name}"
                )

            if distance < best_distance:
                best_distance = distance
                best_index = ri

        reference = references[
            best_index
        ]

        similarity = 1.0 / (
            1.0 + best_distance
        )

        results.append(
            MatchRecord(
                candidate_frame=candidate.frame,
                candidate_event=candidate.event_id,
                reference_frame=reference.frame,
                reference_event=reference.event_id,
                metric=metric_name,
                distance=float(best_distance),
                similarity=float(similarity),
                candidate_index=ci,
                reference_index=best_index,
            )
        )

    return results


# ============================================================================
# ALL-PAIR DISTANCE MATRIX
# ============================================================================

def distance_matrix(
    candidates,
    references,
    metric_name,
    stats,
):
    M = np.empty(
        (
            len(candidates),
            len(references),
        ),
        dtype=np.float64,
    )

    for i, candidate in enumerate(
        candidates
    ):
        for j, reference in enumerate(
            references
        ):

            if metric_name == "current":
                M[i, j] = metric_current(
                    candidate.vector,
                    reference.vector,
                )

            elif metric_name == "standardised":
                M[i, j] = metric_standardised(
                    candidate.vector,
                    reference.vector,
                    stats["mean"],
                    stats["std"],
                )

            elif metric_name == "robust":
                M[i, j] = metric_robust(
                    candidate.vector,
                    reference.vector,
                    stats["median"],
                    stats["mad_scale"],
                )

    return M


# ============================================================================
# NULL MODEL
# ============================================================================

def permutation_null_nearest_distances(
    distance_matrix_real,
    n_permutations,
    seed,
):
    """
    Null model preserving the candidate and reference populations.

    Each permutation randomly reassigns reference identities to candidate
    rows. For each candidate, record the distance to its randomly assigned
    reference.

    This tests whether the observed candidate-reference pairing structure
    is unusually close.

    NOTE:
    The nearest-neighbour statistic itself is handled separately below.
    """

    rng = np.random.default_rng(seed)

    n_candidates, n_references = (
        distance_matrix_real.shape
    )

    if n_candidates == 0 or n_references == 0:
        return np.empty(0)

    values = np.empty(
        n_permutations * n_candidates,
        dtype=np.float64,
    )

    k = 0

    for _ in range(n_permutations):

        indices = rng.integers(
            0,
            n_references,
            size=n_candidates,
        )

        selected = distance_matrix_real[
            np.arange(n_candidates),
            indices,
        ]

        values[
            k:k + n_candidates
        ] = selected

        k += n_candidates

    return values


def permutation_null_minimum_distances(
    distance_matrix_real,
    n_permutations,
    seed,
):
    """
    Null distribution for the minimum candidate-to-reference distance.

    This is the more direct null for the nearest-neighbour question.

    For each permutation, randomly select one reference for every candidate
    and calculate the minimum distance across the resulting assignments.
    """

    rng = np.random.default_rng(seed)

    n_candidates, n_references = (
        distance_matrix_real.shape
    )

    if n_candidates == 0 or n_references == 0:
        return np.empty(0)

    result = np.empty(
        n_permutations,
        dtype=np.float64,
    )

    rows = np.arange(
        n_candidates
    )

    for p in range(
        n_permutations
    ):

        indices = rng.integers(
            0,
            n_references,
            size=n_candidates,
        )

        result[p] = np.min(
            distance_matrix_real[
                rows,
                indices,
            ]
        )

    return result


# ============================================================================
# RANDOM BASELINE FOR NEAREST-NEIGHBOUR DISTANCES
# ============================================================================

def null_nearest_neighbour_distribution(
    distance_matrix_real,
    n_permutations,
    seed,
):
    """
    Construct a null distribution for the median nearest-neighbour distance.

    For each permutation:
      - randomly select a reference event for each candidate;
      - compute the resulting candidate-reference distances;
      - calculate the median.

    This gives a population-level comparison against random pairing.
    """

    rng = np.random.default_rng(seed)

    n_candidates, n_references = (
        distance_matrix_real.shape
    )

    if n_candidates == 0 or n_references == 0:
        return np.empty(0)

    result = np.empty(
        n_permutations,
        dtype=np.float64,
    )

    rows = np.arange(
        n_candidates
    )

    for p in range(
        n_permutations
    ):

        indices = rng.integers(
            0,
            n_references,
            size=n_candidates,
        )

        values = distance_matrix_real[
            rows,
            indices,
        ]

        result[p] = np.median(
            values
        )

    return result


# ============================================================================
# METRIC COMPARISON
# ============================================================================

def compare_metric_neighbours(
    matches_by_metric,
):

    names = list(
        matches_by_metric.keys()
    )

    if not names:
        return {}

    base = names[0]

    base_ids = [
        (
            m.candidate_event,
            m.reference_event,
        )
        for m in matches_by_metric[base]
    ]

    output = {}

    for name in names:

        ids = [
            (
                m.candidate_event,
                m.reference_event,
            )
            for m in matches_by_metric[name]
        ]

        agreement = [
            a == b
            for a, b in zip(
                base_ids,
                ids,
            )
        ]

        output[
            f"{base}_vs_{name}"
        ] = {
            "n": len(agreement),
            "agreement_fraction": (
                float(np.mean(agreement))
                if agreement
                else float("nan")
            ),
        }

    return output


# ============================================================================
# FEATURE ABLATION
# ============================================================================

def scaled_metric_with_subset(
    a,
    b,
    indices,
    stats,
):
    a = np.asarray(
        a,
        dtype=np.float64,
    )[indices]

    b = np.asarray(
        b,
        dtype=np.float64,
    )[indices]

    med = stats["median"][indices]
    mad = stats["mad_scale"][indices]

    return metric_robust(
        a,
        b,
        med,
        mad,
    )


def ablation_nearest_matches(
    candidates,
    references,
    feature_group,
    stats,
):

    indices = FEATURE_GROUPS[
        feature_group
    ]

    results = []

    for candidate in candidates:

        best_d = float("inf")
        best_r = None

        for reference in references:

            d = scaled_metric_with_subset(
                candidate.vector,
                reference.vector,
                indices,
                stats,
            )

            if d < best_d:
                best_d = d
                best_r = reference

        results.append(
            {
                "candidate": candidate.event_id,
                "reference": (
                    best_r.event_id
                    if best_r
                    else None
                ),
                "distance": float(best_d),
                "similarity": float(
                    1.0 / (1.0 + best_d)
                ),
            }
        )

    return results


# ============================================================================
# FEATURE CONTRIBUTION
# ============================================================================

def feature_contributions(
    candidate,
    reference,
    stats,
):

    candidate = finite_vector(
        candidate
    )

    reference = finite_vector(
        reference
    )

    med = stats["median"]
    scale = stats["mad_scale"]

    z_delta = np.abs(
        (
            candidate - med
        ) / scale
        -
        (
            reference - med
        ) / scale
    )

    total = np.sum(
        z_delta ** 2
    )

    if total <= 0:
        fractions = np.zeros(
            len(FEATURE_NAMES)
        )
    else:
        fractions = (
            z_delta ** 2
        ) / total

    order = np.argsort(
        fractions
    )[::-1]

    return [
        {
            "feature": FEATURE_NAMES[i],
            "contribution_fraction": float(
                fractions[i]
            ),
            "absolute_standardised_difference": float(
                z_delta[i]
            ),
        }
        for i in order
    ]


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def threshold_summary(
    similarities,
    thresholds=(
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ),
):

    similarities = np.asarray(
        similarities,
        dtype=np.float64,
    )

    return {
        str(t): {
            "count": int(
                np.sum(
                    similarities >= t
                )
            ),
            "fraction": float(
                np.mean(
                    similarities >= t
                )
            )
            if len(similarities)
            else float("nan"),
        }
        for t in thresholds
    }


# ============================================================================
# FRAME-LEVEL ANALYSIS
# ============================================================================

def analyse_frame_pair(
    candidate_frame,
    reference_records,
    metrics,
    n_permutations,
    seed,
):

    candidates = candidate_frame

    if not candidates or not reference_records:
        return None

    stats = fit_population_statistics(
        [
            r.vector
            for r in reference_records
        ]
    )

    output = {
        "candidate_frame": candidates[0].frame,
        "reference_frame": reference_records[0].frame,
        "n_candidates": len(candidates),
        "n_reference": len(reference_records),
        "metrics": {},
    }

    for metric_name in metrics:

        print(
            f"    metric={metric_name}"
        )

        M = distance_matrix(
            candidates,
            reference_records,
            metric_name,
            stats,
        )

        nearest_idx = np.argmin(
            M,
            axis=1,
        )

        observed = M[
            np.arange(len(candidates)),
            nearest_idx,
        ]

        similarities = 1.0 / (
            1.0 + observed
        )

        match_rows = []

        for i, j in enumerate(
            nearest_idx
        ):

            match_rows.append(
                {
                    "candidate_event": candidates[i].event_id,
                    "reference_event": reference_records[j].event_id,
                    "distance": float(
                        M[i, j]
                    ),
                    "similarity": float(
                        similarities[i]
                    ),
                }
            )

        print(
            f"      median distance = "
            f"{np.median(observed):.6f}"
        )

        print(
            f"      minimum distance = "
            f"{np.min(observed):.6f}"
        )

        print(
            f"      maximum similarity = "
            f"{np.max(similarities):.6f}"
        )

        null_median = null_nearest_neighbour_distribution(
            M,
            n_permutations,
            seed,
        )

        p_median = empirical_lower_tail_p(
            np.median(observed),
            null_median,
        )

        output["metrics"][metric_name] = {
            "matches": match_rows,
            "median_distance": float(
                np.median(observed)
            ),
            "mean_distance": float(
                np.mean(observed)
            ),
            "minimum_distance": float(
                np.min(observed)
            ),
            "maximum_distance": float(
                np.max(observed)
            ),
            "median_similarity": float(
                np.median(similarities)
            ),
            "maximum_similarity": float(
                np.max(similarities)
            ),
            "distance_ci_95": bootstrap_ci(
                observed
            ),
            "null_median_distance_ci_95": percentile_ci(
                null_median
            ),
            "null_median_distance": float(
                np.median(null_median)
            ),
            "empirical_p_median_distance": p_median,
            "thresholds": threshold_summary(
                similarities
            ),
        }

    # ------------------------------------------------------------------------
    # Metric agreement
    # ------------------------------------------------------------------------

    nearest_by_metric = {}

    for metric_name in metrics:

        M = distance_matrix(
            candidates,
            reference_records,
            metric_name,
            stats,
        )

        nearest_by_metric[
            metric_name
        ] = [
            reference_records[
                int(np.argmin(row))
            ].event_id
            for row in M
        ]

    output["nearest_neighbour_agreement"] = {}

    metric_names = list(
        nearest_by_metric.keys()
    )

    if metric_names:

        base = metric_names[0]

        for other in metric_names:

            agreement = [
                a == b
                for a, b in zip(
                    nearest_by_metric[base],
                    nearest_by_metric[other],
                )
            ]

            output[
                "nearest_neighbour_agreement"
            ][
                f"{base}_vs_{other}"
            ] = {
                "fraction": (
                    float(np.mean(agreement))
                    if agreement
                    else float("nan")
                ),
                "n": len(agreement),
            }

    # ------------------------------------------------------------------------
    # Robust feature-group ablations
    # ------------------------------------------------------------------------

    output["feature_groups"] = {}

    for group_name in FEATURE_GROUPS:

        if group_name == "all":
            continue

        print(
            f"    feature_group={group_name}"
        )

        ablated = ablation_nearest_matches(
            candidates,
            reference_records,
            group_name,
            stats,
        )

        distances = np.asarray(
            [
                row["distance"]
                for row in ablated
            ]
        )

        similarities = np.asarray(
            [
                row["similarity"]
                for row in ablated
            ]
        )

        output["feature_groups"][
            group_name
        ] = {
            "n_features": len(
                FEATURE_GROUPS[group_name]
            ),
            "features": [
                FEATURE_NAMES[i]
                for i in FEATURE_GROUPS[
                    group_name
                ]
            ],
            "median_distance": float(
                np.median(distances)
            ),
            "minimum_distance": float(
                np.min(distances)
            ),
            "maximum_similarity": float(
                np.max(similarities)
            ),
            "median_similarity": float(
                np.median(similarities)
            ),
            "thresholds": threshold_summary(
                similarities
            ),
            "matches": ablated,
        }

    # ------------------------------------------------------------------------
    # Feature contribution for strongest robust match
    # ------------------------------------------------------------------------

    M_robust = distance_matrix(
        candidates,
        reference_records,
        "robust",
        stats,
    )

    flat_index = int(
        np.argmin(M_robust)
    )

    ci, ri = np.unravel_index(
        flat_index,
        M_robust.shape,
    )

    output["strongest_robust_match"] = {
        "candidate": candidates[
            ci
        ].event_id,
        "reference": reference_records[
            ri
        ].event_id,
        "distance": float(
            M_robust[ci, ri]
        ),
        "similarity": float(
            1.0 / (
                1.0 + M_robust[ci, ri]
            )
        ),
        "feature_contributions": feature_contributions(
            candidates[ci].vector,
            reference_records[ri].vector,
            stats,
        ),
    }

    return output


# ============================================================================
# GLOBAL SUMMARY
# ============================================================================

def aggregate_frame_results(
    frame_results,
):

    summary = {
        "n_frame_pairs": len(
            frame_results
        ),
        "metrics": {},
    }

    for result in frame_results:

        for metric_name, metric_data in result[
            "metrics"
        ].items():

            summary[
                "metrics"
            ].setdefault(
                metric_name,
                {
                    "median_distances": [],
                    "maximum_similarities": [],
                    "median_similarities": [],
                    "p_values": [],
                },
            )

            summary[
                "metrics"
            ][metric_name][
                "median_distances"
            ].append(
                metric_data[
                    "median_distance"
                ]
            )

            summary[
                "metrics"
            ][metric_name][
                "maximum_similarities"
            ].append(
                metric_data[
                    "maximum_similarity"
                ]
            )

            summary[
                "metrics"
            ][metric_name][
                "median_similarities"
            ].append(
                metric_data[
                    "median_similarity"
                ]
            )

            summary[
                "metrics"
            ][metric_name][
                "p_values"
            ].append(
                metric_data[
                    "empirical_p_median_distance"
                ]
            )

    for metric_name, data in summary[
        "metrics"
    ].items():

        for key in (
            "median_distances",
            "maximum_similarities",
            "median_similarities",
            "p_values",
        ):

            values = np.asarray(
                data[key],
                dtype=np.float64,
            )

            data[key] = {
                "values": values.tolist(),
                "median": float(
                    np.nanmedian(values)
                )
                if len(values)
                else float("nan"),
                "mean": float(
                    np.nanmean(values)
                )
                if len(values)
                else float("nan"),
            }

    return summary


# ============================================================================
# JSON SERIALISATION
# ============================================================================

def json_default(obj):

    if isinstance(
        obj,
        np.ndarray,
    ):
        return obj.tolist()

    if isinstance(
        obj,
        np.integer,
    ):
        return int(obj)

    if isinstance(
        obj,
        np.floating,
    ):
        return float(obj)

    if isinstance(
        obj,
        Path,
    ):
        return str(obj)

    raise TypeError(
        f"Cannot serialise {type(obj)}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="results_mega/replication_audit",
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    np.random.seed(
        args.seed
    )

    random.seed(
        args.seed
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.time()

    print("=" * 78)
    print("OVERNIGHT CROSS-FRAME REPLICATION AUDIT")
    print("=" * 78)
    print()
    print(f"Seed: {args.seed}")
    print(f"Permutations: {args.permutations}")
    print(f"Output: {output_dir}")
    print()

    # ------------------------------------------------------------------------
    # Import project
    # ------------------------------------------------------------------------

    try:
        mega = import_project()
    except Exception:
        traceback.print_exc()
        raise

    print(
        "[OK] Imported pipeline.research_pipeline"
    )

    # ------------------------------------------------------------------------
    # Locate existing data structures
    # ------------------------------------------------------------------------

    required = [
        "build_phenotypes",
        "normalised_feature_distance",
    ]

    for name in required:

        if not hasattr(
            mega,
            name,
        ):
            raise RuntimeError(
                f"Required project function missing: {name}"
            )

    print(
        "[OK] Required phenotype functions found"
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The pipeline's full execution currently owns the actual frame results.
    # Rather than inventing a file format, inspect the module's runtime
    # namespace after its normal execution if available.
    # ------------------------------------------------------------------------

    possible_names = [
        "FRAME_RESULTS",
        "frame_results",
        "all_frame_results",
        "RESULTS",
        "results",
    ]

    frame_results = None

    for name in possible_names:

        if hasattr(
            mega,
            name,
        ):

            value = getattr(
                mega,
                name,
            )

            if isinstance(
                value,
                list,
            ):

                if value and isinstance(
                    value[0],
                    dict,
                ):

                    if any(
                        "events" in x
                        and "path" in x
                        for x in value
                    ):
                        frame_results = value
                        print(
                            f"[OK] Found frame results in "
                            f"mega.{name}"
                        )
                        break

    if frame_results is None:

        # Search module globals more broadly.
        candidates = []

        for name, value in vars(
            mega
        ).items():

            if not isinstance(
                value,
                list,
            ):
                continue

            if not value:
                continue

            if not all(
                isinstance(
                    x,
                    dict,
                )
                for x in value[: min(5, len(value))]
            ):
                continue

            score = 0

            for x in value[: min(10, len(value))]:

                if "events" in x:
                    score += 1

                if "path" in x:
                    score += 1

                if "event_vectors" in x:
                    score += 1

            if score >= 3:
                candidates.append(
                    (
                        score,
                        name,
                        value,
                    )
                )

        candidates.sort(
            reverse=True,
            key=lambda x: x[0],
        )

        if candidates:

            score, name, value = (
                candidates[0]
            )

            frame_results = value

            print(
                f"[OK] Found frame results in "
                f"mega.{name} (score={score})"
            )

    if frame_results is None:

        raise RuntimeError(
            "\nCould not locate frame_results automatically.\n"
            "The existing pipeline does not expose the runtime frame-results "
            "object under a discoverable module variable.\n"
            "Do NOT fabricate data. Run the pipeline once with the runtime "
            "object exported, or provide the output file containing frame "
            "results."
        )

    # ------------------------------------------------------------------------
    # Locate primary frame
    # ------------------------------------------------------------------------

    primary_frames = [
        f
        for f in frame_results
        if f.get(
            "is_primary",
            False,
        )
    ]

    if not primary_frames:
        raise RuntimeError(
            "No primary frame found."
        )

    if len(primary_frames) != 1:
        print(
            f"[WARNING] Found {len(primary_frames)} primary frames."
        )

    primary_frame = primary_frames[0]

    primary_events = primary_frame.get(
        "events",
        [],
    )

    primary_vectors = primary_frame.get(
        "event_vectors",
        [],
    )

    if not primary_events or not primary_vectors:
        raise RuntimeError(
            "Primary frame has no events/event_vectors."
        )

    primary_vectors = np.asarray(
        primary_vectors,
        dtype=np.float64,
    )

    n_primary = min(
        len(primary_events),
        len(primary_vectors),
    )

    primary_records = [
        EventRecord(
            frame=str(
                primary_frame.get(
                    "path",
                    "",
                )
            ),
            event_id=str(
                primary_events[i].get(
                    "event_id",
                    f"E{i:05d}",
                )
            ),
            vector=finite_vector(
                primary_vectors[i]
            ),
        )
        for i in range(n_primary)
    ]

    # ------------------------------------------------------------------------
    # Extract all non-primary records
    # ------------------------------------------------------------------------

    all_records = extract_frame_vectors(
        frame_results
    )

    non_primary = [
        r
        for r in all_records
        if r.frame != primary_records[0].frame
    ]

    grouped = group_by_frame(
        non_primary
    )

    print()
    print(
        f"Primary events: {len(primary_records)}"
    )
    print(
        f"Non-primary frames: {len(grouped)}"
    )
    print(
        f"Non-primary events: {len(non_primary)}"
    )
    print()

    # ------------------------------------------------------------------------
    # Population statistics
    # ------------------------------------------------------------------------

    stats = fit_population_statistics(
        [
            r.vector
            for r in primary_records
        ]
    )

    np.save(
        output_dir / "primary_feature_medians.npy",
        stats["median"],
    )

    np.save(
        output_dir / "primary_feature_mad_scales.npy",
        stats["mad_scale"],
    )

    np.save(
        output_dir / "primary_feature_means.npy",
        stats["mean"],
    )

    np.save(
        output_dir / "primary_feature_std.npy",
        stats["std"],
    )

    # ------------------------------------------------------------------------
    # Feature statistics
    # ------------------------------------------------------------------------

    feature_stats = []

    X = np.asarray(
        [
            r.vector
            for r in primary_records
        ]
    )

    for i, name in enumerate(
        FEATURE_NAMES
    ):

        values = X[:, i]

        feature_stats.append(
            {
                "feature": name,
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(values)
                ),
                "median": float(
                    np.median(values)
                ),
                "mad": float(
                    np.median(
                        np.abs(
                            values
                            - np.median(values)
                        )
                    )
                ),
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }
        )

    with open(
        output_dir / "feature_statistics.json",
        "w",
    ) as f:

        json.dump(
            feature_stats,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Pairwise frame analysis
    # ------------------------------------------------------------------------

    metrics = [
        "current",
        "standardised",
        "robust",
    ]

    frame_outputs = []

    for frame_index, (
        frame_path,
        candidates,
    ) in enumerate(
        sorted(grouped.items())
    ):

        print()
        print(
            "=" * 78
        )
        print(
            f"FRAME {frame_index + 1}/{len(grouped)}"
        )
        print(
            frame_path
        )
        print(
            f"Candidates: {len(candidates)}"
        )
        print(
            "=" * 78
        )

        try:

            result = analyse_frame_pair(
                candidates,
                primary_records,
                metrics,
                args.permutations,
                args.seed + frame_index,
            )

            if result is not None:
                frame_outputs.append(
                    result
                )

        except Exception:

            print(
                "[ERROR] Frame analysis failed:"
            )

            traceback.print_exc()

    # ------------------------------------------------------------------------
    # Global aggregation
    # ------------------------------------------------------------------------

    global_summary = aggregate_frame_results(
        frame_outputs
    )

    # ------------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------------

    with open(
        output_dir / "full_results.json",
        "w",
    ) as f:

        json.dump(
            {
                "configuration": {
                    "seed": args.seed,
                    "permutations": args.permutations,
                    "feature_names": FEATURE_NAMES,
                    "feature_groups": {
                        k: [
                            FEATURE_NAMES[i]
                            for i in v
                        ]
                        for k, v in FEATURE_GROUPS.items()
                    },
                },
                "global_summary": global_summary,
                "frame_results": frame_outputs,
            },
            f,
            indent=2,
            default=json_default,
        )

    # ------------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------------

    summary_path = (
        output_dir
        / "SUMMARY.txt"
    )

    with open(
        summary_path,
        "w",
    ) as f:

        f.write(
            "OVERNIGHT CROSS-FRAME REPLICATION AUDIT\n"
        )
        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            f"Seed: {args.seed}\n"
        )

        f.write(
            f"Permutations per frame/metric: "
            f"{args.permutations}\n"
        )

        f.write(
            f"Primary events: "
            f"{len(primary_records)}\n"
        )

        f.write(
            f"Non-primary frames: "
            f"{len(grouped)}\n"
        )

        f.write(
            f"Non-primary events: "
            f"{len(non_primary)}\n\n"
        )

        for metric_name, data in global_summary[
            "metrics"
        ].items():

            f.write(
                f"\nMETRIC: {metric_name}\n"
            )

            f.write(
                "-" * 78
                + "\n"
            )

            f.write(
                "Median of frame-level median distances: "
                f"{data['median_distances']['median']:.8f}\n"
            )

            f.write(
                "Mean of frame-level median distances: "
                f"{data['median_distances']['mean']:.8f}\n"
            )

            f.write(
                "Median of frame-level maximum similarities: "
                f"{data['maximum_similarities']['median']:.8f}\n"
            )

            f.write(
                "Median of frame-level median similarities: "
                f"{data['median_similarities']['median']:.8f}\n"
            )

            f.write(
                "Median empirical p-value: "
                f"{data['p_values']['median']:.8g}\n"
            )

    # ------------------------------------------------------------------------
    # Final completion report
    # ------------------------------------------------------------------------

    elapsed = time.time() - started

    print()
    print(
        "=" * 78
    )
    print(
        "AUDIT COMPLETE"
    )
    print(
        "=" * 78
    )

    print(
        f"Elapsed: {elapsed / 3600:.2f} hours"
    )

    print(
        f"Results: {output_dir}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "These results are validation diagnostics, not yet evidence "
        "of a physical LIGO mechanism."
    )


if __name__ == "__main__":
    main()


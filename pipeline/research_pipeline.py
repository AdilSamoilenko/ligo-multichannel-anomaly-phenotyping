
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats

try:
    from gwpy.timeseries import TimeSeries
    GWPY_OK = True
except Exception:
    GWPY_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


import torch
import torch.nn as nn
import torch.nn.functional as F


CHANNELS = {
    "WFS_PIT": "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "WFS_YAW": "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "LSC_POP": "H1:LSC-POP_A_LF_OUT_DQ",
    "LSC_REFL": "H1:LSC-REFL_A_RIN_OUT_DQ",
    "OAF_PIT": "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "OAF_YAW": "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "OAF_REFL": "H1:OAF-REFL_A_RIN_PREFILT_OUT_DQ",
    "PEM_MAINS": "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
    "ETMX_L1": "H1:SUS-ETMX_L1_CAL_LINE_OUT_DQ",
    "ETMX_L2": "H1:SUS-ETMX_L2_CAL_LINE_OUT_DQ",
    "ETMX_L3": "H1:SUS-ETMX_L3_CAL_LINE_OUT_DQ",
    "PI_MON": "H1:SUS-PI_PROC_COMPUTE_MODE29_RMSMON",
    "PCALX": "H1:CAL-PCALX_RX_PD_OUT_DQ",
    "PCALY": "H1:CAL-PCALY_RX_PD_OUT_DQ",
}

DERIVED_CHANNELS = {
    "OAF_PIT": "WFS_PIT",
    "OAF_YAW": "WFS_YAW",
    "OAF_REFL": "LSC_REFL",
}

SCALES = (
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
)

TARGET_FS = 1024.0
DEFAULT_SEED = 20260924
EPS = 1e-12


@dataclass
class ChannelRecord:
    label: str
    channel: str
    sample_rate: float
    start_gps: float
    end_gps: float
    samples: int


@dataclass
class FrameRecord:
    path: str
    sha256: str
    start_gps: float
    end_gps: float
    duration_s: float
    channels_loaded: int


def setup_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh = logging.FileHandler(
        log_dir / "RESEARCH_PIPELINE.log",
        encoding="utf-8",
    )
    sh = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def save_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(
    out: Path,
    name: str,
    status: str,
    **extra,
) -> None:

    save_json(
        out / "checkpoints" / f"{name}.json",
        {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "stage": name,
            "status": status,
            **extra,
        },
    )


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def discover_frames(
    data_root: Path,
) -> List[Path]:

    return sorted(
        p
        for p in data_root.rglob("*.gwf")
        if p.is_file()
    )


def read_channel(
    path: Path,
    channel: str,
) -> Tuple[
    np.ndarray,
    float,
    float,
    float,
]:

    if not GWPY_OK:
        raise RuntimeError(
            "GWpy is not available."
        )

    ts = TimeSeries.read(
        str(path),
        channel=channel,
    )

    x = np.asarray(
        ts.value,
        dtype=np.float64,
    )

    fs = float(
        ts.sample_rate.value
    )

    start = float(
        ts.t0.value
    )

    end = start + len(x) / fs

    finite = np.isfinite(x)

    if not np.all(finite):

        if np.any(finite):
            replacement = float(
                np.median(x[finite])
            )
        else:
            replacement = 0.0

        x = np.nan_to_num(
            x,
            nan=replacement,
            posinf=0.0,
            neginf=0.0,
        )

    return x, fs, start, end


def load_frame(
    path: Path,
):
    data = {}
    rates = {}
    starts = {}
    ends = {}
    records = []

    for label, channel in CHANNELS.items():

        try:
            x, fs, start, end = read_channel(
                path,
                channel,
            )

            data[label] = x
            rates[label] = fs
            starts[label] = start
            ends[label] = end

            records.append(
                ChannelRecord(
                    label=label,
                    channel=channel,
                    sample_rate=fs,
                    start_gps=start,
                    end_gps=end,
                    samples=len(x),
                )
            )

        except Exception as exc:
            logging.warning(
                "Unable to load %s: %s",
                label,
                exc,
            )

    if not data:
        raise RuntimeError(
            f"No requested channels loaded from {path}"
        )

    return (
        data,
        rates,
        starts,
        ends,
        records,
    )


def rational_resample(
    x: np.ndarray,
    fs_in: float,
    fs_out: float,
) -> np.ndarray:

    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(
            x,
            dtype=np.float64,
        )

    ratio = Fraction(
        fs_out / fs_in
    ).limit_denominator(10000)

    return signal.resample_poly(
        x,
        ratio.numerator,
        ratio.denominator,
    ).astype(
        np.float64,
        copy=False,
    )


def align_channels(
    data,
    rates,
    starts,
    ends,
    target_fs=TARGET_FS,
):
    common_start = max(
        starts.values()
    )
    common_end = min(
        ends.values()
    )

    if common_end <= common_start:
        raise RuntimeError(
            "No common GPS-time overlap."
        )

    duration = common_end - common_start

    n_target = int(
        math.floor(
            duration * target_fs
        )
    )

    aligned = {}

    for label, x in data.items():

        y = rational_resample(
            x,
            rates[label],
            target_fs,
        )

        offset = int(
            round(
                (common_start - starts[label])
                * target_fs
            )
        )

        stop = offset + n_target

        if offset < 0 or stop > len(y):
            continue

        segment = y[offset:stop]

        if len(segment) == n_target:
            aligned[label] = segment

    if not aligned:
        raise RuntimeError(
            "No channels survived GPS alignment."
        )

    n = min(
        len(x)
        for x in aligned.values()
    )

    aligned = {
        label: x[:n]
        for label, x in aligned.items()
    }

    return aligned, common_start, target_fs


def robust_location_scale(
    x: np.ndarray,
) -> Tuple[float, float]:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    median = float(
        np.median(x)
    )

    mad = float(
        np.median(
            np.abs(x - median)
        )
    )

    scale = 1.4826 * mad

    if not np.isfinite(scale) or scale < EPS:
        scale = float(
            np.std(x)
        )

    return median, max(scale, EPS)


def robust_z(
    x: np.ndarray,
) -> np.ndarray:

    med, scale = robust_location_scale(x)

    return (
        np.asarray(x)
        - med
    ) / scale


def spectral_features(
    x: np.ndarray,
    fs: float,
) -> dict:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if len(x) < 32:
        return {
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "spectral_power": 0.0,
        }

    f, p = signal.periodogram(
        x,
        fs=fs,
        window="hann",
        detrend="constant",
    )

    if len(p) <= 1:
        return {
            "spectral_centroid": 0.0,
            "spectral_entropy": 0.0,
            "dominant_frequency": 0.0,
            "spectral_power": 0.0,
        }

    p = np.maximum(
        p,
        0.0,
    )

    total = float(
        np.trapezoid(
            p,
            f,
        )
    )

    positive = p[1:]
    positive_f = f[1:]

    denom = float(
        np.sum(positive)
    )

    if denom <= EPS:
        centroid = 0.0
        entropy = 0.0
        dominant = 0.0
    else:
        centroid = float(
            np.sum(
                positive_f * positive
            ) / denom
        )

        prob = positive / denom

        entropy = float(
            -np.sum(
                prob
                * np.log2(
                    np.maximum(
                        prob,
                        EPS,
                    )
                )
            )
            / np.log2(
                max(len(prob), 2)
            )
        )

        dominant = float(
            positive_f[
                np.argmax(positive)
            ]
        )

    return {
        "spectral_centroid": centroid,
        "spectral_entropy": entropy,
        "dominant_frequency": dominant,
        "spectral_power": total,
    }


def band_power(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
) -> float:

    nyquist = fs / 2.0

    low = max(
        0.0,
        float(low),
    )

    high = min(
        float(high),
        nyquist,
    )

    if high <= low:
        return 0.0

    f, p = signal.periodogram(
        x,
        fs=fs,
        window="hann",
        detrend="constant",
    )

    mask = (
        (f >= low)
        & (f <= high)
    )

    if np.count_nonzero(mask) < 2:
        return 0.0

    return float(
        np.trapezoid(
            p[mask],
            f[mask],
        )
    )


BANDS = (
    ("0_2", 0.0, 2.0),
    ("2_5", 2.0, 5.0),
    ("5_10", 5.0, 10.0),
    ("10_20", 10.0, 20.0),
    ("20_50", 20.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_300", 100.0, 300.0),
)


def window_features(
    x: np.ndarray,
    fs: float,
) -> dict:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    mean = float(
        np.mean(x)
    )

    median = float(
        np.median(x)
    )

    std = float(
        np.std(x)
    )

    rms = float(
        np.sqrt(
            np.mean(
                x * x
            )
        )
    )

    minimum = float(
        np.min(x)
    )

    maximum = float(
        np.max(x)
    )

    peak_abs = float(
        np.max(
            np.abs(x)
        )
    )

    crest = (
        peak_abs / rms
        if rms > EPS
        else 0.0
    )

    if std > EPS:
        skew = float(
            stats.skew(
                x,
                bias=False,
            )
        )

        kurtosis = float(
            stats.kurtosis(
                x,
                fisher=True,
                bias=False,
            )
        )
    else:
        skew = 0.0
        kurtosis = 0.0

    result = {
        "mean": mean,
        "median": median,
        "std": std,
        "rms": rms,
        "min": minimum,
        "max": maximum,
        "peak_abs": peak_abs,
        "crest_factor": crest,
        "skew": skew,
        "kurtosis": kurtosis,
    }

    result.update(
        spectral_features(
            x,
            fs,
        )
    )

    for name, low, high in BANDS:
        result[
            f"band_{name}"
        ] = band_power(
            x,
            fs,
            low,
            high,
        )

    return result


# ============================================================================
# DETECTION
# ============================================================================

def make_windows(
    n_samples: int,
    fs: float,
    duration: float,
):
    n = int(
        round(
            duration * fs
        )
    )

    if n <= 0:
        return

    for start in range(
        0,
        n_samples - n + 1,
        n,
    ):
        yield start, start + n


def detection_score(
    features: List[dict],
    key: str,
) -> np.ndarray:

    values = np.asarray(
        [
            row[key]
            for row in features
        ],
        dtype=np.float64,
    )

    return np.abs(
        robust_z(values)
    )


def independent_detection(
    x: np.ndarray,
    fs: float,
    gps_start: float,
    scales=SCALES,
    threshold: float = 4.0,
) -> List[dict]:

    detections = []

    for scale in scales:

        features = []
        indices = []

        for i0, i1 in make_windows(
            len(x),
            fs,
            scale,
        ):

            segment = x[i0:i1]

            features.append(
                window_features(
                    segment,
                    fs,
                )
            )

            indices.append(
                (i0, i1)
            )

        if not features:
            continue

        mean_scores = detection_score(
            features,
            "mean",
        )

        rms_scores = detection_score(
            features,
            "rms",
        )

        std_scores = detection_score(
            features,
            "std",
        )

        low_freq = np.asarray(
            [
                row["band_0_2"]
                + row["band_2_5"]
                for row in features
            ]
        )

        low_scores = np.abs(
            robust_z(low_freq)
        )

        for j, (
            i0,
            i1,
        ) in enumerate(indices):

            scores = {
                "mean_score": float(
                    mean_scores[j]
                ),
                "rms_score": float(
                    rms_scores[j]
                ),
                "std_score": float(
                    std_scores[j]
                ),
                "low_frequency_score": float(
                    low_scores[j]
                ),
            }

            independent = (
                scores["mean_score"] >= threshold
                or scores["rms_score"] >= threshold
                or scores["std_score"] >= threshold
                or scores["low_frequency_score"] >= threshold
            )

            if not independent:
                continue

            detections.append(
                {
                    "scale": scale,
                    "start_gps": gps_start + i0 / fs,
                    "end_gps": gps_start + i1 / fs,
                    "i0": i0,
                    "i1": i1,
                    **scores,
                    **features[j],
                }
            )

    return detections


def cluster_events(
    detections: List[dict],
    gap_seconds: float = 1.0,
) -> List[dict]:

    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["start_gps"],
    )

    clusters = []
    current = []

    for row in detections:

        if not current:
            current = [row]
            continue

        previous_end = max(
            x["end_gps"]
            for x in current
        )

        if (
            row["start_gps"]
            <= previous_end + gap_seconds
        ):
            current.append(row)
        else:
            clusters.append(current)
            current = [row]

    if current:
        clusters.append(current)

    events = []

    for event_id, cluster in enumerate(
        clusters,
        start=1,
    ):

        start = min(
            x["start_gps"]
            for x in cluster
        )

        end = max(
            x["end_gps"]
            for x in cluster
        )

        event = {
            "event_id": f"E{event_id:04d}",
            "start_gps": start,
            "end_gps": end,
            "duration": end - start,
            "n_detections": len(cluster),
            "n_scales": len(
                set(
                    x["scale"]
                    for x in cluster
                )
            ),
            "max_mean_score": max(
                x["mean_score"]
                for x in cluster
            ),
            "max_rms_score": max(
                x["rms_score"]
                for x in cluster
            ),
            "max_std_score": max(
                x["std_score"]
                for x in cluster
            ),
            "max_low_frequency_score": max(
                x["low_frequency_score"]
                for x in cluster
            ),
        }

        events.append(event)

    return events


# ============================================================================
# FAST LAGGED CORRELATION
# ============================================================================

def normalised_xcorr(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_samples: int,
) -> Tuple[
    float,
    int,
]:

    n = min(
        len(x),
        len(y),
    )

    if n < 16:
        return 0.0, 0

    x = np.asarray(
        x[:n],
        dtype=np.float64,
    )
    y = np.asarray(
        y[:n],
        dtype=np.float64,
    )

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    if sx < EPS or sy < EPS:
        return 0.0, 0

    x /= sx
    y /= sy

    corr = signal.correlate(
        x,
        y,
        mode="full",
        method="fft",
    ) / n

    lags = signal.correlation_lags(
        n,
        n,
        mode="full",
    )

    mask = (
        np.abs(lags)
        <= max_lag_samples
    )

    if not np.any(mask):
        return 0.0, 0

    subset = corr[mask]
    subset_lags = lags[mask]

    idx = int(
        np.argmax(
            np.abs(subset)
        )
    )

    return (
        float(subset[idx]),
        int(subset_lags[idx]),
    )


def lagged_correlation(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    max_lag_seconds: float = 2.0,
) -> dict:

    r, lag = normalised_xcorr(
        x,
        y,
        int(
            round(
                max_lag_seconds * fs
            )
        ),
    )

    return {
        "r": r,
        "abs_r": abs(r),
        "lag_seconds": lag / fs,
    }


# ============================================================================
# CONTROL WINDOWS
# ============================================================================

def event_free_windows(
    n_samples: int,
    fs: float,
    window_seconds: float,
    event_ranges: List[Tuple[float, float]],
    gps_start: float,
    step_seconds: Optional[float] = None,
) -> List[Tuple[int, int]]:

    if step_seconds is None:
        step_seconds = window_seconds

    n = int(
        round(
            window_seconds * fs
        )
    )

    step = max(
        1,
        int(
            round(
                step_seconds * fs
            )
        ),
    )

    forbidden = [
        (
            start - window_seconds,
            end + window_seconds,
        )
        for start, end in event_ranges
    ]

    windows = []

    for i0 in range(
        0,
        n_samples - n + 1,
        step,
    ):

        i1 = i0 + n

        start_gps = (
            gps_start
            + i0 / fs
        )

        end_gps = (
            gps_start
            + i1 / fs
        )

        overlap = any(
            start_gps < b
            and end_gps > a
            for a, b in forbidden
        )

        if not overlap:
            windows.append(
                (i0, i1)
            )

    return windows


# ============================================================================
# SURROGATE TESTS
# ============================================================================

def circular_shift_surrogate(
    x: np.ndarray,
    rng: np.random.Generator,
    minimum_fraction: float = 0.1,
) -> np.ndarray:

    n = len(x)

    minimum = max(
        1,
        int(
            n * minimum_fraction
        ),
    )

    if minimum >= n:
        return np.roll(
            x,
            n // 2,
        )

    shift = int(
        rng.integers(
            minimum,
            n - minimum + 1,
        )
    )

    if rng.random() < 0.5:
        shift = -shift

    return np.roll(
        x,
        shift,
    )


def circular_surrogate_test(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    n_surrogates: int,
    rng: np.random.Generator,
) -> dict:

    observed = lagged_correlation(
        x,
        y,
        fs,
    )

    null = np.empty(
        n_surrogates,
        dtype=np.float64,
    )

    for i in range(
        n_surrogates
    ):

        ys = circular_shift_surrogate(
            y,
            rng,
        )

        null[i] = lagged_correlation(
            x,
            ys,
            fs,
        )["abs_r"]

    exceed = int(
        np.count_nonzero(
            null >= observed["abs_r"]
        )
    )

    p = (
        exceed + 1
    ) / (
        n_surrogates + 1
    )

    return {
        "observed_r": observed["r"],
        "observed_abs_r": observed["abs_r"],
        "observed_lag_seconds": observed[
            "lag_seconds"
        ],
        "n_surrogates": n_surrogates,
        "null_median": float(
            np.median(null)
        ),
        "null_95": float(
            np.percentile(
                null,
                95,
            )
        ),
        "null_99": float(
            np.percentile(
                null,
                99,
            )
        ),
        "p_value": float(p),
    }


def phase_randomise(
    x: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    n = len(x)

    spectrum = np.fft.rfft(
        x
    )

    phase = rng.uniform(
        0.0,
        2.0 * np.pi,
        len(spectrum),
    )

    phase[0] = 0.0

    if n % 2 == 0:
        phase[-1] = 0.0

    magnitude = np.abs(
        spectrum
    )

    randomised = (
        magnitude
        * np.exp(
            1j * phase
        )
    )

    randomised[0] = spectrum[0]

    if n % 2 == 0:
        randomised[-1] = spectrum[-1]

    result = np.fft.irfft(
        randomised,
        n=n,
    )

    result -= np.mean(result)

    original_std = np.std(x)
    new_std = np.std(result)

    if new_std > EPS:
        result *= (
            original_std
            / new_std
        )

    result += np.mean(x)

    return result


def phase_randomisation_test(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    n_surrogates: int,
    rng: np.random.Generator,
) -> dict:

    observed = lagged_correlation(
        x,
        y,
        fs,
    )

    null = np.empty(
        n_surrogates,
        dtype=np.float64,
    )

    for i in range(
        n_surrogates
    ):

        ys = phase_randomise(
            y,
            rng,
        )

        null[i] = lagged_correlation(
            x,
            ys,
            fs,
        )["abs_r"]

    exceed = int(
        np.count_nonzero(
            null >= observed["abs_r"]
        )
    )

    p = (
        exceed + 1
    ) / (
        n_surrogates + 1
    )

    return {
        "observed_r": observed["r"],
        "observed_abs_r": observed["abs_r"],
        "observed_lag_seconds": observed[
            "lag_seconds"
        ],
        "n_surrogates": n_surrogates,
        "null_median": float(
            np.median(null)
        ),
        "null_95": float(
            np.percentile(
                null,
                95,
            )
        ),
        "null_99": float(
            np.percentile(
                null,
                99,
            )
        ),
        "p_value": float(p),
    }


# ============================================================================
# MULTIPLE TESTING
# ============================================================================

def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[dict]:

    if not p_values:
        return []

    p = np.asarray(
        p_values,
        dtype=np.float64,
    )

    n = len(p)

    order = np.argsort(
        p
    )

    ranked = p[order]

    adjusted = np.empty_like(
        ranked
    )

    running = 1.0

    for i in range(
        n - 1,
        -1,
        -1,
    ):

        rank = i + 1

        value = (
            ranked[i]
            * n
            / rank
        )

        running = min(
            running,
            value,
        )

        adjusted[i] = running

    result = [
        None
    ] * n

    for rank_index, original_index in enumerate(
        order
    ):

        result[original_index] = {
            "p_value": float(
                p[original_index]
            ),
            "q_value": float(
                min(
                    adjusted[rank_index],
                    1.0,
                )
            ),
            "significant": bool(
                adjusted[rank_index]
                <= alpha
            ),
        }

    return result


# ============================================================================
# EVENT EXTRACTION
# ============================================================================

def extract_segment(
    x: np.ndarray,
    fs: float,
    gps_start: float,
    event_start: float,
    event_end: float,
) -> Optional[np.ndarray]:

    i0 = max(
        0,
        int(
            math.floor(
                (event_start - gps_start)
                * fs
            )
        ),
    )

    i1 = min(
        len(x),
        int(
            math.ceil(
                (event_end - gps_start)
                * fs
            )
        ),
    )

    if i1 <= i0:
        return None

    segment = x[
        i0:i1
    ]

    if len(segment) < 32:
        return None

    return segment


def event_channel_analysis(
    events: List[dict],
    aligned: Dict[str, np.ndarray],
    fs: float,
    gps_start: float,
) -> List[dict]:

    rows = []

    for event in events:

        for label, x in aligned.items():

            segment = extract_segment(
                x,
                fs,
                gps_start,
                event["start_gps"],
                event["end_gps"],
            )

            if segment is None:
                continue

            features = window_features(
                segment,
                fs,
            )

            rows.append(
                {
                    "event_id": event[
                        "event_id"
                    ],
                    "channel": label,
                    **features,
                }
            )

    return rows


# ============================================================================
# EVENT CROSS-CHANNEL ANALYSIS
# ============================================================================

def event_cross_channel_analysis(
    events: List[dict],
    aligned: Dict[str, np.ndarray],
    fs: float,
    gps_start: float,
) -> List[dict]:

    rows = []

    labels = list(
        aligned.keys()
    )

    for event in events:

        segments = {}

        for label in labels:

            segment = extract_segment(
                aligned[label],
                fs,
                gps_start,
                event["start_gps"],
                event["end_gps"],
            )

            if segment is not None:
                segments[label] = segment

        for i, label_a in enumerate(
            labels
        ):

            if label_a not in segments:
                continue

            for label_b in labels[
                i + 1:
            ]:

                if label_b not in segments:
                    continue

                a = segments[label_a]
                b = segments[label_b]

                n = min(
                    len(a),
                    len(b),
                )

                if n < 32:
                    continue

                result = lagged_correlation(
                    a[:n],
                    b[:n],
                    fs,
                    max_lag_seconds=2.0,
                )

                rows.append(
                    {
                        "event_id": event[
                            "event_id"
                        ],
                        "channel_a": label_a,
                        "channel_b": label_b,
                        **result,
                        "derived_relationship": (
                            label_a in DERIVED_CHANNELS
                            or label_b in DERIVED_CHANNELS
                        ),
                    }
                )

    return rows


# ============================================================================
# PROVENANCE
# ============================================================================

def provenance_analysis(
    aligned: Dict[str, np.ndarray],
    fs: float,
) -> List[dict]:

    rows = []

    labels = list(
        aligned.keys()
    )

    for i, a_label in enumerate(
        labels
    ):

        for b_label in labels[
            i + 1:
        ]:

            a = aligned[a_label]
            b = aligned[b_label]

            result = lagged_correlation(
                a,
                b,
                fs,
                max_lag_seconds=2.0,
            )

            rows.append(
                {
                    "channel_a": a_label,
                    "channel_b": b_label,
                    **result,
                    "a_derived": (
                        a_label
                        in DERIVED_CHANNELS
                    ),
                    "b_derived": (
                        b_label
                        in DERIVED_CHANNELS
                    ),
                }
            )

    return rows


# ============================================================================
# OAF RESIDUALS
# ============================================================================

def linear_residual(
    target: np.ndarray,
    predictor: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
]:

    n = min(
        len(target),
        len(predictor),
    )

    y = np.asarray(
        target[:n],
        dtype=np.float64,
    )

    x = np.asarray(
        predictor[:n],
        dtype=np.float64,
    )

    X = np.column_stack(
        [
            np.ones(n),
            x,
        ]
    )

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    fitted = X @ beta

    residual = y - fitted

    return residual, float(
        beta[1]
    )


def oaf_residual_analysis(
    aligned: Dict[str, np.ndarray],
    fs: float,
) -> List[dict]:

    rows = []

    relationships = (
        ("OAF_PIT", "WFS_PIT"),
        ("OAF_YAW", "WFS_YAW"),
        ("OAF_REFL", "LSC_REFL"),
    )

    for child, parent in relationships:

        if (
            child not in aligned
            or parent not in aligned
        ):
            continue

        residual, coefficient = linear_residual(
            aligned[child],
            aligned[parent],
        )

        before = float(
            np.std(
                aligned[child]
            )
        )

        after = float(
            np.std(
                residual
            )
        )

        rows.append(
            {
                "child": child,
                "parent": parent,
                "linear_coefficient": coefficient,
                "child_std": before,
                "residual_std": after,
                "variance_reduction_fraction": (
                    1.0
                    - (
                        after * after
                    )
                    / max(
                        before * before,
                        EPS,
                    )
                ),
            }
        )

    return rows


# ============================================================================
# PHENOTYPE REPRESENTATION
# ============================================================================

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


def event_feature_vector(
    event: dict,
    channel_rows: List[dict],
) -> np.ndarray:

    rows = [
        r
        for r in channel_rows
        if r["event_id"]
        == event["event_id"]
    ]

    if not rows:
        return np.zeros(
            len(FEATURE_NAMES),
            dtype=np.float64,
        )

    matrix = []

    for row in rows:

        matrix.append(
            [
                float(
                    row.get(
                        name,
                        0.0,
                    )
                    or 0.0
                )
                for name in FEATURE_NAMES
            ]
        )

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.median(
        matrix,
        axis=0,
    )


def build_phenotypes(
    events: List[dict],
    channel_rows: List[dict],
) -> Tuple[
    List[dict],
    Optional[np.ndarray],
]:

    if not events:
        return [], None

    vectors = np.vstack(
        [
            event_feature_vector(
                event,
                channel_rows,
            )
            for event in events
        ]
    )

    if len(events) < 10:
        return (
            [
                {
                    "event_id": e[
                        "event_id"
                    ],
                    "phenotype": "UNCLUSTERED",
                    "cluster_method": "insufficient_events_for_phenotype_clustering",
                }
                for e in events
            ],
            vectors,
        )

    if not SKLEARN_OK:
        return (
            [
                {
                    "event_id": e[
                        "event_id"
                    ],
                    "phenotype": "UNCLUSTERED",
                    "cluster_method": "sklearn_unavailable",
                }
                for e in events
            ],
            vectors,
        )

    scaler = StandardScaler()
    z = scaler.fit_transform(
        vectors
    )

    n_clusters = min(
        4,
        max(
            2,
            len(events) // 3,
        ),
        len(events),
    )

    if n_clusters >= len(events):
        n_clusters = len(events) - 1

    if n_clusters < 2:
        return (
            [
                {
                    "event_id": e[
                        "event_id"
                    ],
                    "phenotype": "UNCLUSTERED",
                    "cluster_method": "insufficient_events_for_phenotype_clustering",
                }
                for e in events
            ],
            vectors,
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=DEFAULT_SEED,
        n_init=20,
    )

    labels = model.fit_predict(
        z
    )

    phenotype_rows = []

    for event, label in zip(
        events,
        labels,
    ):

        phenotype_rows.append(
            {
                "event_id": event[
                    "event_id"
                ],
                "phenotype": (
                    f"P{int(label) + 1:02d}"
                ),
                "cluster_method": "KMeans",
            }
        )

    return (
        phenotype_rows,
        vectors,
    )


# ============================================================================
# CROSS-FRAME REPLICATION
# ============================================================================

def normalised_feature_distance(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    if len(a) != len(b):
        n = min(
            len(a),
            len(b),
        )
        a = a[:n]
        b = b[:n]

    a = np.asarray(
        a,
        dtype=np.float64,
    )
    b = np.asarray(
        b,
        dtype=np.float64,
    )

    scale = np.maximum(
        np.abs(a),
        np.abs(b),
    )

    scale = np.maximum(
        scale,
        EPS,
    )

    return float(
        np.sqrt(
            np.mean(
                (
                    (a - b)
                    / scale
                ) ** 2
            )
        )
    )


def replicate_events(
    primary_events: List[dict],
    primary_features: Optional[np.ndarray],
    frame_results: List[dict],
    threshold: float = 0.75,
) -> List[dict]:

    if (
        not primary_events
        or primary_features is None
    ):
        return []

    rows = []

    for frame in frame_results:

        if frame.get(
            "is_primary",
            False,
        ):
            continue

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

        for i, event in enumerate(
            events
        ):

            if i >= len(vectors):
                continue

            candidate = vectors[i]

            distances = np.asarray(
                [
                    normalised_feature_distance(
                        candidate,
                        ref,
                    )
                    for ref in primary_features
                ]
            )

            nearest = int(
                np.argmin(
                    distances
                )
            )

            distance = float(
                distances[nearest]
            )

            similarity = float(
                1.0
                / (
                    1.0
                    + distance
                )
            )

            rows.append(
                {
                    "frame": frame[
                        "path"
                    ],
                    "event_id": event[
                        "event_id"
                    ],
                    "nearest_primary_event": primary_events[
                        nearest
                    ][
                        "event_id"
                    ],
                    "feature_distance": distance,
                    "feature_similarity": similarity,
                    "replicated": (
                        similarity
                        >= threshold
                    ),
                }
            )

    return rows


# MACHINE LEARNING

def ml_dataset(
    events: List[dict],
    vectors: Optional[np.ndarray],
) -> dict:

    if vectors is None:
        return {
            "status": "unavailable",
            "reason": "no_event_vectors",
        }

    if len(events) < 10:
        return {
            "status": "deferred",
            "reason": "fewer_than_10_independent_events",
            "n_events": len(events),
        }

    if not SKLEARN_OK:
        return {
            "status": "deferred",
            "reason": "scikit_learn_unavailable",
            "n_events": len(events),
        }

    return {
        "status": "available_for_exploratory_benchmark",
        "n_events": len(events),
        "n_features": vectors.shape[1],
    }


def run_exploratory_ml(
    events: List[dict],
    vectors: Optional[np.ndarray],
) -> dict:

    if vectors is None:
        return {
            "status": "skipped",
            "reason": "no_features",
        }

    if len(events) < 10:
        return {
            "status": "skipped",
            "reason": "insufficient_independent_events",
        }

    if not SKLEARN_OK:
        return {
            "status": "skipped",
            "reason": "scikit_learn_unavailable",
        }

    if len(np.unique(
        np.arange(len(events))
    )) < 4:
        return {
            "status": "skipped",
            "reason": "insufficient_samples",
        }

    return {
        "status": "deferred",
        "reason": "requires_external_or_physical_labels",
        "n_events": len(events),
    }


class TemporalSNN(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 2,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(
        self,
        x,
    ):

        membrane = torch.zeros(
            x.shape[0],
            self.fc1.out_features,
            device=x.device,
        )

        output = None

        for t in range(
            x.shape[1]
        ):

            current = self.fc1(
                x[:, t, :]
            )

            membrane = (
                0.9 * membrane
                + current
            )

            spikes = (
                membrane > 1.0
            ).float()

            membrane = (
                membrane
                * (1.0 - spikes)
            )

            output = self.fc2(
                membrane
            )

        return output


def run_snn(
    events: List[dict],
    vectors: Optional[np.ndarray],
) -> dict:

    if vectors is None:
        return {
            "status": "skipped",
            "reason": "no_features",
        }

    n_events = len(events)

    if n_events < 30:
        return {
            "status": "deferred",
            "reason": "insufficient_independent_events_for_supervised_training",
            "n_events": n_events,
            "minimum_events": 30,
            "model": "TemporalSNN",
            "pytorch_version": torch.__version__,
            "device": (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
        }

    return {
        "status": "deferred",
        "reason": "requires_external_or_physical_labels",
        "n_events": n_events,
        "model": "TemporalSNN",
        "pytorch_version": torch.__version__,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    }


# ============================================================================
# FIGURES
# ============================================================================

def make_figures(
    out: Path,
    aligned: Dict[str, np.ndarray],
    gps_start: float,
    fs: float,
    events: List[dict],
    channel_rows: List[dict],
    provenance_rows: List[dict],
    replication_rows: List[dict],
) -> None:

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except Exception as exc:
        logging.warning(
            "Matplotlib unavailable: %s",
            exc,
        )
        return

    fig_dir = out / "figures"
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = (
        "WFS_PIT"
        if "WFS_PIT" in aligned
        else next(iter(aligned))
    )

    x = aligned[target]

    t = (
        np.arange(len(x))
        / fs
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        t,
        x,
        linewidth=0.6,
    )

    for event in events:

        a = (
            event["start_gps"]
            - gps_start
        )

        b = (
            event["end_gps"]
            - gps_start
        )

        ax.axvspan(
            a,
            b,
            alpha=0.2,
        )

    ax.set_xlabel(
        "Time from frame start (s)"
    )
    ax.set_ylabel(
        target
    )
    ax.set_title(
        "Target channel overview"
    )

    fig.tight_layout()

    fig.savefig(
        fig_dir / "frame_overview.png",
        dpi=180,
    )

    plt.close(fig)

    if events:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        durations = [
            e["duration"]
            for e in events
        ]

        scores = [
            e["max_mean_score"]
            for e in events
        ]

        ax.scatter(
            durations,
            scores,
        )

        for event in events:
            ax.annotate(
                event["event_id"],
                (
                    event["duration"],
                    event["max_mean_score"],
                ),
            )

        ax.set_xlabel(
            "Event duration (s)"
        )
        ax.set_ylabel(
            "Maximum mean deviation score"
        )
        ax.set_title(
            "Candidate event catalogue"
        )

        fig.tight_layout()

        fig.savefig(
            fig_dir / "event_catalogue.png",
            dpi=180,
        )

        plt.close(fig)

    if provenance_rows:

        top = sorted(
            provenance_rows,
            key=lambda r: r["abs_r"],
            reverse=True,
        )[:30]

        labels = [
            f'{r["channel_a"]} / {r["channel_b"]}'
            for r in top
        ]

        values = [
            r["abs_r"]
            for r in top
        ]

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        ax.barh(
            range(len(values)),
            values,
        )

        ax.set_yticks(
            range(len(values))
        )

        ax.set_yticklabels(
            labels,
            fontsize=7,
        )

        ax.invert_yaxis()

        ax.set_xlabel(
            "|maximum lagged correlation|"
        )

        ax.set_title(
            "Channel dependency structure"
        )

        fig.tight_layout()

        fig.savefig(
            fig_dir / "provenance_network.png",
            dpi=180,
        )

        plt.close(fig)

    if replication_rows:

        similarity = [
            r["feature_similarity"]
            for r in replication_rows
        ]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.hist(
            similarity,
            bins=20,
        )

        ax.set_xlabel(
            "Feature similarity"
        )

        ax.set_ylabel(
            "Count"
        )

        ax.set_title(
            "Cross-frame phenotype similarity"
        )

        fig.tight_layout()

        fig.savefig(
            fig_dir / "replication.png",
            dpi=180,
        )

        plt.close(fig)


# ============================================================================
# REPORTS
# ============================================================================

def write_reports(
    out: Path,
    frame_records: List[FrameRecord],
    events: List[dict],
    channel_rows: List[dict],
    provenance_rows: List[dict],
    cross_rows: List[dict],
    control_rows: List[dict],
    residual_rows: List[dict],
    null_rows: List[dict],
    phase_rows: List[dict],
    phenotype_rows: List[dict],
    replication_rows: List[dict],
    ml_result: dict,
    snn_result: dict,
) -> None:

    reports = out / "reports"
    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary = (
        frame_records[0]
        if frame_records
        else None
    )

    methods = f"""# Methods

## Data

The pipeline analyses LIGO auxiliary-channel GWF files discovered beneath
the configured data directory. Every input file is SHA256 hashed before
analysis and rehashed after analysis.

The requested channel inventory contains {len(CHANNELS)} channels.

## Time alignment

Channels are loaded with their native sample rates and GPS start times.
Signals are resampled with polyphase filtering and aligned using GPS time.
No cross-channel calculation uses raw sample indices from channels with
different sample rates.

## Detection

The target channel is analysed using independent windows at the following
scales:

{", ".join(str(x) for x in SCALES)} seconds.

Robust median/MAD statistics are used to construct scale-specific deviation
scores. Mean, RMS, standard deviation and low-frequency power are evaluated
independently.

Overlapping detections are clustered into candidate events.

## Cross-channel analysis

Candidate events are evaluated against the available auxiliary channels
using lagged correlation, spectral features and event morphology.

Processed or derived channels are explicitly marked in the provenance
analysis.

## Null testing

Circular-shift and phase-randomisation tests are used where sufficient
aligned data are available.

P-values use the finite-sample correction:

(p_exceedances + 1) / (N_surrogates + 1)

Multiple comparisons are handled using Benjamini-Hochberg false-discovery
rate correction.

## Replication

Replication is phenotype based. Events in independent frames are compared
using event feature vectors. Absolute GPS-time overlap is not used as a
replication criterion.

## Machine learning

Supervised classification is deferred until independent events and
externally grounded labels are sufficient. Self-generated clusters are not
treated as physical truth.
"""

    results = f"""# Results

## Dataset

Frames analysed: {len(frame_records)}

Primary frame:

{primary.path if primary else "None"}

## Candidate events

Candidate event clusters: {len(events)}

"""

    for event in events:
        results += (
            f'- {event["event_id"]}: '
            f'{event["start_gps"]:.6f} to '
            f'{event["end_gps"]:.6f} GPS, '
            f'duration {event["duration"]:.3f} s, '
            f'{event["n_scales"]} scales, '
            f'max mean score '
            f'{event["max_mean_score"]:.3f}\n'
        )

    results += f"""

## Cross-channel measurements

Event/channel measurements: {len(channel_rows)}

Event cross-channel relationships: {len(cross_rows)}

Provenance relationships: {len(provenance_rows)}

Control comparisons: {len(control_rows)}

Residual analyses: {len(residual_rows)}

Circular null tests: {len(null_rows)}

Phase-randomisation tests: {len(phase_rows)}

Cross-frame replication comparisons: {len(replication_rows)}

## Machine learning

{json.dumps(ml_result, indent=2)}

## SNN

{json.dumps(snn_result, indent=2)}
"""

    discussion = """# Discussion

The analysis is intended to identify candidate transient phenotypes in
auxiliary detector data.

A candidate is not interpreted as a gravitational-wave signal solely from
its presence in an auxiliary channel.

Correlation is not treated as proof of physical causation. Relationships
involving processed channels require provenance-aware interpretation.

High coherence or correlation can arise from deterministic signal-chain
relationships. Independent environmental or calibration channels therefore
provide important controls.

Cross-frame similarity is treated as evidence for a reproducible phenotype
only when the event morphology is sufficiently similar in independent data.
"""

    status = f"""# Scientific Status

Candidate events: {len(events)}

Phenotype records: {len(phenotype_rows)}

Independent frames: {len(frame_records)}

Replication comparisons: {len(replication_rows)}

The computational pipeline has completed the available analyses.

A physical interpretation requires independent frame replication, provenance
validation and, where possible, confirmation against detector-state or
environmental information.

A final scientific claim should be based on reproducible event populations,
not on a single candidate.
"""

    final_report = f"""# Final Research Report

{methods}

{results}

{discussion}

{status}
"""

    (reports / "METHODS_DRAFT.md").write_text(
        methods,
        encoding="utf-8",
    )

    (reports / "RESULTS_DRAFT.md").write_text(
        results,
        encoding="utf-8",
    )

    (reports / "DISCUSSION_DRAFT.md").write_text(
        discussion,
        encoding="utf-8",
    )

    (reports / "SCIENTIFIC_STATUS.md").write_text(
        status,
        encoding="utf-8",
    )

    (reports / "FINAL_RESEARCH_REPORT.md").write_text(
        final_report,
        encoding="utf-8",
    )


# ============================================================================
# QC
# ============================================================================

def quality_control(
    input_hashes_before: Dict[str, str],
    input_hashes_after: Dict[str, str],
    frame_records: List[FrameRecord],
    events: List[dict],
    out: Path,
) -> Tuple[
    bool,
    List[str],
    List[str],
]:

    errors = []
    warnings = []

    for path, before in input_hashes_before.items():

        after = input_hashes_after.get(
            path
        )

        if after != before:
            errors.append(
                f"Input hash changed: {path}"
            )

    if not frame_records:
        errors.append(
            "No frames were successfully analysed."
        )

    if not events:
        warnings.append(
            "No candidate event clusters were detected."
        )

    qc = [
        "MEGA RESEARCH QUALITY CONTROL",
        "=" * 32,
        "",
        f"Frames analysed: {len(frame_records)}",
        f"Candidate events: {len(events)}",
        f"Input files checked: {len(input_hashes_before)}",
        "",
        "Input integrity: "
        + (
            "PASS"
            if not errors
            else "FAIL"
        ),
        "",
    ]

    if warnings:
        qc.append("Warnings:")
        qc.extend(
            f"- {x}"
            for x in warnings
        )
        qc.append("")

    if errors:
        qc.append("Errors:")
        qc.extend(
            f"- {x}"
            for x in errors
        )
        qc.append("")

    qc.append(
        "Overall status: "
        + (
            "PASS"
            if not errors
            else "FAIL"
        )
    )

    (out / "QUALITY_CONTROL.txt").write_text(
        "\n".join(qc),
        encoding="utf-8",
    )

    return (
        not errors,
        errors,
        warnings,
    )


# ============================================================================
# FRAME ANALYSIS
# ============================================================================

def analyse_frame(
    path: Path,
    out: Path,
    surrogates: int,
    rng: np.random.Generator,
) -> dict:

    logging.info(
        "Analysing frame: %s",
        path,
    )

    data, rates, starts, ends, records = load_frame(
        path
    )

    aligned, gps_start, fs = align_channels(
        data,
        rates,
        starts,
        ends,
        TARGET_FS,
    )

    target_label = (
        "WFS_PIT"
        if "WFS_PIT" in aligned
        else next(iter(aligned))
    )

    target = aligned[
        target_label
    ]

    detections = independent_detection(
        target,
        fs,
        gps_start,
    )

    events = cluster_events(
        detections
    )

    channel_rows = event_channel_analysis(
        events,
        aligned,
        fs,
        gps_start,
    )

    cross_rows = event_cross_channel_analysis(
        events,
        aligned,
        fs,
        gps_start,
    )

    provenance_rows = provenance_analysis(
        aligned,
        fs,
    )

    residual_rows = oaf_residual_analysis(
        aligned,
        fs,
    )

    null_rows = []
    phase_rows = []

    null_pairs = [
        (
            target_label,
            "LSC_POP",
        ),
        (
            target_label,
            "WFS_YAW",
        ),
        (
            target_label,
            "LSC_REFL",
        ),
        (
            target_label,
            "OAF_PIT",
        ),
        (
            target_label,
            "OAF_YAW",
        ),
        (
            target_label,
            "OAF_REFL",
        ),
        (
            target_label,
            "PEM_MAINS",
        ),
        (
            target_label,
            "PI_MON",
        ),
    ]

    for a_label, b_label in null_pairs:

        if (
            a_label not in aligned
            or b_label not in aligned
        ):
            continue

        a = aligned[a_label]
        b = aligned[b_label]

        n = min(
            len(a),
            len(b),
        )

        if n < int(
            2 * fs
        ):
            continue

        a = a[:n]
        b = b[:n]

        result = circular_surrogate_test(
            a,
            b,
            fs,
            surrogates,
            rng,
        )

        result.update(
            {
                "channel_a": a_label,
                "channel_b": b_label,
            }
        )

        null_rows.append(
            result
        )

        phase = phase_randomisation_test(
            a,
            b,
            fs,
            surrogates,
            rng,
        )

        phase.update(
            {
                "channel_a": a_label,
                "channel_b": b_label,
            }
        )

        phase_rows.append(
            phase
        )

    event_ranges = [
        (
            e["start_gps"],
            e["end_gps"],
        )
        for e in events
    ]

    control_rows = []

    controls = event_free_windows(
        len(target),
        fs,
        7.0,
        event_ranges,
        gps_start,
        step_seconds=1.0,
    )

    if controls:

        candidate_segments = {}

        for event in events:

            segment = extract_segment(
                target,
                fs,
                gps_start,
                event["start_gps"],
                event["end_gps"],
            )

            if segment is not None:
                candidate_segments[
                    event["event_id"]
                ] = segment

        control_count = min(
            len(controls),
            32,
        )

        selected_controls = controls[
            :control_count
        ]

        for i0, i1 in selected_controls:

            segment = target[
                i0:i1
            ]

            if len(segment) < 32:
                continue

            features = window_features(
                segment,
                fs,
            )

            control_rows.append(
                {
                    "start_gps": gps_start + i0 / fs,
                    "end_gps": gps_start + i1 / fs,
                    **features,
                }
            )

    event_vectors = None

    if events:
        event_vectors = np.vstack(
            [
                event_feature_vector(
                    event,
                    channel_rows,
                )
                for event in events
            ]
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "start_gps": gps_start,
        "end_gps": gps_start + len(target) / fs,
        "duration_s": len(target) / fs,
        "channels_loaded": len(aligned),
        "records": [
            asdict(r)
            for r in records
        ],
        "events": events,
        "detections": detections,
        "channel_rows": channel_rows,
        "cross_rows": cross_rows,
        "provenance_rows": provenance_rows,
        "residual_rows": residual_rows,
        "null_rows": null_rows,
        "phase_rows": phase_rows,
        "control_rows": control_rows,
        "event_vectors": (
            event_vectors.tolist()
            if event_vectors is not None
            else []
        ),
        "aligned_labels": list(
            aligned.keys()
        ),
        "_aligned": aligned,
        "_gps_start": gps_start,
        "_fs": fs,
    }


# ============================================================================
# OUTPUT CONVERSION
# ============================================================================

def strip_internal(
    result: dict,
) -> dict:

    return {
        k: v
        for k, v in result.items()
        if not k.startswith("_")
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data",
        type=Path,
    )

    parser.add_argument(
        "--out",
        default="results_mega",
        type=Path,
    )

    parser.add_argument(
        "--surrogates",
        default=500,
        type=int,
    )

    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
    )

    parser.add_argument(
        "--replication",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    out = args.out

    setup_logging(out)

    start_time = time.time()

    logging.info(
        "Starting mega research pipeline."
    )

    if not GWPY_OK:
        raise RuntimeError(
            "GWpy is required."
        )

    data_root = args.data_root.resolve()

    if not data_root.exists():
        raise FileNotFoundError(
            data_root
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint(
        out,
        "startup",
        "running",
        data_root=str(data_root),
    )

    frames = discover_frames(
        data_root
    )

    if not frames:
        raise RuntimeError(
            f"No GWF files found under {data_root}"
        )

    logging.info(
        "Discovered %d GWF files.",
        len(frames),
    )

    hashes_before = {}

    for frame in frames:

        logging.info(
            "Hashing %s",
            frame,
        )

        hashes_before[
            str(frame)
        ] = sha256_file(
            frame
        )

    save_json(
        out
        / "statistics"
        / "INPUT_HASHES_BEFORE.json",
        hashes_before,
    )

    checkpoint(
        out,
        "input_validation",
        "complete",
        files=len(frames),
    )

    rng = np.random.default_rng(
        args.seed
    )

    frame_results = []

    for index, frame in enumerate(
        frames
    ):

        cache_path = (
            out
            / "checkpoints"
            / f"frame_{index:04d}.json"
        )

        try:

            result = analyse_frame(
                frame,
                out,
                args.surrogates,
                rng,
            )

            clean = strip_internal(
                result
            )

            save_json(
                cache_path,
                clean,
            )

            frame_results.append(
                result
            )

            logging.info(
                "Completed frame %d/%d: %s events",
                index + 1,
                len(frames),
                len(
                    result["events"]
                ),
            )

        except Exception as exc:

            logging.error(
                "Frame failed %s: %s",
                frame,
                exc,
            )

            logging.error(
                traceback.format_exc()
            )

            save_json(
                cache_path,
                {
                    "status": "failed",
                    "path": str(frame),
                    "error": str(exc),
                },
            )

    if not frame_results:
        raise RuntimeError(
            "No frame completed successfully."
        )

    for result in frame_results:
        result[
            "is_primary"
        ] = (
            result["path"]
            == frame_results[0]["path"]
        )

    primary = frame_results[0]

    primary_events = primary[
        "events"
    ]

    primary_channel_rows = primary[
        "channel_rows"
    ]

    primary_vectors = (
        np.asarray(
            primary[
                "event_vectors"
            ],
            dtype=np.float64,
        )
        if primary[
            "event_vectors"
        ]
        else None
    )

    phenotype_rows, phenotype_vectors = build_phenotypes(
        primary_events,
        primary_channel_rows,
    )

    replication_rows = []

    if args.replication and len(
        frame_results
    ) > 1:

        replication_rows = replicate_events(
            primary_events,
            primary_vectors,
            frame_results,
        )

    save_csv(
        out
        / "detection"
        / "ALL_DETECTIONS.csv",
        [
            row
            for frame in frame_results
            for row in frame[
                "detections"
            ]
        ],
    )

    save_csv(
        out
        / "detection"
        / "EVENT_CLUSTERS.csv",
        [
            {
                "frame": frame["path"],
                **event,
            }
            for frame in frame_results
            for event in frame[
                "events"
            ]
        ],
    )

    save_csv(
        out
        / "detection"
        / "MULTISCALE_FEATURES.csv",
        [
            {
                "frame": frame["path"],
                **row,
            }
            for frame in frame_results
            for row in frame[
                "detections"
            ]
        ],
    )

    save_csv(
        out
        / "catalogue"
        / "EVENT_CATALOGUE.csv",
        [
            {
                "frame": frame["path"],
                **event,
            }
            for frame in frame_results
            for event in frame[
                "events"
            ]
        ],
    )

    catalogue_json = []

    for frame in frame_results:

        for event in frame[
            "events"
        ]:

            catalogue_json.append(
                {
                    "frame": frame[
                        "path"
                    ],
                    **event,
                }
            )

    save_json(
        out
        / "catalogue"
        / "EVENT_CATALOGUE.json",
        catalogue_json,
    )

    save_csv(
        out
        / "catalogue"
        / "PHENOTYPES.csv",
        phenotype_rows,
    )

    save_csv(
        out
        / "provenance"
        / "PROVENANCE_RESULTS.csv",
        [
            {
                "frame": frame["path"],
                **row,
            }
            for frame in frame_results
            for row in frame[
                "provenance_rows"
            ]
        ],
    )

    save_csv(
        out
        / "provenance"
        / "DEPENDENCY_MATRIX.csv",
        [
            {
                "frame": frame["path"],
                **row,
            }
            for frame in frame_results
            for row in frame[
                "provenance_rows"
            ]
        ],
    )

    save_json(
        out
        / "provenance"
        / "CHANNEL_GRAPH.json",
        {
            "channels": list(
                CHANNELS.keys()
            ),
            "derived_channels": DERIVED_CHANNELS,
            "relationships": [
                {
                    "channel_a": row[
                        "channel_a"
                    ],
                    "channel_b": row[
                        "channel_b"
                    ],
                    "abs_r": row[
                        "abs_r"
                    ],
                    "lag_seconds": row[
                        "lag_seconds"
                    ],
                }
                for row in primary[
                    "provenance_rows"
                ]
            ],
        },
    )

    save_csv(
        out
        / "provenance"
        / "EVENT_CROSS_CHANNEL.csv",
        [
            {
                "frame": frame["path"],
                **row,
            }
            for frame in frame_results
            for row in frame[
                "cross_rows"
            ]
        ],
    )

    save_csv(
        out
        / "statistics"
        / "CONTROL_COMPARISONS.csv",
        [
            {
                "frame": frame["path"],
                **row,
            }
            for frame in frame_results
            for row in frame[
                "control_rows"
            ]
        ],
    )

    save_csv(
        out
        / "statistics"
        / "OAF_RESIDUAL_ANALYSIS.csv",
        [
            {
                "frame": frame["path"],
                **row,
            }
            for frame in frame_results
            for row in frame[
                "residual_rows"
            ]
        ],
    )

    null_rows = [
        {
            "frame": frame["path"],
            **row,
        }
        for frame in frame_results
        for row in frame[
            "null_rows"
        ]
    ]

    phase_rows = [
        {
            "frame": frame["path"],
            **row,
        }
        for frame in frame_results
        for row in frame[
            "phase_rows"
        ]
    ]

    save_json(
        out
        / "statistics"
        / "NULL_TESTS.json",
        null_rows,
    )

    save_json(
        out
        / "statistics"
        / "PHASE_RANDOMISATION.json",
        phase_rows,
    )

    all_p = [
        row["p_value"]
        for row in null_rows
        if np.isfinite(
            row["p_value"]
        )
    ]

    bh = benjamini_hochberg(
        all_p
    )

    save_json(
        out
        / "statistics"
        / "MULTIPLE_TESTING.json",
        {
            "family": "all_circular_shift_channel_pair_tests",
            "alpha": 0.05,
            "results": bh,
        },
    )

    effects = []

    for row in null_rows:

        effects.append(
            {
                "frame": row[
                    "frame"
                ]
                if "frame" in row
                else None,
                "channel_a": row[
                    "channel_a"
                ],
                "channel_b": row[
                    "channel_b"
                ],
                "observed_abs_r": row[
                    "observed_abs_r"
                ],
                "null_median": row[
                    "null_median"
                ],
                "effect_above_null_median": (
                    row[
                        "observed_abs_r"
                    ]
                    - row[
                        "null_median"
                    ]
                ),
            }
        )

    save_json(
        out
        / "statistics"
        / "EFFECT_SIZES.json",
        effects,
    )

    ml_result = ml_dataset(
        primary_events,
        primary_vectors,
    )

    ml_run = run_exploratory_ml(
        primary_events,
        primary_vectors,
    )

    snn_result = run_snn(
        primary_events,
        primary_vectors,
    )

    save_json(
        out
        / "machine_learning"
        / "DATASET.json",
        ml_result,
    )

    save_json(
        out
        / "machine_learning"
        / "BASELINE_RESULTS.json",
        ml_run,
    )

    save_json(
        out
        / "machine_learning"
        / "SNN_RESULTS.json",
        snn_result,
    )

    save_csv(
        out
        / "replication"
        / "REPLICATION_MATRIX.csv",
        replication_rows,
    )

    frame_summary = []

    for frame in frame_results:

        frame_summary.append(
            {
                "path": frame["path"],
                "sha256": frame["sha256"],
                "start_gps": frame[
                    "start_gps"
                ],
                "end_gps": frame[
                    "end_gps"
                ],
                "duration_s": frame[
                    "duration_s"
                ],
                "channels_loaded": frame[
                    "channels_loaded"
                ],
                "events": len(
                    frame["events"]
                ),
            }
        )

    save_csv(
        out
        / "replication"
        / "FRAME_SUMMARY.csv",
        frame_summary,
    )

    save_json(
        out
        / "replication"
        / "REPLICATION_STATISTICS.json",
        {
            "enabled": args.replication,
            "frames": len(frame_results),
            "primary_events": len(
                primary_events
            ),
            "comparisons": len(
                replication_rows
            ),
            "replicated": sum(
                bool(
                    r["replicated"]
                )
                for r in replication_rows
            ),
        },
    )

    save_json(
        out
        / "statistics"
        / "RUN_METADATA.json",
        {
            "started": start_time,
            "finished": time.time(),
            "seed": args.seed,
            "target_fs": TARGET_FS,
            "scales": SCALES,
            "surrogates": args.surrogates,
            "channels": CHANNELS,
            "derived_channels": DERIVED_CHANNELS,
            "data_root": str(
                data_root
            ),
            "frames_discovered": len(
                frames
            ),
            "frames_completed": len(
                frame_results
            ),
        },
    )

    make_figures(
        out,
        primary["_aligned"],
        primary["_gps_start"],
        primary["_fs"],
        primary_events,
        primary_channel_rows,
        primary[
            "provenance_rows"
        ],
        replication_rows,
    )

    frame_records = []

    for frame in frame_results:

        frame_records.append(
            FrameRecord(
                path=frame["path"],
                sha256=frame["sha256"],
                start_gps=frame[
                    "start_gps"
                ],
                end_gps=frame[
                    "end_gps"
                ],
                duration_s=frame[
                    "duration_s"
                ],
                channels_loaded=frame[
                    "channels_loaded"
                ],
            )
        )

    write_reports(
        out,
        frame_records,
        primary_events,
        primary_channel_rows,
        primary[
            "provenance_rows"
        ],
        primary[
            "cross_rows"
        ],
        primary[
            "control_rows"
        ],
        primary[
            "residual_rows"
        ],
        primary[
            "null_rows"
        ],
        primary[
            "phase_rows"
        ],
        phenotype_rows,
        replication_rows,
        ml_result,
        snn_result,
    )

    checkpoint(
        out,
        "analysis",
        "complete",
        frames=len(
            frame_results
        ),
        events=len(
            primary_events
        ),
    )

    hashes_after = {}

    for frame in frames:
        hashes_after[
            str(frame)
        ] = sha256_file(
            frame
        )

    save_json(
        out
        / "statistics"
        / "INPUT_HASHES_AFTER.json",
        hashes_after,
    )

    qc_pass, errors, warnings = quality_control(
        hashes_before,
        hashes_after,
        frame_records,
        primary_events,
        out,
    )

    final_results = {
        "completed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "runtime_seconds": time.time()
        - start_time,
        "frames_discovered": len(
            frames
        ),
        "frames_completed": len(
            frame_results
        ),
        "channels_loaded_primary": primary[
            "channels_loaded"
        ],
        "candidate_events_primary": len(
            primary_events
        ),
        "detections_primary": len(
            primary[
                "detections"
            ]
        ),
        "cross_channel_results_primary": len(
            primary[
                "cross_rows"
            ]
        ),
        "provenance_results_primary": len(
            primary[
                "provenance_rows"
            ]
        ),
        "control_results_primary": len(
            primary[
                "control_rows"
            ]
        ),
        "circular_null_results_primary": len(
            primary[
                "null_rows"
            ]
        ),
        "phase_randomisation_results_primary": len(
            primary[
                "phase_rows"
            ]
        ),
        "replication_results": len(
            replication_rows
        ),
        "phenotypes": len(
            phenotype_rows
        ),
        "ml": ml_result,
        "snn": snn_result,
        "qc_status": (
            "PASS"
            if qc_pass
            else "FAIL"
        ),
        "errors": errors,
        "warnings": warnings,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
    }

    save_json(
        out
        / "statistics"
        / "FINAL_AUTONOMOUS_RESULTS.json",
        final_results,
    )

    if qc_pass:

        save_json(
            out
            / "COMPLETE.json",
            {
                "completed_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "runtime_seconds": time.time()
                - start_time,
                "qc_status": "PASS",
                "frames_discovered": len(
                    frames
                ),
                "frames_completed": len(
                    frame_results
                ),
                "candidate_events": len(
                    primary_events
                ),
                "errors": errors,
                "warnings": warnings,
            },
        )

        checkpoint(
            out,
            "completion",
            "complete",
            qc="PASS",
        )

        logging.info(
            "MEGA RESEARCH PIPELINE COMPLETE."
        )

    else:

        checkpoint(
            out,
            "completion",
            "failed",
            qc="FAIL",
            errors=errors,
        )

        raise RuntimeError(
            "Quality control failed."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error(
            "Interrupted."
        )
        sys.exit(130)
    except Exception as exc:
        logging.error(
            "Fatal error: %s",
            exc,
        )
        logging.error(
            traceback.format_exc()
        )
        sys.exit(1)

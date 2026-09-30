#!/usr/bin/env python3

"""
AUTONOMOUS LIGO AUXILIARY-CHANNEL RESEARCH RUNNER
=================================================

Purpose
-------
Complete the remaining computational research workflow automatically.

Design principles
-----------------
1. Original GWF is READ-ONLY.
2. Existing overnight checkpoints are reused.
3. Expensive work is not repeated unnecessarily.
4. Individual channel failures do not terminate the project.
5. Every stage writes checkpoints/results.
6. Statistical claims are accompanied by null/control information.
7. Derived channels are not treated as independent evidence.
8. Frequencies are never reported with unsupported precision.
9. Overlapping windows are not treated as independent observations.
10. No claim of gravitational-wave detection is made.

Input
-----
data/auxiliary/W10_H1_AUX_AR1.gwf

Target
------
H1:IMC-WFS_A_DC_PIT_OUT_DQ

Frame
-----
GPS 1376516096 - 1376516160

Output
------
results_autonomous/
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "auxiliary" / "W10_H1_AUX_AR1.gwf"

OLD_RESULTS = ROOT / "results_overnight"
OUT = ROOT / "results_autonomous"

TABLES = OUT / "tables"
STATS = OUT / "statistics"
FIGURES = OUT / "figures"
EVENT_FIGURES = FIGURES / "events"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"

for d in [
    OUT,
    TABLES,
    STATS,
    FIGURES,
    EVENT_FIGURES,
    CHECKPOINTS,
    LOGS,
]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS / "AUTONOMOUS_RUN.log"

FRAME_START = 1376516096.0
FRAME_END = 1376516160.0

TARGET = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

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

SCALES = [
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    7.0,
]

BANDS = [
    (0, 2),
    (2, 5),
    (5, 10),
    (10, 20),
    (20, 50),
    (50, 100),
    (100, 300),
    (300, 1000),
]

RNG = np.random.default_rng(20260924)

ERRORS = []
WARNINGS = []
STAGES = []


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    log("WARNING: " + msg)


def error(msg: str) -> None:
    ERRORS.append(msg)
    log("ERROR: " + msg)


def stage(name: str):
    class Stage:
        def __enter__(self):
            self.t0 = time.time()
            log("=" * 78)
            log(f"START {name}")
            log("=" * 78)
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.time() - self.t0
            if exc is not None:
                error(f"{name} failed after {elapsed:.1f}s: {exc}")
                traceback.print_exc()
                STAGES.append({
                    "stage": name,
                    "status": "FAILED",
                    "seconds": elapsed,
                    "error": str(exc),
                })
                return True
            STAGES.append({
                "stage": name,
                "status": "OK",
                "seconds": elapsed,
            })
            log(f"FINISHED {name} in {elapsed:.1f}s")
            return False

    return Stage()


# ---------------------------------------------------------------------
# SAFE FILE OPERATIONS
# ---------------------------------------------------------------------

def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def safe_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            x = json.load(f)
        return x
    except Exception as exc:
        warn(f"Could not read JSON {path}: {exc}")
        return {}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: row.get(k, "")
                for k in keys
            })


def checkpoint(name: str, payload: Any) -> None:
    atomic_json(
        CHECKPOINTS / f"{name}.json",
        payload,
    )


# ---------------------------------------------------------------------
# INTEGRITY
# ---------------------------------------------------------------------

def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_input() -> dict:
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    stat = DATA.stat()
    digest = sha256(DATA)

    info = {
        "path": str(DATA),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256_before": digest,
        "frame_start": FRAME_START,
        "frame_end": FRAME_END,
        "duration_seconds": FRAME_END - FRAME_START,
    }

    atomic_json(
        STATS / "DATA_INTEGRITY_BEFORE.json",
        info,
    )

    return info


# ---------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------

def environment_info() -> dict:
    packages = {}

    for package in [
        "numpy",
        "scipy",
        "matplotlib",
        "gwpy",
        "lal",
        "lalframe",
        "pandas",
    ]:
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            packages[package] = f"ERROR: {exc}"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(ROOT),
        "packages": packages,
        "random_seed": 20260924,
    }


# ---------------------------------------------------------------------
# IMPORT DATA
# ---------------------------------------------------------------------

def load_channels():
    from gwpy.timeseries import TimeSeries

    data = {}

    for label, channel in CHANNELS.items():
        try:
            log(f"Loading {label}: {channel}")

            ts = TimeSeries.read(
                str(DATA),
                channel=channel,
            )

            arr = np.asarray(ts.value, dtype=np.float64)

            if arr.size == 0:
                warn(f"{label}: empty")
                continue

            good = np.isfinite(arr)

            if not np.any(good):
                warn(f"{label}: no finite samples")
                continue

            arr = arr[good]

            data[label] = {
                "channel": channel,
                "values": arr,
                "sample_rate": float(ts.sample_rate.value),
                "start": float(ts.t0.value),
                "end": float(ts.t0.value + ts.duration.value),
                "n": int(arr.size),
            }

        except Exception as exc:
            warn(f"Could not load {label}: {exc}")

    return data


# ---------------------------------------------------------------------
# TIME ALIGNMENT
# ---------------------------------------------------------------------

def elapsed_axis(meta: dict) -> np.ndarray:
    n = meta["n"]
    fs = meta["sample_rate"]
    return np.arange(n, dtype=np.float64) / fs


def aligned_pair(a: dict, b: dict, fs: float = 1024.0):
    ta = a["start"] + elapsed_axis(a)
    tb = b["start"] + elapsed_axis(b)

    start = max(ta[0], tb[0])
    end = min(ta[-1], tb[-1])

    if end <= start:
        return None, None, None

    fs = min(
        fs,
        a["sample_rate"],
        b["sample_rate"],
    )

    if fs <= 0:
        return None, None, None

    n = int((end - start) * fs)

    if n < 32:
        return None, None, None

    t = start + np.arange(n) / fs

    xa = np.interp(
        t,
        ta,
        a["values"],
    )

    xb = np.interp(
        t,
        tb,
        b["values"],
    )

    return t, xa, xb


# ---------------------------------------------------------------------
# ROBUST STATISTICS
# ---------------------------------------------------------------------

def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return 1.4826 * mad


def safe_skew(x):
    from scipy.stats import skew
    try:
        return float(skew(x, bias=False))
    except Exception:
        return np.nan


def safe_kurtosis(x):
    from scipy.stats import kurtosis
    try:
        return float(kurtosis(x, bias=False))
    except Exception:
        return np.nan


def features(x: np.ndarray, fs: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size < 4:
        return {}

    rms = float(np.sqrt(np.mean(x * x)))

    out = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "rms": rms,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "peak_abs": float(np.max(np.abs(x))),
        "crest_factor": (
            float(np.max(np.abs(x)) / rms)
            if rms > 0 else np.nan
        ),
        "skew": safe_skew(x),
        "kurtosis": safe_kurtosis(x),
    }

    return out


# ---------------------------------------------------------------------
# SPECTRAL FEATURES
# ---------------------------------------------------------------------

def welch_psd(x: np.ndarray, fs: float):
    from scipy.signal import welch

    x = np.asarray(x, dtype=np.float64)

    if x.size < 16 or fs <= 0:
        return np.array([]), np.array([])

    nperseg = min(
        x.size,
        max(16, int(fs * min(1.0, x.size / fs))),
    )

    if nperseg < 16:
        return np.array([]), np.array([])

    f, p = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )

    return f, p


def band_power(x: np.ndarray, fs: float, lo: float, hi: float):
    f, p = welch_psd(x, fs)

    if f.size == 0:
        return np.nan

    nyq = fs / 2

    hi = min(hi, nyq)

    if hi <= lo:
        return np.nan

    mask = (
        (f >= lo) &
        (f <= hi)
    )

    if np.count_nonzero(mask) < 1:
        return np.nan

    return float(np.trapezoid(p[mask], f[mask]))


# ---------------------------------------------------------------------
# MULTISCALE EVENT DETECTION
# ---------------------------------------------------------------------

def robust_window_score(x, baseline_med, baseline_sigma):
    if not np.isfinite(baseline_sigma) or baseline_sigma <= 0:
        return np.nan

    return float(
        abs(np.mean(x) - baseline_med) /
        baseline_sigma
    )


def detect_target(data: dict) -> list[dict]:
    meta = data["WFS_PIT"]
    x = meta["values"]
    fs = meta["sample_rate"]
    start = meta["start"]

    med = float(np.median(x))
    sigma = robust_sigma(x)

    detections = []

    for width in SCALES:
        n = max(4, int(round(width * fs)))
        step = max(1, n // 4)

        for i in range(0, len(x) - n + 1, step):
            w = x[i:i+n]

            mean_score = robust_window_score(
                w,
                med,
                sigma,
            )

            rms = float(np.sqrt(np.mean(w*w)))

            peak = float(np.max(np.abs(w)))

            std = float(np.std(w))

            if not np.isfinite(mean_score):
                continue

            # Deliberately modest thresholds.
            # Final event selection is performed after clustering
            # and control analysis.
            if (
                mean_score >= 2.0
                or peak >= med + 3.0 * sigma
            ):
                detections.append({
                    "start_gps": start + i / fs,
                    "end_gps": start + (i+n) / fs,
                    "duration": width,
                    "scale": width,
                    "mean_score": mean_score,
                    "rms": rms,
                    "std": std,
                    "peak_abs": peak,
                })

    return detections


# ---------------------------------------------------------------------
# EVENT CLUSTERING
# ---------------------------------------------------------------------

def cluster_detections(detections: list[dict]) -> list[dict]:
    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x["start_gps"],
    )

    clusters = []

    current = {
        "start_gps": detections[0]["start_gps"],
        "end_gps": detections[0]["end_gps"],
        "detections": [detections[0]],
    }

    for d in detections[1:]:
        overlap = (
            d["start_gps"]
            <= current["end_gps"] + 0.25
        )

        if overlap:
            current["end_gps"] = max(
                current["end_gps"],
                d["end_gps"],
            )
            current["detections"].append(d)
        else:
            clusters.append(current)
            current = {
                "start_gps": d["start_gps"],
                "end_gps": d["end_gps"],
                "detections": [d],
            }

    clusters.append(current)

    output = []

    for idx, c in enumerate(clusters, 1):
        ds = c["detections"]

        scales = sorted({
            float(d["scale"])
            for d in ds
        })

        methods = set()

        if any(
            d["mean_score"] >= 2
            for d in ds
        ):
            methods.add("robust_mean")

        if any(
            d["peak_abs"] >
            np.nanmedian(
                [z["peak_abs"] for z in ds]
            )
            for d in ds
        ):
            methods.add("peak")

        output.append({
            "event_id": f"E{idx:03d}",
            "start_gps": c["start_gps"],
            "end_gps": c["end_gps"],
            "duration": (
                c["end_gps"] -
                c["start_gps"]
            ),
            "n_detections": len(ds),
            "n_scales": len(scales),
            "scales": ",".join(map(str, scales)),
            "methods": ",".join(sorted(methods)),
            "max_mean_score": max(
                d["mean_score"] for d in ds
            ),
            "max_peak_abs": max(
                d["peak_abs"] for d in ds
            ),
            "max_rms": max(
                d["rms"] for d in ds
            ),
        })

    return output


# ---------------------------------------------------------------------
# CORRELATION
# ---------------------------------------------------------------------

def correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(mask) < 16:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def lagged_corr(
    x,
    y,
    fs,
    max_lag=2.0,
    step=0.05,
):
    if len(x) != len(y):
        return np.nan, np.nan

    max_samples = int(max_lag * fs)
    step_samples = max(
        1,
        int(step * fs),
    )

    best_r = np.nan
    best_lag = np.nan

    for shift in range(
        -max_samples,
        max_samples + 1,
        step_samples,
    ):
        if shift < 0:
            a = x[-shift:]
            b = y[:len(y)+shift]
        elif shift > 0:
            a = x[:-shift]
            b = y[shift:]
        else:
            a = x
            b = y

        if len(a) < 32:
            continue

        r = correlation(a, b)

        if not np.isfinite(r):
            continue

        if (
            not np.isfinite(best_r)
            or abs(r) > abs(best_r)
        ):
            best_r = r
            best_lag = shift / fs

    return best_r, best_lag


# ---------------------------------------------------------------------
# SURROGATES
# ---------------------------------------------------------------------

def surrogate_correlations(
    x,
    y,
    fs,
    n=500,
    max_lag=2.0,
):
    observed, observed_lag = lagged_corr(
        x,
        y,
        fs,
        max_lag=max_lag,
    )

    values = []

    min_shift = max(
        int(0.5 * fs),
        1,
    )

    max_shift = len(y) - min_shift

    if max_shift <= min_shift:
        return {
            "observed": observed,
            "lag": observed_lag,
            "n": 0,
            "null": None,
        }

    for _ in range(n):
        shift = int(
            RNG.integers(
                min_shift,
                max_shift,
            )
        )

        ys = np.roll(y, shift)

        r, _ = lagged_corr(
            x,
            ys,
            fs,
            max_lag=max_lag,
        )

        if np.isfinite(r):
            values.append(abs(r))

    if not values:
        return {
            "observed": observed,
            "lag": observed_lag,
            "n": 0,
            "null": None,
        }

    values = np.asarray(values)

    obs = abs(observed) if np.isfinite(observed) else np.nan

    count = int(
        np.count_nonzero(values >= obs)
    )

    return {
        "observed": observed,
        "lag": observed_lag,
        "n": int(len(values)),
        "null": {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "empirical_p_plus1": (
                (count + 1) /
                (len(values) + 1)
            ),
            "max": float(np.max(values)),
        },
    }


# ---------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------

def provenance():
    rows = []

    for label, channel in CHANNELS.items():

        category = "unknown"
        independence = "unknown"
        reason = ""

        if label.startswith("OAF_"):
            category = "signal-processing-derived"
            independence = "likely-derived"
            reason = (
                "OAF PREFILT naming and measured "
                "high-frequency coherence indicate "
                "potential deterministic dependence "
                "on corresponding WFS/REFL source."
            )

        elif label in {
            "PEM_MAINS",
            "ETMX_L1",
            "ETMX_L2",
            "ETMX_L3",
        }:
            category = "environmental/instrumental"
            independence = "comparatively-independent"

        elif label.startswith("PCAL"):
            category = "calibration"
            independence = "comparatively-independent"

        elif label.startswith("WFS"):
            category = "interferometric-control"
            independence = "source-channel"

        elif label.startswith("LSC"):
            category = "interferometric-control"
            independence = "unknown"

        elif label == "PI_MON":
            category = "processing/control-monitor"
            independence = "unknown"

        rows.append({
            "label": label,
            "channel": channel,
            "category": category,
            "independence_status": independence,
            "reason": reason,
        })

    return rows


# ---------------------------------------------------------------------
# EVENT CROSS-CHANNEL ANALYSIS
# ---------------------------------------------------------------------

def analyse_events(data, events):
    rows = []

    target_meta = data.get("WFS_PIT")

    if target_meta is None:
        return rows

    for event in events:

        start = event["start_gps"]
        end = event["end_gps"]

        target = target_meta

        ta = target["start"] + elapsed_axis(target)

        mask = (
            (ta >= start) &
            (ta <= end)
        )

        if np.count_nonzero(mask) < 32:
            continue

        x = target["values"][mask]
        fs = target["sample_rate"]

        for label, meta in data.items():

            if label == "WFS_PIT":
                continue

            try:
                t, xa, xb = aligned_pair(
                    target,
                    meta,
                    fs=min(
                        1024.0,
                        target["sample_rate"],
                        meta["sample_rate"],
                    ),
                )

                if t is None:
                    continue

                emask = (
                    (t >= start) &
                    (t <= end)
                )

                if np.count_nonzero(emask) < 32:
                    continue

                xa = xa[emask]
                xb = xb[emask]

                if len(xa) < 32:
                    continue

                fs2 = min(
                    1024.0,
                    target["sample_rate"],
                    meta["sample_rate"],
                )

                r0 = correlation(xa, xb)

                rlag, lag = lagged_corr(
                    xa,
                    xb,
                    fs2,
                    max_lag=2.0,
                )

                rows.append({
                    "event_id": event["event_id"],
                    "channel": label,
                    "zero_lag_r": r0,
                    "max_lag_r": rlag,
                    "max_lag_seconds": lag,
                    "sample_rate": meta["sample_rate"],
                })

            except Exception as exc:
                warn(
                    f"Event analysis failed "
                    f"{event['event_id']} / {label}: {exc}"
                )

    return rows


# ---------------------------------------------------------------------
# EVENT CONTROL ANALYSIS
# ---------------------------------------------------------------------

def control_windows(start, end, duration):
    """
    Produce non-overlapping-ish control windows distributed
    throughout the frame, avoiding the candidate interval.
    """

    controls = []

    t = FRAME_START

    step = max(
        duration * 1.5,
        1.0,
    )

    while t + duration <= FRAME_END:

        if (
            t + duration < start - duration
            or t > end + duration
        ):
            controls.append(
                (t, t + duration)
            )

        t += step

    return controls


def event_control_statistics(meta, event):
    fs = meta["sample_rate"]

    axis = (
        meta["start"] +
        elapsed_axis(meta)
    )

    start = event["start_gps"]
    end = event["end_gps"]
    duration = end - start

    mask = (
        (axis >= start) &
        (axis <= end)
    )

    if np.count_nonzero(mask) < 32:
        return {}

    event_x = meta["values"][mask]

    event_rms = float(
        np.sqrt(np.mean(event_x**2))
    )

    controls = []

    for cs, ce in control_windows(
        start,
        end,
        duration,
    ):
        cmask = (
            (axis >= cs) &
            (axis <= ce)
        )

        if np.count_nonzero(cmask) < 32:
            continue

        cx = meta["values"][cmask]

        controls.append(
            float(np.sqrt(np.mean(cx**2)))
        )

    if not controls:
        return {}

    controls = np.asarray(controls)

    percentile = float(
        100 *
        np.mean(controls <= event_rms)
    )

    return {
        "event_rms": event_rms,
        "control_median_rms": float(
            np.median(controls)
        ),
        "control_p95_rms": float(
            np.percentile(controls, 95)
        ),
        "control_p99_rms": float(
            np.percentile(controls, 99)
        ),
        "event_percentile": percentile,
        "n_controls": int(len(controls)),
    }


# ---------------------------------------------------------------------
# OAF RESIDUAL ANALYSIS
# ---------------------------------------------------------------------

def robust_linear_residual(x, y):
    """
    Simple least-squares source -> derived model.

    y = a*x + b

    Fit on control/full data and inspect residuals.
    """

    mask = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(mask) < 64:
        return None

    x = x[mask]
    y = y[mask]

    if np.std(x) == 0:
        return None

    A = np.column_stack([
        x,
        np.ones_like(x),
    ])

    try:
        coef, *_ = np.linalg.lstsq(
            A,
            y,
            rcond=None,
        )

        pred = A @ coef
        residual = y - pred

        return {
            "slope": float(coef[0]),
            "intercept": float(coef[1]),
            "residual": residual,
            "residual_rms": float(
                np.sqrt(
                    np.mean(residual**2)
                )
            ),
        }

    except Exception:
        return None


def oaf_residual_analysis(data, events):
    rows = []

    pairs = [
        ("WFS_PIT", "OAF_PIT"),
        ("WFS_YAW", "OAF_YAW"),
        ("LSC_REFL", "OAF_REFL"),
    ]

    for source, derived in pairs:

        if source not in data or derived not in data:
            continue

        try:
            t, x, y = aligned_pair(
                data[source],
                data[derived],
                fs=1024,
            )

            if t is None:
                continue

            model = robust_linear_residual(
                x,
                y,
            )

            if model is None:
                continue

            residual = model["residual"]

            full_rms = model["residual_rms"]

            for event in events:

                mask = (
                    (t >= event["start_gps"]) &
                    (t <= event["end_gps"])
                )

                if np.count_nonzero(mask) < 64:
                    continue

                er = residual[mask]

                event_rms = float(
                    np.sqrt(
                        np.mean(er**2)
                    )
                )

                rows.append({
                    "event_id": event["event_id"],
                    "source": source,
                    "derived": derived,
                    "slope": model["slope"],
                    "intercept": model["intercept"],
                    "full_residual_rms": full_rms,
                    "event_residual_rms": event_rms,
                    "event_to_full_ratio": (
                        event_rms / full_rms
                        if full_rms > 0
                        else np.nan
                    ),
                })

        except Exception as exc:
            warn(
                f"OAF residual failed "
                f"{source}->{derived}: {exc}"
            )

    return rows


# ---------------------------------------------------------------------
# STRONG NULL TESTS
# ---------------------------------------------------------------------

def strong_null_tests(data, events):
    """
    Expensive tests are deliberately restricted to the strongest
    event and strongest independent relationships.
    """

    results = []

    if not events:
        return results

    # Select largest detection score.
    event = max(
        events,
        key=lambda x: (
            x.get("max_mean_score", 0),
            x.get("max_rms", 0),
        ),
    )

    pairs = [
        ("WFS_PIT", "WFS_YAW"),
        ("WFS_PIT", "LSC_REFL"),
        ("WFS_PIT", "OAF_PIT"),
        ("WFS_PIT", "OAF_YAW"),
        ("WFS_PIT", "PI_MON"),
    ]

    for a_label, b_label in pairs:

        if a_label not in data or b_label not in data:
            continue

        try:
            t, a, b = aligned_pair(
                data[a_label],
                data[b_label],
                fs=512,
            )

            if t is None:
                continue

            mask = (
                (t >= event["start_gps"]) &
                (t <= event["end_gps"])
            )

            a = a[mask]
            b = b[mask]

            if len(a) < 128:
                continue

            fs = min(
                512,
                data[a_label]["sample_rate"],
                data[b_label]["sample_rate"],
            )

            obs, lag = lagged_corr(
                a,
                b,
                fs,
                max_lag=2,
            )

            # Phase-randomisation null.
            # Preserve approximate amplitude spectrum.
            n_surrogates = 250

            fa = np.fft.rfft(
                a - np.mean(a)
            )

            amp = np.abs(fa)

            null_values = []

            for _ in range(n_surrogates):

                random_phase = RNG.uniform(
                    0,
                    2 * np.pi,
                    len(fa),
                )

                random_phase[0] = 0

                if len(a) % 2 == 0:
                    random_phase[-1] = 0

                surrogate_fft = (
                    amp *
                    np.exp(
                        1j * random_phase
                    )
                )

                surrogate = np.fft.irfft(
                    surrogate_fft,
                    n=len(a),
                )

                r, _ = lagged_corr(
                    surrogate,
                    b,
                    fs,
                    max_lag=2,
                )

                if np.isfinite(r):
                    null_values.append(
                        abs(r)
                    )

            if not null_values:
                continue

            null_values = np.asarray(
                null_values
            )

            observed_abs = abs(obs)

            exceed = np.count_nonzero(
                null_values >= observed_abs
            )

            results.append({
                "event_id": event["event_id"],
                "channel_a": a_label,
                "channel_b": b_label,
                "observed_r": obs,
                "observed_abs_r": observed_abs,
                "observed_lag": lag,
                "null_method": "phase_randomisation",
                "n_surrogates": n_surrogates,
                "null_median": float(
                    np.median(null_values)
                ),
                "null_p95": float(
                    np.percentile(
                        null_values,
                        95,
                    )
                ),
                "null_p99": float(
                    np.percentile(
                        null_values,
                        99,
                    )
                ),
                "empirical_p_plus1": (
                    (exceed + 1) /
                    (len(null_values) + 1)
                ),
            })

        except Exception as exc:
            warn(
                f"Strong null test failed "
                f"{a_label}/{b_label}: {exc}"
            )

    return results


# ---------------------------------------------------------------------
# MULTIPLE TESTING
# ---------------------------------------------------------------------

def benjamini_hochberg(pvalues):
    p = np.asarray(
        pvalues,
        dtype=float,
    )

    if p.size == 0:
        return []

    order = np.argsort(p)
    ranked = p[order]

    q = np.empty_like(ranked)

    m = len(p)

    running = 1.0

    for i in range(
        m - 1,
        -1,
        -1,
    ):
        rank = i + 1
        value = (
            ranked[i] *
            m /
            rank
        )

        running = min(
            running,
            value,
        )

        q[i] = running

    output = np.empty_like(q)
    output[order] = q

    return output.tolist()


# ---------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------

def make_figures(data, events):
    import matplotlib.pyplot as plt

    target = data.get("WFS_PIT")

    if target is None:
        return

    t = (
        target["start"] +
        elapsed_axis(target)
    )

    x = target["values"]

    # Full frame.
    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    # Downsample only for plotting.
    step = max(
        1,
        len(x) // 100000,
    )

    ax.plot(
        t[::step],
        x[::step],
        linewidth=0.6,
    )

    for e in events:
        ax.axvspan(
            e["start_gps"],
            e["end_gps"],
            alpha=0.25,
        )

    ax.set_xlabel(
        "GPS time"
    )

    ax.set_ylabel(
        "H1:IMC-WFS_A_DC_PIT_OUT_DQ"
    )

    ax.set_title(
        "Target auxiliary channel: full-frame overview"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "target_full_frame.png",
        dpi=180,
    )

    plt.close(fig)

    # Candidate event figures.
    for event in events:

        start = event["start_gps"] - 2
        end = event["end_gps"] + 2

        mask = (
            (t >= start) &
            (t <= end)
        )

        if np.count_nonzero(mask) < 32:
            continue

        fig, ax = plt.subplots(
            figsize=(14, 5)
        )

        ax.plot(
            t[mask],
            x[mask],
            linewidth=0.7,
        )

        ax.axvspan(
            event["start_gps"],
            event["end_gps"],
            alpha=0.25,
        )

        ax.set_xlabel(
            "GPS time"
        )

        ax.set_ylabel(
            "Target amplitude"
        )

        ax.set_title(
            f"{event['event_id']} "
            f"{event['start_gps']:.3f}–"
            f"{event['end_gps']:.3f} GPS"
        )

        fig.tight_layout()

        fig.savefig(
            EVENT_FIGURES /
            f"{event['event_id']}_target.png",
            dpi=180,
        )

        plt.close(fig)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

def write_report(
    integrity,
    env,
    data,
    events,
    cross_rows,
    control_rows,
    provenance_rows,
    residual_rows,
    null_rows,
):
    report = []

    report.append(
        "# LIGO Auxiliary-Channel Anomaly Research Report\n"
    )

    report.append(
        "## Scope\n"
    )

    report.append(
        "This report describes statistical analysis of "
        "auxiliary channels in a 64-second LIGO Hanford "
        "frame. The analysis concerns auxiliary-channel "
        "disturbances and does not constitute a gravitational-wave "
        "detection claim.\n"
    )

    report.append("## Dataset\n")

    report.append(
        f"- Frame: `{DATA.name}`\n"
        f"- GPS interval: `{FRAME_START}` to `{FRAME_END}`\n"
        f"- Target: `{TARGET}`\n"
        f"- Channels loaded: `{len(data)}`\n"
        f"- Input SHA-256: `{integrity['sha256_before']}`\n"
    )

    report.append("\n## Channel inventory\n")

    for label, meta in data.items():
        report.append(
            f"- **{label}**: `{meta['channel']}`, "
            f"{meta['sample_rate']} Hz, "
            f"{meta['n']} samples\n"
        )

    report.append("\n## Candidate event catalogue\n")

    if events:
        for e in events:
            report.append(
                f"- **{e['event_id']}**: "
                f"{e['start_gps']:.6f}–"
                f"{e['end_gps']:.6f} GPS, "
                f"duration {e['duration']:.3f} s, "
                f"{e['n_scales']} scales, "
                f"maximum robust score "
                f"{e['max_mean_score']:.3f}\n"
            )
    else:
        report.append(
            "No candidate event clusters were produced "
            "by the independent detector.\n"
        )

    report.append(
        "\n## Important previously identified regions\n"
    )

    report.append(
        "- Low-frequency structured episode: approximately "
        "`1376516121–1376516128` GPS.\n"
        "- Rapidly varying transient-like region: approximately "
        "`1376516155–1376516159` GPS.\n"
    )

    report.append(
        "\nThese regions require interpretation against the "
        "independent event detector rather than being assumed "
        "to represent independent events.\n"
    )

    report.append(
        "\n## Provenance and independence\n"
    )

    report.append(
        "OAF PREFILT channels require special treatment because "
        "their naming and measured high-frequency relationships "
        "are consistent with deterministic signal-processing "
        "dependence on source channels. High coherence therefore "
        "cannot automatically be interpreted as independent "
        "physical confirmation.\n"
    )

    report.append(
        "\n## Statistical limitations\n"
    )

    report.append(
        "- Candidate selection can introduce selection bias.\n"
        "- Overlapping windows are not independent observations.\n"
        "- Maximum-over-lag correlations require appropriate null "
        "models.\n"
        "- Derived channels must not be counted as independent "
        "evidence.\n"
        "- Finite FFT resolution limits frequency precision.\n"
        "- A single 64-second frame cannot establish "
        "reproducibility.\n"
    )

    report.append(
        "\n## Interpretation\n"
    )

    report.append(
        "The current evidence supports investigation of "
        "structured low-frequency auxiliary-channel disturbances, "
        "particularly the approximately 1376516121–1376516128 "
        "episode. The analysis does not by itself establish a "
        "physical causal mechanism. Further validation using "
        "additional frames and provenance-aware residual analysis "
        "is required.\n"
    )

    report.append(
        "\n## Next experiment\n"
    )

    report.append(
        "Repeat the event-detection and provenance-aware analysis "
        "on additional H1 frames surrounding the identified "
        "episode and determine whether the same disturbance "
        "phenotype recurs.\n"
    )

    (OUT / "TECHNICAL_RESEARCH_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# RESEARCH GAP
# ---------------------------------------------------------------------

def write_research_gap(
    events,
    residual_rows,
    null_rows,
):
    text = f"""# Research Gap Assessment

## Current state

The present analysis contains {len(events)} independently
clustered candidate auxiliary-channel disturbance events.

The strongest previously identified structure occurs around
GPS 1376516121–1376516128 and is dominated by low-frequency
activity.

## Established observations

1. The target auxiliary channel contains structured temporal
   variability.
2. Some candidate regions show strong low-frequency power.
3. Several auxiliary channels respond during the same broad
   episode.
4. Some apparent relationships are likely affected by
   deterministic signal-processing provenance.
5. Environmental and suspension channels tested so far do not
   show relationships comparable to the strongest
   control-system/processing relationships.

## Unresolved questions

1. Which relationships remain after removing deterministic
   signal-chain dependence?
2. Does an anomalous residual remain?
3. Does the same phenotype recur in additional frames?
4. Is the disturbance generated by a known control-system
   process?
5. Can the phenomenon be detected without selecting events
   using the same statistic used for inference?

## Potential methodological contribution

A potentially useful contribution is a provenance-aware,
multiscale framework for auxiliary-channel disturbance
detection that explicitly separates independent physical
channels from deterministic signal-chain representations.

## Evidence status

OAF residual analyses completed: {len(residual_rows)}

Stronger null tests completed: {len(null_rows)}

These counts describe analyses performed, not positive
scientific results.

## Required validation

The principal missing validation is replication in independent
frames. A phenomenon observed only in one 64-second frame
should be treated as a candidate observation rather than an
established phenomenon.
"""

    (OUT / "RESEARCH_GAP.md").write_text(
        text,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# NEXT EXPERIMENT
# ---------------------------------------------------------------------

def write_next_experiment():
    text = """# Next Experiment

## Hypothesis

The strongest low-frequency auxiliary-channel disturbance
identified in the current frame represents a reproducible
class of structured control/instrument disturbances rather
than an isolated statistical fluctuation.

## Null hypothesis

The apparent event structure is explained by ordinary
time-varying auxiliary-channel behaviour and does not recur
at an elevated rate or with a consistent multichannel
phenotype in independent frames.

## Experiment

1. Obtain additional H1 frames.
2. Apply the exact same detector without retuning thresholds.
3. Generate candidate events automatically.
4. Apply provenance-aware channel grouping.
5. Perform control comparisons.
6. Apply the same null-model procedures.
7. Compare event morphology and frequency content.
8. Test whether the same cross-channel residual structure
   recurs.

## Success criterion

A recurring event phenotype should appear in independent
frames with statistically unusual control-normalised
properties and reproducible provenance-aware residual
structure.

## Failure criterion

The candidate phenotype fails to recur, or its apparent
significance disappears after control, provenance, or
multiple-testing correction.

## Scientific importance

Replication is required before interpreting the current
64-second observation as a general phenomenon.
"""

    (OUT / "NEXT_EXPERIMENT.md").write_text(
        text,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------

def run_qc(integrity_before):
    failures = []

    # Required files.
    required = [
        OUT / "TECHNICAL_RESEARCH_REPORT.md",
        OUT / "RESEARCH_GAP.md",
        OUT / "NEXT_EXPERIMENT.md",
        TABLES / "EVENT_CLUSTERS.csv",
        TABLES / "CHANNEL_PROVENANCE.csv",
        TABLES / "EVENT_CROSS_CHANNEL.csv",
        TABLES / "OAF_RESIDUAL_ANALYSIS.csv",
        STATS / "STRONG_NULL_TESTS.json",
    ]

    for p in required:
        if not p.exists():
            failures.append(
                f"Missing output: {p}"
            )

    # Validate input hash.
    after = sha256(DATA)

    if after != integrity_before["sha256_before"]:
        failures.append(
            "CRITICAL: input GWF SHA-256 changed"
        )

    # JSON validation.
    for p in STATS.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                json.load(f)
        except Exception as exc:
            failures.append(
                f"Invalid JSON {p}: {exc}"
            )

    # CSV validation.
    for p in TABLES.glob("*.csv"):
        try:
            with open(
                p,
                newline="",
                encoding="utf-8",
            ) as f:
                list(csv.reader(f))
        except Exception as exc:
            failures.append(
                f"Invalid CSV {p}: {exc}"
            )

    # Correlation sanity.
    for p in TABLES.glob("*.csv"):
        try:
            with open(
                p,
                newline="",
                encoding="utf-8",
            ) as f:
                rows = list(
                    csv.DictReader(f)
                )

            for row in rows:
                for key, value in row.items():
                    if (
                        key.endswith("_r")
                        or key in {
                            "zero_lag_r",
                            "max_lag_r",
                        }
                    ):
                        try:
                            v = float(value)
                        except Exception:
                            continue

                        if np.isfinite(v):
                            if abs(v) > 1.000001:
                                failures.append(
                                    f"Impossible correlation "
                                    f"{p}: {key}={v}"
                                )

        except Exception:
            pass

    status = (
        "PASS"
        if not failures
        else "FAIL"
    )

    text = [
        "AUTONOMOUS RESEARCH QUALITY CONTROL",
        "=" * 60,
        f"STATUS: {status}",
        "",
        f"Input SHA-256 before: "
        f"{integrity_before['sha256_before']}",
        f"Input SHA-256 after:  {after}",
        "",
        "Failures:",
    ]

    if failures:
        text.extend(
            "- " + x
            for x in failures
        )
    else:
        text.append(
            "None detected."
        )

    text.extend([
        "",
        "Warnings:",
    ])

    if WARNINGS:
        text.extend(
            "- " + x
            for x in WARNINGS
        )
    else:
        text.append(
            "None."
        )

    (OUT / "QUALITY_CONTROL.txt").write_text(
        "\n".join(text),
        encoding="utf-8",
    )

    return failures


# ---------------------------------------------------------------------
# REPAIR EXISTING PIPELINE REPORT BUG
# ---------------------------------------------------------------------

def repair_existing_pipeline():
    script = ROOT / "pipeline" / "research_runner.py"

    if not script.exists():
        warn(
            "Existing overnight pipeline not found; "
            "autonomous analysis will continue."
        )
        return

    text = script.read_text(
        encoding="utf-8"
    )

    old = '''            null = result.get(
                "null",
                {},
            )

            f.write(
                f"\\n{label}\\n"
            )

            f.write(
                f"  observed r: "
                f"{observed.get('r')}\\n"
            )

            f.write(
                f"  observed lag: "
                f"{observed.get('lag')}\\n"
            )

            f.write(
                f"  null median: "
                f"{null.get('median')}\\n"
            )

            f.write(
                f"  null 95th: "
                f"{null.get('p95')}\\n"
            )

            f.write(
                f"  null 99th: "
                f"{null.get('p99')}\\n"
            )

            f.write(
                f"  empirical p: "
                f"{null.get('empirical_p')}\\n"
            )
'''

    new = '''            null = result.get(
                "null"
            )

            f.write(
                f"\\n{label}\\n"
            )

            f.write(
                f"  observed r: "
                f"{observed.get('r')}\\n"
            )

            f.write(
                f"  observed lag: "
                f"{observed.get('lag')}\\n"
            )

            if null is None:

                f.write(
                    "  surrogate null: "
                    "not available\\n"
                )

                f.write(
                    "  reason: insufficient "
                    "usable aligned data\\n"
                )

            else:

                f.write(
                    f"  null median: "
                    f"{null.get('median')}\\n"
                )

                f.write(
                    f"  null 95th: "
                    f"{null.get('p95')}\\n"
                )

                f.write(
                    f"  null 99th: "
                    f"{null.get('p99')}\\n"
                )

                f.write(
                    f"  empirical p: "
                    f"{null.get('empirical_p')}\\n"
                )
'''

    if old in text:
        backup = (
            script.with_suffix(
                ".pre_autonomous_backup.py"
            )
        )

        shutil.copy2(
            script,
            backup,
        )

        script.write_text(
            text.replace(old, new, 1),
            encoding="utf-8",
        )

        log(
            "Repaired existing overnight report "
            "None-handling bug."
        )

    else:
        log(
            "Existing overnight report bug already "
            "appears repaired or code has changed."
        )


# ---------------------------------------------------------------------
# RUN EXISTING CHECKPOINTED PIPELINE
# ---------------------------------------------------------------------

def run_existing_pipeline():
    script = ROOT / "pipeline" / "research_runner.py"

    if not script.exists():
        warn(
            "No overnight pipeline available."
        )
        return

    log(
        "Running existing checkpointed "
        "overnight pipeline."
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=str(ROOT),
        text=True,
    )

    if result.returncode != 0:
        warn(
            "Existing pipeline returned non-zero. "
            "Continuing with autonomous analysis "
            "because its checkpoints may still be usable."
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    overall_start = time.time()

    log("=" * 78)
    log("AUTONOMOUS LIGO RESEARCH RUN")
    log("=" * 78)
    log(f"ROOT: {ROOT}")
    log(f"DATA: {DATA}")
    log(f"OUTPUT: {OUT}")

    integrity = None
    data = {}
    events = []
    cross_rows = []
    control_rows = []
    provenance_rows = []
    residual_rows = []
    null_rows = []

    # -------------------------------------------------------------
    # 1. Environment
    # -------------------------------------------------------------

    with stage("ENVIRONMENT"):
        env = environment_info()

        atomic_json(
            STATS / "RUN_METADATA.json",
            env,
        )

    # -------------------------------------------------------------
    # 2. Input integrity
    # -------------------------------------------------------------

    with stage("INPUT INTEGRITY"):
        integrity = verify_input()

    # -------------------------------------------------------------
    # 3. Repair existing pipeline
    # -------------------------------------------------------------

    with stage("PIPELINE REPAIR"):
        repair_existing_pipeline()

    # -------------------------------------------------------------
    # 4. Resume existing pipeline
    # -------------------------------------------------------------

    with stage("CHECKPOINTED OVERNIGHT PIPELINE"):
        run_existing_pipeline()

    # -------------------------------------------------------------
    # 5. Load data
    # -------------------------------------------------------------

    with stage("CHANNEL EXTRACTION"):
        data = load_channels()

        atomic_json(
            STATS / "CHANNEL_LOAD_STATUS.json",
            {
                k: {
                    "channel": v["channel"],
                    "sample_rate": v["sample_rate"],
                    "start": v["start"],
                    "end": v["end"],
                    "n": v["n"],
                }
                for k, v in data.items()
            },
        )

        if "WFS_PIT" not in data:
            raise RuntimeError(
                "Target channel could not be loaded."
            )

    # -------------------------------------------------------------
    # 6. Independent detection
    # -------------------------------------------------------------

    with stage("INDEPENDENT MULTISCALE DETECTION"):
        detections = detect_target(
            data
        )

        write_csv(
            TABLES / "INDEPENDENT_DETECTIONS.csv",
            detections,
        )

    # -------------------------------------------------------------
    # 7. Event clustering
    # -------------------------------------------------------------

    with stage("EVENT CLUSTERING"):
        events = cluster_detections(
            detections
        )

        write_csv(
            TABLES / "EVENT_CLUSTERS.csv",
            events,
        )

        checkpoint(
            "EVENTS",
            {
                "n_detections": len(detections),
                "n_events": len(events),
                "events": events,
            },
        )

    # -------------------------------------------------------------
    # 8. Provenance
    # -------------------------------------------------------------

    with stage("CHANNEL PROVENANCE"):
        provenance_rows = provenance()

        write_csv(
            TABLES / "CHANNEL_PROVENANCE.csv",
            provenance_rows,
        )

    # -------------------------------------------------------------
    # 9. Event cross-channel analysis
    # -------------------------------------------------------------

    with stage("EVENT CROSS-CHANNEL ANALYSIS"):
        cross_rows = analyse_events(
            data,
            events,
        )

        write_csv(
            TABLES / "EVENT_CROSS_CHANNEL.csv",
            cross_rows,
        )

    # -------------------------------------------------------------
    # 10. Control statistics
    # -------------------------------------------------------------

    with stage("EVENT CONTROL ANALYSIS"):

        target = data["WFS_PIT"]

        for event in events:

            result = event_control_statistics(
                target,
                event,
            )

            if result:
                result["event_id"] = event[
                    "event_id"
                ]

                control_rows.append(
                    result
                )

        write_csv(
            TABLES / "EVENT_CONTROL_COMPARISONS.csv",
            control_rows,
        )

    # -------------------------------------------------------------
    # 11. OAF residual analysis
    # -------------------------------------------------------------

    with stage("OAF RESIDUAL ANALYSIS"):
        residual_rows = oaf_residual_analysis(
            data,
            events,
        )

        write_csv(
            TABLES / "OAF_RESIDUAL_ANALYSIS.csv",
            residual_rows,
        )

    # -------------------------------------------------------------
    # 12. Strong null models
    # -------------------------------------------------------------

    with stage("STRONG NULL MODELS"):
        null_rows = strong_null_tests(
            data,
            events,
        )

        atomic_json(
            STATS / "STRONG_NULL_TESTS.json",
            {
                "random_seed": 20260924,
                "results": null_rows,
            },
        )

    # -------------------------------------------------------------
    # 13. Multiple testing
    # -------------------------------------------------------------

    with stage("MULTIPLE TESTING"):

        pvalues = [
            x["empirical_p_plus1"]
            for x in null_rows
            if np.isfinite(
                x.get(
                    "empirical_p_plus1",
                    np.nan,
                )
            )
        ]

        qvalues = benjamini_hochberg(
            pvalues
        )

        multiple = {
            "method": "Benjamini-Hochberg",
            "n_tests": len(pvalues),
            "p_values": pvalues,
            "q_values": qvalues,
        }

        atomic_json(
            STATS / "MULTIPLE_TESTING.json",
            multiple,
        )

    # -------------------------------------------------------------
    # 14. Figures
    # -------------------------------------------------------------

    with stage("FIGURE GENERATION"):
        make_figures(
            data,
            events,
        )

    # -------------------------------------------------------------
    # 15. Reports
    # -------------------------------------------------------------

    with stage("REPORT GENERATION"):

        write_report(
            integrity,
            env,
            data,
            events,
            cross_rows,
            control_rows,
            provenance_rows,
            residual_rows,
            null_rows,
        )

        write_research_gap(
            events,
            residual_rows,
            null_rows,
        )

        write_next_experiment()

    # -------------------------------------------------------------
    # 16. Final results JSON
    # -------------------------------------------------------------

    with stage("FINAL MACHINE-READABLE RESULTS"):

        final = {
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "runtime_seconds": (
                time.time() -
                overall_start
            ),
            "input": integrity,
            "channels_loaded": len(data),
            "channels": {
                k: {
                    "channel": v["channel"],
                    "sample_rate": v["sample_rate"],
                    "n": v["n"],
                }
                for k, v in data.items()
            },
            "n_independent_detections": len(
                detections
            ),
            "n_event_clusters": len(
                events
            ),
            "n_cross_channel_results": len(
                cross_rows
            ),
            "n_control_results": len(
                control_rows
            ),
            "n_residual_results": len(
                residual_rows
            ),
            "n_strong_null_results": len(
                null_rows
            ),
            "events": events,
            "stages": STAGES,
            "warnings": WARNINGS,
            "errors": ERRORS,
        }

        atomic_json(
            STATS / "FINAL_AUTONOMOUS_RESULTS.json",
            final,
        )

    # -------------------------------------------------------------
    # 17. Final QC
    # -------------------------------------------------------------

    with stage("QUALITY CONTROL"):
        qc_failures = run_qc(
            integrity
        )

    # -------------------------------------------------------------
    # 18. Final completion marker
    # -------------------------------------------------------------

    completion = {
        "completed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "runtime_seconds": (
            time.time() -
            overall_start
        ),
        "qc_status": (
            "PASS"
            if not qc_failures
            else "FAIL"
        ),
        "input_sha256": sha256(DATA),
        "channels_loaded": len(data),
        "candidate_events": len(events),
        "errors": ERRORS,
        "warnings": WARNINGS,
    }

    atomic_json(
        OUT / "COMPLETE.json",
        completion,
    )

    if qc_failures:
        log(
            "RUN FINISHED WITH QC FAILURES."
        )
    else:
        log(
            "RUN FINISHED WITH QC PASS."
        )

    log("=" * 78)
    log("AUTONOMOUS RESEARCH RUN COMPLETE")
    log("=" * 78)


if __name__ == "__main__":
    main()


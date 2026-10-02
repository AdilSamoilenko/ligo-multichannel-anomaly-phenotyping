#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from gwpy.timeseries import TimeSeries
except Exception:
    TimeSeries = None

try:
    from scipy import signal
except Exception:
    signal = None


REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
AUX_DIR = DATA_DIR / "auxiliary"

OUT_DIR = REPO / "results_overnight" / "independent_replication"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DISCOVERY_R = 0.797704
DISCOVERY_LAG = 0.208984

DISCOVERY_START = 1376516121.0
DISCOVERY_END = 1376516128.0
DISCOVERY_FILE = "W10_H1_AUX_AR1.gwf"

WINDOW_SECONDS = 7.0
STEP_SECONDS = 1.0

MAX_LAG_SECONDS = 0.5
FROZEN_LAG_TOLERANCE = 0.0625

TARGET_SAMPLE_RATE = 1024.0

MIN_SAMPLES = 1024

PRIMARY_THRESHOLD = 0.70

SURROGATES_PER_WINDOW = 250

RNG_SEED = 20261002

MIN_WINDOW_SEPARATION = 7.0


@dataclass
class WindowResult:
    file: str
    start: float
    end: float

    r_max: float
    lag_max: float

    r_frozen: float
    frozen_lag: float

    passes_primary: bool
    passes_secondary: bool

    surrogate_count: int
    surrogate_mean: float
    surrogate_std: float
    surrogate_95: float
    surrogate_99: float

    empirical_p: float
    z_score: float

    significant_uncorrected: bool
    significant_fdr: bool

    channel_wfs_yaw: str
    channel_lsc_pop: str


def log(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"

    print(line, flush=True)

    with (OUT_DIR / "replication.log").open(
        "a",
        encoding="utf-8"
    ) as handle:
        handle.write(line + "\n")


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    finite = np.isfinite(x)

    if finite.sum() < MIN_SAMPLES:
        raise ValueError("Not enough finite samples")

    if not finite.all():
        indices = np.arange(len(x))
        good = np.flatnonzero(finite)

        x = np.interp(
            indices,
            good,
            x[good]
        )

    if signal is not None:
        x = signal.detrend(x, type="linear")
    else:
        x = x - np.mean(x)

    x = x - np.mean(x)

    standard_deviation = np.std(x)

    if not np.isfinite(standard_deviation):
        raise ValueError("Invalid standard deviation")

    if standard_deviation == 0:
        raise ValueError("Constant signal")

    return x / standard_deviation


def correlation_at_lag(
    x: np.ndarray,
    y: np.ndarray,
    lag_samples: int
) -> float:

    if lag_samples > 0:
        a = x[:-lag_samples]
        b = y[lag_samples:]

    elif lag_samples < 0:
        k = -lag_samples
        a = x[k:]
        b = y[:-k]

    else:
        a = x
        b = y

    if len(a) < MIN_SAMPLES:
        return float("nan")

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = math.sqrt(
        float(np.sum(a * a)) *
        float(np.sum(b * b))
    )

    if denominator == 0:
        return float("nan")

    return float(np.sum(a * b) / denominator)


def correlation_scan(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float,
    maximum_lag: float
) -> tuple[float, float]:

    maximum_lag_samples = int(
        round(maximum_lag * sample_rate)
    )

    best_r = float("-inf")
    best_lag = float("nan")

    for lag in range(
        -maximum_lag_samples,
        maximum_lag_samples + 1
    ):
        r = correlation_at_lag(
            x,
            y,
            lag
        )

        if np.isfinite(r) and r > best_r:
            best_r = r
            best_lag = lag / sample_rate

    if not np.isfinite(best_r):
        return float("nan"), float("nan")

    return float(best_r), float(best_lag)


def frozen_correlation(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float
) -> tuple[float, float]:

    centre = int(
        round(DISCOVERY_LAG * sample_rate)
    )

    tolerance = int(
        round(
            FROZEN_LAG_TOLERANCE *
            sample_rate
        )
    )

    best_r = float("-inf")
    best_lag = float("nan")

    for lag in range(
        centre - tolerance,
        centre + tolerance + 1
    ):
        r = correlation_at_lag(
            x,
            y,
            lag
        )

        if np.isfinite(r) and r > best_r:
            best_r = r
            best_lag = lag / sample_rate

    if not np.isfinite(best_r):
        return float("nan"), float("nan")

    return float(best_r), float(best_lag)


def phase_randomised_surrogate(
    x: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:

    n = len(x)

    spectrum = np.fft.rfft(x)

    phase = rng.uniform(
        0,
        2 * np.pi,
        len(spectrum)
    )

    phase[0] = 0.0

    if n % 2 == 0:
        phase[-1] = 0.0

    surrogate_spectrum = (
        np.abs(spectrum) *
        np.exp(1j * phase)
    )

    surrogate = np.fft.irfft(
        surrogate_spectrum,
        n=n
    )

    return normalise(surrogate)


def surrogate_distribution(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float,
    rng: np.random.Generator
) -> np.ndarray:

    values = np.empty(
        SURROGATES_PER_WINDOW,
        dtype=float
    )

    for i in range(SURROGATES_PER_WINDOW):

        surrogate = phase_randomised_surrogate(
            y,
            rng
        )

        r, _ = correlation_scan(
            x,
            surrogate,
            sample_rate,
            MAX_LAG_SECONDS
        )

        values[i] = r

    return values[np.isfinite(values)]


def apply_fdr(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    valid = np.isfinite(p_values)

    result = np.zeros(
        len(p_values),
        dtype=bool
    )

    indices = np.flatnonzero(valid)

    if len(indices) == 0:
        return result

    values = p_values[indices]

    order = np.argsort(values)
    sorted_values = values[order]

    m = len(sorted_values)

    thresholds = (
        np.arange(1, m + 1) *
        alpha /
        m
    )

    passing = sorted_values <= thresholds

    if not np.any(passing):
        return result

    last = np.flatnonzero(passing)[-1]

    selected = order[:last + 1]

    result[
        indices[selected]
    ] = True

    return result


def find_channel_names_in_text() -> tuple[list[str], list[str]]:

    wfs = set()
    lsc = set()

    roots = [
        DATA_DIR,
        REPO / "results",
        REPO / "results_overnight",
        REPO / "pipeline",
        REPO
    ]

    pattern = re.compile(
        r"H1:[A-Za-z0-9_.:\-]+"
    )

    checked = set()

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*"):

            if path in checked:
                continue

            checked.add(path)

            if not path.is_file():
                continue

            if path.suffix.lower() in {
                ".gwf",
                ".npy",
                ".npz",
                ".hdf5",
                ".h5",
                ".png",
                ".jpg",
                ".jpeg",
                ".pdf",
                ".zip"
            }:
                continue

            try:
                if path.stat().st_size > 5_000_000:
                    continue

                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

            except Exception:
                continue

            if (
                "WFS_YAW" not in text
                and "LSC_POP" not in text
            ):
                continue

            for channel in pattern.findall(text):

                upper = channel.upper()

                if (
                    "WFS" in upper
                    and "YAW" in upper
                ):
                    wfs.add(channel)

                if (
                    "LSC" in upper
                    and "POP" in upper
                ):
                    lsc.add(channel)

    return sorted(wfs), sorted(lsc)


def inspect_gwf_channels(path: Path) -> list[str]:

    if TimeSeries is None:
        return []

    try:
        from gwpy.timeseries import TimeSeriesDict

        data = TimeSeriesDict.read(
            str(path),
            "*",
            verbose=False
        )

        return sorted(data.keys())

    except Exception as exc:
        log(
            f"Could not inspect {path.name}: {exc}"
        )

        return []


def discover_channels(
    gwf_files: list[Path]
) -> tuple[str, str]:

    wfs, lsc = find_channel_names_in_text()

    log(f"WFS candidates: {wfs}")
    log(f"LSC candidates: {lsc}")

    if wfs and lsc:
        return wfs[0], lsc[0]

    for path in gwf_files:

        channels = inspect_gwf_channels(path)

        for channel in channels:

            upper = channel.upper()

            if (
                not wfs
                and "WFS" in upper
                and "YAW" in upper
            ):
                wfs.append(channel)

            if (
                not lsc
                and "LSC" in upper
                and "POP" in upper
            ):
                lsc.append(channel)

        if wfs and lsc:
            return wfs[0], lsc[0]

    raise RuntimeError(
        "Could not identify WFS_YAW and LSC_POP."
    )


def discover_gwf_files() -> list[Path]:

    files = sorted(
        AUX_DIR.rglob("*.gwf")
    )

    if not files:
        files = sorted(
            DATA_DIR.rglob("*.gwf")
        )

    if not files:
        raise RuntimeError(
            f"No GWF files found under {DATA_DIR}"
        )

    return files


def read_channel(
    path: Path,
    channel: str,
    start: float,
    end: float
) -> tuple[np.ndarray, float]:

    if TimeSeries is None:
        raise RuntimeError(
            "GWpy is not installed."
        )

    series = TimeSeries.read(
        str(path),
        channel,
        start=start,
        end=end
    )

    values = np.asarray(
        series.value,
        dtype=float
    )

    sample_rate = float(
        series.sample_rate.value
    )

    values, sample_rate = resample(
        values,
        sample_rate
    )

    return values, sample_rate


def resample(
    values: np.ndarray,
    sample_rate: float
) -> tuple[np.ndarray, float]:

    if abs(
        sample_rate -
        TARGET_SAMPLE_RATE
    ) < 1e-6:
        return values, sample_rate

    if signal is None:
        raise RuntimeError(
            "SciPy is required for resampling."
        )

    duration = (
        len(values) /
        sample_rate
    )

    new_length = int(
        round(
            duration *
            TARGET_SAMPLE_RATE
        )
    )

    values = signal.resample(
        values,
        new_length
    )

    return values, TARGET_SAMPLE_RATE


def get_common_time_range(
    path: Path,
    channel_a: str,
    channel_b: str
) -> Optional[tuple[float, float]]:

    if TimeSeries is None:
        return None

    try:
        from gwpy.timeseries import TimeSeriesDict

        data = TimeSeriesDict.read(
            str(path),
            [channel_a, channel_b],
            verbose=False
        )

        if (
            channel_a not in data
            or channel_b not in data
        ):
            return None

        a = data[channel_a]
        b = data[channel_b]

        start = max(
            float(a.t0.value),
            float(b.t0.value)
        )

        end = min(
            float(a.t0.value + a.duration.value),
            float(b.t0.value + b.duration.value)
        )

        if end - start < WINDOW_SECONDS:
            return None

        return start, end

    except Exception:
        return None


def make_windows(
    start: float,
    end: float
) -> list[tuple[float, float]]:

    windows = []

    current = start

    while current + WINDOW_SECONDS <= end:

        windows.append(
            (
                current,
                current + WINDOW_SECONDS
            )
        )

        current += STEP_SECONDS

    return windows


def is_discovery_window(
    filename: str,
    start: float,
    end: float
) -> bool:

    if filename != DISCOVERY_FILE:
        return False

    overlap = min(
        end,
        DISCOVERY_END
    ) - max(
        start,
        DISCOVERY_START
    )

    return overlap > 0


def analyse_window(
    path: Path,
    start: float,
    end: float,
    channel_wfs: str,
    channel_lsc: str,
    rng: np.random.Generator
) -> Optional[WindowResult]:

    try:

        x, fs_x = read_channel(
            path,
            channel_wfs,
            start,
            end
        )

        y, fs_y = read_channel(
            path,
            channel_lsc,
            start,
            end
        )

        if abs(fs_x - fs_y) > 1e-6:
            raise ValueError(
                "Channel sample rates do not match."
            )

        n = min(
            len(x),
            len(y)
        )

        x = x[:n]
        y = y[:n]

        if n < MIN_SAMPLES:
            return None

        x = normalise(x)
        y = normalise(y)

        r_max, lag_max = correlation_scan(
            x,
            y,
            fs_x,
            MAX_LAG_SECONDS
        )

        r_frozen, frozen_lag = frozen_correlation(
            x,
            y,
            fs_x
        )

        surrogates = surrogate_distribution(
            x,
            y,
            fs_x,
            rng
        )

        if len(surrogates) == 0:
            return None

        surrogate_mean = float(
            np.mean(surrogates)
        )

        surrogate_std = float(
            np.std(surrogates, ddof=1)
        )

        surrogate_95 = float(
            np.quantile(
                surrogates,
                0.95
            )
        )

        surrogate_99 = float(
            np.quantile(
                surrogates,
                0.99
            )
        )

        empirical_p = float(
            (
                np.sum(
                    surrogates >= r_max
                ) + 1
            ) /
            (
                len(surrogates) + 1
            )
        )

        if surrogate_std > 0:
            z_score = float(
                (
                    r_max -
                    surrogate_mean
                ) /
                surrogate_std
            )
        else:
            z_score = float("nan")

        return WindowResult(
            file=path.name,
            start=float(start),
            end=float(end),

            r_max=float(r_max),
            lag_max=float(lag_max),

            r_frozen=float(r_frozen),
            frozen_lag=float(frozen_lag),

            passes_primary=bool(
                r_frozen >= PRIMARY_THRESHOLD
            ),

            passes_secondary=bool(
                r_max >= DISCOVERY_R
            ),

            surrogate_count=len(surrogates),
            surrogate_mean=surrogate_mean,
            surrogate_std=surrogate_std,
            surrogate_95=surrogate_95,
            surrogate_99=surrogate_99,

            empirical_p=empirical_p,
            z_score=z_score,

            significant_uncorrected=bool(
                empirical_p < 0.05
            ),

            significant_fdr=False,

            channel_wfs_yaw=channel_wfs,
            channel_lsc_pop=channel_lsc
        )

    except Exception as exc:

        log(
            f"Window failed in {path.name} "
            f"{start:.3f}: {exc}"
        )

        return None


def write_csv(
    results: list[WindowResult],
    path: Path
) -> None:

    if not results:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(results[0]).keys()
            )
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                asdict(result)
            )


def write_json(
    results: list[WindowResult],
    path: Path
) -> None:

    payload = [
        asdict(result)
        for result in results
    ]

    path.write_text(
        json.dumps(
            payload,
            indent=2
        ),
        encoding="utf-8"
    )


def make_figures(
    results: list[WindowResult]
) -> None:

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(
            f"Could not import matplotlib: {exc}"
        )
        return

    if not results:
        return

    r_values = np.array(
        [r.r_frozen for r in results],
        dtype=float
    )

    p_values = np.array(
        [r.empirical_p for r in results],
        dtype=float
    )

    valid = np.isfinite(r_values)

    if np.any(valid):

        plt.figure(
            figsize=(11, 6)
        )

        plt.hist(
            r_values[valid],
            bins=30
        )

        plt.axvline(
            PRIMARY_THRESHOLD,
            linestyle="--",
            label="Primary threshold"
        )

        plt.xlabel(
            "Correlation at frozen lag"
        )

        plt.ylabel(
            "Number of windows"
        )

        plt.title(
            "Independent replication scan"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIG_DIR / "frozen_lag_distribution.png",
            dpi=180
        )

        plt.close()

    valid_p = np.isfinite(p_values)

    if np.any(valid_p):

        plt.figure(
            figsize=(11, 6)
        )

        values = np.clip(
            p_values[valid_p],
            1e-5,
            1.0
        )

        plt.hist(
            -np.log10(values),
            bins=30
        )

        plt.xlabel(
            "-log10 empirical p-value"
        )

        plt.ylabel(
            "Number of windows"
        )

        plt.title(
            "Replication-window significance distribution"
        )

        plt.tight_layout()

        plt.savefig(
            FIG_DIR / "p_value_distribution.png",
            dpi=180
        )

        plt.close()

    lag_values = np.array(
        [r.lag_max for r in results],
        dtype=float
    )

    valid_lag = np.isfinite(lag_values)

    if np.any(valid_lag):

        plt.figure(
            figsize=(11, 6)
        )

        plt.hist(
            lag_values[valid_lag],
            bins=30
        )

        plt.axvline(
            DISCOVERY_LAG,
            linestyle="--",
            label="Discovery lag"
        )

        plt.xlabel(
            "Maximum-correlation lag (s)"
        )

        plt.ylabel(
            "Number of windows"
        )

        plt.title(
            "Lag distribution"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIG_DIR / "lag_distribution.png",
            dpi=180
        )

        plt.close()


def summarise(
    results: list[WindowResult]
) -> dict:

    if not results:
        return {
            "windows": 0
        }

    frozen = np.array(
        [r.r_frozen for r in results],
        dtype=float
    )

    maximum = np.array(
        [r.r_max for r in results],
        dtype=float
    )

    p_values = np.array(
        [r.empirical_p for r in results],
        dtype=float
    )

    primary = (
        np.isfinite(frozen)
        & (frozen >= PRIMARY_THRESHOLD)
    )

    secondary = (
        np.isfinite(maximum)
        & (maximum >= DISCOVERY_R)
    )

    significant = (
        np.isfinite(p_values)
        & (p_values < 0.05)
    )

    return {
        "windows": len(results),

        "primary_passes": int(
            np.sum(primary)
        ),

        "secondary_passes": int(
            np.sum(secondary)
        ),

        "uncorrected_significant": int(
            np.sum(significant)
        ),

        "primary_fraction": float(
            np.mean(primary)
        ),

        "secondary_fraction": float(
            np.mean(secondary)
        ),

        "median_frozen_r": float(
            np.nanmedian(frozen)
        ),

        "median_max_r": float(
            np.nanmedian(maximum)
        ),

        "maximum_frozen_r": float(
            np.nanmax(frozen)
        ),

        "maximum_r": float(
            np.nanmax(maximum)
        ),

        "median_empirical_p": float(
            np.nanmedian(p_values)
        ),

        "minimum_empirical_p": float(
            np.nanmin(p_values)
        )
    }


def write_report(
    results: list[WindowResult],
    summary: dict,
    files: list[Path],
    channel_wfs: str,
    channel_lsc: str
) -> None:

    lines = []

    lines.append(
        "INDEPENDENT REPLICATION ANALYSIS"
    )
    lines.append(
        "=" * 42
    )
    lines.append("")
    lines.append(
        "This analysis tests a fixed cross-channel "
        "phenotype identified during the discovery analysis."
    )
    lines.append("")
    lines.append(
        "Discovery phenotype"
    )
    lines.append(
        f"  WFS channel: {channel_wfs}"
    )
    lines.append(
        f"  LSC channel: {channel_lsc}"
    )
    lines.append(
        f"  Discovery correlation: {DISCOVERY_R:.6f}"
    )
    lines.append(
        f"  Discovery lag: {DISCOVERY_LAG:.6f} s"
    )
    lines.append(
        f"  Discovery file: {DISCOVERY_FILE}"
    )
    lines.append(
        f"  Discovery interval: "
        f"{DISCOVERY_START:.3f} to "
        f"{DISCOVERY_END:.3f}"
    )
    lines.append("")
    lines.append(
        "Validation settings"
    )
    lines.append(
        f"  Window length: {WINDOW_SECONDS:.1f} s"
    )
    lines.append(
        f"  Window step: {STEP_SECONDS:.1f} s"
    )
    lines.append(
        f"  Lag search: +/- {MAX_LAG_SECONDS:.3f} s"
    )
    lines.append(
        f"  Frozen lag tolerance: "
        f"+/- {FROZEN_LAG_TOLERANCE:.4f} s"
    )
    lines.append(
        f"  Primary correlation threshold: "
        f"{PRIMARY_THRESHOLD:.2f}"
    )
    lines.append(
        f"  Surrogates per window: "
        f"{SURROGATES_PER_WINDOW}"
    )
    lines.append("")
    lines.append(
        "Input files"
    )

    for path in files:
        lines.append(
            f"  {path}"
        )

    lines.append("")
    lines.append(
        "Summary"
    )
    lines.append(
        f"  Windows analysed: "
        f"{summary.get('windows', 0)}"
    )
    lines.append(
        f"  Primary passes: "
        f"{summary.get('primary_passes', 0)}"
    )
    lines.append(
        f"  Primary fraction: "
        f"{summary.get('primary_fraction', float('nan')):.6f}"
    )
    lines.append(
        f"  Secondary passes: "
        f"{summary.get('secondary_passes', 0)}"
    )
    lines.append(
        f"  Secondary fraction: "
        f"{summary.get('secondary_fraction', float('nan')):.6f}"
    )
    lines.append(
        f"  Uncorrected significant windows: "
        f"{summary.get('uncorrected_significant', 0)}"
    )
    lines.append(
        f"  Median frozen-lag correlation: "
        f"{summary.get('median_frozen_r', float('nan')):.6f}"
    )
    lines.append(
        f"  Maximum frozen-lag correlation: "
        f"{summary.get('maximum_frozen_r', float('nan')):.6f}"
    )
    lines.append(
        f"  Median maximum correlation: "
        f"{summary.get('median_max_r', float('nan')):.6f}"
    )
    lines.append(
        f"  Maximum correlation: "
        f"{summary.get('maximum_r', float('nan')):.6f}"
    )
    lines.append(
        f"  Median empirical p-value: "
        f"{summary.get('median_empirical_p', float('nan')):.6g}"
    )
    lines.append(
        f"  Minimum empirical p-value: "
        f"{summary.get('minimum_empirical_p', float('nan')):.6g}"
    )
    lines.append("")

    if results:

        ordered = sorted(
            results,
            key=lambda r: (
                r.r_frozen
                if np.isfinite(r.r_frozen)
                else -np.inf
            ),
            reverse=True
        )

        lines.append(
            "Strongest validation windows"
        )

        for result in ordered[:20]:

            lines.append(
                f"  {result.file} "
                f"{result.start:.3f} "
                f"r_frozen={result.r_frozen:+.6f} "
                f"lag={result.frozen_lag:+.6f}s "
                f"r_max={result.r_max:+.6f} "
                f"max_lag={result.lag_max:+.6f}s "
                f"p={result.empirical_p:.6g}"
            )

    lines.append("")
    lines.append(
        "Interpretation"
    )
    lines.append(
        "  The discovery phenotype was fixed before "
        "the validation scan."
    )
    lines.append(
        "  The primary test evaluates correlation near "
        "the discovery lag rather than selecting a new "
        "lag for each validation window."
    )
    lines.append(
        "  The maximum-correlation statistic is reported "
        "separately because it involves an additional "
        "lag search."
    )
    lines.append(
        "  The surrogate test uses phase-randomised "
        "versions of the second channel."
    )
    lines.append(
        "  The empirical p-values are per-window values "
        "and are not sufficient by themselves to establish "
        "global discovery significance."
    )
    lines.append(
        "  The scan does not establish causality, a "
        "gravitational-wave origin, a new instrumental "
        "coupling, or novelty."
    )
    lines.append(
        "  Any candidate surviving this analysis should "
        "be tested against independent controls and "
        "physical channel provenance before stronger "
        "claims are made."
    )
    lines.append("")

    report_path = OUT_DIR / "INDEPENDENT_REPLICATION_REPORT.txt"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main() -> int:

    log("Starting independent replication scan.")

    if TimeSeries is None:
        log(
            "GWpy is not available in the current environment."
        )
        return 1

    if signal is None:
        log(
            "SciPy is not available in the current environment."
        )
        return 1

    files = discover_gwf_files()

    log(
        f"Found {len(files)} GWF files."
    )

    for path in files:
        log(
            f"Input: {path}"
        )

    channel_wfs, channel_lsc = discover_channels(
        files
    )

    log(
        f"WFS channel: {channel_wfs}"
    )

    log(
        f"LSC channel: {channel_lsc}"
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    results: list[WindowResult] = []

    for file_index, path in enumerate(files, start=1):

        log(
            f"Processing file {file_index}/{len(files)}: "
            f"{path.name}"
        )

        common_range = get_common_time_range(
            path,
            channel_wfs,
            channel_lsc
        )

        if common_range is None:
            log(
                f"No usable common range in {path.name}"
            )
            continue

        start, end = common_range

        windows = make_windows(
            start,
            end
        )

        log(
            f"Found {len(windows)} possible windows."
        )

        for index, (
            window_start,
            window_end
        ) in enumerate(windows, start=1):

            if is_discovery_window(
                path.name,
                window_start,
                window_end
            ):
                continue

            result = analyse_window(
                path,
                window_start,
                window_end,
                channel_wfs,
                channel_lsc,
                rng
            )

            if result is None:
                continue

            results.append(result)

            if index % 20 == 0:
                log(
                    f"{path.name}: "
                    f"{index}/{len(windows)} windows checked, "
                    f"{len(results)} usable results"
                )

    if not results:

        log(
            "No usable validation windows were produced."
        )

        return 1

    p_values = np.array(
        [
            result.empirical_p
            for result in results
        ],
        dtype=float
    )

    fdr_significant = apply_fdr(
        p_values,
        alpha=0.05
    )

    for result, significant in zip(
        results,
        fdr_significant
    ):
        result.significant_fdr = bool(
            significant
        )

    summary = summarise(
        results
    )

    summary[
        "fdr_significant"
    ] = int(
        np.sum(fdr_significant)
    )

    write_csv(
        results,
        OUT_DIR / "replication_windows.csv"
    )

    write_json(
        results,
        OUT_DIR / "replication_windows.json"
    )

    (OUT_DIR / "replication_summary.json").write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    make_figures(
        results
    )

    write_report(
        results,
        summary,
        files,
        channel_wfs,
        channel_lsc
    )

    log("")
    log("Independent replication scan complete.")
    log(
        f"Windows analysed: {summary['windows']}"
    )
    log(
        f"Primary passes: {summary['primary_passes']}"
    )
    log(
        f"FDR-significant windows: "
        f"{summary['fdr_significant']}"
    )
    log(
        f"Maximum frozen-lag r: "
        f"{summary['maximum_frozen_r']:.6f}"
    )
    log(
        f"Minimum empirical p: "
        f"{summary['minimum_empirical_p']:.6g}"
    )
    log(
        f"Report: "
        f"{OUT_DIR / 'INDEPENDENT_REPLICATION_REPORT.txt'}"
    )

    return 0


if __name__ == "__main__":

    try:
        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        log(
            "Stopped by user."
        )

        sys.exit(130)

    except Exception as exc:

        log(
            f"Fatal error: {exc}"
        )

        traceback_path = (
            OUT_DIR /
            "fatal_error.txt"
        )

        traceback_path.write_text(
            __import__("traceback").format_exc(),
            encoding="utf-8"
        )

        sys.exit(1)

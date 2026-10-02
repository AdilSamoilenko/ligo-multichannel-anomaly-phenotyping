#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    from gwpy.timeseries import TimeSeries
except ImportError:
    TimeSeries = None


ROOT = Path.cwd()

INPUT = ROOT / "results_overnight" / "independent_replication" / "replication_windows.csv"
OUTPUT = ROOT / "results_overnight" / "next_stage"

EPISODES_DIR = OUTPUT / "episodes"
CONTROLS_DIR = OUTPUT / "controls"
STRAIN_DIR = OUTPUT / "strain"
PROVENANCE_DIR = OUTPUT / "provenance"
STATISTICS_DIR = OUTPUT / "statistics"
FIGURES_DIR = OUTPUT / "figures"

AUX_DIR = ROOT / "data" / "auxiliary"
STRAIN_DATA_DIR = ROOT / "data" / "strain_files"

WFS_CHANNEL = "H1:IMC-WFS_A_DC_YAW_OUT_DQ"
LSC_CHANNEL = "H1:LSC-POP_A_LF_OUT_DQ"

DISCOVERY_FILE = "W10_H1_AUX_AR1.gwf"
DISCOVERY_START = 1376516121.0
DISCOVERY_END = 1376516128.0

SAMPLE_RATE = 1024.0
WINDOW_SECONDS = 7.0

FROZEN_LAG = 0.208984
LAG_TOLERANCE = 0.0625
MAX_LAG = 0.5

PRIMARY_THRESHOLD = 0.70

EPISODE_GAP = 1.0
CONTROL_MARGIN = 7.0
CONTROL_COUNT = 5

SURROGATES = 2000
SEED = 20261003

BANDS = {
    "0_2": (0.0, 2.0),
    "2_5": (2.0, 5.0),
    "5_10": (5.0, 10.0),
    "10_20": (10.0, 20.0),
    "20_50": (20.0, 50.0),
    "50_100": (50.0, 100.0),
    "100_300": (100.0, 300.0),
    "300_1000": (300.0, 1000.0),
}


def log(message: str) -> None:
    print(message, flush=True)


def make_directories() -> None:
    for directory in [
        OUTPUT,
        EPISODES_DIR,
        CONTROLS_DIR,
        STRAIN_DIR,
        PROVENANCE_DIR,
        STATISTICS_DIR,
        FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def read_input() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "file",
        "start",
        "end",
        "r_frozen",
        "frozen_lag",
        "r_max",
        "lag_max",
        "empirical_p",
        "passes_primary",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(sorted(missing))
        )

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["r_frozen"] = pd.to_numeric(df["r_frozen"], errors="coerce")
    df["frozen_lag"] = pd.to_numeric(df["frozen_lag"], errors="coerce")
    df["r_max"] = pd.to_numeric(df["r_max"], errors="coerce")
    df["lag_max"] = pd.to_numeric(df["lag_max"], errors="coerce")
    df["empirical_p"] = pd.to_numeric(
        df["empirical_p"], errors="coerce"
    )

    df["passes_primary"] = (
        df["passes_primary"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return df


def merge_windows(df: pd.DataFrame) -> pd.DataFrame:
    passing = df[
        df["passes_primary"]
        & np.isfinite(df["start"])
        & np.isfinite(df["end"])
    ].copy()

    if passing.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "file",
                "start",
                "end",
                "duration",
                "window_count",
                "maximum_r_frozen",
                "median_r_frozen",
                "maximum_r_max",
                "median_lag",
                "lag_std",
                "minimum_empirical_p",
            ]
        )

    rows = []

    for filename, group in passing.groupby("file"):
        group = group.sort_values("start")

        current = []
        current_end = None
        episode_number = 0

        def emit(items):
            nonlocal episode_number

            if not items:
                return

            block = pd.DataFrame(items)

            start = float(block["start"].min())
            end = float(block["end"].max())

            lags = block["frozen_lag"].dropna().to_numpy()

            rows.append(
                {
                    "episode_id": f"E{episode_number:04d}",
                    "file": filename,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                    "window_count": len(block),
                    "maximum_r_frozen": float(
                        block["r_frozen"].max()
                    ),
                    "median_r_frozen": float(
                        block["r_frozen"].median()
                    ),
                    "maximum_r_max": float(block["r_max"].max()),
                    "median_lag": (
                        float(np.median(lags))
                        if len(lags)
                        else np.nan
                    ),
                    "lag_std": (
                        float(np.std(lags, ddof=1))
                        if len(lags) > 1
                        else 0.0
                    ),
                    "minimum_empirical_p": float(
                        block["empirical_p"].min()
                    ),
                }
            )

            episode_number += 1

        for _, row in group.iterrows():
            start = float(row["start"])
            end = float(row["end"])

            if current_end is None:
                current = [row]
                current_end = end
                continue

            if start <= current_end + EPISODE_GAP:
                current.append(row)
                current_end = max(current_end, end)
            else:
                emit(current)
                current = [row]
                current_end = end

        emit(current)

    episodes = pd.DataFrame(rows)

    if not episodes.empty:
        episodes = episodes.sort_values(
            ["file", "start"]
        ).reset_index(drop=True)

        episode_ids = []

        for index, (_, row) in enumerate(episodes.iterrows()):
            episode_ids.append(f"E{index + 1:04d}")

        episodes["episode_id"] = episode_ids

    return episodes


def make_control_windows(
    df: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    rng = np.random.default_rng(SEED)

    for filename, group in df.groupby("file"):
        group = group.sort_values("start")

        usable = group[
            np.isfinite(group["start"])
            & np.isfinite(group["end"])
        ].copy()

        if usable.empty:
            continue

        forbidden = []

        for _, episode in episodes[
            episodes["file"] == filename
        ].iterrows():

            forbidden.append(
                (
                    float(episode["start"]) - CONTROL_MARGIN,
                    float(episode["end"]) + CONTROL_MARGIN,
                )
            )

        candidates = []

        for _, row in usable.iterrows():
            start = float(row["start"])
            end = float(row["end"])

            blocked = any(
                start < forbidden_end
                and end > forbidden_start
                for forbidden_start, forbidden_end in forbidden
            )

            if not blocked:
                candidates.append((start, end))

        if not candidates:
            continue

        rng.shuffle(candidates)

        selected = candidates[:CONTROL_COUNT]

        for number, (start, end) in enumerate(selected, start=1):
            rows.append(
                {
                    "control_id": f"{filename}_C{number:02d}",
                    "file": filename,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )

    return pd.DataFrame(rows)


def read_auxiliary_segment(
    filename: str,
    start: float,
    end: float,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:

    if TimeSeries is None:
        return None

    path = AUX_DIR / filename

    if not path.exists():
        return None

    try:
        wfs = TimeSeries.read(
            str(path),
            WFS_CHANNEL,
            start=start,
            end=end,
        )

        lsc = TimeSeries.read(
            str(path),
            LSC_CHANNEL,
            start=start,
            end=end,
        )
    except Exception:
        return None

    if len(wfs) < 128 or len(lsc) < 128:
        return None

    wfs = wfs.resample(SAMPLE_RATE)
    lsc = lsc.resample(SAMPLE_RATE)

    n = min(len(wfs), len(lsc))

    if n < 128:
        return None

    x = np.asarray(wfs.value[:n], dtype=float)
    y = np.asarray(lsc.value[:n], dtype=float)

    return x, y, SAMPLE_RATE


def frozen_correlation(
    x: np.ndarray,
    y: np.ndarray,
    lag: float,
    sample_rate: float,
) -> float:

    shift = int(round(lag * sample_rate))

    if abs(shift) >= min(len(x), len(y)):
        return np.nan

    if shift > 0:
        a = x[:-shift]
        b = y[shift:]
    elif shift < 0:
        a = x[-shift:]
        b = y[:shift]
    else:
        a = x
        b = y

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 64:
        return np.nan

    a = a[mask]
    b = b[mask]

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def lag_scan(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float,
    centre: float,
    tolerance: float,
) -> tuple[float, float]:

    low = centre - tolerance
    high = centre + tolerance

    lags = np.arange(
        low,
        high + 1.0 / sample_rate,
        1.0 / sample_rate,
    )

    values = []

    for lag in lags:
        values.append(
            frozen_correlation(
                x,
                y,
                float(lag),
                sample_rate,
            )
        )

    values = np.asarray(values)

    if not np.any(np.isfinite(values)):
        return np.nan, np.nan

    index = np.nanargmax(np.abs(values))

    return float(values[index]), float(lags[index])


def power_spectrum(
    x: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:

    nperseg = min(4096, len(x))

    return signal.welch(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        detrend="constant",
    )


def calculate_bandpowers(
    x: np.ndarray,
    sample_rate: float,
) -> dict:

    frequencies, powers = power_spectrum(
        x,
        sample_rate,
    )

    result = {}

    for name, (low, high) in BANDS.items():
        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        if mask.sum() < 2:
            result[name] = np.nan
        else:
            result[name] = float(
                np.trapezoid(
                    powers[mask],
                    frequencies[mask],
                )
            )

    return result


def calculate_coherence(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float,
) -> dict:

    nperseg = min(4096, len(x), len(y))

    frequencies, coherence = signal.coherence(
        x,
        y,
        fs=sample_rate,
        nperseg=nperseg,
    )

    result = {
        "maximum_coherence": np.nan,
        "frequency_at_maximum_coherence": np.nan,
    }

    mask = (
        np.isfinite(frequencies)
        & np.isfinite(coherence)
        & (frequencies >= 1.0)
        & (frequencies <= 500.0)
    )

    if not mask.any():
        return result

    selected_frequencies = frequencies[mask]
    selected_coherence = coherence[mask]

    index = np.argmax(selected_coherence)

    result["maximum_coherence"] = float(
        selected_coherence[index]
    )

    result["frequency_at_maximum_coherence"] = float(
        selected_frequencies[index]
    )

    return result


def circular_shift_null(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float,
    lag: float,
    n_surrogates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:

    observed = frozen_correlation(
        x,
        y,
        lag,
        sample_rate,
    )

    if not np.isfinite(observed):
        return np.nan, np.nan

    minimum_shift = int(
        max(
            sample_rate,
            abs(lag) * sample_rate + 1,
        )
    )

    maximum_shift = len(y) - minimum_shift

    if maximum_shift <= minimum_shift:
        return observed, np.nan

    null_values = np.empty(n_surrogates)

    for index in range(n_surrogates):
        shift = int(
            rng.integers(
                minimum_shift,
                maximum_shift,
            )
        )

        shifted = np.roll(y, shift)

        null_values[index] = frozen_correlation(
            x,
            shifted,
            lag,
            sample_rate,
        )

    null_values = null_values[np.isfinite(null_values)]

    if len(null_values) == 0:
        return observed, np.nan

    p_value = (
        1.0
        + np.sum(
            np.abs(null_values)
            >= abs(observed)
        )
    ) / (len(null_values) + 1.0)

    return observed, float(p_value)


def analyse_auxiliary_windows(
    episodes: pd.DataFrame,
    controls: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(SEED)

    episode_results = []
    control_results = []

    for _, row in episodes.iterrows():

        filename = row["file"]
        start = float(row["start"])
        end = float(row["end"])

        log(
            f"Episode {row['episode_id']}: "
            f"{filename} {start:.3f} to {end:.3f}"
        )

        data = read_auxiliary_segment(
            filename,
            start,
            end,
        )

        if data is None:
            episode_results.append(
                {
                    "episode_id": row["episode_id"],
                    "file": filename,
                    "start": start,
                    "end": end,
                    "r_frozen": np.nan,
                    "r_lag_local": np.nan,
                    "lag_local": np.nan,
                    "null_p": np.nan,
                    "maximum_coherence": np.nan,
                    "coherence_frequency": np.nan,
                }
            )
            continue

        x, y, fs = data

        r_frozen = frozen_correlation(
            x,
            y,
            FROZEN_LAG,
            fs,
        )

        r_local, lag_local = lag_scan(
            x,
            y,
            fs,
            FROZEN_LAG,
            LAG_TOLERANCE,
        )

        _, null_p = circular_shift_null(
            x,
            y,
            fs,
            FROZEN_LAG,
            SURROGATES,
            rng,
        )

        coh = calculate_coherence(
            x,
            y,
            fs,
        )

        row_result = {
            "episode_id": row["episode_id"],
            "file": filename,
            "start": start,
            "end": end,
            "duration": end - start,
            "r_frozen": r_frozen,
            "r_lag_local": r_local,
            "lag_local": lag_local,
            "null_p": null_p,
            "maximum_coherence": coh[
                "maximum_coherence"
            ],
            "coherence_frequency": coh[
                "frequency_at_maximum_coherence"
            ],
        }

        for name, value in calculate_bandpowers(x, fs).items():
            row_result[f"wfs_bp_{name}"] = value

        for name, value in calculate_bandpowers(y, fs).items():
            row_result[f"lsc_bp_{name}"] = value

        episode_results.append(row_result)

    for _, row in controls.iterrows():

        filename = row["file"]
        start = float(row["start"])
        end = float(row["end"])

        data = read_auxiliary_segment(
            filename,
            start,
            end,
        )

        if data is None:
            continue

        x, y, fs = data

        r_frozen = frozen_correlation(
            x,
            y,
            FROZEN_LAG,
            fs,
        )

        r_local, lag_local = lag_scan(
            x,
            y,
            fs,
            FROZEN_LAG,
            LAG_TOLERANCE,
        )

        coh = calculate_coherence(
            x,
            y,
            fs,
        )

        result = {
            "control_id": row["control_id"],
            "file": filename,
            "start": start,
            "end": end,
            "duration": end - start,
            "r_frozen": r_frozen,
            "r_lag_local": r_local,
            "lag_local": lag_local,
            "maximum_coherence": coh[
                "maximum_coherence"
            ],
            "coherence_frequency": coh[
                "frequency_at_maximum_coherence"
            ],
        }

        for name, value in calculate_bandpowers(x, fs).items():
            result[f"wfs_bp_{name}"] = value

        for name, value in calculate_bandpowers(y, fs).items():
            result[f"lsc_bp_{name}"] = value

        control_results.append(result)

    return (
        pd.DataFrame(episode_results),
        pd.DataFrame(control_results),
    )


def compare_episode_controls(
    episodes: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict:

    result = {
        "episode_count": int(len(episodes)),
        "control_count": int(len(controls)),
        "tests": {},
    }

    if episodes.empty or controls.empty:
        return result

    metrics = [
        "r_frozen",
        "r_lag_local",
        "lag_local",
        "maximum_coherence",
    ]

    for metric in metrics:

        if metric not in episodes or metric not in controls:
            continue

        a = pd.to_numeric(
            episodes[metric],
            errors="coerce",
        ).dropna().to_numpy()

        b = pd.to_numeric(
            controls[metric],
            errors="coerce",
        ).dropna().to_numpy()

        if len(a) < 2 or len(b) < 2:
            continue

        test = stats.mannwhitneyu(
            a,
            b,
            alternative="two-sided",
        )

        result["tests"][metric] = {
            "episode_median": float(np.median(a)),
            "control_median": float(np.median(b)),
            "episode_count": int(len(a)),
            "control_count": int(len(b)),
            "mann_whitney_u": float(test.statistic),
            "p_value": float(test.pvalue),
        }

    return result


def strain_candidates() -> list[Path]:

    if not STRAIN_DATA_DIR.exists():
        return []

    return sorted(
        [
            path
            for path in STRAIN_DATA_DIR.iterdir()
            if path.suffix.lower()
            in {".hdf5", ".h5", ".gwf"}
        ]
    )


def read_strain(
    path: Path,
    start: float,
    end: float,
) -> Optional[tuple[np.ndarray, float]]:

    if TimeSeries is None:
        return None

    try:
        strain = TimeSeries.read(
            str(path),
            start=start,
            end=end,
        )
    except Exception:
        return None

    if len(strain) < 128:
        return None

    strain = strain.resample(SAMPLE_RATE)

    values = np.asarray(
        strain.value,
        dtype=float,
    )

    if len(values) < 128:
        return None

    return values, SAMPLE_RATE


def strain_band_rms(
    x: np.ndarray,
    sample_rate: float,
    low: float,
    high: float,
) -> float:

    if len(x) < 128:
        return np.nan

    nyquist = sample_rate / 2.0

    high = min(high, nyquist * 0.99)

    if low <= 0 or high <= low:
        return np.nan

    sos = signal.butter(
        4,
        [low, high],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    filtered = signal.sosfiltfilt(
        sos,
        x,
    )

    return float(np.sqrt(np.mean(filtered ** 2)))


def analyse_strain(
    episodes: pd.DataFrame,
) -> pd.DataFrame:

    files = strain_candidates()

    if not files:
        log("No strain files found.")
        return pd.DataFrame()

    results = []

    for _, episode in episodes.iterrows():

        start = float(episode["start"])
        end = float(episode["end"])

        found = False

        for path in files:

            data = read_strain(
                path,
                start,
                end,
            )

            if data is None:
                continue

            strain, fs = data

            row = {
                "episode_id": episode["episode_id"],
                "auxiliary_file": episode["file"],
                "strain_file": path.name,
                "start": start,
                "end": end,
                "strain_rms": float(
                    np.sqrt(
                        np.mean(
                            strain ** 2
                        )
                    )
                ),
            }

            frequencies, powers = power_spectrum(
                strain,
                fs,
            )

            for name, (low, high) in BANDS.items():

                mask = (
                    (frequencies >= low)
                    & (frequencies < high)
                )

                if mask.sum() >= 2:
                    row[f"strain_bp_{name}"] = float(
                        np.trapezoid(
                            powers[mask],
                            frequencies[mask],
                        )
                    )
                else:
                    row[f"strain_bp_{name}"] = np.nan

                row[f"strain_rms_{name}"] = strain_band_rms(
                    strain,
                    fs,
                    low,
                    high,
                )

            results.append(row)
            found = True
            break

        if not found:
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "auxiliary_file": episode["file"],
                    "strain_file": None,
                    "start": start,
                    "end": end,
                    "strain_rms": np.nan,
                }
            )

    return pd.DataFrame(results)


def search_repository_for_provenance() -> dict:

    patterns = [
        "IMC-WFS_A_DC_YAW_OUT_DQ",
        "LSC-POP_A_LF_OUT_DQ",
        "WFS_YAW",
        "LSC_POP",
        "OAF",
        "GPS",
        "derived",
        "prefilt",
        "filter",
    ]

    extensions = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".csv",
    }

    results = {}

    for pattern in patterns:

        matches = []

        for path in ROOT.rglob("*"):

            if not path.is_file():
                continue

            if path.is_relative_to(ROOT / ".git"):
                continue

            if path.suffix.lower() not in extensions:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            if pattern.lower() in text.lower():
                matches.append(
                    str(path.relative_to(ROOT))
                )

        results[pattern] = sorted(set(matches))

    return results


def calculate_fdr(
    p_values: pd.Series,
) -> np.ndarray:

    values = np.asarray(
        p_values,
        dtype=float,
    )

    valid = np.isfinite(values)

    corrected = np.full(
        len(values),
        np.nan,
    )

    if not valid.any():
        return corrected

    p = values[valid]

    order = np.argsort(p)
    ranked = p[order]

    m = len(ranked)

    adjusted = ranked * m / np.arange(
        1,
        m + 1,
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    inverse = np.empty_like(order)
    inverse[order] = np.arange(m)

    corrected[valid] = adjusted[inverse]

    return corrected


def write_report(
    discovery: pd.DataFrame,
    episodes: pd.DataFrame,
    episode_results: pd.DataFrame,
    controls: pd.DataFrame,
    control_results: pd.DataFrame,
    comparison: dict,
    strain_results: pd.DataFrame,
    provenance: dict,
) -> None:

    report = OUTPUT / "NEXT_STAGE_RESEARCH_REPORT.txt"

    with report.open("w", encoding="utf-8") as handle:

        handle.write(
            "LIGO MULTICHANNEL ANOMALY PHENOTYPING\n"
        )
        handle.write(
            "Next-stage validation report\n\n"
        )

        handle.write("Discovery definition\n")
        handle.write("-------------------\n")
        handle.write(
            f"WFS channel: {WFS_CHANNEL}\n"
        )
        handle.write(
            f"LSC channel: {LSC_CHANNEL}\n"
        )
        handle.write(
            f"Discovery file: {DISCOVERY_FILE}\n"
        )
        handle.write(
            f"Discovery interval: "
            f"{DISCOVERY_START:.3f} - "
            f"{DISCOVERY_END:.3f}\n"
        )
        handle.write(
            f"Frozen lag: {FROZEN_LAG:.6f} s\n"
        )
        handle.write(
            f"Lag tolerance: "
            f"{LAG_TOLERANCE:.6f} s\n"
        )
        handle.write(
            f"Primary threshold: "
            f"r >= {PRIMARY_THRESHOLD:.2f}\n\n"
        )

        handle.write("Window to episode conversion\n")
        handle.write("---------------------------\n")
        handle.write(
            f"Primary-pass windows: "
            f"{int(discovery['passes_primary'].sum())}\n"
        )
        handle.write(
            f"Candidate episodes: "
            f"{len(episodes)}\n\n"
        )

        if not episodes.empty:
            handle.write(
                episodes.to_string(index=False)
            )
            handle.write("\n\n")

        handle.write("Episode analysis\n")
        handle.write("----------------\n")

        if episode_results.empty:
            handle.write("No usable episode data.\n\n")
        else:
            handle.write(
                episode_results.to_string(index=False)
            )
            handle.write("\n\n")

        handle.write("Control analysis\n")
        handle.write("----------------\n")
        handle.write(
            f"Control windows selected: "
            f"{len(controls)}\n\n"
        )

        if not control_results.empty:
            handle.write(
                control_results.to_string(index=False)
            )
            handle.write("\n\n")

        handle.write("Episode-control comparison\n")
        handle.write("--------------------------\n")
        handle.write(
            json.dumps(
                comparison,
                indent=2,
            )
        )
        handle.write("\n\n")

        handle.write("Strain analysis\n")
        handle.write("---------------\n")

        if strain_results.empty:
            handle.write(
                "No usable strain analysis was completed.\n"
            )
        else:
            handle.write(
                strain_results.to_string(
                    index=False
                )
            )
            handle.write("\n")

        handle.write("\nProvenance search\n")
        handle.write("-----------------\n")

        for pattern, matches in provenance.items():
            handle.write(
                f"\n{pattern}\n"
            )

            for match in matches:
                handle.write(
                    f"  {match}\n"
                )

        handle.write("\nInterpretation\n")
        handle.write("--------------\n")
        handle.write(
            "The episode-level analysis is intended to "
            "determine whether the previously identified "
            "window-level relationship forms distinct "
            "recurring episodes. Overlapping windows are "
            "not treated as independent observations.\n\n"
        )
        handle.write(
            "The control comparison is descriptive and "
            "does not by itself establish a causal "
            "relationship or a new detector coupling.\n\n"
        )
        handle.write(
            "A relationship with auxiliary channels does "
            "not establish an astrophysical origin. "
            "Channel provenance and known detector "
            "control pathways must be examined before "
            "any novelty claim is made.\n"
        )


def main() -> None:

    make_directories()

    log("Loading replication results.")
    discovery = read_input()

    log("Identifying candidate episodes.")
    episodes = merge_windows(discovery)

    episodes.to_csv(
        EPISODES_DIR / "episodes.csv",
        index=False,
    )

    save_json(
        EPISODES_DIR / "episodes.json",
        episodes.to_dict(orient="records"),
    )

    log(
        f"Candidate episodes: {len(episodes)}"
    )

    controls = make_control_windows(
        discovery,
        episodes,
    )

    controls.to_csv(
        CONTROLS_DIR / "control_windows.csv",
        index=False,
    )

    save_json(
        CONTROLS_DIR / "control_windows.json",
        controls.to_dict(orient="records"),
    )

    log(
        f"Control windows: {len(controls)}"
    )

    log("Analysing auxiliary episodes and controls.")

    episode_results, control_results = (
        analyse_auxiliary_windows(
            episodes,
            controls,
        )
    )

    if not episode_results.empty:
        episode_results["fdr_p"] = calculate_fdr(
            episode_results["null_p"]
        )

        episode_results[
            "fdr_significant"
        ] = (
            episode_results["fdr_p"] <= 0.05
        )

    episode_results.to_csv(
        EPISODES_DIR / "episode_analysis.csv",
        index=False,
    )

    control_results.to_csv(
        CONTROLS_DIR / "control_analysis.csv",
        index=False,
    )

    save_json(
        EPISODES_DIR / "episode_analysis.json",
        episode_results.to_dict(
            orient="records"
        ),
    )

    save_json(
        CONTROLS_DIR / "control_analysis.json",
        control_results.to_dict(
            orient="records"
        ),
    )

    comparison = compare_episode_controls(
        episode_results,
        control_results,
    )

    save_json(
        STATISTICS_DIR / "episode_control_comparison.json",
        comparison,
    )

    log("Analysing strain coupling.")

    strain_results = analyse_strain(
        episodes
    )

    strain_results.to_csv(
        STRAIN_DIR / "strain_episode_analysis.csv",
        index=False,
    )

    save_json(
        STRAIN_DIR / "strain_episode_analysis.json",
        strain_results.to_dict(
            orient="records"
        ),
    )

    log("Searching repository for channel provenance.")

    provenance = search_repository_for_provenance()

    save_json(
        PROVENANCE_DIR / "channel_provenance_search.json",
        provenance,
    )

    write_report(
        discovery,
        episodes,
        episode_results,
        controls,
        control_results,
        comparison,
        strain_results,
        provenance,
    )

    summary = {
        "input": str(INPUT),
        "candidate_windows": int(
            discovery["passes_primary"].sum()
        ),
        "candidate_episodes": int(
            len(episodes)
        ),
        "control_windows": int(
            len(controls)
        ),
        "episode_results": int(
            len(episode_results)
        ),
        "control_results": int(
            len(control_results)
        ),
        "strain_results": int(
            len(strain_results)
        ),
        "configuration": {
            "wfs_channel": WFS_CHANNEL,
            "lsc_channel": LSC_CHANNEL,
            "frozen_lag": FROZEN_LAG,
            "lag_tolerance": LAG_TOLERANCE,
            "primary_threshold": PRIMARY_THRESHOLD,
            "surrogates": SURROGATES,
            "seed": SEED,
        },
    }

    save_json(
        OUTPUT / "next_stage_summary.json",
        summary,
    )

    log("")
    log("Next-stage analysis complete.")
    log(
        f"Candidate windows: "
        f"{summary['candidate_windows']}"
    )
    log(
        f"Candidate episodes: "
        f"{summary['candidate_episodes']}"
    )
    log(
        f"Control windows: "
        f"{summary['control_windows']}"
    )
    log(
        f"Strain results: "
        f"{summary['strain_results']}"
    )
    log(
        f"Report: "
        f"{OUTPUT / 'NEXT_STAGE_RESEARCH_REPORT.txt'}"
    )


if __name__ == "__main__":
    main()

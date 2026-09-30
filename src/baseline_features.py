from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats


WINDOW_DIR = Path("data/windows")
OUTPUT_FILE = Path("data/H1_O4a_baseline_features.csv")

SAMPLE_RATE = 4096


def extract_features(x):
    if not np.all(np.isfinite(x)):
        raise ValueError("Window contains non-finite samples.")

    features = {}

    features["mean"] = np.mean(x)
    features["std"] = np.std(x)
    features["rms"] = np.sqrt(np.mean(x**2))
    features["peak_abs"] = np.max(np.abs(x))

    if features["rms"] != 0:
        features["peak_to_rms"] = (
            features["peak_abs"] / features["rms"]
        )
    else:
        features["peak_to_rms"] = np.nan

    features["skewness"] = stats.skew(x)
    features["kurtosis"] = stats.kurtosis(x)

    frequencies, psd = signal.welch(
        x,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=16384,
        noverlap=8192,
        scaling="density",
    )

    mask = (
        (frequencies >= 10)
        & (frequencies <= 2000)
    )

    f = frequencies[mask]
    p = psd[mask]

    total_power = np.trapezoid(p, f)
    features["spectral_power"] = total_power

    if total_power > 0:
        spectral_centroid = (
            np.trapezoid(f * p, f) / total_power
        )

        spectral_bandwidth = np.sqrt(
            np.trapezoid(
                ((f - spectral_centroid) ** 2) * p,
                f,
            )
            / total_power
        )

        features["spectral_centroid"] = spectral_centroid
        features["spectral_bandwidth"] = spectral_bandwidth

    else:
        features["spectral_centroid"] = np.nan
        features["spectral_bandwidth"] = np.nan

    bands = {
        "10_30Hz": (10, 30),
        "30_100Hz": (30, 100),
        "100_300Hz": (100, 300),
        "300_1000Hz": (300, 1000),
        "1000_2000Hz": (1000, 2000),
    }

    for name, (low, high) in bands.items():
        band_mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        if np.any(band_mask):
            features[f"power_{name}"] = np.trapezoid(
                psd[band_mask],
                frequencies[band_mask],
            )
        else:
            features[f"power_{name}"] = np.nan

    return features


def main():
    files = sorted(WINDOW_DIR.glob("*.npy"))

    if len(files) != 30:
        raise RuntimeError(
            f"Expected 30 windows, found {len(files)}"
        )

    rows = []
    rejected = []

    for i, file in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {file.name}")

        x = np.load(file)

        if not np.all(np.isfinite(x)):
            print("    REJECTED: non-finite samples")
            rejected.append(
                {
                    "window": i,
                    "file": file.name,
                    "reason": "non-finite samples",
                }
            )
            continue

        features = extract_features(x)
        features["window"] = i
        features["file"] = file.name

        rows.append(features)

    table = pd.DataFrame(rows)

    identifier_columns = ["window", "file"]

    other_columns = [
        column
        for column in table.columns
        if column not in identifier_columns
    ]

    table = table[
        identifier_columns + other_columns
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Baseline feature extraction complete.")
    print(f"Valid windows: {len(table)}")
    print(f"Rejected windows: {len(rejected)}")
    print(f"Features: {len(table.columns) - 2}")
    print(f"Saved: {OUTPUT_FILE}")

    if rejected:
        print()
        print("Rejected windows:")
        for item in rejected:
            print(
                f"  Window {item['window']}: "
                f"{item['file']} "
                f"({item['reason']})"
            )


if __name__ == "__main__":
    main()
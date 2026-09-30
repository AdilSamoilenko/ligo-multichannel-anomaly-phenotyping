from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
import csv

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"
CHANNEL = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

ts = TimeSeries.read(PATH, channel=CHANNEL)
x = np.asarray(ts.value, dtype=np.float64)

fs = float(ts.sample_rate.value)
gps_start = float(ts.t0.value)

print("Loaded:", CHANNEL)
print("Samples:", len(x))
print("Sample rate:", fs, "Hz")
print("Duration:", len(x) / fs, "s")

median = np.median(x)
mad = np.median(np.abs(x - median))
robust_sigma = 1.4826 * mad

print("\nRobust baseline")
print("Median:", median)
print("MAD:", mad)
print("Robust sigma:", robust_sigma)

window_seconds = 0.25
window = int(window_seconds * fs)
n_windows = len(x) // window

results = []

for i in range(n_windows):
    start = i * window
    end = start + window
    segment = x[start:end]

    mean = np.mean(segment)
    std = np.std(segment)
    rms = np.sqrt(np.mean(segment ** 2))
    maximum = np.max(segment)
    minimum = np.min(segment)

    mean_score = abs(mean - median) / robust_sigma
    rms_score = rms / np.sqrt(np.mean(x ** 2))

    results.append({
        "window": i,
        "gps_start": gps_start + start / fs,
        "gps_end": gps_start + end / fs,
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": minimum,
        "max": maximum,
        "mean_score": mean_score,
        "rms_score": rms_score
    })

by_mean = sorted(results, key=lambda r: r["mean_score"], reverse=True)
by_rms = sorted(results, key=lambda r: r["rms_score"], reverse=True)

print("\nTop 10 windows by mean deviation")

for r in by_mean[:10]:
    print(
        f"GPS {r['gps_start']:.6f} - {r['gps_end']:.6f} | "
        f"mean={r['mean']:.6g} | "
        f"std={r['std']:.6g} | "
        f"score={r['mean_score']:.3f}"
    )

print("\nTop 10 windows by RMS")

for r in by_rms[:10]:
    print(
        f"GPS {r['gps_start']:.6f} - {r['gps_end']:.6f} | "
        f"RMS={r['rms']:.6g} | "
        f"score={r['rms_score']:.3f}"
    )

with open("anomaly_windows.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "window",
            "gps_start",
            "gps_end",
            "mean",
            "std",
            "rms",
            "min",
            "max",
            "mean_score",
            "rms_score"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nSaved: anomaly_windows.csv")

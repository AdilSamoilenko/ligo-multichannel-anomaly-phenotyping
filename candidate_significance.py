from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

target_name = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"
reference_name = "H1:LSC-POP_A_LF_OUT_DQ"

target = TimeSeries.read(PATH, channel=target_name)
reference = TimeSeries.read(PATH, channel=reference_name)

x = np.asarray(target.value, dtype=float)
y = np.asarray(reference.value, dtype=float)

n = min(len(x), len(y))
x = x[:n]
y = y[:n]

fs = float(target.sample_rate.value)
gps_start = float(target.t0.value)

window_seconds = 0.25
window = int(window_seconds * fs)

candidate_times = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

def corr(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

    if denom == 0:
        return np.nan

    return np.sum(a * b) / denom

rows = []

for i in range(n // window):
    start = i * window
    end = start + window

    gps = gps_start + start / fs

    a = x[start:end]
    b = y[start:end]

    if len(a) != window:
        continue

    c = corr(a, b)

    rows.append({
        "index": i,
        "gps": gps,
        "correlation": c,
        "abs_correlation": abs(c),
        "mean": np.mean(a),
        "rms": np.sqrt(np.mean(a ** 2))
    })

abs_corr = np.array([r["abs_correlation"] for r in rows])

print("Total windows:", len(rows))
print()
print("Frame-wide |correlation| statistics")
print("Median:", np.median(abs_corr))
print("95th percentile:", np.percentile(abs_corr, 95))
print("99th percentile:", np.percentile(abs_corr, 99))
print()

print("Candidate results")
print()

for candidate in candidate_times:

    index = round((candidate - gps_start) / window_seconds)

    matches = [r for r in rows if r["index"] == index]

    if not matches:
        print("No matching window:", candidate)
        continue

    r = matches[0]

    percentile = (
        np.sum(abs_corr <= r["abs_correlation"])
        / len(abs_corr)
        * 100
    )

    print(
        f"GPS {r['gps']:.2f} - {r['gps'] + window_seconds:.2f}\n"
        f"  correlation = {r['correlation']:.6f}\n"
        f"  percentile = {percentile:.2f}\n"
        f"  mean = {r['mean']:.6g}\n"
        f"  RMS = {r['rms']:.6g}"
    )
    print()


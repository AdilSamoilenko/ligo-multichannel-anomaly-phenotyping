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

candidate_times = np.array([
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
])

def correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0:
        return np.nan
    return np.sum(a * b) / denom

def get_window(gps):
    start = int(round((gps - gps_start) * fs))
    end = start + window
    if start < 0 or end > n:
        return None
    return x[start:end], y[start:end]

print("CANDIDATE WINDOWS")
print("=================")

candidate_results = []

for gps in candidate_times:
    result = get_window(gps)

    if result is None:
        print(f"GPS {gps:.2f}: OUT OF RANGE")
        continue

    a, b = result
    c = correlation(a, b)
    candidate_results.append((gps, c))

    print(
        f"GPS {gps:.2f} - {gps + window_seconds:.2f} | "
        f"correlation = {c:.6f} | "
        f"|correlation| = {abs(c):.6f}"
    )

null = []

for start in range(0, n - window + 1, window):
    end = start + window
    a = x[start:end]
    b = y[start:end]

    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        continue

    c = correlation(a, b)

    if np.isfinite(c):
        null.append(abs(c))

null = np.asarray(null)

print()
print("NULL DISTRIBUTION")
print("=================")
print("Windows:", len(null))
print("Median:", np.median(null))
print("95th percentile:", np.percentile(null, 95))
print("99th percentile:", np.percentile(null, 99))
print("Maximum:", np.max(null))

print()
print("CANDIDATE EMPIRICAL PERCENTILES")
print("================================")

for gps, c in candidate_results:
    percentile = 100 * np.mean(null <= abs(c))
    exceedances = np.sum(null >= abs(c))

    print(
        f"GPS {gps:.2f} | "
        f"|r| = {abs(c):.6f} | "
        f"percentile = {percentile:.2f}% | "
        f"null exceedances = {exceedances}/{len(null)}"
    )

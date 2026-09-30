from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

target = TimeSeries.read(PATH, channel="H1:IMC-WFS_A_DC_PIT_OUT_DQ")
reference = TimeSeries.read(PATH, channel="H1:LSC-POP_A_LF_OUT_DQ")

x = np.asarray(target.value, dtype=float)
y = np.asarray(reference.value, dtype=float)

fs_x = float(target.sample_rate.value)
fs_y = float(reference.sample_rate.value)

t0 = float(target.t0.value)

window_seconds = 0.25
window_x = int(window_seconds * fs_x)
window_y = int(window_seconds * fs_y)

max_lag = int(2.0 * fs_y)
lag_step = 100

def corr(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))

    if denom == 0:
        return np.nan

    return np.sum(a*b) / denom

def max_corr(gps):
    ix = int(round((gps - t0) * fs_x))
    iy = int(round((gps - t0) * fs_y))

    a = x[ix:ix + window_x]

    if len(a) != window_x:
        return np.nan

    best = np.nan

    for lag in range(-max_lag, max_lag + 1, lag_step):
        j = iy + lag

        if j < 0 or j + window_y > len(y):
            continue

        b = y[j:j + window_y]

        t_a = np.linspace(0, window_seconds, window_x, endpoint=False)
        t_b = np.linspace(0, window_seconds, window_y, endpoint=False)

        b_interp = np.interp(t_a, t_b, b)

        r = corr(a, b_interp)

        if np.isfinite(r):
            if np.isnan(best) or abs(r) > abs(best):
                best = r

    return abs(best)

values = []

for i in range(0, len(x) - window_x + 1, window_x):
    gps = t0 + i / fs_x
    c = max_corr(gps)

    if np.isfinite(c):
        values.append(c)

values = np.asarray(values)

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

print("NULL DISTRIBUTION")
print("=================")
print("Windows:", len(values))
print("Median:", np.median(values))
print("95th percentile:", np.percentile(values, 95))
print("99th percentile:", np.percentile(values, 99))
print("Maximum:", np.max(values))

print()
print("CANDIDATES")
print("==========")

for gps in candidates:
    c = max_corr(gps)
    percentile = 100 * np.mean(values <= c)
    exceedances = np.sum(values >= c)

    print(
        f"GPS {gps:.2f} | "
        f"|r|max = {c:.6f} | "
        f"percentile = {percentile:.2f}% | "
        f"exceedances = {exceedances}/{len(values)}"
    )

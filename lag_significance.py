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

fs_x = float(target.sample_rate.value)
fs_y = float(reference.sample_rate.value)

t0_x = float(target.t0.value)
t0_y = float(reference.t0.value)

window_seconds = 0.25
window_x = int(window_seconds * fs_x)
window_y = int(window_seconds * fs_y)

max_lag_seconds = 2.0
step_seconds = 1.0 / fs_y * 100

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

def correlation(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))

    if denom == 0:
        return np.nan

    return np.sum(a * b) / denom

def max_correlation(gps_start):
    ix = int(round((gps_start - t0_x) * fs_x))
    iy = int(round((gps_start - t0_y) * fs_y))

    a = x[ix:ix + window_x]

    best_abs = -1.0
    best_r = np.nan
    best_lag = np.nan

    for lag_y in range(
        -int(max_lag_seconds * fs_y),
        int(max_lag_seconds * fs_y) + 1,
        100
    ):
        j = iy + lag_y

        if j < 0 or j + window_y > len(y):
            continue

        b = y[j:j + window_y]

        b_interp = np.interp(
            np.linspace(0, window_seconds, len(a), endpoint=False),
            np.linspace(0, window_seconds, len(b), endpoint=False),
            b
        )

        r = correlation(a, b_interp)

        if np.isfinite(r) and abs(r) > best_abs:
            best_abs = abs(r)
            best_r = r
            best_lag = lag_y / fs_y

    return best_abs, best_r, best_lag

print("CHANNEL TIMING")
print("==============")
print("Target fs:", fs_x)
print("Reference fs:", fs_y)
print("Target t0:", t0_x)
print("Reference t0:", t0_y)

print()
print("CANDIDATE WINDOWS")
print("=================")

for gps in candidates:
    c, r, lag = max_correlation(gps)

    print(
        f"GPS {gps:.2f} | "
        f"|r|max = {c:.6f} | "
        f"r = {r:.6f} | "
        f"lag = {lag:.4f} s"
    )

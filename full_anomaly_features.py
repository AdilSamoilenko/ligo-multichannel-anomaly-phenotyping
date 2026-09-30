from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

ts = TimeSeries.read(PATH, channel="H1:IMC-WFS_A_DC_PIT_OUT_DQ")

x = np.asarray(ts.value, dtype=float)
fs = float(ts.sample_rate.value)
t0 = float(ts.t0.value)

window = int(0.25 * fs)

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

def band_power(data, lo, hi):
    data = data - np.mean(data)

    n = len(data)

    spec = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1 / fs)

    power = np.abs(spec) ** 2 / n**2

    mask = (freqs >= lo) & (freqs < hi)

    return np.sum(power[mask])

features = []

for i in range(0, len(x) - window + 1, window):
    data = x[i:i + window]

    features.append([
        t0 + i / fs,
        np.mean(data),
        np.sqrt(np.mean(data ** 2)),
        np.std(data),
        band_power(data, 0, 20),
        band_power(data, 20, 30),
        band_power(data, 75, 85),
        band_power(data, 115, 125)
    ])

features = np.asarray(features)

names = [
    "Mean",
    "RMS",
    "STD",
    "0-20 Hz",
    "20-30 Hz",
    "75-85 Hz",
    "115-125 Hz"
]

print("FULL FRAME FEATURE DISTRIBUTIONS")
print("================================")

for j, name in enumerate(names, start=1):
    values = features[:, j]

    print()
    print(name)
    print(f"  median = {np.median(values):.8e}")
    print(f"  95th   = {np.percentile(values, 95):.8e}")
    print(f"  99th   = {np.percentile(values, 99):.8e}")
    print(f"  max    = {np.max(values):.8e}")

print()
print("CANDIDATES")
print("==========")

for gps in candidates:
    idx = np.argmin(np.abs(features[:, 0] - gps))
    row = features[idx]

    print()
    print(f"GPS {gps:.2f}")

    for j, name in enumerate(names, start=1):
        values = features[:, j]
        value = row[j]

        percentile = 100 * np.mean(values <= value)

        print(
            f"  {name:10s} | "
            f"value = {value:.8e} | "
            f"percentile = {percentile:.2f}%"
        )

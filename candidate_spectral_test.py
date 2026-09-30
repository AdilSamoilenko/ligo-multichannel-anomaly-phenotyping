from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channel = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

ts = TimeSeries.read(PATH, channel=channel)

fs = float(ts.sample_rate.value)
t0 = float(ts.t0.value)

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

frequencies = [
    25.25,
    25.50,
    79.00,
    79.25,
    120.00
]

window_seconds = 0.25
window = int(window_seconds * fs)

def power_at_frequency(data, frequency):
    data = data - np.mean(data)

    n = len(data)
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1 / fs)

    idx = np.argmin(np.abs(freqs - frequency))

    return np.abs(spectrum[idx]) ** 2 / n**2

print("CANDIDATE SPECTRAL POWER")
print("========================")

for gps in candidates:
    i = int(round((gps - t0) * fs))
    data = np.asarray(ts.value[i:i + window], dtype=float)

    print()
    print(f"GPS {gps:.2f}")

    for f in frequencies:
        p = power_at_frequency(data, f)
        print(f"  {f:7.2f} Hz | power = {p:.8e}")

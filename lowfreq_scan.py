from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

ts = TimeSeries.read(
    PATH,
    channel="H1:IMC-WFS_A_DC_PIT_OUT_DQ"
)

x = np.asarray(ts.value, dtype=float)
fs = float(ts.sample_rate.value)
t0 = float(ts.t0.value)

WINDOW = 4.5
STEP = 0.25

print("LOW-FREQUENCY POWER SCAN")
print("========================")
print("Band: 1-5 Hz")
print(f"Window: {WINDOW} s")
print(f"Step: {STEP} s")
print()

results = []

start = t0

while start + WINDOW <= t0 + len(x) / fs:
    i = int(round((start - t0) * fs))
    j = i + int(round(WINDOW * fs))

    data = x[i:j]

    data = data - np.mean(data)

    n = len(data)
    window = np.hanning(n)

    spectrum = np.fft.rfft(data * window)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    power = np.abs(spectrum) ** 2

    mask = (freqs >= 1.0) & (freqs <= 5.0)

    band_power = np.sum(power[mask])

    peak_idx = np.argmax(power[mask])
    peak_freq = freqs[mask][peak_idx]

    rms = np.sqrt(np.mean(data ** 2))

    results.append(
        (
            start,
            band_power,
            peak_freq,
            rms
        )
    )

    start += STEP

results = np.array(results)

order = np.argsort(results[:, 1])[::-1]

print("TOP 20 WINDOWS BY 1-5 Hz POWER")
print("==============================")

for k in range(min(20, len(order))):
    r = results[order[k]]

    print(
        f"{r[0]:.2f} | "
        f"1-5Hz power={r[1]:.8e} | "
        f"peak={r[2]:.4f} Hz | "
        f"RMS={r[3]:.8e}"
    )

print()
print("FULL-FRAME DISTRIBUTION")
print("=======================")

p = results[:, 1]

for q in [50, 90, 95, 99, 100]:
    print(
        f"{q:>3}th percentile: "
        f"{np.percentile(p, q):.8e}"
    )

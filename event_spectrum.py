from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

ts = TimeSeries.read(
    PATH,
    channel="H1:IMC-WFS_A_DC_PIT_OUT_DQ"
)

fs = float(ts.sample_rate.value)
t0 = float(ts.t0.value)

start = 1376516155.0
end = 1376516159.5

i = int(round((start - t0) * fs))
j = int(round((end - t0) * fs))

data = np.asarray(ts.value[i:j], dtype=float)

data -= np.mean(data)

n = len(data)

window = np.hanning(n)

spectrum = np.fft.rfft(data * window)

freqs = np.fft.rfftfreq(n, 1 / fs)

power = np.abs(spectrum) ** 2

mask = (freqs >= 1) & (freqs <= 300)

f = freqs[mask]
p = power[mask]

order = np.argsort(p)[::-1]

print("EVENT SPECTRUM")
print("==============")
print(f"Duration: {len(data) / fs:.3f} s")
print(f"Frequency resolution: {fs / n:.4f} Hz")

print()
print("TOP 30 FREQUENCY PEAKS")
print("======================")

for k in range(30):
    idx = order[k]

    print(
        f"{f[idx]:10.4f} Hz | "
        f"power = {p[idx]:.8e}"
    )

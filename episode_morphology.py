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

centre = 1376516124.0

print("LOW-FREQUENCY EPISODE MORPHOLOGY")
print("================================")
print(f"Centre: {centre:.6f}")
print()

for duration in [0.25, 0.5, 1, 2, 4, 6, 8, 10]:
    start = centre - duration / 2
    end = centre + duration / 2

    if start < t0 or end > t0 + len(x) / fs:
        continue

    i = int(round((start - t0) * fs))
    j = int(round((end - t0) * fs))

    data = x[i:j]

    print(f"Window: {duration:.2f} s")
    print(f"GPS: {start:.6f} - {end:.6f}")
    print(f"Mean: {np.mean(data):+.8e}")
    print(f"STD:  {np.std(data):.8e}")
    print(f"RMS:  {np.sqrt(np.mean(data**2)):.8e}")
    print(f"Min:  {np.min(data):+.8e}")
    print(f"Max:  {np.max(data):+.8e}")
    print()

print("LOCAL 0.25 SECOND WINDOWS")
print("==========================")

for k in range(-12, 13):
    start = centre + k * 0.25

    if start < t0 or start + 0.25 > t0 + len(x) / fs:
        continue

    i = int(round((start - t0) * fs))
    j = i + int(round(0.25 * fs))

    data = x[i:j]

    print(
        f"{start:.2f} | "
        f"mean={np.mean(data):+.7e} | "
        f"std={np.std(data):.7e} | "
        f"rms={np.sqrt(np.mean(data**2)):.7e}"
    )

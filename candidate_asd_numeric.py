from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channel = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

ts = TimeSeries.read(PATH, channel=channel)

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

bands = [
    ("20-30 Hz", 20, 30),
    ("75-85 Hz", 75, 85),
    ("115-125 Hz", 115, 125),
    ("0-20 Hz", 0, 20),
    ("30-75 Hz", 30, 75),
    ("85-115 Hz", 85, 115),
    ("125-300 Hz", 125, 300)
]

print("CANDIDATE ASD BAND ANALYSIS")
print("===========================")

results = []

for gps in candidates:
    segment = ts.crop(gps, gps + 4.0)

    asd = segment.asd(
        fftlength=1,
        overlap=0.5
    )

    f = np.asarray(asd.frequencies.value)
    a = np.asarray(asd.value)

    row = []

    print()
    print(f"GPS {gps:.2f}")

    for name, lo, hi in bands:
        mask = (f >= lo) & (f < hi) & np.isfinite(a)

        if np.any(mask):
            value = np.sqrt(np.mean(a[mask] ** 2))
        else:
            value = np.nan

        row.append(value)

        print(f"  {name:12s} | RMS ASD = {value:.8e}")

    results.append(row)

print()
print("RELATIVE TO CANDIDATE MEDIAN")
print("============================")

results = np.asarray(results)

for j, (name, _, _) in enumerate(bands):
    median = np.median(results[:, j])

    print()
    print(name)

    for i, gps in enumerate(candidates):
        ratio = results[i, j] / median
        print(f"  {gps:.2f} | {ratio:.3f}x")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

target = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

channels = [
    "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "H1:OAF-REFL_A_RIN_PREFILT_OUT_DQ",
    "H1:LSC-REFL_A_RIN_OUT_DQ",
    "H1:LSC-POP_A_LF_OUT_DQ",
    "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
    "H1:SUS-ETMX_L1_CAL_LINE_OUT_DQ",
    "H1:SUS-ETMX_L2_CAL_LINE_OUT_DQ",
    "H1:SUS-ETMX_L3_CAL_LINE_OUT_DQ",
]

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

target_ts = TimeSeries.read(PATH, channel=target)

print("Target:", target)
print()

for gps in candidates:
    start = gps - 0.5
    end = gps + 0.5

    x = target_ts.crop(start, end).value
    x = np.asarray(x, dtype=float)

    print("=" * 70)
    print(f"Candidate GPS: {gps:.2f}")
    print("=" * 70)

    for channel in channels:
        try:
            ts = TimeSeries.read(PATH, channel=channel)

            y = ts.crop(start, end).value
            y = np.asarray(y, dtype=float)

            n = min(len(x), len(y))
            a = x[:n]
            b = y[:n]

            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]

            if len(a) < 10:
                print(f"{channel}: insufficient data")
                continue

            corr = np.corrcoef(a, b)[0, 1]

            print(f"{channel}")
            print(f"  zero-lag correlation: {corr:.4f}")

        except Exception as e:
            print(f"{channel}: ERROR {e}")

    print()

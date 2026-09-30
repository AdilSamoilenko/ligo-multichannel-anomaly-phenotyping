from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channels = [
    "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "H1:OAF-REFL_A_RIN_PREFILT_OUT_DQ",
    "H1:LSC-POP_A_LF_OUT_DQ",
    "H1:LSC-REFL_A_RIN_OUT_DQ",
    "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
    "H1:SUS-ETMX_L1_CAL_LINE_OUT_DQ",
    "H1:SUS-ETMX_L2_CAL_LINE_OUT_DQ",
    "H1:SUS-ETMX_L3_CAL_LINE_OUT_DQ",
    "H1:SUS-PI_PROC_COMPUTE_MODE29_RMSMON",
]

centre = 1376516126.0
window = 0.5

print("CROSS-CHANNEL EPISODE SCAN")
print("==========================")
print(f"Window: {centre-window/2:.6f} - {centre+window/2:.6f}")
print()

for channel in channels:

    try:
        ts = TimeSeries.read(
            PATH,
            channel=channel,
            start=centre - window / 2,
            end=centre + window / 2
        )

        data = np.asarray(ts.value, dtype=float)
        data = data[np.isfinite(data)]

        if len(data) == 0:
            print(channel)
            print("  NO FINITE DATA")
            print()
            continue

        mean = np.mean(data)
        std = np.std(data)
        rms = np.sqrt(np.mean(data ** 2))
        minimum = np.min(data)
        maximum = np.max(data)

        print(channel)
        print(f"  samples = {len(data)}")
        print(f"  mean    = {mean:+.8e}")
        print(f"  std     = {std:.8e}")
        print(f"  rms     = {rms:.8e}")
        print(f"  min     = {minimum:+.8e}")
        print(f"  max     = {maximum:+.8e}")
        print()

    except Exception as e:
        print(channel)
        print(f"  ERROR: {e}")
        print()


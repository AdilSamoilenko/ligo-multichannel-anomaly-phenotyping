from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channels = [
    "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "H1:LSC-POP_A_LF_OUT_DQ",
    "H1:LSC-REFL_A_RIN_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "H1:OAF-REFL_A_RIN_PREFILT_OUT_DQ",
    "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
    "H1:SUS-ETMX_L1_CAL_LINE_OUT_DQ",
    "H1:SUS-ETMX_L2_CAL_LINE_OUT_DQ",
    "H1:SUS-ETMX_L3_CAL_LINE_OUT_DQ",
    "H1:SUS-PI_PROC_COMPUTE_MODE29_RMSMON",
]

start = 1376516121.0
end = 1376516128.0
step = 0.25

print("MULTICHANNEL EPISODE MAP")
print("========================")
print(f"GPS: {start:.2f} - {end:.2f}")
print()

for channel in channels:

    try:
        ts = TimeSeries.read(
            PATH,
            channel=channel,
            start=start,
            end=end
        )

        x = np.asarray(ts.value, dtype=float)
        fs = float(ts.sample_rate.value)

        print()
        print(channel)
        print("-" * len(channel))

        for k in range(int((end - start) / step)):

            a = start + k * step
            b = a + step

            i = int(round((a - float(ts.t0.value)) * fs))
            j = int(round((b - float(ts.t0.value)) * fs))

            if i < 0 or j > len(x):
                continue

            d = x[i:j]
            d = d[np.isfinite(d)]

            if len(d) == 0:
                continue

            print(
                f"{a:.2f} "
                f"mean={np.mean(d):+.5e} "
                f"std={np.std(d):.5e} "
                f"rms={np.sqrt(np.mean(d*d)):.5e}"
            )

    except Exception as e:
        print()
        print(channel)
        print("ERROR:", e)


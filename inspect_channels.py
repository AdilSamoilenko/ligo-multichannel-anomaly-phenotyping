from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channels = [
    "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "H1:LSC-REFL_A_RIN_OUT_DQ",
    "H1:OAF-REFL_A_RIN_PREFILT_OUT_DQ",
    "H1:SUS-PI_PROC_COMPUTE_MODE29_RMSMON",
]

for channel in channels:
    print("\n" + "=" * 70)
    print(channel)
    print("=" * 70)

    try:
        ts = TimeSeries.read(PATH, channel=channel)

        print("sample rate:", ts.sample_rate)
        print("start:", ts.t0)
        print("number of samples:", len(ts))
        print("unit:", repr(ts.unit))
        print("dtype:", ts.value.dtype)

    except Exception as e:
        print("ERROR:", type(e).__name__, str(e))

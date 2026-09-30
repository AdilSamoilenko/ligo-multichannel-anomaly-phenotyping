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

    print()
    print("=" * 80)
    print(channel)
    print("=" * 80)

    try:
        ts = TimeSeries.read(
            PATH,
            channel=channel,
            start=1376516096,
            end=1376516160
        )

        fs = float(ts.sample_rate)
        n = len(ts)
        start = float(ts.t0)
        duration = n / fs
        end = start + duration

        print("sample rate :", fs)
        print("samples     :", n)
        print("start GPS   :", start)
        print("end GPS     :", end)
        print("duration    :", duration)
        print("unit        :", ts.unit)
        print("dtype       :", ts.dtype)
        print("shape       :", ts.shape)

        print()
        print("metadata:")

        try:
            metadata = ts.metadata
            if metadata:
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print("  <empty>")
        except Exception as e:
            print("  <metadata unavailable:", repr(e), ">")

    except Exception as e:
        print("ERROR:", repr(e))

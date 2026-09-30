from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
import csv

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
    "H1:SUS-ETMX_L3_CAL_LINE_OUT_DQ"
]

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

MAX_LAG = 2.0
LAG_STEP = 0.01

target_ts = TimeSeries.read(PATH, channel=target)
fs = float(target_ts.sample_rate.value)

rows = []

for gps in candidates:
    start = gps - 2.5
    end = gps + 2.5

    x = np.asarray(
        target_ts.crop(start, end).value,
        dtype=float
    )

    print()
    print("=" * 70)
    print(f"Candidate: {gps:.2f}")
    print("=" * 70)

    for channel in channels:
        try:
            ts = TimeSeries.read(PATH, channel=channel)

            y = np.asarray(
                ts.crop(start, end).value,
                dtype=float
            )

            n = min(len(x), len(y))
            a = x[:n]
            b = y[:n]

            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]

            if len(a) < 100:
                continue

            a -= np.mean(a)
            b -= np.mean(b)

            best_corr = 0.0
            best_lag = 0.0

            for lag in np.arange(-MAX_LAG, MAX_LAG + LAG_STEP, LAG_STEP):
                shift = int(round(lag * fs))

                if shift < 0:
                    aa = a[:shift]
                    bb = b[-shift:]
                elif shift > 0:
                    aa = a[shift:]
                    bb = b[:-shift]
                else:
                    aa = a
                    bb = b

                if len(aa) < 100:
                    continue

                denom = np.sqrt(
                    np.sum(aa ** 2) *
                    np.sum(bb ** 2)
                )

                if denom == 0:
                    continue

                corr = np.sum(aa * bb) / denom

                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            rows.append({
                "candidate": gps,
                "channel": channel,
                "correlation": best_corr,
                "lag_seconds": best_lag
            })

            print(
                f"{channel}\n"
                f"  correlation = {best_corr:.5f}\n"
                f"  lag = {best_lag:.3f} s"
            )

        except Exception as e:
            print(f"{channel}: ERROR {e}")

with open("wide_lag_correlations.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "candidate",
            "channel",
            "correlation",
            "lag_seconds"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print("Saved: wide_lag_correlations.csv")

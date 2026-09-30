from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

target_name = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

channels = [
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

print("EPISODE CROSS-CORRELATION")
print("=========================")
print(f"GPS interval: {start:.2f} - {end:.2f}")
print()

target = TimeSeries.read(
    PATH,
    channel=target_name,
    start=start,
    end=end
)

# Resample everything to a common rate.
# 1024 Hz is safely below the target's 16384 Hz and above the
# reference channel's 2048 Hz Nyquist constraint.
target = target.resample(1024)

x = np.asarray(target.value, dtype=float)

print("Target sample rate:", float(target.sample_rate.value))
print("Target samples:", len(x))
print()

for name in channels:

    try:
        ref = TimeSeries.read(
            PATH,
            channel=name,
            start=start,
            end=end
        )

        ref = ref.resample(1024)

        y = np.asarray(ref.value, dtype=float)

        n = min(len(x), len(y))

        xx = x[:n].copy()
        yy = y[:n].copy()

        finite = np.isfinite(xx) & np.isfinite(yy)

        xx = xx[finite]
        yy = yy[finite]

        if len(xx) < 1000:
            print(name)
            print("  insufficient data")
            print()
            continue

        xx -= np.mean(xx)
        yy -= np.mean(yy)

        xx /= np.std(xx)
        yy /= np.std(yy)

        # Correlation over ±2 seconds.
        max_lag = int(2.0 * 1024)

        corr = np.correlate(xx, yy, mode="full")
        lags = np.arange(-len(xx) + 1, len(xx))

        mask = np.abs(lags) <= max_lag

        c = corr[mask] / len(xx)
        l = lags[mask]

        idx = np.argmax(np.abs(c))

        best_corr = c[idx]
        best_lag = l[idx] / 1024.0

        zero_idx = np.argmin(np.abs(l))
        zero_corr = c[zero_idx]

        print(name)
        print(f"  zero-lag correlation = {zero_corr:+.6f}")
        print(f"  max |correlation|     = {best_corr:+.6f}")
        print(f"  lag                   = {best_lag:+.6f} s")
        print()

    except Exception as e:
        print(name)
        print("  ERROR:", e)
        print()


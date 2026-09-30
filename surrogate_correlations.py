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

FS = 1024
MAX_LAG = int(2 * FS)
N_SURROGATES = 500

rng = np.random.default_rng(12345)

target = TimeSeries.read(
    PATH,
    channel=target_name,
    start=start,
    end=end
).resample(FS)

x = np.asarray(target.value, dtype=float)

def clean(z):
    z = np.asarray(z, dtype=float)
    return z[np.isfinite(z)]

def max_corr(x, y):

    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    if sx == 0 or sy == 0:
        return np.nan, np.nan

    x /= sx
    y /= sy

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)

    mask = np.abs(lags) <= MAX_LAG

    corr = corr[mask] / n
    lags = lags[mask]

    idx = np.argmax(np.abs(corr))

    return corr[idx], lags[idx] / FS

print("SURROGATE CROSS-CORRELATION TEST")
print("================================")
print(f"Surrogates: {N_SURROGATES}")
print()

for name in channels:

    try:

        ref = TimeSeries.read(
            PATH,
            channel=name,
            start=start,
            end=end
        ).resample(FS)

        y = np.asarray(ref.value, dtype=float)

        n = min(len(x), len(y))

        xx = x[:n]
        yy = y[:n]

        finite = np.isfinite(xx) & np.isfinite(yy)

        xx = xx[finite]
        yy = yy[finite]

        observed_r, observed_lag = max_corr(xx, yy)

        null = []

        # Use shifts much larger than the ±2 s search region.
        min_shift = int(2.5 * FS)
        max_shift = int((len(yy) - 2.5 * FS))

        if max_shift <= min_shift:
            print(name)
            print("  insufficient length for surrogate test")
            print()
            continue

        for _ in range(N_SURROGATES):

            shift = rng.integers(
                min_shift,
                max_shift
            )

            shifted = np.roll(yy, shift)

            r, lag = max_corr(xx, shifted)

            if np.isfinite(r):
                null.append(abs(r))

        null = np.asarray(null)

        percentile = 100 * np.mean(
            null <= abs(observed_r)
        )

        p_empirical = (
            1 + np.sum(null >= abs(observed_r))
        ) / (len(null) + 1)

        print(name)
        print(f"  observed |r| = {abs(observed_r):.6f}")
        print(f"  observed lag = {observed_lag:+.6f} s")
        print(f"  null median  = {np.percentile(null, 50):.6f}")
        print(f"  null 95%     = {np.percentile(null, 95):.6f}")
        print(f"  null 99%     = {np.percentile(null, 99):.6f}")
        print(f"  percentile   = {percentile:.2f}%")
        print(f"  empirical p  = {p_empirical:.5f}")
        print()

    except Exception as e:

        print(name)
        print("  ERROR:", e)
        print()


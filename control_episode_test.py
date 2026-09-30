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

FS = 1024
DURATION = 7.0
STEP = 1.0

frame_start = 1376516096.0
frame_end = 1376516160.0

episode_start = 1376516121.0

print("CONTROL EPISODE COMPARISON")
print("==========================")
print(f"Target episode: {episode_start:.2f} - {episode_start + DURATION:.2f}")
print()

def max_corr(x, y):

    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    finite = np.isfinite(x) & np.isfinite(y)

    x = x[finite]
    y = y[finite]

    if len(x) < 1000:
        return np.nan, np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    if sx == 0 or sy == 0:
        return np.nan, np.nan

    x /= sx
    y /= sy

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    max_lag = int(2.0 * FS)

    mask = np.abs(lags) <= max_lag

    corr = corr[mask] / len(x)
    lags = lags[mask]

    i = np.argmax(np.abs(corr))

    return corr[i], lags[i] / FS


# Load all channels once.
data = {}

for name in [target_name] + channels:

    try:

        ts = TimeSeries.read(
            PATH,
            channel=name,
            start=frame_start,
            end=frame_end
        ).resample(FS)

        data[name] = np.asarray(ts.value, dtype=float)

        print("Loaded:", name)

    except Exception as e:

        print("ERROR loading", name, e)

print()

# Seven-second windows, spaced one second apart.
starts = np.arange(
    frame_start,
    frame_end - DURATION + 0.001,
    STEP
)

results = []

for s in starts:

    row = {
        "start": s
    }

    i0 = int(round((s - frame_start) * FS))
    i1 = int(round((s + DURATION - frame_start) * FS))

    x = data[target_name][i0:i1]

    row["target_rms"] = np.sqrt(
        np.nanmean(x * x)
    )

    row["target_std"] = np.nanstd(x)

    for name in channels:

        y = data[name][i0:i1]

        r, lag = max_corr(x, y)

        key = name.replace("H1:", "").replace(":", "_")

        row[key + "_r"] = r
        row[key + "_lag"] = lag

    results.append(row)


# Print compact results for the most important relationships.
important = [
    "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "H1:SUS-PI_PROC_COMPUTE_MODE29_RMSMON",
    "H1:LSC-REFL_A_RIN_OUT_DQ",
]

print()
print("IMPORTANT CHANNELS")
print("==================")

for name in important:

    key = name.replace("H1:", "").replace(":", "_") + "_r"

    vals = np.array([
        row[key]
        for row in results
        if np.isfinite(row[key])
    ])

    if len(vals) == 0:
        continue

    absvals = np.abs(vals)

    print()
    print(name)
    print(f"  median |r| = {np.median(absvals):.6f}")
    print(f"  95th |r|   = {np.percentile(absvals, 95):.6f}")
    print(f"  max |r|    = {np.max(absvals):.6f}")

    episode_rows = [
        row for row in results
        if abs(row["start"] - episode_start) < 0.001
    ]

    if episode_rows:

        er = episode_rows[0][key]

        percentile = (
            100 *
            np.mean(absvals <= abs(er))
        )

        print(f"  episode |r| = {abs(er):.6f}")
        print(f"  percentile  = {percentile:.2f}%")


print()
print("ALL WINDOWS")
print("===========")

# Print every window with target RMS and the strongest control relationship.
for row in results:

    print(
        f"{row['start']:.2f} "
        f"RMS={row['target_rms']:.5e} "
        f"YAW={row.get('IMC-WFS_A_DC_YAW_OUT_DQ_r', np.nan):+.3f} "
        f"OAF_PIT={row.get('OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ_r', np.nan):+.3f} "
        f"OAF_YAW={row.get('OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ_r', np.nan):+.3f} "
        f"PI={row.get('SUS-PI_PROC_COMPUTE_MODE29_RMSMON_r', np.nan):+.3f}"
    )


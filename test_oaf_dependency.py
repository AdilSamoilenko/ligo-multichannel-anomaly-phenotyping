from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
from scipy import signal
from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

FRAME_START = 1376516096.0
FRAME_END = 1376516160.0

pairs = [
    (
        "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
        "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
        "PIT"
    ),
    (
        "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
        "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
        "YAW"
    ),
]


def load_channel(channel):
    ts = TimeSeries.read(
        PATH,
        channel=channel,
        start=FRAME_START,
        end=FRAME_END
    )

    data = np.asarray(ts.value, dtype=np.float64)

    sample_rate = ts.sample_rate

    try:
        fs = float(sample_rate.value)
    except AttributeError:
        fs = float(sample_rate)

    return data, fs


def analyse(x, y, fs, label, start, end):

    i0 = int(round((start - FRAME_START) * fs))
    i1 = int(round((end - FRAME_START) * fs))

    if i0 < 0:
        raise ValueError("Analysis start is before frame start")

    if i1 > len(x) or i1 > len(y):
        raise ValueError("Analysis interval extends beyond available data")

    x = x[i0:i1]
    y = y[i0:i1]

    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    if n < 4:
        raise ValueError("Not enough samples for analysis")

    x = x - np.mean(x)
    y = y - np.mean(y)

    nperseg = min(16384, n)

    if nperseg < 256:
        raise ValueError("Analysis interval is too short")

    noverlap = nperseg // 2

    f, Pxy = signal.csd(
        x,
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant"
    )

    _, Pxx = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant"
    )

    _, Pyy = signal.welch(
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend="constant"
    )

    denominator = Pxx * Pyy

    coherence = np.zeros_like(denominator)

    valid = denominator > 0

    coherence[valid] = (
        np.abs(Pxy[valid]) ** 2
        / denominator[valid]
    )

    coherence = np.clip(coherence, 0.0, 1.0)

    transfer = np.zeros_like(Pxy, dtype=np.complex128)

    valid_x = Pxx > 0

    transfer[valid_x] = (
        Pxy[valid_x] / Pxx[valid_x]
    )

    bands = [
        (0, 2),
        (2, 5),
        (5, 10),
        (10, 20),
        (20, 50),
        (50, 100),
        (100, 300),
        (300, 1000),
    ]

    print()
    print("=" * 80)
    print(f"{label}: {start:.2f} - {end:.2f}")
    print("=" * 80)

    print(f"Samples:      {n}")
    print(f"Sample rate:  {fs:.2f} Hz")
    print(
        f"Correlation:  "
        f"{np.corrcoef(x, y)[0, 1]:.6f}"
    )

    print()
    print(
        "Frequency band coherence and transfer-function magnitude:"
    )

    for lo, hi in bands:

        mask = (f >= lo) & (f < hi)

        if not np.any(mask):
            continue

        c = coherence[mask]
        h = np.abs(transfer[mask])

        print(
            f"{lo:>4}-{hi:<4} Hz | "
            f"coh median = {np.median(c):.6f} | "
            f"coh max = {np.max(c):.6f} | "
            f"|H| median = {np.median(h):.6e}"
        )

    print()
    print("Highest coherence frequencies:")

    idx = np.argsort(coherence)[-10:][::-1]

    for k in idx:

        print(
            f"{f[k]:10.3f} Hz | "
            f"coherence = {coherence[k]:.6f} | "
            f"|H| = {abs(transfer[k]):.6e} | "
            f"phase = {np.angle(transfer[k]):+.6f} rad"
        )


def main():

    for x_channel, y_channel, label in pairs:

        print()
        print("#" * 80)
        print(label)
        print("#" * 80)

        print()
        print("Loading:")
        print("  X:", x_channel)
        print("  Y:", y_channel)

        x, fs_x = load_channel(x_channel)
        y, fs_y = load_channel(y_channel)

        print()
        print(f"X sample rate: {fs_x:.2f} Hz")
        print(f"Y sample rate: {fs_y:.2f} Hz")
        print(f"X samples:     {len(x)}")
        print(f"Y samples:     {len(y)}")

        if not np.isclose(fs_x, fs_y):
            raise ValueError(
                f"Sample rates differ: {fs_x} vs {fs_y}"
            )

        fs = fs_x

        # Control interval
        analyse(
            x,
            y,
            fs,
            label,
            1376516146.0,
            1376516153.0
        )

        # Episode of interest
        analyse(
            x,
            y,
            fs,
            label,
            1376516121.0,
            1376516128.0
        )

        # Entire frame
        analyse(
            x,
            y,
            fs,
            label,
            FRAME_START,
            FRAME_END
        )


if __name__ == "__main__":
    main()

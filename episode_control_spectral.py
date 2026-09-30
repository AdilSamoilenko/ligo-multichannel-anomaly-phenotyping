from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
from scipy import signal
from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

FRAME_START = 1376516096.0
FRAME_END = 1376516160.0

EP_START = 1376516121.0
EP_END = 1376516128.0

WINDOW = 7.0

CHANNELS = [
    (
        "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
        "WFS PIT"
    ),
    (
        "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
        "WFS YAW"
    ),
    (
        "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
        "OAF PIT"
    ),
    (
        "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
        "OAF YAW"
    ),
]


BANDS = [
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 50.0),
    (50.0, 100.0),
]


def load(channel, start, end):

    ts = TimeSeries.read(
        PATH,
        channel=channel,
        start=start,
        end=end
    )

    data = np.asarray(ts.value, dtype=np.float64)

    try:
        fs = float(ts.sample_rate.value)
    except AttributeError:
        fs = float(ts.sample_rate)

    return data, fs


def band_powers(data, fs):

    nperseg = int(round(2.0 * fs))
    noverlap = int(round(1.0 * fs))

    f, Pxx = signal.welch(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant"
    )

    results = []

    for low, high in BANDS:

        mask = (f >= low) & (f < high)

        if np.any(mask):
            power = np.trapezoid(
                Pxx[mask],
                f[mask]
            )
        else:
            power = np.nan

        results.append(power)

    return np.array(results)


def analyse_window(channel, start):

    data, fs = load(
        channel,
        start,
        start + WINDOW
    )

    powers = band_powers(data, fs)

    return powers


def percentile(value, distribution):

    return 100.0 * np.mean(
        np.asarray(distribution) <= value
    )


def main():

    control_starts = np.arange(
        FRAME_START,
        FRAME_END - WINDOW + 0.001,
        1.0
    )

    control_starts = [
        float(x)
        for x in control_starts
        if not (
            x < EP_END and
            x + WINDOW > EP_START
        )
    ]

    print("=" * 90)
    print("EPISODE VS FULL-FRAME CONTROL SPECTRAL ANALYSIS")
    print("=" * 90)

    print()
    print("Episode:")
    print(EP_START, "-", EP_END)

    print()
    print("Control windows:", len(control_starts))

    for channel, name in CHANNELS:

        print()
        print("=" * 90)
        print(name)
        print("=" * 90)

        episode = analyse_window(
            channel,
            EP_START
        )

        controls = []

        for start in control_starts:

            try:
                powers = analyse_window(
                    channel,
                    start
                )

                controls.append(powers)

            except Exception as error:

                print(
                    "Control window failed:",
                    start,
                    type(error).__name__,
                    str(error)
                )

        controls = np.asarray(controls)

        print()
        print(
            f"{'Band':>12} "
            f"{'Episode':>15} "
            f"{'Median':>15} "
            f"{'95th':>15} "
            f"{'Ratio':>12} "
            f"{'Percentile':>12}"
        )

        print("-" * 90)

        for i, (low, high) in enumerate(BANDS):

            distribution = controls[:, i]

            median = np.median(distribution)
            p95 = np.percentile(distribution, 95)

            value = episode[i]

            ratio = value / median if median != 0 else np.inf

            pct = percentile(
                value,
                distribution
            )

            print(
                f"{low:5.1f}-{high:5.1f} Hz "
                f"{value:15.6e} "
                f"{median:15.6e} "
                f"{p95:15.6e} "
                f"{ratio:12.3f} "
                f"{pct:11.2f}%"
            )


if __name__ == "__main__":
    main()

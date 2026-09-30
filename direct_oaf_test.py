from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
from scipy import signal
from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

START = 1376516096.0
END = 1376516160.0

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


def load(channel):
    ts = TimeSeries.read(
        PATH,
        channel=channel,
        start=START,
        end=END
    )

    x = np.asarray(ts.value, dtype=np.float64)

    return x


def analyse(x, y, label, start, end):

    fs = 16384.0

    i0 = int(round((start - START) * fs))
    i1 = int(round((end - START) * fs))

    x = x[i0:i1]
    y = y[i0:i1]

    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    print()
    print("=" * 80)
    print(label, start, end)
    print("=" * 80)

    print("samples:", n)

    print()
    print("Basic statistics")

    print("X mean:", np.mean(x))
    print("Y mean:", np.mean(y))

    print("X std:", np.std(x))
    print("Y std:", np.std(y))

    print("X RMS:", np.sqrt(np.mean(x ** 2)))
    print("Y RMS:", np.sqrt(np.mean(y ** 2)))

    print()

    # Direct difference
    difference = y - x

    print("Y - X")
    print("mean:", np.mean(difference))
    print("std:", np.std(difference))
    print("RMS:", np.sqrt(np.mean(difference ** 2)))

    # Sum/difference for possible sign inversion
    difference_minus = y - x
    difference_plus = y + x

    print()
    print("RMS comparison")
    print("RMS(Y - X):", np.sqrt(np.mean(difference_minus ** 2)))
    print("RMS(Y + X):", np.sqrt(np.mean(difference_plus ** 2)))

    # Linear regression Y = aX + b
    A = np.column_stack([x, np.ones(n)])

    coefficients, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    a, b = coefficients

    prediction = a * x + b
    residual = y - prediction

    print()
    print("Linear model")
    print("Y = aX + b")
    print("a:", a)
    print("b:", b)

    print()
    print("Residual")
    print("mean:", np.mean(residual))
    print("std:", np.std(residual))
    print("RMS:", np.sqrt(np.mean(residual ** 2)))

    signal_rms = np.sqrt(np.mean(y ** 2))
    residual_rms = np.sqrt(np.mean(residual ** 2))

    print()
    print("Residual / Y RMS:", residual_rms / signal_rms)

    # Find best integer-sample lag over +/- 100 ms
    max_lag = int(0.1 * fs)

    x0 = x - np.mean(x)
    y0 = y - np.mean(y)

    corr = signal.correlate(
        y0,
        x0,
        mode="full",
        method="fft"
    )

    lags = signal.correlation_lags(
        len(y0),
        len(x0),
        mode="full"
    )

    mask = np.abs(lags) <= max_lag

    corr_window = corr[mask]
    lag_window = lags[mask]

    best = np.argmax(np.abs(corr_window))

    best_lag_samples = lag_window[best]
    best_lag_seconds = best_lag_samples / fs

    normalisation = np.sqrt(
        np.sum(x0 ** 2) * np.sum(y0 ** 2)
    )

    best_r = corr_window[best] / normalisation

    print()
    print("Best lag")
    print("lag samples:", best_lag_samples)
    print("lag seconds:", best_lag_seconds)
    print("correlation:", best_r)

    # Compare exact samples at beginning
    print()
    print("First 10 X samples:")
    print(x[:10])

    print()
    print("First 10 Y samples:")
    print(y[:10])

    print()
    print("First 10 Y-X:")
    print((y - x)[:10])


def main():

    for x_channel, y_channel, label in pairs:

        print()
        print("#" * 80)
        print(label)
        print("#" * 80)

        x = load(x_channel)
        y = load(y_channel)

        analyse(
            x,
            y,
            label,
            1376516146.0,
            1376516153.0
        )

        analyse(
            x,
            y,
            label,
            1376516121.0,
            1376516128.0
        )

        analyse(
            x,
            y,
            label,
            START,
            END
        )


if __name__ == "__main__":
    main()

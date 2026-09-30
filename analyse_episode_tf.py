from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

EP_START = 1376516121.0
EP_END = 1376516128.0

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


def load_channel(channel):
    ts = TimeSeries.read(
        PATH,
        channel=channel,
        start=EP_START,
        end=EP_END
    )

    data = np.asarray(ts.value, dtype=np.float64)

    try:
        fs = float(ts.sample_rate.value)
    except AttributeError:
        fs = float(ts.sample_rate)

    return data, fs


def make_spectrogram(data, fs, name):

    nperseg = int(round(2.0 * fs))
    noverlap = int(round(1.0 * fs))

    if len(data) < nperseg:
        raise ValueError(
            f"Not enough samples for 2-second STFT: "
            f"{len(data)} samples available"
        )

    f, t, Z = signal.stft(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        boundary=None,
        padded=False
    )

    power = np.abs(Z) ** 2

    frequency_limit = 100.0
    frequency_mask = f <= frequency_limit

    f_plot = f[frequency_mask]
    power_plot = power[frequency_mask, :]

    gps_time = EP_START + t

    power_db = 10.0 * np.log10(
        np.maximum(power_plot, 1e-30)
    )

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    filename = f"{safe_name}_spectrogram.png"

    plt.figure(figsize=(12, 6))

    plt.pcolormesh(
        gps_time,
        f_plot,
        power_db,
        shading="auto"
    )

    plt.axvline(
        EP_START,
        linestyle="--"
    )

    plt.axvline(
        EP_END,
        linestyle="--"
    )

    plt.xlabel("GPS time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(
        f"{name}: time-frequency power, "
        f"{EP_START:.0f}-{EP_END:.0f}"
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.ylim(0, frequency_limit)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return filename, f, t, power


def print_band_summary(f, t, power, name):

    bands = [
        (0.0, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
        (10.0, 20.0),
        (20.0, 50.0),
        (50.0, 100.0),
    ]

    print()
    print("Band power summary:")

    for low, high in bands:

        mask = (f >= low) & (f < high)

        if not np.any(mask):
            continue

        band_power = np.sum(
            power[mask, :],
            axis=0
        )

        peak_index = np.argmax(band_power)

        print(
            f"{low:5.1f}-{high:5.1f} Hz | "
            f"median={np.median(band_power):.6e} | "
            f"max={np.max(band_power):.6e} | "
            f"peak time={EP_START + t[peak_index]:.3f}"
        )

    usable = f <= 100.0

    restricted_power = power[usable, :]

    peak_flat = np.argmax(restricted_power)

    peak_row, peak_col = np.unravel_index(
        peak_flat,
        restricted_power.shape
    )

    peak_frequency = f[usable][peak_row]
    peak_time = EP_START + t[peak_col]
    peak_power = restricted_power[peak_row, peak_col]

    print()
    print("Strongest time-frequency bin below 100 Hz:")
    print("frequency:", peak_frequency, "Hz")
    print("time:", peak_time, "GPS")
    print("power:", peak_power)


def main():

    print("=" * 80)
    print("LIGO EPISODE TIME-FREQUENCY ANALYSIS")
    print("=" * 80)
    print()
    print("Episode:")
    print(EP_START, "-", EP_END)
    print()

    for channel, name in CHANNELS:

        print()
        print("=" * 80)
        print(name)
        print("=" * 80)

        try:

            data, fs = load_channel(channel)

            print("Channel:", channel)
            print("Sample rate:", fs, "Hz")
            print("Samples:", len(data))
            print(
                "Duration:",
                len(data) / fs,
                "s"
            )

            filename, f, t, power = make_spectrogram(
                data,
                fs,
                name
            )

            print_band_summary(
                f,
                t,
                power,
                name
            )

            print()
            print("Saved:", filename)

        except Exception as error:

            print()
            print("ERROR")
            print(type(error).__name__)
            print(str(error))


if __name__ == "__main__":
    main()

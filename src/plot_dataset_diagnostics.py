from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


WINDOW_DIR = Path("data/windows")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 4096


def main():
    files = sorted(WINDOW_DIR.glob("*.npy"))

    if len(files) != 30:
        raise RuntimeError(f"Expected 30 windows, found {len(files)}")

    data = [np.load(f) for f in files]

    # ------------------------------------------------------------
    # Figure 1: all windows in the time domain
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, x in enumerate(data):
        t = np.arange(len(x)) / SAMPLE_RATE
        ax.plot(t, x, alpha=0.35, linewidth=0.6)

    ax.set_xlabel("Time within window (s)")
    ax.set_ylabel("Strain")
    ax.set_title("H1 O4a strain windows: time-domain overview")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "H1_O4a_all_windows_time_domain.png",
        dpi=200,
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 2: RMS / standard deviation
    # ------------------------------------------------------------

    rms = np.array([
        np.sqrt(np.mean(x**2))
        for x in data
    ])

    std = np.array([
        np.std(x)
        for x in data
    ])

    fig, ax = plt.subplots(figsize=(12, 6))

    indices = np.arange(len(data))

    ax.plot(
        indices,
        rms,
        marker="o",
        label="RMS",
    )

    ax.plot(
        indices,
        std,
        marker="s",
        label="Standard deviation",
    )

    ax.set_xlabel("Window index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Window-level amplitude statistics")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "H1_O4a_window_statistics.png",
        dpi=200,
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 3: representative time-domain windows
    # ------------------------------------------------------------

    selected = [0, 7, 14, 21, 29]

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(14, 12),
        sharex=True,
    )

    for ax, i in zip(axes, selected):
        x = data[i]
        t = np.arange(len(x)) / SAMPLE_RATE

        ax.plot(t, x, linewidth=0.6)
        ax.set_ylabel(f"W{i}")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time within window (s)")

    fig.suptitle(
        "Representative H1 O4a strain windows",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "H1_O4a_representative_windows.png",
        dpi=200,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 4: spectrogram
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(14, 14),
        sharex=True,
    )

    for ax, i in zip(axes, selected):
        x = data[i]

        frequencies, times, power = signal.spectrogram(
            x,
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=2048,
            noverlap=1536,
            scaling="density",
        )

        # Restrict display to 10-2000 Hz.
        mask = (frequencies >= 10) & (frequencies <= 2000)

        power_db = 10 * np.log10(
            power[mask] + np.finfo(float).tiny
        )

        ax.pcolormesh(
            times,
            frequencies[mask],
            power_db,
            shading="auto",
        )

        ax.set_yscale("log")
        ax.set_ylabel(f"W{i}\nFrequency (Hz)")

    axes[-1].set_xlabel("Time within window (s)")

    fig.suptitle(
        "H1 O4a representative strain spectrograms",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "H1_O4a_representative_spectrograms.png",
        dpi=200,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 5: PSD
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(14, 14),
        sharex=True,
    )

    for ax, i in zip(axes, selected):
        x = data[i]

        frequencies, psd = signal.welch(
            x,
            fs=SAMPLE_RATE,
            window="hann",
            nperseg=16384,
            noverlap=8192,
            scaling="density",
        )

        mask = (
            (frequencies >= 10)
            & (frequencies <= 2000)
        )

        ax.loglog(
            frequencies[mask],
            psd[mask],
            linewidth=0.8,
        )

        ax.set_ylabel(f"W{i}\nPSD")
        ax.grid(True, which="both", alpha=0.2)

    axes[-1].set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "H1 O4a representative strain power spectral densities",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "H1_O4a_representative_psds.png",
        dpi=200,
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------

    table = pd.DataFrame(
        {
            "window": np.arange(len(data)),
            "rms": rms,
            "std": std,
            "mean": [np.mean(x) for x in data],
            "minimum": [np.min(x) for x in data],
            "maximum": [np.max(x) for x in data],
        }
    )

    table.to_csv(
        "data/H1_O4a_window_statistics.csv",
        index=False,
    )

    print("Diagnostic analysis complete.")
    print()
    print("Figures:")
    print(" figures/H1_O4a_all_windows_time_domain.png")
    print(" figures/H1_O4a_window_statistics.png")
    print(" figures/H1_O4a_representative_windows.png")
    print(" figures/H1_O4a_representative_spectrograms.png")
    print(" figures/H1_O4a_representative_psds.png")
    print()
    print("Statistics:")
    print(" data/H1_O4a_window_statistics.csv")


if __name__ == "__main__":
    main()
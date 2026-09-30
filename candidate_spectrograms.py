from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"
CHANNEL = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

ts = TimeSeries.read(PATH, channel=CHANNEL)

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

for gps in candidates:
    segment = ts.crop(gps - 2.0, gps + 2.0)

    spec = segment.spectrogram(
        0.125,
        fftlength=0.125,
        overlap=0.0625
    )

    fig = spec.plot(
        norm="log",
        vmin=spec.value.min(),
        vmax=spec.value.max()
    )

    ax = fig.gca()
    ax.set_title(
        f"{CHANNEL}\nGPS {gps:.2f}"
    )
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("GPS time [s]")

    filename = f"spectrogram_{gps:.2f}.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Created:", filename)

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
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
    start = gps - 1.0
    end = gps + 1.0

    segment = ts.crop(start, end)

    fig = segment.plot()
    ax = fig.gca()

    ax.set_title(
        f"{CHANNEL}\n"
        f"GPS {start:.2f} to {end:.2f}"
    )

    ax.set_xlabel("GPS time [s]")
    ax.set_ylabel("Amplitude")

    filename = f"candidate_{gps:.2f}.png"
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Created:", filename)

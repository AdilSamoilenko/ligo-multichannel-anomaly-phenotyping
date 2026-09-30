from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channel = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

ts = TimeSeries.read(PATH, channel=channel)

candidates = [
    1376516111.75,
    1376516129.75,
    1376516135.25,
    1376516139.50,
    1376516151.75,
    1376516157.50
]

for gps in candidates:
    segment = ts.crop(gps, gps + 4.0)

    asd = segment.asd(
        fftlength=1,
        overlap=0.5
    )

    fig = asd.plot()
    ax = fig.gca()

    ax.set_xlim(1, 300)
    ax.set_title(f"IMC WFS PIT ASD at GPS {gps:.2f}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD")

    filename = f"asd_{gps:.2f}.png"

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Created", filename)

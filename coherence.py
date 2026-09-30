from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

target = TimeSeries.read(
    PATH,
    channel="H1:IMC-WFS_A_DC_PIT_OUT_DQ"
)

reference = TimeSeries.read(
    PATH,
    channel="H1:LSC-POP_A_LF_OUT_DQ"
)

target = target.resample(2048)
target = target.crop(
    max(float(target.t0.value), float(reference.t0.value)),
    min(
        float(target.t0.value + target.duration.value),
        float(reference.t0.value + reference.duration.value)
    )
)

reference = reference.crop(
    float(target.t0.value),
    float(target.t0.value + target.duration.value)
)

coh = target.coherence(reference, fftlength=4, overlap=2)

fig = coh.plot()
ax = fig.gca()
ax.set_title("Magnitude-Squared Coherence: IMC WFS PIT vs LSC POP LF")
ax.set_ylabel("Coherence")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylim(0, 1)

fig.savefig("coherence_imc_pop.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Created coherence_imc_pop.png")
print("Frequency bins:", len(coh))
print("Maximum coherence:", np.nanmax(coh.value))

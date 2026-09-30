from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

target_name = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"
reference_name = "H1:LSC-POP_A_LF_OUT_DQ"

target = TimeSeries.read(PATH, channel=target_name)
reference = TimeSeries.read(PATH, channel=reference_name)

x = np.asarray(target.value, dtype=float)
y = np.asarray(reference.value, dtype=float)

n = min(len(x), len(y))

x = x[:n]
y = y[:n]

x -= np.mean(x)
y -= np.mean(y)

corr = np.correlate(x, y, mode="full")
lags = np.arange(-n + 1, n)

best = np.argmax(np.abs(corr))
best_lag_samples = lags[best]
best_lag_seconds = best_lag_samples / float(target.sample_rate.value)

print("Target:", target_name)
print("Reference:", reference_name)
print("Samples:", n)
print("Sample rate:", float(target.sample_rate.value), "Hz")
print("Best lag:", best_lag_seconds, "s")

print("Target std:", np.std(x))
print("Reference std:", np.std(y))

scale = np.std(x) / np.std(y)

print("Approximate scale ratio:", scale)

if best_lag_samples > 0:
    aligned_x = x[best_lag_samples:]
    aligned_y = y[:-best_lag_samples]
elif best_lag_samples < 0:
    shift = abs(best_lag_samples)
    aligned_x = x[:-shift]
    aligned_y = y[shift:]
else:
    aligned_x = x
    aligned_y = y

corr_coeff = np.corrcoef(aligned_x, aligned_y)[0, 1]

print("Aligned correlation:", corr_coeff)

fig = target.plot()
ax = fig.gca()
ax.set_title("Target channel")
fig.savefig("relationship_target.png", dpi=200, bbox_inches="tight")
plt.close(fig)

fig = reference.plot()
ax = fig.gca()
ax.set_title("LSC-POP channel")
fig.savefig("relationship_reference.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("Created relationship_target.png")
print("Created relationship_reference.png")

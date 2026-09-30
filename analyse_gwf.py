from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"
CHANNEL = "H1:IMC-WFS_A_DC_PIT_OUT_DQ"

ts = TimeSeries.read(PATH, channel=CHANNEL)

x = np.asarray(ts.value, dtype=np.float64)
fs = float(ts.sample_rate.value)
gps_start = float(ts.t0.value)

duration = len(x) / fs
gps_end = gps_start + duration

print("=== CHANNEL SUMMARY ===")
print("Channel:", CHANNEL)
print("Samples:", len(x))
print("Sample rate:", fs, "Hz")
print("Duration:", duration, "s")
print("GPS start:", gps_start)
print("GPS end:", gps_end)
print("Mean:", np.mean(x))
print("Std:", np.std(x))
print("RMS:", np.sqrt(np.mean(x**2)))
print("Min:", np.min(x))
print("Max:", np.max(x))
print("P01:", np.percentile(x, 1))
print("Median:", np.median(x))
print("P99:", np.percentile(x, 99))
print("NaN:", np.isnan(x).sum())
print("Inf:", np.isinf(x).sum())

# Time-domain plot
time = np.arange(len(x)) / fs

plt.figure(figsize=(12, 5))
plt.plot(time, x, linewidth=0.5)
plt.xlabel("Time since GPS start (s)")
plt.ylabel("Channel value")
plt.title(CHANNEL + " - Time Domain")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gwf_time_domain.png", dpi=200)
plt.close()

print("\nCalculating ASD...")

# Amplitude spectral density
asd = ts.asd(fftlength=4)

plt.figure(figsize=(12, 5))
plt.loglog(asd.frequencies.value, asd.value)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude spectral density")
plt.title(CHANNEL + " - Amplitude Spectral Density")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("gwf_asd.png", dpi=200)
plt.close()

print("\nCreated:")
print("  gwf_time_domain.png")
print("  gwf_asd.png")

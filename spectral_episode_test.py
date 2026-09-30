from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

from gwpy.timeseries import TimeSeries
import numpy as np
from scipy.signal import coherence, csd, welch

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"

channels = {
    "TARGET": "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "OAF_PIT": "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "OAF_YAW": "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
}

FS = 1024

# Candidate episode and a control interval of equal duration.
windows = {
    "EPISODE": (1376516121.0, 1376516128.0),
    "CONTROL": (1376516146.0, 1376516153.0),
}

data = {}

for label, channel in channels.items():

    data[label] = {}

    for wname, (start, end) in windows.items():

        ts = TimeSeries.read(
            PATH,
            channel=channel,
            start=start,
            end=end
        ).resample(FS)

        x = np.asarray(ts.value, dtype=float)

        x = x[np.isfinite(x)]

        data[label][wname] = x


print("SPECTRAL EPISODE TEST")
print("=====================")

for wname in windows:

    print()
    print(wname)
    print("-" * len(wname))

    x = data["TARGET"][wname]
    pit = data["OAF_PIT"][wname]
    yaw = data["OAF_YAW"][wname]

    n = min(len(x), len(pit), len(yaw))

    x = x[:n]
    pit = pit[:n]
    yaw = yaw[:n]

    # Remove DC.
    x = x - np.mean(x)
    pit = pit - np.mean(pit)
    yaw = yaw - np.mean(yaw)

    # 1-second FFT segments with 50% overlap.
    nperseg = 1024
    noverlap = 512

    f, px = welch(
        x,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    _, ppit = welch(
        pit,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    _, pyaw = welch(
        yaw,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    fc_pit, coh_pit = coherence(
        x,
        pit,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    fc_yaw, coh_yaw = coherence(
        x,
        yaw,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    fc, cross_pit = csd(
        x,
        pit,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    fc2, cross_yaw = csd(
        x,
        yaw,
        fs=FS,
        nperseg=nperseg,
        noverlap=noverlap
    )

    phase_pit = np.angle(cross_pit)
    phase_yaw = np.angle(cross_yaw)

    # Frequency bands.
    bands = [
        ("0-2 Hz", 0, 2),
        ("2-5 Hz", 2, 5),
        ("5-10 Hz", 5, 10),
        ("10-20 Hz", 10, 20),
        ("20-30 Hz", 20, 30),
        ("30-50 Hz", 30, 50),
        ("50-100 Hz", 50, 100),
        ("100-300 Hz", 100, 300),
    ]

    print()
    print("BAND POWER")

    for name, lo, hi in bands:

        mask = (f >= lo) & (f < hi)

        if not np.any(mask):
            continue

        target_power = np.mean(px[mask])
        pit_power = np.mean(ppit[mask])
        yaw_power = np.mean(pyaw[mask])

        print(
            f"{name:10s} "
            f"target={target_power:.5e} "
            f"oaf_pit={pit_power:.5e} "
            f"oaf_yaw={yaw_power:.5e}"
        )

    print()
    print("COHERENCE")

    for name, lo, hi in bands:

        mask = (fc_pit >= lo) & (fc_pit < hi)

        if not np.any(mask):
            continue

        pit_max = np.max(coh_pit[mask])
        yaw_max = np.max(coh_yaw[mask])

        print(
            f"{name:10s} "
            f"target-OAF_PIT={pit_max:.5f} "
            f"target-OAF_YAW={yaw_max:.5f}"
        )

    # Top coherence frequencies.
    print()
    print("TOP OAF PIT COHERENCE")

    indices = np.argsort(coh_pit)[-10:][::-1]

    for i in indices:

        print(
            f"f={fc_pit[i]:8.3f} Hz "
            f"coherence={coh_pit[i]:.6f} "
            f"phase={phase_pit[i]:+.5f} rad"
        )

    print()
    print("TOP OAF YAW COHERENCE")

    indices = np.argsort(coh_yaw)[-10:][::-1]

    for i in indices:

        print(
            f"f={fc_yaw[i]:8.3f} Hz "
            f"coherence={coh_yaw[i]:.6f} "
            f"phase={phase_yaw[i]:+.5f} rad"
        )


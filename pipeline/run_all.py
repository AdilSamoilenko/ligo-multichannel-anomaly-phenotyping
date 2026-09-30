import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"
OUT = REPO_ROOT / "results"

FRAME_START = 1376516096.0
FRAME_END = 1376516160.0
EP_START = 1376516121.0
EP_END = 1376516128.0

CHANNELS = {
    "WFS_PIT": "H1:IMC-WFS_A_DC_PIT_OUT_DQ",
    "WFS_YAW": "H1:IMC-WFS_A_DC_YAW_OUT_DQ",
    "LSC_POP": "H1:LSC-POP_A_LF_OUT_DQ",
    "LSC_REFL": "H1:LSC-REFL_A_RIN_OUT_DQ",
    "OAF_PIT": "H1:OAF-IMC_WFS_A_DC_PIT_PREFILT_OUT_DQ",
    "OAF_YAW": "H1:OAF-IMC_WFS_A_DC_YAW_PREFILT_OUT_DQ",
    "OAF_REFL": "H1:OAF-REFL_A_RIN_PREFILT_OUT_DQ",
    "PEM_MAINS": "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
    "ETMX_L1": "H1:SUS-ETMX_L1_CAL_LINE_OUT_DQ",
    "ETMX_L2": "H1:SUS-ETMX_L2_CAL_LINE_OUT_DQ",
    "ETMX_L3": "H1:SUS-ETMX_L3_CAL_LINE_OUT_DQ",
    "PI_MON": "H1:SUS-PI_PROC_COMPUTE_MODE29_RMSMON",
    "PCALX": "H1:CAL-PCALX_RX_PD_OUT_DQ",
    "PCALY": "H1:CAL-PCALY_RX_PD_OUT_DQ",
}


def load(name, start=FRAME_START, end=FRAME_END):
    channel_map = dict(CHANNELS)
    if name not in channel_map:
        raise KeyError(f"Unknown channel label: {name}")
    channel = channel_map[name]
    ts = TimeSeries.read(PATH, channel=channel, start=start, end=end)
    x = np.asarray(ts.value, dtype=np.float64)
    try:
        fs = float(ts.sample_rate.value)
    except AttributeError:
        fs = float(ts.sample_rate)
    return x, fs


def clean(x):
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if not np.all(good):
        med = np.nanmedian(x)
        x = np.where(good, x, med)
    return x


def stats(x):
    x = clean(x)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x*x))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p01": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def bandpower(x, fs, low, high):
    x = clean(x)
    nperseg = min(len(x), max(256, int(fs * 2)))
    noverlap = nperseg // 2
    f, p = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )
    m = (f >= low) & (f < high)
    if not np.any(m):
        return np.nan
    return float(np.trapezoid(p[m], f[m]))


def window_features(x, fs):
    bands = [(0,2),(2,5),(5,10),(10,20),(20,50),(50,100)]
    out = stats(x)
    for a,b in bands:
        out[f"bp_{a}_{b}"] = bandpower(x, fs, a, b)
    return out


def candidate_scan(x, fs, start):
    win = int(round(0.25 * fs))
    rows = []
    n = len(x)
    for i in range(0, n-win+1, win):
        z = x[i:i+win]
        rows.append({
            "gps": start + i/fs,
            "mean": float(np.mean(z)),
            "std": float(np.std(z)),
            "rms": float(np.sqrt(np.mean(z*z))),
        })
    return rows


def percentile(v, arr):
    arr = np.asarray(arr, dtype=float)
    return float(100*np.mean(arr <= v))


def correlation(a, fa, b, fb):
    duration = min(len(a)/fa, len(b)/fb)
    fs = min(fa, fb, 1024.0)
    n = int(duration * fs)
    if n < 100:
        return np.nan
    ta = np.linspace(0, duration, len(a), endpoint=False)
    tb = np.linspace(0, duration, len(b), endpoint=False)
    t = np.arange(n)/fs
    aa = np.interp(t, ta, a[:len(ta)])
    bb = np.interp(t, tb, b[:len(tb)])
    aa -= np.mean(aa)
    bb -= np.mean(bb)
    da = np.std(aa)
    db = np.std(bb)
    if da == 0 or db == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0,1])


def direct_dependency(x, y):
    n = min(len(x), len(y))
    x = clean(x[:n])
    y = clean(y[:n])

    X = np.column_stack([x, np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred

    xr = np.sqrt(np.mean(x*x))
    yr = np.sqrt(np.mean(y*y))
    rr = np.sqrt(np.mean(resid*resid))

    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "y_rms": float(yr),
        "residual_rms": float(rr),
        "residual_fraction_of_y": float(rr/yr) if yr else np.nan,
        "corr": float(np.corrcoef(x,y)[0,1]),
        "plus_identity_rms": float(np.sqrt(np.mean((y-x)**2))),
        "minus_identity_rms": float(np.sqrt(np.mean((y+x)**2))),
    }


def save_json(obj, path):
    def conv(v):
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def main():
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("tables").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("statistics").mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LIGO OVERNIGHT ANALYSIS")
    print("="*80)
    print("Reading:", PATH)
    print("Original GWF will not be modified.")
    print()

    data = {}
    metadata = {}

    for label, channel in CHANNELS.items():
        print("Loading", label)
        try:
            x, fs = load(label)
            x = clean(x)
            data[label] = x
            metadata[label] = {
                "channel": channel,
                "sample_rate": fs,
                "samples": len(x),
                "duration": len(x)/fs,
            }
        except Exception as e:
            print("FAILED:", e)

    save_json(metadata, OUT/"tables"/"channel_metadata.json")

    print()
    print("Loaded", len(data), "channels.")

    # Full-frame statistics
    full = {}
    for label, x in data.items():
        full[label] = stats(x)

    save_json(full, OUT/"tables"/"full_frame_statistics.json")

    # Candidate scan on target
    target = data["WFS_PIT"]
    fs = metadata["WFS_PIT"]["sample_rate"]

    candidates = candidate_scan(target, fs, FRAME_START)

    means = np.array([r["mean"] for r in candidates])
    rms = np.array([r["rms"] for r in candidates])
    stds = np.array([r["std"] for r in candidates])

    for r in candidates:
        r["mean_percentile"] = percentile(r["mean"], means)
        r["rms_percentile"] = percentile(r["rms"], rms)
        r["std_percentile"] = percentile(r["std"], stds)

    candidates_sorted = sorted(
        candidates,
        key=lambda r: max(
            abs(r["mean_percentile"]-50),
            abs(r["rms_percentile"]-50),
            abs(r["std_percentile"]-50)
        ),
        reverse=True
    )

    save_json(
        candidates_sorted,
        OUT/"tables"/"candidate_windows.json"
    )

    # Episode vs controls
    controls = []
    for start in np.arange(FRAME_START, FRAME_END-7+0.001, 1.0):
        end = start + 7
        if start < EP_END and end > EP_START:
            continue
        controls.append(float(start))

    episode_results = {}
    control_results = {}

    for label, x in data.items():
        fs = metadata[label]["sample_rate"]

        i0 = int(round((EP_START-FRAME_START)*fs))
        i1 = int(round((EP_END-FRAME_START)*fs))

        episode = x[i0:i1]
        episode_results[label] = window_features(episode, fs)

        vals = []
        for start in controls:
            a = int(round((start-FRAME_START)*fs))
            b = int(round((start+7-FRAME_START)*fs))
            vals.append(window_features(x[a:b], fs))

        control_results[label] = vals

    comparison = {}

    for label in episode_results:
        comparison[label] = {}

        for key, value in episode_results[label].items():
            if not key.startswith("bp_"):
                continue

            arr = np.array([
                z[key] for z in control_results[label]
                if np.isfinite(z[key])
            ])

            med = np.median(arr)
            p95 = np.percentile(arr, 95)

            comparison[label][key] = {
                "episode": float(value),
                "control_median": float(med),
                "control_95th": float(p95),
                "ratio_to_median": float(value/med) if med else np.nan,
                "control_percentile": percentile(value, arr),
            }

    save_json(
        comparison,
        OUT/"statistics"/"episode_vs_controls.json"
    )

    # Direct OAF dependency
    dependencies = {}

    pairs = [
        ("WFS_PIT", "OAF_PIT"),
        ("WFS_YAW", "OAF_YAW"),
    ]

    for a_label, b_label in pairs:
        fs_a = metadata[a_label]["sample_rate"]
        fs_b = metadata[b_label]["sample_rate"]

        ia0 = int(round((EP_START-FRAME_START)*fs_a))
        ia1 = int(round((EP_END-FRAME_START)*fs_a))
        ib0 = int(round((EP_START-FRAME_START)*fs_b))
        ib1 = int(round((EP_END-FRAME_START)*fs_b))

        dependencies[f"{a_label}_vs_{b_label}"] = {
            "episode": direct_dependency(
                data[a_label][ia0:ia1],
                data[b_label][ib0:ib1]
            )
        }

    save_json(
        dependencies,
        OUT/"statistics"/"oaf_dependency.json"
    )

    # Cross-channel episode correlations
    correlations = {}

    fs_target = metadata["WFS_PIT"]["sample_rate"]
    t0 = int(round((EP_START-FRAME_START)*fs_target))
    t1 = int(round((EP_END-FRAME_START)*fs_target))

    for label, x in data.items():
        if label == "WFS_PIT":
            continue

        fs_x = metadata[label]["sample_rate"]
        j0 = int(round((EP_START-FRAME_START)*fs_x))
        j1 = int(round((EP_END-FRAME_START)*fs_x))

        correlations[label] = correlation(
            data["WFS_PIT"][t0:t1],
            fs_target,
            x[j0:j1],
            fs_x
        )

    save_json(
        correlations,
        OUT/"statistics"/"episode_correlations.json"
    )

    # Spectrograms
    for label in ["WFS_PIT","WFS_YAW","OAF_PIT","OAF_YAW"]:
        x = data[label]
        fs = metadata[label]["sample_rate"]

        i0 = int(round((EP_START-FRAME_START)*fs))
        i1 = int(round((EP_END-FRAME_START)*fs))
        z = x[i0:i1]

        nperseg = int(round(2*fs))
        noverlap = int(round(fs))

        f, t, Z = signal.stft(
            z,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            boundary=None,
            padded=False,
        )

        m = f <= 100
        power = np.abs(Z[m])**2
        db = 10*np.log10(np.maximum(power, 1e-30))

        plt.figure(figsize=(12,6))
        plt.pcolormesh(
            EP_START+t,
            f[m],
            db,
            shading="auto"
        )
        plt.xlabel("GPS time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"{label}: 6121–6128 time-frequency power")
        plt.ylim(0,100)
        plt.colorbar(label="Power (dB)")
        plt.tight_layout()
        plt.savefig(
            OUT/"figures"/f"{label.lower()}_spectrogram.png",
            dpi=200
        )
        plt.close()

    # Candidate plot
    times = np.array([r["gps"] for r in candidates])
    rmsvals = np.array([r["rms"] for r in candidates])

    plt.figure(figsize=(12,5))
    plt.plot(times, rmsvals)
    plt.axvspan(EP_START, EP_END, alpha=0.2)
    plt.xlabel("GPS time (s)")
    plt.ylabel("0.25 s RMS")
    plt.title("WFS PIT full-frame RMS scan")
    plt.tight_layout()
    plt.savefig(
        OUT/"figures"/"wfs_pit_rms_scan.png",
        dpi=200
    )
    plt.close()

    # Human-readable summary
    strongest = candidates_sorted[:20]

    with open(OUT/"ANALYSIS_SUMMARY.txt", "w") as f:
        f.write("LIGO AUXILIARY CHANNEL ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Frame: {FRAME_START} - {FRAME_END}\n")
        f.write(f"Episode: {EP_START} - {EP_END}\n")
        f.write(f"Channels loaded: {len(data)}\n\n")

        f.write("TOP CANDIDATE WINDOWS\n")
        f.write("-"*70 + "\n")

        for r in strongest:
            f.write(
                f"{r['gps']:.3f} | "
                f"mean={r['mean']:.6e} | "
                f"rms={r['rms']:.6e} | "
                f"std={r['std']:.6e} | "
                f"mean_pct={r['mean_percentile']:.2f} | "
                f"rms_pct={r['rms_percentile']:.2f} | "
                f"std_pct={r['std_percentile']:.2f}\n"
            )

        f.write("\nEPISODE VS CONTROLS\n")
        f.write("-"*70 + "\n")

        for label in comparison:
            f.write(f"\n{label}\n")
            for key, v in comparison[label].items():
                f.write(
                    f"{key}: "
                    f"episode={v['episode']:.6e}, "
                    f"median={v['control_median']:.6e}, "
                    f"95th={v['control_95th']:.6e}, "
                    f"ratio={v['ratio_to_median']:.3f}, "
                    f"percentile={v['control_percentile']:.2f}\n"
                )

        f.write("\nDIRECT OAF DEPENDENCY\n")
        f.write("-"*70 + "\n")

        for pair, result in dependencies.items():
            f.write(f"\n{pair}\n")
            for k,v in result["episode"].items():
                f.write(f"{k}: {v}\n")

        f.write("\nEPISODE CORRELATIONS WITH WFS PIT\n")
        f.write("-"*70 + "\n")

        for label, r in correlations.items():
            f.write(f"{label}: {r:.6f}\n")

    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Results:")
    print(OUT)
    print()
    print("Summary:")
    print(OUT/"ANALYSIS_SUMMARY.txt")
    print()
    print("Original GWF was read-only.")
    print("You can now leave this running environment alone.")


if __name__ == "__main__":
    main()

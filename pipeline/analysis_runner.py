import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from gwpy.timeseries import TimeSeries

PATH = REPO_ROOT / "data/auxiliary/W10_H1_AUX_AR1.gwf"
ROOT = REPO_ROOT / "results_overnight"

FRAME_START = 1376516096.0
FRAME_END = 1376516160.0

EP_START = 1376516121.0
EP_END = 1376516128.0

BANDS = [
    (0, 2),
    (2, 5),
    (5, 10),
    (10, 20),
    (20, 50),
    (50, 100),
]

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

WINDOWS = [0.25, 0.5, 1.0, 2.0, 4.0, 7.0]
SURROGATES = 5000
LAG_SECONDS = 2.0
LAG_STEP = 0.25


def mkdirs():
    for d in [
        ROOT,
        ROOT / "tables",
        ROOT / "figures",
        ROOT / "statistics",
        ROOT / "spectrograms",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def load_channel(label):
    channel = CHANNELS[label]

    ts = TimeSeries.read(
        PATH,
        channel=channel,
        start=FRAME_START,
        end=FRAME_END,
    )

    x = np.asarray(ts.value, dtype=np.float64)

    try:
        fs = float(ts.sample_rate.value)
    except AttributeError:
        fs = float(ts.sample_rate)

    x = np.where(np.isfinite(x), x, np.nanmedian(x))

    return x, fs


def basic_stats(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p01": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
    }


def welch_bands(x, fs):
    nperseg = min(len(x), max(256, int(round(2 * fs))))

    if nperseg < 16:
        return {f"bp_{a}_{b}": np.nan for a, b in BANDS}

    noverlap = nperseg // 2

    f, p = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )

    result = {}

    for low, high in BANDS:
        m = (f >= low) & (f < high)

        if np.any(m):
            result[f"bp_{low}_{high}"] = float(
                np.trapezoid(p[m], f[m])
            )
        else:
            result[f"bp_{low}_{high}"] = np.nan

    return result


def window_feature(x, fs):
    result = basic_stats(x)
    result.update(welch_bands(x, fs))
    return result


def scan_channel(label, x, fs, window):
    nwin = int(round(window * fs))

    if nwin > len(x):
        return []

    rows = []

    for start in range(0, len(x) - nwin + 1, nwin):
        z = x[start:start + nwin]

        row = window_feature(z, fs)
        row["gps_start"] = FRAME_START + start / fs
        row["gps_end"] = FRAME_START + (start + nwin) / fs
        row["duration"] = window

        rows.append(row)

    return rows


def percentile(value, distribution):
    distribution = np.asarray(distribution, dtype=float)
    distribution = distribution[np.isfinite(distribution)]

    if len(distribution) == 0:
        return np.nan

    return float(
        100 * np.mean(distribution <= value)
    )


def save_json(path, obj):
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)


def aligned_series(a, fa, b, fb):
    duration = min(len(a) / fa, len(b) / fb)

    fs = min(fa, fb, 1024.0)

    n = int(duration * fs)

    if n < 100:
        return None, None, None

    ta = np.arange(len(a)) / fa
    tb = np.arange(len(b)) / fb
    t = np.arange(n) / fs

    aa = np.interp(t, ta, a[:len(ta)])
    bb = np.interp(t, tb, b[:len(tb)])

    return aa, bb, fs


def max_lag_correlation(a, fa, b, fb):
    aligned = aligned_series(a, fa, b, fb)

    if aligned[0] is None:
        return np.nan, np.nan

    aa, bb, fs = aligned

    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)

    sa = np.std(aa)
    sb = np.std(bb)

    if sa == 0 or sb == 0:
        return np.nan, np.nan

    max_lag = int(round(LAG_SECONDS * fs))
    step = max(1, int(round(LAG_STEP * fs)))

    best_r = 0.0
    best_lag = 0.0

    for lag in range(-max_lag, max_lag + 1, step):

        if lag < 0:
            x = aa[-lag:]
            y = bb[:len(bb) + lag]
        elif lag > 0:
            x = aa[:len(aa) - lag]
            y = bb[lag:]
        else:
            x = aa
            y = bb

        if len(x) < 100:
            continue

        r = np.corrcoef(x, y)[0, 1]

        if np.isfinite(r) and abs(r) > abs(best_r):
            best_r = r
            best_lag = lag / fs

    return float(best_r), float(best_lag)


def direct_dependency(x, y):
    n = min(len(x), len(y))

    x = x[:n]
    y = y[:n]

    X = np.column_stack([
        x,
        np.ones(n),
    ])

    coef, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    prediction = X @ coef
    residual = y - prediction

    yrms = np.sqrt(np.mean(y * y))
    rrms = np.sqrt(np.mean(residual * residual))

    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "y_rms": float(yrms),
        "residual_rms": float(rrms),
        "residual_fraction": float(rrms / yrms) if yrms else np.nan,
        "identity_rms": float(
            np.sqrt(np.mean((y - x) ** 2))
        ),
        "negative_identity_rms": float(
            np.sqrt(np.mean((y + x) ** 2))
        ),
    }


def surrogate_test(a, fa, b, fb, observed):
    aligned = aligned_series(a, fa, b, fb)

    if aligned[0] is None:
        return {
            "observed": np.nan,
            "surrogate_median": np.nan,
            "surrogate_95": np.nan,
            "surrogate_99": np.nan,
            "empirical_p": np.nan,
        }

    aa, bb, fs = aligned

    aa -= np.mean(aa)
    bb -= np.mean(bb)

    n = len(bb)

    rng = np.random.default_rng(20260923)

    values = np.empty(SURROGATES)

    minimum_shift = max(1, int(0.5 * fs))
    possible = np.arange(
        minimum_shift,
        n - minimum_shift
    )

    for i in range(SURROGATES):

        shift = int(
            rng.choice(possible)
        )

        shifted = np.roll(bb, shift)

        r, _ = max_lag_correlation(
            aa,
            fs,
            shifted,
            fs
        )

        values[i] = abs(r)

    observed_abs = abs(observed)

    p = (
        1 + np.sum(values >= observed_abs)
    ) / (
        SURROGATES + 1
    )

    return {
        "observed": float(observed_abs),
        "surrogate_median": float(np.median(values)),
        "surrogate_95": float(np.percentile(values, 95)),
        "surrogate_99": float(np.percentile(values, 99)),
        "empirical_p": float(p),
    }


def spectrogram(label, x, fs):
    start = int(round((EP_START - FRAME_START) * fs))
    end = int(round((EP_END - FRAME_START) * fs))

    z = x[start:end]

    nperseg = int(round(2 * fs))
    noverlap = int(round(fs))

    if len(z) < nperseg:
        return

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

    power = np.abs(Z[m]) ** 2
    db = 10 * np.log10(
        np.maximum(power, 1e-30)
    )

    plt.figure(figsize=(12, 6))

    plt.pcolormesh(
        EP_START + t,
        f[m],
        db,
        shading="auto",
    )

    plt.xlabel("GPS time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(
        f"{label}: 6121–6128 s"
    )

    plt.ylim(0, 100)

    plt.colorbar(
        label="Power (dB)"
    )

    plt.tight_layout()

    plt.savefig(
        ROOT
        / "spectrograms"
        / f"{label.lower()}_spectrogram.png",
        dpi=200,
    )

    plt.close()


def main():
    mkdirs()

    t0 = time.time()

    print("=" * 80)
    print("OVERNIGHT LIGO ANALYSIS")
    print("=" * 80)
    print("Channels:", len(CHANNELS))
    print("Windows:", WINDOWS)
    print("Surrogates:", SURROGATES)
    print()

    data = {}
    metadata = {}

    for label, channel in CHANNELS.items():

        print(f"[LOAD] {label}")

        try:
            x, fs = load_channel(label)

            data[label] = x

            metadata[label] = {
                "channel": channel,
                "sample_rate": fs,
                "samples": len(x),
                "duration": len(x) / fs,
            }

        except Exception as e:

            print(
                f"[FAILED] {label}: "
                f"{type(e).__name__}: {e}"
            )

    save_json(
        ROOT / "tables" / "metadata.json",
        metadata
    )

    print()
    print(
        f"Successfully loaded {len(data)} "
        f"of {len(CHANNELS)} channels."
    )

    if "WFS_PIT" not in data:
        raise RuntimeError(
            "WFS_PIT failed to load."
        )

    # ------------------------------------------------------------
    # FULL FRAME STATISTICS
    # ------------------------------------------------------------

    print()
    print("[1/7] Full-frame statistics")

    full_stats = {}

    for label, x in data.items():
        full_stats[label] = basic_stats(x)

    save_json(
        ROOT / "tables" / "full_frame_statistics.json",
        full_stats
    )

    # ------------------------------------------------------------
    # MULTI-SCALE WINDOW SCAN
    # ------------------------------------------------------------

    print()
    print("[2/7] Multi-scale anomaly scan")

    candidate_summary = {}

    for label, x in data.items():

        candidate_summary[label] = {}

        fs = metadata[label]["sample_rate"]

        for window in WINDOWS:

            print(
                f"  {label}: {window:.2f}s"
            )

            rows = scan_channel(
                label,
                x,
                fs,
                window
            )

            candidate_summary[label][str(window)] = rows

            save_json(
                ROOT
                / "tables"
                / f"{label.lower()}_{window:.2f}s.json",
                rows
            )

    # ------------------------------------------------------------
    # EPISODE VS CONTROLS
    # ------------------------------------------------------------

    print()
    print("[3/7] Episode versus control windows")

    controls = []

    for start in np.arange(
        FRAME_START,
        FRAME_END - 7 + 0.001,
        1.0
    ):

        end = start + 7

        if start < EP_END and end > EP_START:
            continue

        controls.append(float(start))

    comparison = {}

    for label, x in data.items():

        fs = metadata[label]["sample_rate"]

        i0 = int(
            round((EP_START - FRAME_START) * fs)
        )
        i1 = int(
            round((EP_END - FRAME_START) * fs)
        )

        episode = x[i0:i1]

        ep = window_feature(
            episode,
            fs
        )

        comparison[label] = {}

        for key, value in ep.items():

            if not key.startswith("bp_"):
                continue

            values = []

            for start in controls:

                a = int(
                    round((start - FRAME_START) * fs)
                )

                b = int(
                    round((start + 7 - FRAME_START) * fs)
                )

                values.append(
                    window_feature(
                        x[a:b],
                        fs
                    )[key]
                )

            values = np.asarray(values)

            median = np.nanmedian(values)

            comparison[label][key] = {
                "episode": float(value),
                "median": float(median),
                "p95": float(
                    np.nanpercentile(values, 95)
                ),
                "ratio": float(
                    value / median
                ) if median else np.nan,
                "percentile": percentile(
                    value,
                    values
                ),
            }

    save_json(
        ROOT
        / "statistics"
        / "episode_vs_controls.json",
        comparison
    )

    # ------------------------------------------------------------
    # DIRECT OAF DEPENDENCY
    # ------------------------------------------------------------

    print()
    print("[4/7] OAF dependency analysis")

    dependencies = {}

    for a_label, b_label in [
        ("WFS_PIT", "OAF_PIT"),
        ("WFS_YAW", "OAF_YAW"),
    ]:

        fa = metadata[a_label]["sample_rate"]
        fb = metadata[b_label]["sample_rate"]

        a0 = int(
            round((EP_START - FRAME_START) * fa)
        )
        a1 = int(
            round((EP_END - FRAME_START) * fa)
        )

        b0 = int(
            round((EP_START - FRAME_START) * fb)
        )
        b1 = int(
            round((EP_END - FRAME_START) * fb)
        )

        dependencies[
            f"{a_label}_vs_{b_label}"
        ] = direct_dependency(
            data[a_label][a0:a1],
            data[b_label][b0:b1],
        )

    save_json(
        ROOT
        / "statistics"
        / "oaf_dependency.json",
        dependencies
    )

    # ------------------------------------------------------------
    # CROSS CHANNEL CORRELATION + SURROGATES
    # ------------------------------------------------------------

    print()
    print(
        "[5/7] Cross-channel lag correlations "
        f"and {SURROGATES} surrogate tests"
    )

    target_label = "WFS_PIT"

    fs_target = metadata[target_label]["sample_rate"]

    a0 = int(
        round((EP_START - FRAME_START) * fs_target)
    )
    a1 = int(
        round((EP_END - FRAME_START) * fs_target)
    )

    target_episode = data[target_label][a0:a1]

    cross = {}

    for label, x in data.items():

        if label == target_label:
            continue

        fs = metadata[label]["sample_rate"]

        b0 = int(
            round((EP_START - FRAME_START) * fs)
        )
        b1 = int(
            round((EP_END - FRAME_START) * fs)
        )

        other = x[b0:b1]

        r, lag = max_lag_correlation(
            target_episode,
            fs_target,
            other,
            fs
        )

        print(
            f"  {label}: r={r:.6f}, lag={lag:.3f}s"
        )

        cross[label] = {
            "observed_r": r,
            "lag_seconds": lag,
        }

        cross[label]["surrogates"] = surrogate_test(
            target_episode,
            fs_target,
            other,
            fs,
            r
        )

    save_json(
        ROOT
        / "statistics"
        / "cross_channel_surrogates.json",
        cross
    )

    # ------------------------------------------------------------
    # SPECTROGRAMS
    # ------------------------------------------------------------

    print()
    print("[6/7] Generating spectrograms")

    for label in [
        "WFS_PIT",
        "WFS_YAW",
        "OAF_PIT",
        "OAF_YAW",
    ]:

        if label in data:
            spectrogram(
                label,
                data[label],
                metadata[label]["sample_rate"]
            )

    # ------------------------------------------------------------
    # TARGET RMS FIGURE
    # ------------------------------------------------------------

    x = data["WFS_PIT"]
    fs = metadata["WFS_PIT"]["sample_rate"]

    rows = candidate_summary[
        "WFS_PIT"
    ]["0.25"]

    t = np.array([
        r["gps_start"] for r in rows
    ])

    rms = np.array([
        r["rms"] for r in rows
    ])

    plt.figure(figsize=(14, 5))

    plt.plot(
        t,
        rms,
    )

    plt.axvspan(
        EP_START,
        EP_END,
        alpha=0.2,
    )

    plt.xlabel("GPS time (s)")
    plt.ylabel("0.25 s RMS")
    plt.title("WFS PIT full-frame RMS scan")

    plt.tight_layout()

    plt.savefig(
        ROOT
        / "figures"
        / "wfs_pit_full_frame_rms.png",
        dpi=200,
    )

    plt.close()

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    print()
    print("[7/7] Writing final report")

    elapsed = time.time() - t0

    with open(
        ROOT / "OVERNIGHT_REPORT.txt",
        "w"
    ) as f:

        f.write(
            "LIGO AUXILIARY CHANNEL OVERNIGHT ANALYSIS\n"
        )

        f.write("=" * 80 + "\n\n")

        f.write(
            f"Frame: {FRAME_START} - {FRAME_END}\n"
        )

        f.write(
            f"Episode: {EP_START} - {EP_END}\n"
        )

        f.write(
            f"Channels successfully loaded: "
            f"{len(data)}/{len(CHANNELS)}\n"
        )

        f.write(
            f"Surrogates per cross-channel test: "
            f"{SURROGATES}\n"
        )

        f.write(
            f"Runtime: {elapsed:.1f} seconds\n\n"
        )

        f.write(
            "IMPORTANT SCIENTIFIC STATUS\n"
        )

        f.write("-" * 80 + "\n")

        f.write(
            "This pipeline identifies statistical and "
            "signal-processing anomalies. It does not "
            "classify an auxiliary-channel feature as "
            "a gravitational-wave detection.\n\n"
        )

        f.write(
            "OAF DEPENDENCY\n"
        )

        f.write("-" * 80 + "\n")

        for pair, result in dependencies.items():

            f.write(
                f"\n{pair}\n"
            )

            for key, value in result.items():

                f.write(
                    f"  {key}: {value}\n"
                )

        f.write(
            "\nCROSS-CHANNEL SURROGATE RESULTS\n"
        )

        f.write("-" * 80 + "\n")

        for label, result in cross.items():

            s = result["surrogates"]

            f.write(
                f"\n{label}\n"
            )

            f.write(
                f"  observed |r|: "
                f"{s['observed']:.8f}\n"
            )

            f.write(
                f"  surrogate median: "
                f"{s['surrogate_median']:.8f}\n"
            )

            f.write(
                f"  surrogate 95th: "
                f"{s['surrogate_95']:.8f}\n"
            )

            f.write(
                f"  surrogate 99th: "
                f"{s['surrogate_99']:.8f}\n"
            )

            f.write(
                f"  empirical p: "
                f"{s['empirical_p']:.8f}\n"
            )

        f.write(
            "\nRESULT DIRECTORY\n"
        )

        f.write("-" * 80 + "\n")

        f.write(
            str(ROOT) + "\n"
        )

    print()
    print("=" * 80)
    print("OVERNIGHT ANALYSIS COMPLETE")
    print("=" * 80)
    print("Results:", ROOT)
    print("Report:", ROOT / "OVERNIGHT_REPORT.txt")
    print("Original GWF was not modified.")
    print()


if __name__ == "__main__":
    main()

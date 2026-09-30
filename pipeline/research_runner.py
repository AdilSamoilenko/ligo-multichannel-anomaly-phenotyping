import csv
import json
import math
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from gwpy.timeseries import TimeSeries


PATH = Path(
    REPO_ROOT / ""
    "data/auxiliary/W10_H1_AUX_AR1.gwf"
)

ROOT = Path(
    REPO_ROOT / ""
    "results_overnight"
)

FRAME_START = 1376516096.0
FRAME_END = 1376516160.0

EPISODE_START = 1376516121.0
EPISODE_END = 1376516128.0

BANDS = [
    (0, 2),
    (2, 5),
    (5, 10),
    (10, 20),
    (20, 50),
    (50, 100),
    (100, 300),
    (300, 1000),
]

WINDOWS = [
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    7.0,
]

SURROGATES = 2000

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


def setup():
    for d in [
        ROOT,
        ROOT / "tables",
        ROOT / "statistics",
        ROOT / "figures",
        ROOT / "figures" / "spectrograms",
        ROOT / "figures" / "correlations",
        ROOT / "figures" / "distributions",
        ROOT / "figures" / "candidates",
        ROOT / "checkpoints",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")

    def conv(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v).__name__)

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=conv)

    tmp.replace(path)


def load_json(path, default=None):
    if not path.exists():
        return default

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_channel(label):
    channel = CHANNELS[label]

    ts = TimeSeries.read(
        str(PATH),
        channel=channel,
        start=FRAME_START,
        end=FRAME_END,
    )

    x = np.asarray(ts.value, dtype=np.float64)

    try:
        fs = float(ts.sample_rate.value)
    except AttributeError:
        fs = float(ts.sample_rate)

    finite = np.isfinite(x)

    if not np.all(finite):
        replacement = np.nanmedian(x)
        x = np.where(finite, x, replacement)

    return x, fs


def robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return 1.4826 * mad


def basic_features(x):
    x = np.asarray(x, dtype=np.float64)

    mean = np.mean(x)
    median = np.median(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x * x))

    if std > 0:
        skew = stats.skew(x, bias=False)
        kurt = stats.kurtosis(
            x,
            fisher=True,
            bias=False,
        )
    else:
        skew = 0.0
        kurt = 0.0

    peak = np.max(np.abs(x))

    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "rms": float(rms),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "peak_abs": float(peak),
        "crest_factor": float(
            peak / rms
        ) if rms > 0 else np.nan,
    }


def spectral_features(x, fs):
    x = np.asarray(x, dtype=np.float64)

    if len(x) < 64:
        return {
            "spectral_centroid": np.nan,
            "spectral_entropy": np.nan,
            "dominant_frequency": np.nan,
        }

    nperseg = min(
        len(x),
        max(64, int(round(2 * fs)))
    )

    f, p = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )

    total = np.trapezoid(p, f)

    if total <= 0:
        return {
            "spectral_centroid": np.nan,
            "spectral_entropy": np.nan,
            "dominant_frequency": np.nan,
        }

    centroid = (
        np.trapezoid(f * p, f) / total
    )

    q = p / np.sum(p)
    q = q[q > 0]

    entropy = -np.sum(q * np.log(q))
    entropy /= math.log(len(q))

    dominant = f[np.argmax(p)]

    return {
        "spectral_centroid": float(centroid),
        "spectral_entropy": float(entropy),
        "dominant_frequency": float(dominant),
    }


def band_features(x, fs):
    x = np.asarray(x, dtype=np.float64)

    nperseg = min(
        len(x),
        max(64, int(round(2 * fs)))
    )

    if nperseg < 64:
        return {
            f"power_{lo}_{hi}": np.nan
            for lo, hi in BANDS
        }

    f, p = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
    )

    result = {}

    for lo, hi in BANDS:

        mask = (
            (f >= lo) &
            (f < hi)
        )

        if np.any(mask):
            result[
                f"power_{lo}_{hi}"
            ] = float(
                np.trapezoid(
                    p[mask],
                    f[mask],
                )
            )
        else:
            result[
                f"power_{lo}_{hi}"
            ] = np.nan

    return result


def all_features(x, fs):
    result = basic_features(x)

    result.update(
        spectral_features(x, fs)
    )

    result.update(
        band_features(x, fs)
    )

    return result


def multiscale_scan(label, x, fs):
    results = {}

    for window in WINDOWS:

        n = int(round(window * fs))

        if n > len(x):
            continue

        rows = []

        for start in range(
            0,
            len(x) - n + 1,
            n,
        ):

            z = x[start:start + n]

            row = all_features(
                z,
                fs,
            )

            row["gps_start"] = (
                FRAME_START +
                start / fs
            )

            row["gps_end"] = (
                FRAME_START +
                (start + n) / fs
            )

            row["window_seconds"] = window

            rows.append(row)

        results[str(window)] = rows

    return results


def robust_scores(rows):
    if not rows:
        return rows

    numeric_keys = [
        "mean",
        "std",
        "rms",
        "ptp",
        "peak_abs",
        "crest_factor",
        "spectral_centroid",
        "spectral_entropy",
        "power_0_2",
        "power_2_5",
        "power_5_10",
        "power_10_20",
        "power_20_50",
        "power_50_100",
    ]

    arrays = {}

    for key in numeric_keys:
        values = np.array([
            r.get(key, np.nan)
            for r in rows
        ])

        finite = np.isfinite(values)

        if not np.any(finite):
            continue

        med = np.nanmedian(values)
        sig = robust_sigma(
            values[finite]
        )

        if not np.isfinite(sig) or sig == 0:
            sig = np.nanstd(
                values[finite]
            )

        if not np.isfinite(sig) or sig == 0:
            sig = 1.0

        arrays[key] = (
            values,
            med,
            sig,
        )

    for i, row in enumerate(rows):

        scores = []

        for key, (
            values,
            med,
            sig,
        ) in arrays.items():

            value = values[i]

            if np.isfinite(value):
                scores.append(
                    abs(value - med) / sig
                )

        row["aggregate_anomaly_score"] = (
            float(np.median(scores))
            if scores
            else np.nan
        )

    return rows


def save_csv(path, rows):
    if not rows:
        return

    keys = sorted({
        k
        for row in rows
        for k in row
    })

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def resample_pair(a, fa, b, fb, target_fs=1024.0):
    duration = min(
        len(a) / fa,
        len(b) / fb,
    )

    fs = min(
        float(target_fs),
        float(fa),
        float(fb),
    )

    n = int(
        duration * fs
    )

    if n < 1000:
        return None

    ta = np.arange(len(a)) / fa
    tb = np.arange(len(b)) / fb
    t = np.arange(n) / fs

    aa = np.interp(
        t,
        ta,
        a,
    )

    bb = np.interp(
        t,
        tb,
        b,
    )

    return aa, bb, fs


def lagged_correlation(
    a,
    fa,
    b,
    fb,
    max_lag=2.0,
    lag_step=0.05,
):
    aligned = resample_pair(
        a,
        fa,
        b,
        fb,
    )

    if aligned is None:
        return {
            "r": np.nan,
            "lag": np.nan,
        }

    aa, bb, fs = aligned

    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)

    sa = np.std(aa)
    sb = np.std(bb)

    if sa == 0 or sb == 0:
        return {
            "r": np.nan,
            "lag": np.nan,
        }

    max_samples = int(
        round(max_lag * fs)
    )

    step = max(
        1,
        int(round(lag_step * fs))
    )

    best_r = 0.0
    best_lag = 0.0

    for lag in range(
        -max_samples,
        max_samples + 1,
        step,
    ):

        if lag < 0:
            x = aa[-lag:]
            y = bb[:len(bb) + lag]

        elif lag > 0:
            x = aa[:-lag]
            y = bb[lag:]

        else:
            x = aa
            y = bb

        if len(x) < 500:
            continue

        r = np.corrcoef(
            x,
            y,
        )[0, 1]

        if np.isfinite(r):
            if abs(r) > abs(best_r):
                best_r = r
                best_lag = lag / fs

    return {
        "r": float(best_r),
        "lag": float(best_lag),
    }


def band_correlation(
    a,
    fa,
    b,
    fb,
    lo,
    hi,
):
    """
    Band-limited Pearson correlation.

    Handles:
      * low-pass bands beginning at 0 Hz
      * ordinary band-pass regions
      * bands approaching Nyquist
      * channels with different sample rates

    The original GWF is never modified.
    """

    aligned = resample_pair(
        a,
        fa,
        b,
        fb,
    )

    if aligned is None:
        return np.nan

    aa, bb, fs = aligned

    nyquist = fs / 2.0

    # Keep a small safety margin below Nyquist.
    hi_safe = min(
        float(hi),
        nyquist * 0.95,
    )

    lo_safe = max(
        0.0,
        float(lo),
    )

    # Invalid / unresolved band.
    if hi_safe <= 0:
        return np.nan

    if lo_safe >= hi_safe:
        return np.nan

    # A zero lower edge means LOW-PASS, not band-pass.
    if lo_safe <= 0:

        # Very-low-frequency filtering needs enough
        # samples to avoid edge artefacts.
        if len(aa) < max(
            128,
            int(round(8 * fs / max(hi_safe, 1e-6)))
        ):
            return np.nan

        try:
            sos = signal.butter(
                4,
                hi_safe,
                btype="lowpass",
                fs=fs,
                output="sos",
            )

            aa_f = signal.sosfiltfilt(
                sos,
                aa,
            )

            bb_f = signal.sosfiltfilt(
                sos,
                bb,
            )

        except (ValueError, RuntimeError):
            return np.nan

    else:

        try:
            sos = signal.butter(
                4,
                [lo_safe, hi_safe],
                btype="bandpass",
                fs=fs,
                output="sos",
            )

            aa_f = signal.sosfiltfilt(
                sos,
                aa,
            )

            bb_f = signal.sosfiltfilt(
                sos,
                bb,
            )

        except (ValueError, RuntimeError):
            return np.nan

    aa_f = aa_f - np.mean(aa_f)
    bb_f = bb_f - np.mean(bb_f)

    sa = np.std(aa_f)
    sb = np.std(bb_f)

    if (
        not np.isfinite(sa)
        or not np.isfinite(sb)
        or sa == 0
        or sb == 0
    ):
        return np.nan

    r = np.corrcoef(
        aa_f,
        bb_f,
    )[0, 1]

    if not np.isfinite(r):
        return np.nan

    return float(r)


def dependency_metrics(x, y):
    n = min(
        len(x),
        len(y),
    )

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

    yrms = np.sqrt(
        np.mean(y * y)
    )

    rrms = np.sqrt(
        np.mean(residual * residual)
    )

    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "correlation": float(
            np.corrcoef(x, y)[0, 1]
        ),
        "r_squared": float(
            np.corrcoef(x, y)[0, 1] ** 2
        ),
        "y_rms": float(yrms),
        "residual_rms": float(rrms),
        "residual_fraction": float(
            rrms / yrms
        ) if yrms else np.nan,
        "identity_rms": float(
            np.sqrt(
                np.mean((y - x) ** 2)
            )
        ),
        "negative_identity_rms": float(
            np.sqrt(
                np.mean((y + x) ** 2)
            )
        ),
    }


def episode_slice(
    x,
    fs,
    start=EPISODE_START,
    end=EPISODE_END,
):
    i0 = int(
        round(
            (start - FRAME_START) * fs
        )
    )

    i1 = int(
        round(
            (end - FRAME_START) * fs
        )
    )

    return x[i0:i1]


def make_control_starts():
    starts = []

    for start in np.arange(
        FRAME_START,
        FRAME_END - 7.0 + 0.0001,
        1.0,
    ):

        end = start + 7.0

        if (
            start < EPISODE_END
            and end > EPISODE_START
        ):
            continue

        starts.append(
            float(start)
        )

    return starts


def extract_interval(
    x,
    fs,
    start,
    duration,
):
    i0 = int(
        round(
            (start - FRAME_START) * fs
        )
    )

    i1 = int(
        round(
            (start + duration - FRAME_START)
            * fs
        )
    )

    if i0 < 0 or i1 > len(x):
        return None

    return x[i0:i1]


def control_statistics(
    label,
    x,
    fs,
):
    starts = make_control_starts()

    episode = episode_slice(
        x,
        fs,
    )

    episode_features = all_features(
        episode,
        fs,
    )

    controls = []

    for start in starts:

        z = extract_interval(
            x,
            fs,
            start,
            7.0,
        )

        if z is None:
            continue

        row = all_features(
            z,
            fs,
        )

        row["gps_start"] = start

        controls.append(row)

    comparison = {}

    for key, episode_value in episode_features.items():

        values = np.array([
            row.get(key, np.nan)
            for row in controls
        ])

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            continue

        median = np.median(values)
        mad = np.median(
            np.abs(values - median)
        )

        sigma = 1.4826 * mad

        if sigma == 0:
            sigma = np.std(values)

        comparison[key] = {
            "episode": float(
                episode_value
            ),
            "control_median": float(
                median
            ),
            "control_p95": float(
                np.percentile(
                    values,
                    95,
                )
            ),
            "control_p99": float(
                np.percentile(
                    values,
                    99,
                )
            ),
            "percentile": float(
                100 *
                np.mean(
                    values <= episode_value
                )
            ),
            "robust_z": float(
                (episode_value - median)
                / sigma
            ) if sigma else np.nan,
        }

    return comparison


def full_channel_network(
    data,
    metadata,
):
    labels = list(data)

    result = {
        "zero_lag": {},
        "max_lag": {},
        "band_correlations": {},
    }

    for i, a_label in enumerate(labels):

        result["zero_lag"][a_label] = {}
        result["max_lag"][a_label] = {}
        result["band_correlations"][a_label] = {}

        for j, b_label in enumerate(labels):

            if j < i:
                continue

            fa = metadata[a_label]["fs"]
            fb = metadata[b_label]["fs"]

            aligned = resample_pair(
                data[a_label],
                fa,
                data[b_label],
                fb,
            )

            if aligned is None:
                continue

            aa, bb, fs = aligned

            r0 = np.corrcoef(
                aa - np.mean(aa),
                bb - np.mean(bb),
            )[0, 1]

            lag = lagged_correlation(
                data[a_label],
                fa,
                data[b_label],
                fb,
                max_lag=2.0,
                lag_step=0.05,
            )

            key = f"{a_label}__{b_label}"

            result["zero_lag"][a_label][b_label] = float(r0)

            result["max_lag"][a_label][b_label] = lag

            result["band_correlations"][a_label][b_label] = {}

            for lo, hi in BANDS:

                result[
                    "band_correlations"
                ][a_label][b_label][
                    f"{lo}_{hi}"
                ] = band_correlation(
                    data[a_label],
                    fa,
                    data[b_label],
                    fb,
                    lo,
                    hi,
                )

    return result


def episode_network(
    data,
    metadata,
):
    labels = list(data)

    result = {}

    for i, a_label in enumerate(labels):

        result[a_label] = {}

        for j, b_label in enumerate(labels):

            if j < i:
                continue

            aa = episode_slice(
                data[a_label],
                metadata[a_label]["fs"],
            )

            bb = episode_slice(
                data[b_label],
                metadata[b_label]["fs"],
            )

            lag = lagged_correlation(
                aa,
                metadata[a_label]["fs"],
                bb,
                metadata[b_label]["fs"],
                max_lag=2.0,
                lag_step=0.025,
            )

            key = f"{a_label}__{b_label}"

            result[a_label][b_label] = lag

    return result


def surrogate_distribution(
    a,
    fa,
    b,
    fb,
    observed,
    rng,
):
    aligned = resample_pair(
        a,
        fa,
        b,
        fb,
    )

    if aligned is None:
        return None

    aa, bb, fs = aligned

    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)

    n = len(bb)

    minimum_shift = max(
        1,
        int(round(0.5 * fs))
    )

    possible = np.arange(
        minimum_shift,
        n - minimum_shift,
        max(
            1,
            int(round(0.05 * fs))
        ),
    )

    values = np.empty(
        SURROGATES,
        dtype=np.float64,
    )

    for k in range(
        SURROGATES
    ):

        shift = int(
            rng.choice(possible)
        )

        shifted = np.roll(
            bb,
            shift,
        )

        r = lagged_correlation(
            aa,
            fs,
            shifted,
            fs,
            max_lag=2.0,
            lag_step=0.05,
        )["r"]

        values[k] = abs(r)

    observed_abs = abs(observed)

    p = (
        1 +
        np.sum(
            values >= observed_abs
        )
    ) / (
        SURROGATES + 1
    )

    return {
        "observed_abs_r": float(
            observed_abs
        ),
        "median": float(
            np.median(values)
        ),
        "p95": float(
            np.percentile(values, 95)
        ),
        "p99": float(
            np.percentile(values, 99)
        ),
        "maximum": float(
            np.max(values)
        ),
        "empirical_p": float(p),
    }


def plot_distribution(
    values,
    observed,
    title,
    path,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        values,
        bins=50,
    )

    plt.axvline(
        observed,
        linewidth=2,
    )

    plt.xlabel(
        "Absolute maximum lag correlation"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def plot_rms_scan(
    rows,
    path,
):
    if not rows:
        return

    t = np.array([
        r["gps_start"]
        for r in rows
    ])

    rms = np.array([
        r["rms"]
        for r in rows
    ])

    score = np.array([
        r.get(
            "aggregate_anomaly_score",
            np.nan,
        )
        for r in rows
    ])

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    ax[0].plot(
        t,
        rms,
    )

    ax[0].axvspan(
        EPISODE_START,
        EPISODE_END,
        alpha=0.2,
    )

    ax[0].set_ylabel(
        "RMS"
    )

    ax[0].set_title(
        "WFS PIT full-frame multiscale scan"
    )

    ax[1].plot(
        t,
        score,
    )

    ax[1].axvspan(
        EPISODE_START,
        EPISODE_END,
        alpha=0.2,
    )

    ax[1].set_xlabel(
        "GPS time (s)"
    )

    ax[1].set_ylabel(
        "Aggregate anomaly score"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
    )

    plt.close()


def plot_spectrogram(
    label,
    x,
    fs,
):
    z = episode_slice(
        x,
        fs,
    )

    nperseg = int(
        round(2 * fs)
    )

    noverlap = int(
        round(fs)
    )

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

    mask = f <= 100

    power = np.abs(
        Z[mask]
    ) ** 2

    db = 10 * np.log10(
        np.maximum(
            power,
            1e-30,
        )
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.pcolormesh(
        EPISODE_START + t,
        f[mask],
        db,
        shading="auto",
    )

    plt.xlabel(
        "GPS time (s)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        f"{label}: episode spectrogram"
    )

    plt.ylim(
        0,
        100,
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.tight_layout()

    plt.savefig(
        ROOT
        / "figures"
        / "spectrograms"
        / f"{label.lower()}.png",
        dpi=200,
    )

    plt.close()


def checkpoint(name, obj):
    atomic_json(
        ROOT
        / "checkpoints"
        / f"{name}.json",
        obj,
    )


def main():
    setup()

    start_time = time.time()

    print("=" * 80)
    print("OVERNIGHT RESEARCH PIPELINE")
    print("=" * 80)
    print(f"Input: {PATH}")
    print(f"Output: {ROOT}")
    print(f"Channels: {len(CHANNELS)}")
    print(f"Window scales: {len(WINDOWS)}")
    print(f"Surrogates: {SURROGATES}")
    print()

    data = {}
    metadata = {}

    # ==========================================================
    # PHASE 1: LOAD
    # ==========================================================

    print("[PHASE 1] Loading all channels")

    for label, channel in CHANNELS.items():

        checkpoint_path = (
            ROOT
            / "checkpoints"
            / f"channel_{label}.npz"
        )

        if checkpoint_path.exists():
            try:
                loaded = np.load(
                    checkpoint_path
                )

                data[label] = loaded["x"]

                metadata[label] = {
                    "channel": channel,
                    "fs": float(
                        loaded["fs"]
                    ),
                    "samples": len(
                        loaded["x"]
                    ),
                }

                print(
                    f"  {label}: checkpoint"
                )

                continue

            except Exception:
                pass

        print(
            f"  Loading {label}"
        )

        x, fs = load_channel(
            label
        )

        data[label] = x

        metadata[label] = {
            "channel": channel,
            "fs": fs,
            "samples": len(x),
            "duration": len(x) / fs,
        }

        np.savez_compressed(
            checkpoint_path,
            x=x,
            fs=fs,
        )

    atomic_json(
        ROOT / "tables" / "metadata.json",
        metadata,
    )

    print(
        f"Loaded {len(data)}/{len(CHANNELS)} channels"
    )

    if len(data) != len(CHANNELS):
        missing = [
            x for x in CHANNELS
            if x not in data
        ]

        raise RuntimeError(
            f"Missing channels: {missing}"
        )

    # ==========================================================
    # PHASE 2: MULTISCALE
    # ==========================================================

    print()
    print("[PHASE 2] Multiscale anomaly scan")

    all_scan_results = {}

    for label in data:

        output = (
            ROOT
            / "checkpoints"
            / f"scan_{label}.json"
        )

        if output.exists():

            result = load_json(
                output
            )

            if result is not None:
                all_scan_results[label] = result

                print(
                    f"  {label}: checkpoint"
                )

                continue

        print(
            f"  Scanning {label}"
        )

        result = multiscale_scan(
            label,
            data[label],
            metadata[label]["fs"],
        )

        for key in result:
            result[key] = robust_scores(
                result[key]
            )

            save_csv(
                ROOT
                / "tables"
                / f"{label}_{key}s.csv",
                result[key],
            )

        checkpoint(
            f"scan_{label}",
            result,
        )

        all_scan_results[label] = result

    # ==========================================================
    # PHASE 3: CONTROL POPULATION
    # ==========================================================

    print()
    print("[PHASE 3] Episode versus control population")

    controls = {}

    control_file = (
        ROOT
        / "checkpoints"
        / "controls.json"
    )

    if control_file.exists():

        controls = load_json(
            control_file,
            {},
        )

        print(
            "  Control results: checkpoint"
        )

    else:

        for label in data:

            print(
                f"  Controls: {label}"
            )

            controls[label] = control_statistics(
                label,
                data[label],
                metadata[label]["fs"],
            )

        checkpoint(
            "controls",
            controls,
        )

    # ==========================================================
    # PHASE 4: FULL CHANNEL NETWORK
    # ==========================================================

    print()
    print("[PHASE 4] Full 14-channel network")

    network_file = (
        ROOT
        / "checkpoints"
        / "full_network.json"
    )

    if network_file.exists():

        full_network = load_json(
            network_file
        )

        print(
            "  Network: checkpoint"
        )

    else:

        full_network = full_channel_network(
            data,
            metadata,
        )

        checkpoint(
            "full_network",
            full_network,
        )

    # ==========================================================
    # PHASE 5: EPISODE NETWORK
    # ==========================================================

    print()
    print("[PHASE 5] Episode network")

    episode_network_file = (
        ROOT
        / "checkpoints"
        / "episode_network.json"
    )

    if episode_network_file.exists():

        ep_network = load_json(
            episode_network_file
        )

        print(
            "  Episode network: checkpoint"
        )

    else:

        ep_network = episode_network(
            data,
            metadata,
        )

        checkpoint(
            "episode_network",
            ep_network,
        )

    # ==========================================================
    # PHASE 6: OAF DEPENDENCY
    # ==========================================================

    print()
    print("[PHASE 6] OAF dependency analysis")

    dependency_file = (
        ROOT
        / "statistics"
        / "oaf_dependency.json"
    )

    if dependency_file.exists():

        dependencies = load_json(
            dependency_file
        )

        print(
            "  OAF dependency: checkpoint"
        )

    else:

        dependencies = {}

        pairs = [
            ("WFS_PIT", "OAF_PIT"),
            ("WFS_YAW", "OAF_YAW"),
        ]

        for a_label, b_label in pairs:

            a_episode = episode_slice(
                data[a_label],
                metadata[a_label]["fs"],
            )

            b_episode = episode_slice(
                data[b_label],
                metadata[b_label]["fs"],
            )

            aligned = resample_pair(
                a_episode,
                metadata[a_label]["fs"],
                b_episode,
                metadata[b_label]["fs"],
            )

            if aligned is None:
                continue

            aa, bb, fs = aligned

            dependencies[
                f"{a_label}_vs_{b_label}"
            ] = dependency_metrics(
                aa,
                bb,
            )

        atomic_json(
            dependency_file,
            dependencies,
        )

    # ==========================================================
    # PHASE 7: TARGET CROSS-CHANNEL SURROGATES
    # ==========================================================

    print()
    print(
        "[PHASE 7] Cross-channel surrogate experiments"
    )

    surrogate_file = (
        ROOT
        / "statistics"
        / "surrogates.json"
    )

    if surrogate_file.exists():

        surrogate_results = load_json(
            surrogate_file
        )

        print(
            "  Surrogates: checkpoint"
        )

    else:

        surrogate_results = {}

        target = episode_slice(
            data["WFS_PIT"],
            metadata["WFS_PIT"]["fs"],
        )

        rng = np.random.default_rng(
            20260924
        )

        for label in data:

            if label == "WFS_PIT":
                continue

            print(
                f"  {label}: {SURROGATES} surrogates"
            )

            other = episode_slice(
                data[label],
                metadata[label]["fs"],
            )

            observed = lagged_correlation(
                target,
                metadata["WFS_PIT"]["fs"],
                other,
                metadata[label]["fs"],
                max_lag=2.0,
                lag_step=0.05,
            )

            result = surrogate_distribution(
                target,
                metadata["WFS_PIT"]["fs"],
                other,
                metadata[label]["fs"],
                observed["r"],
                rng,
            )

            surrogate_results[label] = {
                "observed": observed,
                "null": result,
            }

            atomic_json(
                surrogate_file,
                surrogate_results,
            )

    # ==========================================================
    # PHASE 8: SPECTROGRAMS
    # ==========================================================

    print()
    print("[PHASE 8] Time-frequency products")

    for label in [
        "WFS_PIT",
        "WFS_YAW",
        "OAF_PIT",
        "OAF_YAW",
        "LSC_POP",
        "LSC_REFL",
        "OAF_REFL",
    ]:

        print(
            f"  Spectrogram: {label}"
        )

        plot_spectrogram(
            label,
            data[label],
            metadata[label]["fs"],
        )

    # ==========================================================
    # PHASE 9: RMS / ANOMALY FIGURES
    # ==========================================================

    print()
    print("[PHASE 9] Candidate figures")

    if "0.25" in all_scan_results["WFS_PIT"]:

        plot_rms_scan(
            all_scan_results[
                "WFS_PIT"
            ]["0.25"],
            ROOT
            / "figures"
            / "wfs_pit_multiscale_scan.png",
        )

    # ==========================================================
    # PHASE 10: CANDIDATE EXTRACTION
    # ==========================================================

    print()
    print("[PHASE 10] Candidate convergence analysis")

    candidate_results = {}

    for label in data:

        candidate_results[label] = {}

        for window, rows in all_scan_results[label].items():

            ordered = sorted(
                rows,
                key=lambda r:
                    (
                        -r.get(
                            "aggregate_anomaly_score",
                            -np.inf,
                        )
                    ),
            )

            candidate_results[label][
                window
            ] = ordered[:20]

            save_csv(
                ROOT
                / "figures"
                / "candidates"
                / f"{label}_{window}s_top20.csv",
                ordered[:20],
            )

    checkpoint(
        "candidates",
        candidate_results,
    )

    # ==========================================================
    # PHASE 11: FINAL SUMMARY
    # ==========================================================

    print()
    print("[PHASE 11] Final summary")

    strongest = {}

    for label in data:

        strongest[label] = {}

        for window, rows in candidate_results[label].items():

            if rows:

                strongest[label][window] = rows[0]

    final = {
        "input": str(PATH),
        "frame_start": FRAME_START,
        "frame_end": FRAME_END,
        "episode_start": EPISODE_START,
        "episode_end": EPISODE_END,
        "channels_loaded": len(data),
        "channels": metadata,
        "window_scales": WINDOWS,
        "surrogates": SURROGATES,
        "controls": controls,
        "oaf_dependencies": dependencies,
        "surrogates_results": surrogate_results,
        "strongest_candidates": strongest,
        "runtime_seconds": (
            time.time() - start_time
        ),
    }

    atomic_json(
        ROOT
        / "statistics"
        / "FINAL_RESULTS.json",
        final,
    )

    with open(
        ROOT / "FINAL_RESEARCH_REPORT.txt",
        "w",
    ) as f:

        f.write(
            "LIGO AUXILIARY CHANNEL OVERNIGHT "
            "RESEARCH ANALYSIS\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"Input file:\n{PATH}\n\n"
        )

        f.write(
            f"Frame:\n"
            f"{FRAME_START} - {FRAME_END}\n\n"
        )

        f.write(
            f"Episode:\n"
            f"{EPISODE_START} - {EPISODE_END}\n\n"
        )

        f.write(
            f"Channels loaded: "
            f"{len(data)}/{len(CHANNELS)}\n\n"
        )

        f.write(
            f"Window scales:\n"
            f"{WINDOWS}\n\n"
        )

        f.write(
            f"Surrogates per cross-channel test: "
            f"{SURROGATES}\n\n"
        )

        f.write(
            "OAF DEPENDENCY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for name, result in dependencies.items():

            f.write(
                f"\n{name}\n"
            )

            for key, value in result.items():

                f.write(
                    f"  {key}: {value}\n"
                )

        f.write(
            "\nCROSS-CHANNEL SURROGATES\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for label, result in surrogate_results.items():

            observed = result.get(
                "observed",
                {},
            )

            null = result.get(
                "null"
            )

            f.write(
                f"\n{label}\n"
            )

            f.write(
                f"  observed r: "
                f"{observed.get('r')}\n"
            )

            f.write(
                f"  observed lag: "
                f"{observed.get('lag')}\n"
            )

            if null is None:

                f.write(
                    "  surrogate null: "
                    "not available\n"
                )

                f.write(
                    "  reason: insufficient "
                    "usable aligned data\n"
                )

            else:

                f.write(
                    f"  null median: "
                    f"{null.get('median')}\n"
                )

                f.write(
                    f"  null 95th: "
                    f"{null.get('p95')}\n"
                )

                f.write(
                    f"  null 99th: "
                    f"{null.get('p99')}\n"
                )

                f.write(
                    f"  empirical p: "
                    f"{null.get('empirical_p')}\n"
                )

        f.write(
            "\nIMPORTANT INTERPRETATION LIMITATION\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "These analyses identify statistical and "
            "signal-processing structure in auxiliary "
            "channels. They do not establish a "
            "gravitational-wave detection or a physical "
            "cause. OAF channels may contain processed "
            "versions of WFS signals and therefore are "
            "not automatically independent evidence.\n"
        )

        f.write(
            "\nRUNTIME\n"
        )

        f.write(
            f"{time.time() - start_time:.1f} seconds\n"
        )

        f.write(
            "\nOriginal GWF was read only.\n"
        )

    print()
    print("=" * 80)
    print("OVERNIGHT PIPELINE COMPLETE")
    print("=" * 80)
    print(
        f"Results: {ROOT}"
    )
    print(
        f"Report: {ROOT / 'FINAL_RESEARCH_REPORT.txt'}"
    )
    print(
        f"Runtime: "
        f"{time.time() - start_time:.1f} s"
    )
    print(
        "Original GWF was not modified."
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted. "
            "Completed checkpoints are preserved."
        )

    except Exception:

        print(
            "\nPIPELINE FAILED"
        )

        traceback.print_exc()

        print(
            "\nCompleted checkpoints have been preserved."
        )

        raise

from pathlib import Path

import pandas as pd
from scipy.stats import percentileofscore

FEATURE_FILE = Path("data/H1_O4a_baseline_features.csv")
PCA_FILE = Path("results/H1_O4a_pca_scores.csv")

features = pd.read_csv(FEATURE_FILE)
pca = pd.read_csv(PCA_FILE)

df = features.merge(
    pca[["window", "anomaly_score", "anomaly_flag"]],
    on="window",
    how="left",
)

candidate_windows = [10, 0, 7]

metrics = [
    "std",
    "peak_to_rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "power_10_30Hz",
    "power_30_100Hz",
    "power_100_300Hz",
    "power_300_1000Hz",
    "power_1000_2000Hz",
]

for window in candidate_windows:
    row = df[df["window"] == window].iloc[0]

    print()
    print("=" * 70)
    print(f"WINDOW {window}")
    print("=" * 70)
    print(f"Anomaly score: {row['anomaly_score']:.4f}")

    for metric in metrics:
        population = df[metric].dropna()
        value = row[metric]

        percentile = percentileofscore(
            population,
            value,
            kind="rank",
        )

        print(
            f"{metric:24s} "
            f"value={value:.6e} "
            f"percentile={percentile:5.1f}"
        )
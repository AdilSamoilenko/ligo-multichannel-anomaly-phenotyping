from pathlib import Path
import pandas as pd

PCA_FILE = Path("results/H1_O4a_pca_scores.csv")
FEATURE_FILE = Path("data/H1_O4a_baseline_features.csv")
OUTPUT_FILE = Path("results/H1_O4a_candidate_summary.csv")

pca = pd.read_csv(PCA_FILE)
features = pd.read_csv(FEATURE_FILE)

df = pca.merge(
    features,
    on=["window", "file"],
    how="left",
)

df = df.sort_values(
    "anomaly_score",
    ascending=False,
).reset_index(drop=True)

df["rank"] = df.index + 1

columns = [
    "rank",
    "window",
    "anomaly_score",
    "anomaly_flag",
    "std",
    "rms",
    "peak_abs",
    "peak_to_rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "power_10_30Hz",
    "power_30_100Hz",
    "power_100_300Hz",
    "power_300_1000Hz",
    "power_1000_2000Hz",
]

df[columns].to_csv(
    OUTPUT_FILE,
    index=False,
)

print("Candidate summary created.")
print(f"Windows analysed: {len(df)}")
print(f"Saved: {OUTPUT_FILE}")
print()
print(df[columns].head(10).to_string(index=False))
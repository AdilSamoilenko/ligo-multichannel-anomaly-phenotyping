from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope


FEATURE_FILE = Path("data/H1_O4a_baseline_features.csv")
RESULTS_FILE = Path("results/H1_O4a_pca_scores.csv")
FIGURE_FILE = Path("figures/H1_O4a_pca_baseline.png")

RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(FEATURE_FILE)

    identifier_columns = ["window", "file"]
    feature_columns = [
        column
        for column in df.columns
        if column not in identifier_columns
    ]

    X = df[feature_columns].copy()

    if not np.all(np.isfinite(X.to_numpy())):
        raise RuntimeError(
            "Feature matrix still contains non-finite values."
        )

    # Log-transform strictly positive power features.
    power_columns = [
        column
        for column in feature_columns
        if column == "spectral_power"
        or column.startswith("power_")
    ]

    for column in power_columns:
        X[column] = np.log10(
            np.maximum(X[column], np.finfo(float).tiny)
        )

    # Standardise every feature so large numerical scales
    # do not dominate PCA.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA.
    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_

    print("PCA complete.")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print()
    print("Explained variance:")
    
    for i, value in enumerate(explained, start=1):
        print(
            f"  PC{i}: {value:.4f} "
            f"({value * 100:.2f}%)"
        )

    print()
    print(
        "PC1 + PC2 explained variance: "
        f"{(explained[0] + explained[1]) * 100:.2f}%"
    )

    # Robust multivariate anomaly detector.
    detector = EllipticEnvelope(
        contamination=0.10,
        random_state=42,
    )

    detector.fit(scores[:, :min(5, scores.shape[1])])

    anomaly_label = detector.predict(
        scores[:, :min(5, scores.shape[1])]
    )

    anomaly_score = -detector.decision_function(
        scores[:, :min(5, scores.shape[1])]
    )

    results = df[identifier_columns].copy()

    for i in range(min(5, scores.shape[1])):
        results[f"PC{i + 1}"] = scores[:, i]

    results["anomaly_score"] = anomaly_score
    results["anomaly_flag"] = anomaly_label == -1

    results = results.sort_values(
        "anomaly_score",
        ascending=False,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("Top candidate windows:")
    print(
        results[
            [
                "window",
                "anomaly_score",
                "anomaly_flag",
            ]
        ].head(10).to_string(index=False)
    )

    # PCA visualisation.
    fig, ax = plt.subplots(figsize=(10, 7))

    normal = results["anomaly_flag"] == False
    anomalous = results["anomaly_flag"] == True

    ax.scatter(
        results.loc[normal, "PC1"],
        results.loc[normal, "PC2"],
        alpha=0.75,
        label="Not flagged",
    )

    ax.scatter(
        results.loc[anomalous, "PC1"],
        results.loc[anomalous, "PC2"],
        marker="x",
        s=80,
        label="Flagged",
    )

    for _, row in results[anomalous].iterrows():
        ax.annotate(
            str(int(row["window"])),
            (
                row["PC1"],
                row["PC2"],
            ),
        )

    ax.set_xlabel(
        f"PC1 ({explained[0] * 100:.1f}% variance)"
    )

    ax.set_ylabel(
        f"PC2 ({explained[1] * 100:.1f}% variance)"
    )

    ax.set_title(
        "H1 O4a Baseline PCA"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=200)
    plt.close(fig)

    print()
    print(f"Results: {RESULTS_FILE}")
    print(f"Figure: {FIGURE_FILE}")


if __name__ == "__main__":
    main()
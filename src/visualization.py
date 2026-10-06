"""Visualisasi hasil Elbow Method dan cluster (PCA 2D)."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def plot_elbow(k_range, wcss, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, wcss, marker="o", linestyle="--", color="b")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.xticks(k_range)
    plt.grid(True)
    _save_or_show(save_path)


def plot_pca_clusters(X, labels, save_path=None):
    """Reduksi fitur ke 2 dimensi dengan PCA lalu plot cluster."""
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(X)

    pca_df = pd.DataFrame(data=pca_features, columns=["PC1", "PC2"])
    pca_df["KMeans_Cluster"] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="PC1", y="PC2",
        hue="KMeans_Cluster",
        palette="viridis",
        data=pca_df,
        legend="full",
        alpha=0.7,
    )
    plt.title("K-Means Clustering Visualization (PCA)", fontsize=14)
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

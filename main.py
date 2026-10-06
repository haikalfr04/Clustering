"""PriceRunner product clustering pipeline using KMeans.

Usage:
    python main.py
"""
import os

from src.clustering import compute_silhouette_scores, compute_wcss, train_kmeans
from src.data_loader import explore_data, load_data
from src.preprocessing import preprocess
from src.visualization import plot_elbow, plot_pca_clusters

DATA_PATH = "data/pricerunner_aggregate.csv"
OUTPUT_PATH = "outputs/pricerunner_clustered.csv"
K_OPTIMAL = 7


def main():
    # 1. Load & explore data
    df = load_data(DATA_PATH)
    explore_data(df)

    # 2. Preprocessing
    scaled_df, _ = preprocess(df)
    print("\nScaled features shape:", scaled_df.shape)

    # 3. Determine the optimal number of clusters
    k_range, wcss = compute_wcss(scaled_df)
    plot_elbow(k_range, wcss, save_path="images/elbow_method.png")
    compute_silhouette_scores(scaled_df)

    # 4. Train KMeans with the optimal k
    _, labels = train_kmeans(scaled_df, K_OPTIMAL)
    df["KMeans_Cluster"] = labels

    # 5. Visualize clusters with PCA
    plot_pca_clusters(scaled_df, labels, save_path="images/pca_clusters.png")

    # 6. Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nClustering results saved to {OUTPUT_PATH}")
    print(df["KMeans_Cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()

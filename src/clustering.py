"""Penentuan jumlah cluster optimal dan training model KMeans."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


def compute_wcss(X, k_range=range(2, 11)):
    """Menghitung WCSS (inertia) untuk setiap k (Elbow Method)."""
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return list(k_range), wcss


def compute_silhouette_scores(X, k_range=range(2, 8), sample_size=5000):
    """Menghitung Silhouette Score pada sampel data agar hemat memori/waktu."""
    sample_indices = np.random.RandomState(RANDOM_STATE).choice(
        X.shape[0], size=sample_size, replace=False
    )
    sample_data = X.iloc[sample_indices]

    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(sample_data)
        scores[k] = silhouette_score(sample_data, labels)
        print(f"For k={k}, Silhouette Score: {scores[k]:.4f}")
    return scores


def train_kmeans(X, n_clusters):
    """Training KMeans dengan k optimal dan mengembalikan model beserta label cluster."""
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

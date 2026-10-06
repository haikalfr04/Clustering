"""Optimal cluster selection and KMeans model training."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


def compute_wcss(X, k_range=range(2, 11)):
    """Compute the WCSS (inertia) for each k (Elbow Method)."""
    wcss = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    return list(k_range), wcss


def compute_silhouette_scores(X, k_range=range(2, 8), sample_size=5000):
    """Compute Silhouette Scores on a data sample to save memory and time."""
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
    """Train KMeans with the chosen k and return the model and cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

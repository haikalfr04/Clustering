"""Preprocessing fitur untuk clustering."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Kolom identifier/granular dan ground truth yang tidak dipakai untuk training
DROP_COLUMNS = ["Product ID", "Product Title", "Cluster ID", "Cluster Label"]
CATEGORICAL_COLUMNS = ["Category Label"]


def preprocess(df):
    """Drop kolom identifier, one-hot encoding kategori, lalu standarisasi fitur.

    Returns:
        scaled_df (pd.DataFrame): fitur yang sudah di-scale.
        scaler (StandardScaler): scaler yang sudah di-fit.
    """
    features_df = df.drop(columns=DROP_COLUMNS)
    features_encoded = pd.get_dummies(
        features_df, columns=CATEGORICAL_COLUMNS, drop_first=True
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_encoded)
    scaled_df = pd.DataFrame(scaled_features, columns=features_encoded.columns)
    return scaled_df, scaler

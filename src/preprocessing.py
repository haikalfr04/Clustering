"""Feature preprocessing for clustering."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Identifier and ground-truth columns that are excluded from training
DROP_COLUMNS = ["Product ID", "Product Title", "Cluster ID", "Cluster Label"]
CATEGORICAL_COLUMNS = ["Category Label"]


def preprocess(df):
    """Drop identifier columns, one-hot encode categories, and standardize features.

    Returns:
        scaled_df (pd.DataFrame): the scaled features.
        scaler (StandardScaler): the fitted scaler.
    """
    features_df = df.drop(columns=DROP_COLUMNS)
    features_encoded = pd.get_dummies(
        features_df, columns=CATEGORICAL_COLUMNS, drop_first=True
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_encoded)
    scaled_df = pd.DataFrame(scaled_features, columns=features_encoded.columns)
    return scaled_df, scaler

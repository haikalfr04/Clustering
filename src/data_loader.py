"""Load and explore the PriceRunner dataset."""
import pandas as pd


def load_data(path):
    """Read the CSV dataset and strip whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def explore_data(df):
    """Print the data structure, data types, missing values, and descriptive statistics."""
    print("First few rows of the DataFrame:")
    print(df.head())

    print("\nDataFrame Information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive Statistics:")
    print(df.describe(include="all"))

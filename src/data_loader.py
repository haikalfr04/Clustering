"""Load dan eksplorasi awal dataset PriceRunner."""
import pandas as pd


def load_data(path):
    """Membaca dataset CSV dan merapikan nama kolom (strip whitespace)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def explore_data(df):
    """Menampilkan ringkasan struktur, tipe data, missing value, dan statistik deskriptif."""
    print("First few rows of the DataFrame:")
    print(df.head())

    print("\nDataFrame Information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive Statistics:")
    print(df.describe(include="all"))

# Product Clustering — PriceRunner Dataset

This project groups e-commerce products from the **PriceRunner Product Aggregate** dataset using **K-Means Clustering**. It covers data exploration, preprocessing, selecting the optimal number of clusters (Elbow Method and Silhouette Score), model training, and cluster visualization with PCA.

## 📁 Repository Structure

```
Clustering/
├── data/
│   └── pricerunner_aggregate.csv   # Dataset
├── images/
│   ├── elbow_method.png            # Elbow Method plot
│   └── pca_clusters.png            # Cluster visualization (2D PCA)
├── src/
│   ├── data_loader.py              # Data loading and exploration
│   ├── preprocessing.py            # Feature encoding and scaling
│   ├── clustering.py               # Elbow, Silhouette, and KMeans training
│   └── visualization.py            # Elbow and PCA plots
├── main.py                         # Main pipeline
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
git clone https://github.com/haikalfr04/Clustering.git
cd Clustering
pip install -r requirements.txt
python main.py
```

The plots are saved to the `images/` folder, and the dataset with its cluster labels is saved to `outputs/pricerunner_clustered.csv`.

## 📊 Dataset

| Description | Value |
|---|---|
| Number of rows | 35,311 |
| Number of columns | 7 |
| Missing values | 0 |
| Unique product categories | 10 |
| Unique product titles | 30,993 |
| Unique cluster labels (ground truth) | 12,849 |
| Most frequent category | Fridge Freezers (5,501 products) |

Columns: `Product ID`, `Product Title`, `Merchant ID`, `Cluster ID`, `Cluster Label`, `Category ID`, `Category Label`.

Product categories: Mobile Phones, TVs, CPUs, Digital Cameras, Microwaves, Dishwashers, Washing Machines, Freezers, Fridge Freezers, and Fridges.

## ⚙️ Methodology

### 1. Preprocessing
- Removed extra whitespace from column names.
- Checked for missing values (none were found).
- Dropped identifier columns (`Product ID`, `Product Title`) and ground-truth columns (`Cluster ID`, `Cluster Label`) to prevent *data leakage*.
- Applied *One-Hot Encoding* to `Category Label` (`drop_first=True`).
- Standardized all features with `StandardScaler`, resulting in **11 features**.

### 2. Choosing the Optimal Number of Clusters

**Elbow Method** (WCSS for k = 2–10):

![Elbow Method](images/elbow_method.png)

**Silhouette Score** (calculated on a sample of 5,000 rows):

| k | Silhouette Score |
|---|---|
| 2 | 0.2219 |
| 3 | 0.2955 |
| 4 | 0.3620 |
| 5 | 0.4385 |
| 6 | 0.4884 |
| **7** | **0.5809** |

The Silhouette Score increased steadily and reached its highest value at **k = 7**, so seven clusters were selected.

### 3. Model Training
A `KMeans(n_clusters=7, random_state=42, n_init=10)` model was trained on the full scaled dataset. The resulting cluster labels were added to the original data as the `KMeans_Cluster` column.

### 4. Cluster Visualization (PCA)

The features were reduced to two principal components (PC1 and PC2) using PCA:

![PCA Clusters](images/pca_clusters.png)

## 🔍 Results and Insights

- The dataset is clean (no missing values) and contains 35,311 products across 10 categories.
- The optimal number of clusters is **7**, with a Silhouette Score of **0.5809**, which indicates a fairly strong cluster structure.
- The PCA plot shows seven clearly separated clusters in two-dimensional space.
- Since most features come from the one-hot encoded `Category Label`, the clusters mainly group products by category and `Merchant ID`.

## 📌 Next Steps

- **Cluster profiling:** group the data by `KMeans_Cluster` to examine the category and merchant composition of each cluster.
- **Evaluation against ground truth:** compare the KMeans results with `Cluster ID` / `Category Label` using metrics such as the *Adjusted Rand Index (ARI)* or *Normalized Mutual Information (NMI)*.
- **Wider range of k:** the Silhouette Score was still increasing at k = 7, so values of k greater than 7 should also be tested.
- **Text features:** use `Product Title` (for example, with TF-IDF) to make the clustering more informative.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

## 📚 Data Source

[PriceRunner Product Classification and Clustering — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering)

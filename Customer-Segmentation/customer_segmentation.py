"""Thiranex Internship - Customer Segmentation Project
Run: python customer_segmentation.py
Dependencies: pandas, numpy, scikit-learn
This script trains K-Means on a synthetic dataset included in customer_segmentation_dataset.csv.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("customer_segmentation_dataset.csv")
features = ["Age", "Annual Income (₹000s)", "Spending Score (1-100)",
            "Purchase Frequency (orders/year)", "Average Order Value (₹)"]
X = df[features].fillna(df[features].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)
score = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette score: {score:.3f}")
summary = df.groupby("Cluster").agg(
    Customers=("Customer ID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (₹000s)", "mean"),
    Avg_Spending_Score=("Spending Score (1-100)", "mean"),
    Avg_Frequency=("Purchase Frequency (orders/year)", "mean")
)
print(summary)
df.to_csv("customer_segments_output.csv", index=False)
print("Saved customer_segments_output.csv")

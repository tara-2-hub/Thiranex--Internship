import pandas as pd
# Task 4: automated data cleaning and reporting
raw = pd.read_csv("raw_sales_data_messy.csv")
rows_before = len(raw)
duplicates_removed = int(raw.duplicated().sum())
df = raw.drop_duplicates().copy()
missing_before = int(df.isna().sum().sum())
for col in ["Customer Name", "Product"]:
    df[col] = df[col].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
df["Region"] = df["Region"].astype("string").str.strip().str.title().replace("", pd.NA).fillna("Unknown")
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Unit Price (₹)"] = pd.to_numeric(df["Unit Price (₹)"], errors="coerce")
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median()).round().astype(int)
df["Unit Price (₹)"] = df["Unit Price (₹)"].fillna(df["Unit Price (₹)"].median()).round(2)
df["Order Date"] = df["Order Date"].fillna(df["Order Date"].dropna().median())
df["Revenue (₹)"] = (df["Quantity"] * df["Unit Price (₹)"]).round(2)
df.to_csv("cleaned_sales_data.csv", index=False, date_format="%Y-%m-%d")
region = df.groupby("Region")["Revenue (₹)"].sum().sort_values(ascending=False)
product = df.groupby("Product")["Revenue (₹)"].sum().sort_values(ascending=False)
with open("automated_data_quality_report.txt","w",encoding="utf-8") as f:
 f.write(f"Rows before cleaning: {rows_before}\nDuplicates removed: {duplicates_removed}\nRows after cleaning: {len(df)}\nMissing before imputation: {missing_before}\nMissing after cleaning: {int(df.isna().sum().sum())}\nTotal revenue: ₹{df['Revenue (₹)'].sum():,.2f}\nTop region: {region.index[0]}\nTop product: {product.index[0]}\n")
print("Cleaned data and report generated.")

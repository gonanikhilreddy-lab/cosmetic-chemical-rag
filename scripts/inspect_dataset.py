import pandas as pd

from src.config.settings import CSV_PATH


df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("\nCOLUMN TYPES")
print("-" * 60)
print(df.dtypes)
print("\nMISSING VALUES")
print("-" * 60)
print(df.isna().sum())
print("\nDUPLICATE ROWS")
print("-" * 60)
print(f"Duplicate rows: {df.duplicated().sum():,}")
print("\nSAMPLE DATA")
print("-" * 60)
print(df.head(5).to_string())

print("\nUNIQUE COUNTS")
print("-" * 60)
for column in (
    "CDPHId", "ProductName", "CompanyId", "CompanyName", "BrandName",
    "PrimaryCategory", "SubCategory", "CasNumber", "ChemicalId", "ChemicalName",
):
    print(f"{column}: {df[column].nunique(dropna=True):,}")
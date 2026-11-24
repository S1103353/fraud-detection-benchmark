from pathlib import Path
import pandas as pd

# ============================================================
# Missing Value Summary for Raw Fraud Detection Datasets
# This script calculates missing-value statistics BEFORE preprocessing.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "tables" / "step21"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Raw/original dataset filenames
datasets = {
    "Bank Fraud Base": "Base.csv",
    "Credit Card Fraud": "creditcard.csv",
    "PaySim": "PS_20174392719_1491204439457_log.csv",
    "IEEE-CIS Identity": "train_identity.csv",
    "E-Commerce Fraud": "ecommerce.csv",
    "Fraud Transactions": "fraud_transactions_train.csv",
}


def find_file(filename: str):
    """Search for a file anywhere inside the project folder."""
    matches = list(PROJECT_ROOT.rglob(filename))
    if not matches:
        return None
    return matches[0]


def read_dataset(path: Path):
    """Read CSV or Parquet files safely."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    else:
        return pd.read_csv(path, low_memory=False)


summary = []

for dataset_name, filename in datasets.items():
    path = find_file(filename)

    if path is None:
        print(f"File not found: {filename}")
        continue

    print(f"Reading: {dataset_name} -> {path}")

    df = read_dataset(path)

    missing_rate = df.isna().mean() * 100

    total_features = df.shape[1]
    total_rows = df.shape[0]
    features_with_missing = int((missing_rate > 0).sum())
    highest_missing_rate = round(float(missing_rate.max()), 2)

    if features_with_missing > 0:
        average_missing_rate = round(float(missing_rate[missing_rate > 0].mean()), 2)
    else:
        average_missing_rate = 0.00

    total_missing_values = int(df.isna().sum().sum())

    summary.append({
        "Dataset": dataset_name,
        "Total Rows": total_rows,
        "Total Features": total_features,
        "Features with Missing Values": features_with_missing,
        "Total Missing Values": total_missing_values,
        "Highest Missing Rate (%)": highest_missing_rate,
        "Average Missing Rate (%)": average_missing_rate
    })


summary_df = pd.DataFrame(summary)

print("\nMissing Value Summary for Raw Datasets:")
print(summary_df.to_string(index=False))

csv_path = OUTPUT_DIR / "raw_missing_value_summary.csv"
xlsx_path = OUTPUT_DIR / "raw_missing_value_summary.xlsx"

summary_df.to_csv(csv_path, index=False)
summary_df.to_excel(xlsx_path, index=False)

print(f"\nSaved CSV to: {csv_path}")
print(f"Saved Excel to: {xlsx_path}")
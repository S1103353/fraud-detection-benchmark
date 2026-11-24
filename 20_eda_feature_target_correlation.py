# ============================================================
# Step 20 - EDA: Feature-Target Correlation Analysis
# Master Thesis: Credit Card Fraud Detection Using Machine Learning
# Author: Narges Maleki
# ============================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Data Sets"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"

TABLE_DIR = OUTPUT_DIR / "tables" / "step20_eda_feature_target_correlation"
FIGURE_DIR = OUTPUT_DIR / "figures" / "step20_eda_feature_target_correlation"
REPORT_DIR = OUTPUT_DIR / "reports" / "step20_eda_feature_target_correlation"

for folder in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Dataset configuration
# ------------------------------------------------------------
DATASETS = {
    "bank_fraud_base": {
        "path": DATA_DIR / "Bank Fraud" / "Base.csv",
        "target": "fraud_bool"
    },
    "bank_fraud_variant_i": {
        "path": DATA_DIR / "Bank Fraud" / "Variant I.csv",
        "target": "fraud_bool"
    },
    "bank_fraud_variant_ii": {
        "path": DATA_DIR / "Bank Fraud" / "Variant II.csv",
        "target": "fraud_bool"
    },
    "bank_fraud_variant_iii": {
        "path": DATA_DIR / "Bank Fraud" / "Variant III.csv",
        "target": "fraud_bool"
    },
    "bank_fraud_variant_iv": {
        "path": DATA_DIR / "Bank Fraud" / "Variant IV.csv",
        "target": "fraud_bool"
    },
    "bank_fraud_variant_v": {
        "path": DATA_DIR / "Bank Fraud" / "Variant V.csv",
        "target": "fraud_bool"
    },
    "creditcard": {
        "path": DATA_DIR / "Credit Card" / "creditcard.csv",
        "target": "Class"
    },
    "paysim": {
        "path": DATA_DIR / "PaySim" / "PS_20174392719_1491204439457_log.csv",
        "target": "isFraud"
    },
    "fraud_transactions_train": {
        "path": DATA_DIR / "Fraud Transactions" / "fraudTrain.csv",
        "target": "is_fraud"
    },
    "fraud_transactions_test": {
        "path": DATA_DIR / "Fraud Transactions" / "fraudTest.csv",
        "target": "is_fraud"
    },
    "ecommerce": {
        "path": DATA_DIR / "Ecommerce" / "transactions.csv",
        "target": "isFraud"
    },
    "ieee_transaction": {
        "path": DATA_DIR / "IEEE Fraud" / "train_transaction.csv",
        "target": "isFraud"
    },
    "elliptic": {
        "features_path": DATA_DIR / "Elliptic" / "elliptic_txs_features.csv",
        "classes_path": DATA_DIR / "Elliptic" / "elliptic_txs_classes.csv",
        "target": "class"
    }
}


# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df


def infer_target(df: pd.DataFrame, preferred_target: str):
    if preferred_target in df.columns:
        return preferred_target

    possible_targets = [
        "fraud_bool", "isFraud", "is_fraud", "Class", "class",
        "label", "target", "fraud"
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    return None


def prepare_dataframe(df: pd.DataFrame, target_col: str, max_rows: int = 200000):
    df = clean_feature_names(df)

    target_col = infer_target(df, target_col)

    if target_col is None:
        raise ValueError("Target column could not be identified.")

    df = df.dropna(subset=[target_col])

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    y = df[target_col]

    if y.dtype == "object":
        y = y.astype(str)
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=df.index)

    X = df.drop(columns=[target_col])

    # Convert categorical columns to numeric labels for MI
    X_encoded = X.copy()

    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object":
            X_encoded[col] = X_encoded[col].astype(str).fillna("missing")
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])

    # Keep only numeric columns
    X_encoded = X_encoded.select_dtypes(include=[np.number])

    # Remove constant columns
    nunique = X_encoded.nunique(dropna=False)
    X_encoded = X_encoded.loc[:, nunique > 1]

    # Fill missing values
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))

    return X_encoded, y, target_col


def compute_correlation_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    results = []

    for col in X.columns:
        try:
            corr = np.corrcoef(X[col], y)[0, 1]
            if np.isnan(corr):
                corr = 0
        except Exception:
            corr = 0

        results.append({
            "Feature": col,
            "Pearson_Correlation": corr,
            "Absolute_Correlation": abs(corr)
        })

    corr_df = pd.DataFrame(results)
    corr_df = corr_df.sort_values("Absolute_Correlation", ascending=False)

    return corr_df


def compute_mutual_information_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    try:
        mi_scores = mutual_info_classif(
            X,
            y,
            discrete_features="auto",
            random_state=42
        )

        mi_df = pd.DataFrame({
            "Feature": X.columns,
            "Mutual_Information": mi_scores
        }).sort_values("Mutual_Information", ascending=False)

        return mi_df

    except Exception as e:
        print(f"Mutual information failed: {e}")
        return pd.DataFrame(columns=["Feature", "Mutual_Information"])


def plot_top_features(df, value_col, title, output_path, top_n=20):
    if df.empty:
        return

    plot_df = df.head(top_n).copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["Feature"], plot_df[value_col])
    plt.title(title)
    plt.xlabel(value_col.replace("_", " "))
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_target_correlation_heatmap(X, y, corr_df, dataset_name, output_path, top_n=10):
    if corr_df.empty:
        return

    top_features = corr_df.head(top_n)["Feature"].tolist()

    heatmap_df = X[top_features].copy()
    heatmap_df["target"] = y.values

    corr_matrix = heatmap_df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
    plt.title(f"Correlation Heatmap: {dataset_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# 4. Elliptic special loader
# ------------------------------------------------------------
def load_elliptic_dataset(config):
    features_path = config["features_path"]
    classes_path = config["classes_path"]

    features = safe_read_csv(features_path)
    classes = safe_read_csv(classes_path)

    features = clean_feature_names(features)
    classes = clean_feature_names(classes)

    # Try to identify transaction ID columns
    feature_id_col = features.columns[0]
    class_id_col = classes.columns[0]

    df = features.merge(
        classes,
        left_on=feature_id_col,
        right_on=class_id_col,
        how="inner"
    )

    # Elliptic class may include "unknown"; remove it
    target_col = config["target"]
    target_col = infer_target(df, target_col)

    if target_col is None:
        raise ValueError("Elliptic target column not found.")

    df = df[df[target_col].astype(str).str.lower() != "unknown"]

    return df


# ------------------------------------------------------------
# 5. Run analysis for all datasets
# ------------------------------------------------------------
summary_rows = []

for dataset_name, config in DATASETS.items():
    print(f"\n==============================")
    print(f"Processing dataset: {dataset_name}")
    print(f"==============================")

    try:
        if dataset_name == "elliptic":
            df = load_elliptic_dataset(config)
        else:
            df = safe_read_csv(config["path"])

        X, y, target_col = prepare_dataframe(df, config["target"])

        print(f"Rows used: {len(X)}")
        print(f"Features used: {X.shape[1]}")
        print(f"Target column: {target_col}")

        if X.empty:
            print(f"Skipping {dataset_name}: no valid numeric features.")
            continue

        corr_df = compute_correlation_scores(X, y)
        mi_df = compute_mutual_information_scores(X, y)

        corr_output = TABLE_DIR / f"{dataset_name}_feature_target_correlation.csv"
        mi_output = TABLE_DIR / f"{dataset_name}_mutual_information.csv"

        corr_df.to_csv(corr_output, index=False)
        mi_df.to_csv(mi_output, index=False)

        plot_top_features(
            corr_df,
            value_col="Absolute_Correlation",
            title=f"Top Feature-Target Correlations: {dataset_name}",
            output_path=FIGURE_DIR / f"{dataset_name}_top_correlation_features.png",
            top_n=20
        )

        plot_top_features(
            mi_df,
            value_col="Mutual_Information",
            title=f"Top Mutual Information Features: {dataset_name}",
            output_path=FIGURE_DIR / f"{dataset_name}_top_mutual_information_features.png",
            top_n=20
        )

        plot_target_correlation_heatmap(
            X,
            y,
            corr_df,
            dataset_name,
            FIGURE_DIR / f"{dataset_name}_correlation_heatmap.png",
            top_n=10
        )

        top_corr_feature = corr_df.iloc[0]["Feature"] if not corr_df.empty else None
        top_corr_value = corr_df.iloc[0]["Absolute_Correlation"] if not corr_df.empty else None

        top_mi_feature = mi_df.iloc[0]["Feature"] if not mi_df.empty else None
        top_mi_value = mi_df.iloc[0]["Mutual_Information"] if not mi_df.empty else None

        summary_rows.append({
            "Dataset": dataset_name,
            "Rows_Used": len(X),
            "Number_of_Features": X.shape[1],
            "Target": target_col,
            "Top_Correlation_Feature": top_corr_feature,
            "Top_Absolute_Correlation": top_corr_value,
            "Top_MI_Feature": top_mi_feature,
            "Top_MI_Score": top_mi_value
        })

        print(f"Completed: {dataset_name}")

    except Exception as e:
        print(f"Failed for {dataset_name}: {e}")

        summary_rows.append({
            "Dataset": dataset_name,
            "Rows_Used": None,
            "Number_of_Features": None,
            "Target": config.get("target", None),
            "Top_Correlation_Feature": None,
            "Top_Absolute_Correlation": None,
            "Top_MI_Feature": None,
            "Top_MI_Score": None,
            "Error": str(e)
        })


# ------------------------------------------------------------
# 6. Save overall summary
# ------------------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLE_DIR / "all_datasets_feature_target_summary.csv", index=False)

with open(REPORT_DIR / "step20_eda_feature_target_correlation_summary.txt", "w") as f:
    f.write("Step 20 - EDA Feature-Target Correlation Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(
        "This analysis evaluates the relationship between the fraud label "
        "and input features across all datasets using Pearson correlation "
        "and mutual information. Pearson correlation captures linear "
        "relationships, while mutual information captures both linear and "
        "non-linear dependencies.\n\n"
    )
    f.write(summary_df.to_string(index=False))

print("\nAll datasets processed.")
print(f"Tables saved to: {TABLE_DIR}")
print(f"Figures saved to: {FIGURE_DIR}")
print(f"Report saved to: {REPORT_DIR}")
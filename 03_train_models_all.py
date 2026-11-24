"""
Step 03: Train Baseline Models (FULL DATA, NO SAMPLING)

What this script does
---------------------
This step trains a small set of baseline ML classifiers on ALL cleaned datasets produced by Step 02.
It is designed to be:
- Robust across different scikit-learn versions (fixes OneHotEncoder sparse/sparse_output breaking change)
- Safe for large datasets (uses sparse one-hot, avoids unnecessary copies)
- Reproducible (fixed random seed)
- Consistent output format across multiple datasets

Expected inputs (from Step 02 outputs in ./Outputs)
---------------------------------------------------
- bank_fraud_base_clean.csv
- creditcard_clean.csv
- ecommerce_clean.csv
- fraud_transactions_train_clean.csv
- elliptic_features_clean.csv (optional / no target)
- elliptic_classes_clean.csv  (optional / has labels but may require join)
- elliptic_edgelist_clean.csv (optional / graph edges, not used here)
- ieee_train_clean.parquet
- paysim_clean.csv.gz

Important note about Elliptic
-----------------------------
Elliptic is a graph dataset. Step 02 may save components separately if it cannot merge.
This Step 03 focuses on tabular supervised datasets with a clear "target" column.
If Elliptic is not merged into a single table with a "target", we skip it and log a note.

Outputs (saved under ./Outputs)
-------------------------------
- step03_results.csv           : metrics table for all datasets/models
- Models saved under ./Models  : trained pipelines (joblib)
- Optional confusion matrices  : saved as small CSV per run

How to run
----------
python3 03_train_models_all.py
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# joblib is the recommended serializer for sklearn pipelines
import joblib


# -----------------------------
# Basic pretty-print utilities
# -----------------------------
def divider(title: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def info(msg: str) -> None:
    print(f"ℹ️  {msg}")


def warn(msg: str) -> None:
    print(f"⚠️  {msg}")


# -----------------------------
# Paths / configuration
# -----------------------------
# ROOT = Path.home() / Downloads / "Fraud Thesis"
from SRC.utils.paths import get_project_root

ROOT = get_project_root()
OUTPUTS_DIR = ROOT / "Outputs"
MODELS_DIR = ROOT / "Models"
RANDOM_STATE = 42

@dataclass
class DatasetSpec:
    name: str
    path: Path
    kind: str  # "csv", "csv.gz", "parquet"


def resolve_datasets() -> List[DatasetSpec]:
    """
    Map known Step 02 outputs to dataset specs.
    We only include datasets that exist on disk.
    """
    candidates = [
        DatasetSpec("bank_fraud", OUTPUTS_DIR / "bank_fraud_base_clean.csv", "csv"),
        DatasetSpec("creditcard", OUTPUTS_DIR / "creditcard_clean.csv", "csv"),
        DatasetSpec("ecommerce", OUTPUTS_DIR / "ecommerce_clean.csv", "csv"),
        DatasetSpec("fraud_transactions", OUTPUTS_DIR / "fraud_transactions_train_clean.csv", "csv"),
        DatasetSpec("ieee", OUTPUTS_DIR / "ieee_train_clean.parquet", "parquet"),
        DatasetSpec("paysim", OUTPUTS_DIR / "paysim_clean.csv.gz", "csv.gz"),
        # Elliptic components exist but may not be merged into a single supervised table:
        DatasetSpec("elliptic_features", OUTPUTS_DIR / "elliptic_features_clean.csv", "csv"),
        DatasetSpec("elliptic_classes", OUTPUTS_DIR / "elliptic_classes_clean.csv", "csv"),
    ]

    found = []
    for ds in candidates:
        if ds.path.exists():
            found.append(ds)

    return found


def approx_free_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return float("nan")


# -----------------------------
# Robust OneHotEncoder creation
# -----------------------------
def make_ohe(handle_unknown: str = "ignore"):
    """
    scikit-learn changed OneHotEncoder:
      - Old versions: OneHotEncoder(..., sparse=True/False)
      - Newer versions: OneHotEncoder(..., sparse_output=True/False)
    This helper works across both.
    """
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=True)


# -----------------------------
# Loading utilities
# -----------------------------
def load_dataset(ds: DatasetSpec) -> pd.DataFrame:
    if ds.kind == "csv":
        return pd.read_csv(ds.path)
    if ds.kind == "csv.gz":
        return pd.read_csv(ds.path, compression="gzip")
    if ds.kind == "parquet":
        return pd.read_parquet(ds.path)
    raise ValueError(f"Unknown dataset kind: {ds.kind}")


def has_target(df: pd.DataFrame) -> bool:
    return "target" in df.columns


def ensure_binary_target(y: pd.Series) -> pd.Series:
    """
    Make sure target is 0/1 numeric.
    If already numeric-like, keep it.
    If boolean, cast to int.
    """
    if y.dtype == bool:
        return y.astype(int)

    # Convert strings like "0"/"1" safely if needed
    if y.dtype == object:
        y2 = y.astype(str).str.strip()
        # Try to map common binary strings
        mapping = {"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1}
        lower = y2.str.lower()
        if set(lower.unique()).issubset(set(mapping.keys())):
            return lower.map(mapping).astype(int)

    # Fall back: try numeric conversion
    return pd.to_numeric(y, errors="coerce").fillna(0).astype(int)


# -----------------------------
# Preprocessing & models
# -----------------------------
def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Determine categorical vs numeric columns.
    We treat:
    - object/category/bool as categorical
    - everything else numeric
    """
    cat_cols = [c for c in df.columns if df[c].dtype.name in ("object", "category", "bool")]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor_standard(X: pd.DataFrame) -> ColumnTransformer:
    """
    Standard preprocessing:
    - Numeric: median impute + standardize
    - Categorical: most_frequent impute + one-hot (sparse)
    """
    num_cols, cat_cols = split_columns(X)

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # with_mean=False is safer when combined with sparse parts
        ]
    )

    ohe = make_ohe(handle_unknown="ignore")
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,  # encourages sparse output when helpful
    )
    return pre


def get_models() -> Dict[str, object]:
    """
    Baseline models that can work with sparse input.
    (RandomForest typically does NOT accept sparse matrices.)
    """
    models = {}

    # Logistic Regression is a solid baseline; saga handles large sparse features.
    models["logreg_saga"] = LogisticRegression(
        solver="saga",
        max_iter=2000,
        n_jobs=-2, # only 'saga' support parallel computation
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # Another simple baseline: logistic with liblinear (good for smaller sets)
    models["logreg_liblinear"] = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return models


# -----------------------------
# Metrics
# -----------------------------
def safe_roc_auc(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> Optional[float]:
    if y_score is None:
        return None
    # roc_auc requires both classes present
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None



def safe_ap(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> Optional[float]:
    if y_score is None:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


def get_scores(model, X_test):
    """
    Try to get probability scores. If not available, fall back to decision_function.
    """
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            return proba
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            return scores
        except Exception:
            pass
    return None


# -----------------------------
# Training loop
# -----------------------------
def train_models_for_dataset(ds_name: str, df: pd.DataFrame) -> List[Dict]:
    """
    Train all baseline models for one dataset and return a list of metric rows.
    """
    results = []

    if not has_target(df):
        warn(f"{ds_name}: no 'target' column found. Skipping.")
        return results

    # Basic cleanup
    df = df.dropna(axis=0, subset=["target"]).copy()
    y = ensure_binary_target(df["target"])
    X = df.drop(columns=["target"])

    # If target has only one class, training metrics are meaningless
    if len(np.unique(y.values)) < 2:
        warn(f"{ds_name}: target has <2 classes after cleaning. Skipping.")
        return results

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y.values,
    )

    pre = make_preprocessor_standard(X_train)
    models = get_models()

    for model_name, clf in models.items():
        info(f"{ds_name}: training {model_name} ...")

        pipe = Pipeline(
            steps=[
                ("preprocess", pre),
                ("model", clf),
            ]
        )

        # Train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        # Predict
        y_pred = pipe.predict(X_test)
        y_score = get_scores(pipe, X_test)

        # Metrics
        row = {
            "dataset": ds_name,
            "model": model_name,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "pos_rate_train": float(np.mean(y_train)),
            "pos_rate_test": float(np.mean(y_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": safe_roc_auc(y_test, y_score),
            "avg_precision": safe_ap(y_test, y_score),
        }
        results.append(row)

        # Save confusion matrix (small CSV)
        cm = confusion_matrix(y_test, y_pred)
        cm_path = OUTPUTS_DIR / f"step03_cm_{ds_name}_{model_name}.csv"
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(cm_path, index=True)

        # Save model pipeline
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{ds_name}_{model_name}.joblib"
        joblib.dump(pipe, model_path)

        ok(f"{ds_name}: saved model -> {model_path.name} | cm -> {cm_path.name}")

    return results


def try_build_elliptic_supervised(
    features_path: Path, classes_path: Path
) -> Optional[pd.DataFrame]:
    """
    Attempt to build a supervised Elliptic table:
    - features file usually has 'txId' (or sometimes 'txId' / 'txId ' etc.)
    - classes file typically has txId + class label
    If join key is missing, return None.
    """
    try:
        feats = pd.read_csv(features_path)
        cls = pd.read_csv(classes_path)
    except Exception:
        return None

    # Normalize column names (strip spaces)
    feats.columns = [c.strip() for c in feats.columns]
    cls.columns = [c.strip() for c in cls.columns]

    # Find likely join key
    key_candidates = ["txId", "txid", "transaction_id", "TransactionID"]
    key = None
    for k in key_candidates:
        if k in feats.columns and k in cls.columns:
            key = k
            break

    if key is None:
        return None

    # Find label column
    label_candidates = ["class", "Class", "label", "Label"]
    label_col = None
    for lc in label_candidates:
        if lc in cls.columns:
            label_col = lc
            break

    if label_col is None:
        return None

    merged = feats.merge(cls[[key, label_col]], on=key, how="inner")

    # Map to binary if possible: elliptic classes might be {1=illicit, 2=licit, 3=unknown}
    y_raw = merged[label_col]
    # Keep only rows where label is known binary-ish
    # Common case: 'unknown' exists; we drop it.
    if y_raw.dtype == object:
        lower = y_raw.astype(str).str.lower().str.strip()
        keep = lower.isin(["0", "1", "illicit", "licit"])
        merged = merged.loc[keep].copy()
        lower = merged[label_col].astype(str).str.lower().str.strip()
        merged["target"] = lower.map({"0": 0, "1": 1, "licit": 0, "illicit": 1}).astype(int)
    else:
        # numeric case (often 1/2/3)
        merged = merged.loc[y_raw.isin([1, 2])].copy()
        merged["target"] = (merged[label_col] == 1).astype(int)

    # Drop original label col
    merged = merged.drop(columns=[label_col])
    return merged


def main() -> None:
    divider("Step 03: Train Baseline Models (FULL DATA, NO SAMPLING)")

    info(f"Root: {ROOT}")
    info(f"Outputs: {OUTPUTS_DIR}")
    info(f"Models:  {MODELS_DIR}")
    info(f"Free disk space (approx): {approx_free_gb(ROOT):.2f} GB")
    info("This run uses FULL datasets. It may take a long time on PaySim/IEEE.")

    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Outputs folder not found: {OUTPUTS_DIR} (run Step 02 first)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = resolve_datasets()
    if not datasets:
        warn("No Step 02 outputs found. Please run Step 02 first.")
        return

    # Build a clean supervised elliptic table if possible (optional)
    elliptic_feats = OUTPUTS_DIR / "elliptic_features_clean.csv"
    elliptic_cls = OUTPUTS_DIR / "elliptic_classes_clean.csv"
    elliptic_supervised = None
    if elliptic_feats.exists() and elliptic_cls.exists():
        info("Elliptic: attempting to build supervised table (features + classes) ...")
        elliptic_supervised = try_build_elliptic_supervised(elliptic_feats, elliptic_cls)
        if elliptic_supervised is not None and "target" in elliptic_supervised.columns:
            ok(f"Elliptic: supervised table built. shape={elliptic_supervised.shape}")
        else:
            warn("Elliptic: could not build supervised table. Will skip Elliptic training.")

    all_results: List[Dict] = []

    # Train for each dataset that has a target
    for ds in datasets:
        # Skip elliptic components (we handle merged supervised version separately)
        if ds.name in ("elliptic_features", "elliptic_classes"):
            continue

        divider(f"Loading dataset: {ds.name}")
        info(f"File: {ds.path}")

        df = load_dataset(ds)
        ok(f"Loaded {ds.name}: rows={len(df)} cols={df.shape[1]}")

        all_results.extend(train_models_for_dataset(ds.name, df))

    # Train Elliptic if we built it successfully
    if elliptic_supervised is not None and "target" in elliptic_supervised.columns:
        divider("Training dataset: elliptic (supervised)")
        all_results.extend(train_models_for_dataset("elliptic", elliptic_supervised))

    # Save results
    if not all_results:
        warn("No results produced. Check Step 02 outputs and whether 'target' exists.")
        return

    results_df = pd.DataFrame(all_results)
    results_path = OUTPUTS_DIR / "step03_results.csv" # change destination path
    results_df.to_csv(results_path, index=False)

    # Save a JSON summary too (handy for quick inspection)
    summary = {
        "n_rows": int(results_df.shape[0]),
        "datasets": sorted(results_df["dataset"].unique().tolist()),
        "models": sorted(results_df["model"].unique().tolist()),
    }
    (OUTPUTS_DIR / "step03_summary.json").write_text(json.dumps(summary, indent=2))

    divider("Step 03 Summary")
    ok(f"Saved results -> {results_path}")
    info(f"Datasets trained: {summary['datasets']}")
    info(f"Models trained:   {summary['models']}")
    ok("Step 03 completed successfully.")


if __name__ == "__main__":
    main()
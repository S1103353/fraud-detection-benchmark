"""
04_evaluate_models.py
============================================================
Fraud Thesis - Step 04: Evaluate Trained Models (FULL DATA, NO SAMPLING)

What this script does
---------------------
This script evaluates all trained models produced in Step 03 across all available datasets.
It reads:
  - Cleaned datasets from:   <ROOT>/Outputs/*.csv or *.parquet
  - Trained models from:     <ROOT>/Models/*.joblib
  - (Optionally) confusion matrices already saved in Step 03 from: <ROOT>/Outputs/cm_*.csv

It then produces:
  1) A single consolidated evaluation table:
        <ROOT>/Outputs/step04_results.csv
     containing per-dataset & per-model metrics:
        - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
        - Confusion matrix components (TN, FP, FN, TP)
        - Fit/eval timing metadata where possible

  2) A JSON summary:
        <ROOT>/Outputs/step04_summary.json

  3) Publication-ready plots:
        <ROOT>/Outputs/figures/
        - ROC curves (per dataset; one figure per dataset)
        - Precision-Recall curves (per dataset; one figure per dataset)

  4) Publication-ready tables:
        <ROOT>/Outputs/tables/
        - Best model per dataset (by F1 and PR-AUC)
        - Full comparison table (CSV)

Important notes
---------------
- FULL DATA: No sampling is performed. This may take time for large datasets (PaySim/IEEE).
- Memory: ROC/PR curves require probability scores. For huge datasets, this script
  computes curves efficiently, but still needs to hold y_true and scores in memory.
  If you run into memory pressure, we can add chunked curve computation later.
- Target column: expects standardized target column name "target" (as produced in Step 02).
- This script is model-agnostic but assumes sklearn-compatible estimators saved via joblib.

How to run
----------
From terminal (recommended):
  python -u "<repo>/04_evaluate_models.py"

Or via your pipeline runner (Main.py) after enabling Step 04.

Author
------
Narges Maleki - Fraud Thesis Pipeline
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROOT = Path.home() / "Downloads" / "Fraud Thesis"
OUTPUTS_DIR = ROOT / "Outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = ROOT / "Models"

TARGET_COL = "target"

# Datasets produced by Step 02 (expected filenames in Outputs)
DATASET_FILES: Dict[str, str] = {
    "bank_fraud": "bank_fraud_base_clean.csv",
    "creditcard": "creditcard_clean.csv",
    "ecommerce": "ecommerce_clean.csv",
    "fraud_transactions": "fraud_transactions_train_clean.csv",
    "ieee": "ieee_train_clean.parquet",
    "paysim": "paysim_clean.csv.gz",
    # Elliptic is intentionally excluded here because Step 02 saved components separately
    # and Step 03 may skip elliptic supervised training if it cannot build a labeled table.
}

# Model filename pattern: <dataset>_<modelname>.joblib
# Example: bank_fraud_logreg_saga.joblib
SUPPORTED_MODEL_SUFFIXES = [
    "logreg_saga",
    "logreg_liblinear",
]


# -----------------------------------------------------------------------------
# Utility printing helpers
# -----------------------------------------------------------------------------
def divider(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def info(msg: str) -> None:
    print(f"[i] {msg}")


def ok(msg: str) -> None:
    print(f"[✓] {msg}")


def warn(msg: str) -> None:
    print(f"[!] {msg}")


# -----------------------------------------------------------------------------
# Core IO helpers
# -----------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create required output directories for Step 04."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ok(f"Figures dir ready: {FIGURES_DIR}")
    ok(f"Tables dir ready:  {TABLES_DIR}")


def load_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """
    Load a cleaned dataset produced by Step 02.
    Supports .csv, .csv.gz, and .parquet.
    """
    if dataset_name not in DATASET_FILES:
        warn(f"Dataset '{dataset_name}' is not configured.")
        return None

    path = OUTPUTS_DIR / DATASET_FILES[dataset_name]
    if not path.exists():
        warn(f"Missing dataset file: {path}")
        return None

    info(f"Loading dataset: {dataset_name}")
    info(f"File: {path}")

    t0 = time.time()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffixes[-2:] == [".csv", ".gz"]:
        df = pd.read_csv(path, compression="gzip")
    else:
        # fallback
        df = pd.read_csv(path)

    dt = time.time() - t0
    ok(f"Loaded {dataset_name}: rows={len(df)} cols={df.shape[1]} in {dt:.2f}s")

    # Basic checks
    if TARGET_COL not in df.columns:
        warn(f"Dataset '{dataset_name}' has no '{TARGET_COL}' column. Skipping.")
        return None

    return df


def find_models_for_dataset(dataset_name: str) -> List[Path]:
    """
    Find all model files for a given dataset in MODELS_DIR.
    Expected naming: <dataset>_<model_suffix>.joblib
    """
    models = []
    for suffix in SUPPORTED_MODEL_SUFFIXES:
        p = MODELS_DIR / f"{dataset_name}_{suffix}.joblib"
        if p.exists():
            models.append(p)
    return models


def load_model(path: Path):
    """Load a joblib model."""
    info(f"Loading model: {path.name}")
    return joblib.load(path)


# -----------------------------------------------------------------------------
# Metrics + curve computation
# -----------------------------------------------------------------------------
@dataclass
class EvalResult:
    dataset: str
    model: str
    n_rows: int
    n_features: int
    pos_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    tn: int
    fp: int
    fn: int
    tp: int
    eval_seconds: float


def safe_predict_proba(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Return probability for the positive class if available.
    If predict_proba is not available, try decision_function and map to [0,1] using sigmoid.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba is None:
            return None
        # assume binary classification => class 1 is positive
        if proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores).ravel()
        # sigmoid mapping (monotonic); suitable for curves when proba is unavailable
        return 1.0 / (1.0 + np.exp(-scores))

    return None


def evaluate_one_model(
    dataset_name: str,
    df: pd.DataFrame,
    model_path: Path,
) -> EvalResult:
    """
    Evaluate a single trained model on a dataset.
    Assumes df already contains the "target" column.
    """
    model_name = model_path.stem.replace(f"{dataset_name}_", "")

    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    # Track class balance
    pos_rate = float(np.mean(y))

    model = load_model(model_path)

    t0 = time.time()
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    # Core metrics
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    # Probabilistic metrics (ROC-AUC, PR-AUC)
    y_score = safe_predict_proba(model, X)
    roc_auc = None
    pr_auc = None
    if y_score is not None:
        # Guard: ROC-AUC requires both classes present
        if len(np.unique(y)) == 2:
            roc_auc = float(roc_auc_score(y, y_score))
        pr_auc = float(average_precision_score(y, y_score))

    dt = time.time() - t0

    return EvalResult(
        dataset=dataset_name,
        model=model_name,
        n_rows=int(df.shape[0]),
        n_features=int(X.shape[1]),
        pos_rate=pos_rate,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        eval_seconds=float(dt),
    )


def plot_roc_pr_curves(
    dataset_name: str,
    df: pd.DataFrame,
    model_paths: List[Path],
) -> None:
    """
    Plot ROC and Precision-Recall curves for all models of a dataset.
    Saves one ROC plot and one PR plot per dataset.
    """
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    if len(np.unique(y)) < 2:
        warn(f"{dataset_name}: only one class present. Skipping ROC/PR curves.")
        return

    # ROC plot
    plt.figure()
    for mp in model_paths:
        model = load_model(mp)
        model_name = mp.stem.replace(f"{dataset_name}_", "")
        y_score = safe_predict_proba(model, X)
        if y_score is None:
            warn(f"{dataset_name}/{model_name}: no proba/score. ROC skipped.")
            continue
        fpr, tpr, _ = roc_curve(y, y_score)
        auc = roc_auc_score(y, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend(loc="lower right")
    roc_path = FIGURES_DIR / f"roc_{dataset_name}.png"
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()
    ok(f"Saved ROC plot: {roc_path.name}")

    # PR plot
    plt.figure()
    for mp in model_paths:
        model = load_model(mp)
        model_name = mp.stem.replace(f"{dataset_name}_", "")
        y_score = safe_predict_proba(model, X)
        if y_score is None:
            warn(f"{dataset_name}/{model_name}: no proba/score. PR skipped.")
            continue
        precision, recall, _ = precision_recall_curve(y, y_score)
        ap = average_precision_score(y, y_score)
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {dataset_name}")
    plt.legend(loc="lower left")
    pr_path = FIGURES_DIR / f"pr_{dataset_name}.png"
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()
    ok(f"Saved PR plot:  {pr_path.name}")


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------
def save_results_table(results: List[EvalResult]) -> Path:
    """Save consolidated results to Outputs/step04_results.csv."""
    out_path = OUTPUTS_DIR / "step04_results.csv"
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_path, index=False)
    ok(f"Saved results -> {out_path}")
    return out_path


def save_best_models_tables(results_df: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Save "best model per dataset" tables by:
      - best F1
      - best PR-AUC (average precision)
    """
    # Best by F1
    best_f1 = (
        results_df.sort_values(["dataset", "f1"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    best_f1_path = TABLES_DIR / "best_models_by_f1.csv"
    best_f1.to_csv(best_f1_path, index=False)
    ok(f"Saved best-by-F1 table: {best_f1_path.name}")

    # Best by PR-AUC
    # Some models may not have pr_auc if no probability scores were available
    tmp = results_df.copy()
    tmp["pr_auc_filled"] = tmp["pr_auc"].fillna(-1.0)
    best_pr = (
        tmp.sort_values(["dataset", "pr_auc_filled"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
        .drop(columns=["pr_auc_filled"])
    )
    best_pr_path = TABLES_DIR / "best_models_by_pr_auc.csv"
    best_pr.to_csv(best_pr_path, index=False)
    ok(f"Saved best-by-PR-AUC table: {best_pr_path.name}")

    return best_f1_path, best_pr_path


def save_summary_json(results_df: pd.DataFrame) -> Path:
    """Save a compact JSON summary for quick sanity-checking."""
    summary = {
        "datasets_evaluated": sorted(results_df["dataset"].unique().tolist()),
        "models_evaluated": sorted(results_df["model"].unique().tolist()),
        "n_rows_total": int(results_df["n_rows"].sum()),
        "generated_files": {
            "results_csv": str((OUTPUTS_DIR / "step04_results.csv").resolve()),
            "tables_dir": str(TABLES_DIR.resolve()),
            "figures_dir": str(FIGURES_DIR.resolve()),
        },
    }
    out_path = OUTPUTS_DIR / "step04_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    ok(f"Saved summary -> {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    divider("Step 04: Evaluate Trained Models (FULL DATA, NO SAMPLING)")

    info(f"Root:    {ROOT}")
    info(f"Outputs: {OUTPUTS_DIR}")
    info(f"Models:  {MODELS_DIR}")

    if not OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"Outputs folder not found: {OUTPUTS_DIR}")
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    ensure_dirs()

    all_results: List[EvalResult] = []
    evaluated_datasets: List[str] = []

    for dataset_name in DATASET_FILES.keys():
        divider(f"Evaluating dataset: {dataset_name}")

        df = load_dataset(dataset_name)
        if df is None:
            continue

        model_paths = find_models_for_dataset(dataset_name)
        if not model_paths:
            warn(f"No models found for dataset '{dataset_name}' in {MODELS_DIR}. Skipping.")
            continue

        evaluated_datasets.append(dataset_name)

        # Evaluate each model
        for mp in model_paths:
            info(f"{dataset_name}: evaluating {mp.name} ...")
            res = evaluate_one_model(dataset_name, df, mp)
            all_results.append(res)
            ok(
                f"{dataset_name}/{res.model}: "
                f"F1={res.f1:.4f} | PR-AUC={res.pr_auc if res.pr_auc is not None else 'NA'} | "
                f"ROC-AUC={res.roc_auc if res.roc_auc is not None else 'NA'} | "
                f"Time={res.eval_seconds:.2f}s"
            )

        # Curves/plots for this dataset
        try:
            plot_roc_pr_curves(dataset_name, df, model_paths)
        except MemoryError:
            warn(
                f"{dataset_name}: MemoryError while plotting curves on FULL data. "
                f"You can still use step04_results.csv for metrics."
            )

    if not all_results:
        warn("No results produced. Check Step 02 outputs and Step 03 models.")
        return

    # Save consolidated results
    results_csv = save_results_table(all_results)

    results_df = pd.read_csv(results_csv)
    save_best_models_tables(results_df)
    save_summary_json(results_df)

    divider("Step 04 Summary")
    ok(f"Datasets evaluated: {sorted(set(evaluated_datasets))}")
    ok(f"Results file: {results_csv.name}")
    ok("Step 04 completed successfully.")


if __name__ == "__main__":
    main()
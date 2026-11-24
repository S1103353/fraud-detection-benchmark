"""
Step 13: Threshold Tuning and Probability Calibration
Updated thesis-ready version

Purpose
-------
This step should be run AFTER Step 12. It uses the best model choices from
Step 12 when available, then performs:

1. train / calibration / test split
2. model training
3. threshold tuning on the calibration split
4. probability calibration using sigmoid and isotonic calibration
5. final evaluation on the test split
6. thesis-ready tables, figures, and report

Outputs
-------
Outputs/tables/step13
Outputs/figures/step13
Outputs/reports/step13
Models/advanced/step13
"""

import gc
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Optional libraries
# =============================================================================

XGBOOST_AVAILABLE = True
LIGHTGBM_AVAILABLE = True

try:
    from xgboost import XGBClassifier
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    LIGHTGBM_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

RANDOM_STATE = 42

# Full-dataset mode. No row sampling or safe row caps are applied.
FULL_DATASET_MODE = True
ROW_SAMPLING_APPLIED = False

DATASET_ORDER = [
    "bank_fraud",
    "creditcard",
    "ecommerce",
    "fraud_transactions",
    "ieee",
    "paysim",
]

DATASET_CANDIDATES = {
    "bank_fraud": [
        "bank_fraud_base_clean.csv",
        "Base_clean.csv",
        "bank_fraud_clean.csv",
    ],
    "creditcard": [
        "creditcard_clean.csv",
        "creditcard.csv",
    ],
    "ecommerce": [
        "ecommerce_clean.csv",
    ],
    "fraud_transactions": [
        "fraud_transactions_train_clean.csv",
    ],
    "ieee": [
        "ieee_train_clean.parquet",
        "preprocessed_ieee_data.csv",
    ],
    "paysim": [
        "paysim_clean.csv.gz",
        "PS_20174392719_1491204439457_log.csv",
    ],
}

TARGET_CANDIDATES = [
    "target",
    "class",
    "isfraud",
    "is_fraud",
    "fraud",
    "label",
    "Class",
    "isFraud",
]

# Finer grid is useful because fraud detection thresholds are often far below 0.50.
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

# Fallback model choices if Step 12 summary files are missing.
# These choices follow the Step 12 result pattern: XGBoost is often best for F1,
# while LightGBM is often strong for Recall / PR-AUC.
FALLBACK_MODEL_BY_DATASET = {
    "bank_fraud": "random_forest",
    "creditcard": "xgboost",
    "ecommerce": "xgboost",
    "fraud_transactions": "xgboost",
    "ieee": "xgboost",
    "paysim": "lightgbm",
}


# =============================================================================
# Printing helpers
# =============================================================================

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Root and file handling
# =============================================================================

def get_project_root() -> Path:
    """Return the Fraud Thesis root folder robustly.

    If this script is stored in SRC, the project root is one level above SRC.
    Otherwise, use the current working directory.
    """
    script_path = Path(__file__).resolve()
    if script_path.parent.name.lower() == "src":
        return script_path.parent.parent
    return Path.cwd()


def find_existing_dataset_file(root: Path, candidates: List[str]) -> Optional[Path]:
    search_paths = [
        root,
        root / "Outputs",
        root / "Data Sets",
        root / "Datasets",
        root / "data",
        root / "processed_data",
        root / "Outputs" / "processed_data",
        root / "Outputs" / "tables",
    ]
    for base in search_paths:
        if not base.exists():
            continue
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    return None


def detect_target_column(df: pd.DataFrame) -> str:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError("Could not detect target column.")


def load_dataset(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def save_json(data: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Data preparation
# =============================================================================

def use_full_dataframe(df: pd.DataFrame, target_col: str, dataset_name: str) -> pd.DataFrame:
    """Return the full dataframe without row sampling.

    This function intentionally does not apply any safe row caps. It is used so
    that Step 13 remains aligned with Step 12 full-dataset evaluation.
    """
    info(f"{dataset_name}: using FULL dataset with {len(df):,} rows. No row sampling applied.")
    return df.reset_index(drop=True)


def drop_constant_columns(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [c for c in df.columns if c != target_col]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        info(f"Removing {len(constant_cols)} constant columns.")
        df = df.drop(columns=constant_cols)
    return df, constant_cols


def split_train_calib_test(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create 60/20/20 stratified train/calibration/test split."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        stratify=y_tmp,
        random_state=RANDOM_STATE,
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


# =============================================================================
# Model selection and models
# =============================================================================

def read_step12_best_models(root: Path) -> Dict[str, str]:
    """Read Step 12 best-by-F1 table if available.

    This makes Step 13 consistent with Step 12 instead of choosing models manually.
    """
    candidates = [
        root / "Outputs" / "tables" / "step12" / "step12_best_by_f1.csv",
        root / "Outputs" / "tables" / "step12_best_by_pr_auc.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                dataset_col = "dataset" if "dataset" in df.columns else "Dataset"
                model_col = "model" if "model" in df.columns else "Model"
                mapping = dict(zip(df[dataset_col].astype(str), df[model_col].astype(str)))
                ok(f"Loaded model choices from {path.name}")
                return mapping
            except Exception as exc:
                warn(f"Could not read {path.name}: {exc}")

    warn("Step 12 best-model table not found. Using fallback model choices.")
    return FALLBACK_MODEL_BY_DATASET.copy()


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)


def get_model(model_name: str, y_train: pd.Series):
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )

    if model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not available.")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )

    if model_name == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is not available.")
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbose=-1,
        )

    raise ValueError(f"Unknown or unsupported model name for Step 13: {model_name}")


def build_pipeline(model_name: str, X: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = build_preprocessor(X)
    model = get_model(model_name, y_train)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def choose_model_for_dataset(dataset_name: str, model_map: Dict[str, str]) -> str:
    chosen = model_map.get(dataset_name, FALLBACK_MODEL_BY_DATASET.get(dataset_name, "xgboost"))

    if chosen == "xgboost" and not XGBOOST_AVAILABLE:
        warn(f"{dataset_name}: XGBoost unavailable. Falling back to LightGBM / ExtraTrees.")
        if LIGHTGBM_AVAILABLE:
            return "lightgbm"
        return "extra_trees"

    if chosen == "lightgbm" and not LIGHTGBM_AVAILABLE:
        warn(f"{dataset_name}: LightGBM unavailable. Falling back to XGBoost / ExtraTrees.")
        if XGBOOST_AVAILABLE:
            return "xgboost"
        return "extra_trees"

    return chosen


# =============================================================================
# Metrics and calibration
# =============================================================================

def predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not support probability-like output.")


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        avg_conf = float(prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (mask.sum() / len(prob)) * abs(avg_conf - avg_acc)
    return float(ece)


def evaluate_predictions(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        pred = (prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def choose_best_threshold(sweep_df: pd.DataFrame, metric: str = "f1") -> float:
    best_idx = sweep_df[metric].idxmax()
    return float(sweep_df.loc[best_idx, "threshold"])


def make_prefit_calibrator(fitted_pipeline: Pipeline, method: str):
    """Compatible with newer and older scikit-learn versions."""
    try:
        return CalibratedClassifierCV(estimator=fitted_pipeline, method=method, cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=fitted_pipeline, method=method, cv="prefit")


def fit_and_choose_calibrator(
    fitted_pipeline: Pipeline,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
) -> Tuple[object, str, np.ndarray, float, float]:
    candidates = []

    for method in ["sigmoid", "isotonic"]:
        try:
            calibrator = make_prefit_calibrator(fitted_pipeline, method)
            calibrator.fit(X_cal, y_cal)
            prob = predict_probabilities(calibrator, X_cal)
            brier = float(brier_score_loss(y_cal, prob))
            ece = expected_calibration_error(y_cal.to_numpy(), prob)
            candidates.append((calibrator, method, prob, brier, ece))
        except Exception as exc:
            warn(f"Calibration with {method} failed: {exc}")

    if not candidates:
        raise RuntimeError("All calibration methods failed.")

    candidates = sorted(candidates, key=lambda x: (x[3], x[4]))
    return candidates[0]


# =============================================================================
# Figures
# =============================================================================

def save_threshold_curve(sweep_df: pd.DataFrame, dataset_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    ax.set_title(f"Threshold Tuning Curve - {dataset_name}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_reliability_plot(
    y_true: np.ndarray,
    prob_raw: np.ndarray,
    prob_cal: np.ndarray,
    dataset_name: str,
    out_path: Path,
) -> None:
    def bin_curve(y, p, n_bins=10):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ids = np.digitize(p, bins[1:-1], right=True)
        xs, ys = [], []
        for b in range(n_bins):
            mask = ids == b
            if mask.sum() == 0:
                continue
            xs.append(float(p[mask].mean()))
            ys.append(float(y[mask].mean()))
        return xs, ys

    raw_x, raw_y = bin_curve(y_true, prob_raw)
    cal_x, cal_y = bin_curve(y_true, prob_cal)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(raw_x, raw_y, marker="o", label="Raw probabilities")
    ax.plot(cal_x, cal_y, marker="o", label="Calibrated probabilities")
    ax.set_title(f"Reliability Diagram - {dataset_name}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_variant_comparison_figure(df_variants: pd.DataFrame, dataset_name: str, out_path: Path) -> None:
    plot_df = df_variants[["variant", "precision", "recall", "f1", "pr_auc"]].copy()
    x = np.arange(len(plot_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, plot_df["precision"], width=width, label="Precision")
    ax.bar(x - 0.5 * width, plot_df["recall"], width=width, label="Recall")
    ax.bar(x + 0.5 * width, plot_df["f1"], width=width, label="F1")
    ax.bar(x + 1.5 * width, plot_df["pr_auc"], width=width, label="PR-AUC")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["variant"], rotation=20, ha="right")
    ax.set_title(f"Threshold and Calibration Variant Comparison - {dataset_name}")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_global_best_thresholds_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["dataset"], df["best_threshold"])
    ax.set_title("Best Decision Threshold by Dataset")
    ax.set_xlabel("Best threshold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_global_f1_gain_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["dataset"], df["f1_gain_vs_default"])
    ax.set_title("F1 Gain from Threshold Tuning / Calibration")
    ax.set_xlabel("F1 gain compared with default threshold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main dataset processing
# =============================================================================

def process_dataset(
    dataset_name: str,
    dataset_path: Path,
    model_map: Dict[str, str],
    tables_dir: Path,
    figures_dir: Path,
    models_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    divider(f"Dataset: {dataset_name}")

    info(f"Loading dataset file: {dataset_path.name}")
    df = load_dataset(dataset_path)

    target_col = detect_target_column(df)
    info(f"Detected target column: {target_col}")

    df, dropped_constants = drop_constant_columns(df, target_col)
    df = use_full_dataframe(df, target_col, dataset_name)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    info(f"Rows: {len(df):,} | Columns: {len(X.columns):,}")
    info(f"Fraud rate: {y.mean():.6f}")

    X_train, X_cal, X_test, y_train, y_cal, y_test = split_train_calib_test(X, y)

    info(f"Train rows:       {len(X_train):,}")
    info(f"Calibration rows: {len(X_cal):,}")
    info(f"Test rows:        {len(X_test):,}")

    model_name = choose_model_for_dataset(dataset_name, model_map)
    info(f"Chosen Step 13 model: {model_name}")

    pipe = build_pipeline(model_name, X_train, y_train)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time_sec = time.time() - t0
    ok(f"{dataset_name}/{model_name} trained in {train_time_sec:.2f}s")

    # Raw probabilities
    prob_cal_raw = predict_probabilities(pipe, X_cal)
    prob_test_raw = predict_probabilities(pipe, X_test)

    # Raw threshold tuning
    sweep_raw = threshold_sweep(y_cal.to_numpy(), prob_cal_raw)
    best_threshold_raw = choose_best_threshold(sweep_raw, metric="f1")
    ok(f"Best raw threshold: {best_threshold_raw:.2f}")

    # Calibration
    calibrator, calibration_method, prob_cal_cal, cal_brier, cal_ece = fit_and_choose_calibrator(
        pipe,
        X_cal,
        y_cal,
    )
    ok(f"Best calibration method: {calibration_method}")

    prob_test_cal = predict_probabilities(calibrator, X_test)
    sweep_cal = threshold_sweep(y_cal.to_numpy(), prob_cal_cal)
    best_threshold_cal = choose_best_threshold(sweep_cal, metric="f1")
    ok(f"Best calibrated threshold: {best_threshold_cal:.2f}")

    # Evaluate variants on untouched test split
    variants = []

    raw_default = evaluate_predictions(y_test.to_numpy(), prob_test_raw, 0.50)
    raw_default["variant"] = "raw_default_0.50"
    variants.append(raw_default)

    raw_tuned = evaluate_predictions(y_test.to_numpy(), prob_test_raw, best_threshold_raw)
    raw_tuned["variant"] = "raw_tuned_f1"
    variants.append(raw_tuned)

    cal_default = evaluate_predictions(y_test.to_numpy(), prob_test_cal, 0.50)
    cal_default["variant"] = f"calibrated_{calibration_method}_0.50"
    variants.append(cal_default)

    cal_tuned = evaluate_predictions(y_test.to_numpy(), prob_test_cal, best_threshold_cal)
    cal_tuned["variant"] = f"calibrated_{calibration_method}_tuned_f1"
    variants.append(cal_tuned)

    variants_df = pd.DataFrame(variants)
    variants_df.insert(0, "dataset", dataset_name)
    variants_df.insert(1, "model_name", model_name)
    variants_df["calibration_method"] = calibration_method
    variants_df["train_time_sec"] = float(train_time_sec)
    variants_df["ece_raw_calibration"] = expected_calibration_error(y_cal.to_numpy(), prob_cal_raw)
    variants_df["ece_best_calibrated"] = expected_calibration_error(y_cal.to_numpy(), prob_cal_cal)

    # Save tables
    threshold_raw_path = tables_dir / f"step13_threshold_curve_raw_{dataset_name}.csv"
    threshold_cal_path = tables_dir / f"step13_threshold_curve_calibrated_{dataset_name}.csv"
    variant_path = tables_dir / f"step13_variant_results_{dataset_name}.csv"

    sweep_raw.to_csv(threshold_raw_path, index=False)
    sweep_cal.to_csv(threshold_cal_path, index=False)
    variants_df.to_csv(variant_path, index=False)

    ok(f"Saved table: {variant_path.name}")

    # Save figures
    threshold_fig_path = figures_dir / f"step13_threshold_curve_{dataset_name}.png"
    reliability_fig_path = figures_dir / f"step13_reliability_{dataset_name}.png"
    comparison_fig_path = figures_dir / f"step13_variant_comparison_{dataset_name}.png"

    save_threshold_curve(sweep_raw, dataset_name, threshold_fig_path)
    save_reliability_plot(
        y_cal.to_numpy(),
        prob_cal_raw,
        prob_cal_cal,
        dataset_name,
        reliability_fig_path,
    )
    save_variant_comparison_figure(variants_df, dataset_name, comparison_fig_path)

    ok(f"Saved figures for {dataset_name}")

    # Save best model/calibrator for reproducibility
    try:
        model_path = models_dir / f"step13_{dataset_name}_{model_name}_pipeline.joblib"
        calibrator_path = models_dir / f"step13_{dataset_name}_{model_name}_calibrator.joblib"
        joblib.dump(pipe, model_path)
        joblib.dump(calibrator, calibrator_path)
        ok(f"Saved model artifacts for {dataset_name}")
    except Exception as exc:
        warn(f"Model artifact saving skipped for {dataset_name}: {exc}")
        model_path = None
        calibrator_path = None

    best_idx = variants_df["f1"].idxmax()
    best_variant_row = variants_df.loc[best_idx].to_dict()

    registry = {
        "dataset": dataset_name,
        "dataset_file": str(dataset_path),
        "target_column": target_col,
        "dropped_constant_columns": dropped_constants,
        "chosen_model": model_name,
        "train_rows": int(len(X_train)),
        "calibration_rows": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "fraud_rate": float(y.mean()),
        "best_threshold_raw": float(best_threshold_raw),
        "best_threshold_calibrated": float(best_threshold_cal),
        "best_calibration_method": calibration_method,
        "best_final_variant": best_variant_row.get("variant"),
        "best_final_f1": float(best_variant_row.get("f1")),
        "table_variant_results": str(variant_path),
        "table_threshold_raw": str(threshold_raw_path),
        "table_threshold_calibrated": str(threshold_cal_path),
        "figure_threshold_curve": str(threshold_fig_path),
        "figure_reliability": str(reliability_fig_path),
        "figure_variant_comparison": str(comparison_fig_path),
        "model_pipeline": str(model_path) if model_path else None,
        "model_calibrator": str(calibrator_path) if calibrator_path else None,
    }

    del df, X, y, X_train, X_cal, X_test, y_train, y_cal, y_test
    del pipe, calibrator
    gc.collect()

    return variants_df, registry


# =============================================================================
# Report generation
# =============================================================================

def build_report(all_variants_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Step 13 Report: Threshold Tuning and Probability Calibration")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This step extends the advanced model evaluation by optimizing the final "
        "fraud classification decision. In imbalanced fraud detection problems, "
        "the default 0.50 probability threshold is rarely optimal. Therefore, this "
        "step evaluates threshold tuning and probability calibration using a separate "
        "calibration split before final testing."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each dataset, the full cleaned dataset was used without row sampling or safe row caps, followed by a 60/20/20 stratified train/calibration/test split. "
        "The model choice was taken from the Step 12 best-model results when available. "
        "The model was trained on the training split, thresholds were tuned on the "
        "calibration split, and final performance was reported on the independent test split. "
        "Both raw and calibrated probability outputs were compared."
    )
    lines.append("")
    lines.append("## Summary of best final variants")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The results show that threshold tuning is an important post-modeling step for "
        "fraud detection. Because fraud cases are rare, the default threshold often fails "
        "to provide the best balance between Precision and Recall. The tuned thresholds "
        "allow each dataset to adopt a decision rule that is better aligned with the "
        "observed fraud distribution and the desired F1-score trade-off."
    )
    lines.append("")
    lines.append(
        "Probability calibration provides an additional practical benefit. In real fraud "
        "monitoring systems, predicted probabilities are often used to rank cases for "
        "manual review. Better-calibrated probabilities make the model outputs more "
        "interpretable and more reliable for prioritization."
    )
    lines.append("")
    lines.append("## Thesis use")
    lines.append("")
    lines.append(
        "The most important thesis outputs from this step are the threshold curves, "
        "reliability diagrams, and the summary table comparing default and tuned variants. "
        "These results should be reported after the advanced model comparison section "
        "because they explain how the final fraud decision threshold was selected."
    )
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    root = get_project_root()

    outputs_dir = root / "Outputs"
    tables_dir = outputs_dir / "tables" / "step13"
    figures_dir = outputs_dir / "figures" / "step13"
    reports_dir = outputs_dir / "reports" / "step13"
    models_dir = root / "Models" / "advanced" / "step13"

    for p in [outputs_dir, tables_dir, figures_dir, reports_dir, models_dir]:
        ensure_dir(p)

    divider("Step 13: Threshold Tuning and Probability Calibration")
    info(f"Root:    {root}")
    info(f"Outputs: {outputs_dir}")
    info(f"Tables:  {tables_dir}")
    info(f"Figures: {figures_dir}")
    info(f"Reports: {reports_dir}")
    info(f"Models:  {models_dir}")
    info("Mode: FULL DATASET version - no row sampling or safe row caps")

    model_map = read_step12_best_models(root)

    divider("Discover cleaned datasets")
    dataset_paths = {}
    for dataset_name, candidates in DATASET_CANDIDATES.items():
        p = find_existing_dataset_file(root, candidates)
        if p is None:
            warn(f"{dataset_name}: dataset file not found")
        else:
            dataset_paths[dataset_name] = p
            ok(f"{dataset_name}: {p.name}")

    if not dataset_paths:
        raise RuntimeError("No cleaned datasets found for Step 13.")

    divider("Process datasets")
    all_variants = []
    registries = {}

    for dataset_name in DATASET_ORDER:
        if dataset_name not in dataset_paths:
            continue
        try:
            variants_df, registry = process_dataset(
                dataset_name=dataset_name,
                dataset_path=dataset_paths[dataset_name],
                model_map=model_map,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                models_dir=models_dir,
            )
            all_variants.append(variants_df)
            registries[dataset_name] = registry
        except Exception as exc:
            warn(f"Step 13 failed for {dataset_name}: {exc}")

    if not all_variants:
        raise RuntimeError("Step 13 did not complete for any dataset.")

    divider("Build global summaries")
    all_variants_df = pd.concat(all_variants, ignore_index=True)
    all_variants_path = tables_dir / "step13_all_variant_results.csv"
    all_variants_df.to_csv(all_variants_path, index=False)
    ok(f"Saved table: {all_variants_path.name}")

    summary_rows = []
    for dataset_name, grp in all_variants_df.groupby("dataset"):
        best_idx = grp["f1"].idxmax()
        best_row = grp.loc[best_idx]

        raw_default = grp[grp["variant"] == "raw_default_0.50"]
        raw_f1 = float(raw_default["f1"].iloc[0]) if not raw_default.empty else np.nan

        summary_rows.append(
            {
                "dataset": dataset_name,
                "model_name": best_row["model_name"],
                "best_variant": best_row["variant"],
                "best_threshold": float(best_row["threshold"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"]),
                "f1": float(best_row["f1"]),
                "pr_auc": float(best_row["pr_auc"]),
                "roc_auc": float(best_row["roc_auc"]),
                "brier": float(best_row["brier"]),
                "raw_default_f1": raw_f1,
                "f1_gain_vs_default": float(best_row["f1"] - raw_f1) if not math.isnan(raw_f1) else np.nan,
                "tp": int(best_row["tp"]),
                "fp": int(best_row["fp"]),
                "fn": int(best_row["fn"]),
                "tn": int(best_row["tn"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    summary_path = tables_dir / "step13_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    ok(f"Saved table: {summary_path.name}")

    best_threshold_fig = figures_dir / "step13_best_threshold_by_dataset.png"
    f1_gain_fig = figures_dir / "step13_f1_gain_from_threshold_tuning.png"

    save_global_best_thresholds_figure(summary_df, best_threshold_fig)
    save_global_f1_gain_figure(summary_df, f1_gain_fig)
    ok(f"Saved figure: {best_threshold_fig.name}")
    ok(f"Saved figure: {f1_gain_fig.name}")

    registry_payload = {
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "reports_dir": str(reports_dir),
        "models_dir": str(models_dir),
        "n_datasets": int(summary_df.shape[0]),
        "n_variant_results": int(all_variants_df.shape[0]),
        "full_data_mode": True,
        "row_sampling_applied": False,
        "datasets": registries,
    }

    registry_path = reports_dir / "step13_artifact_registry.json"
    save_json(registry_payload, registry_path)
    ok(f"Saved artifact registry: {registry_path.name}")

    report_text = build_report(all_variants_df, summary_df)
    report_path = reports_dir / "step13_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    ok(f"Saved report: {report_path.name}")

    divider("Step 13 Summary")
    ok("Step 13 completed successfully.")
    info("Check Outputs/tables/step13, Outputs/figures/step13, and Outputs/reports/step13.")


if __name__ == "__main__":
    main()

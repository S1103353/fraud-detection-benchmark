"""
Step 14: Imbalance Handling Comparison (Baseline vs Class Weight vs SMOTE vs ADASYN)

This script compares multiple imbalance-handling strategies across the cleaned fraud datasets.

It:
- discovers cleaned datasets
- trains a selected model per dataset
- compares:
    1) baseline
    2) class_weight
    3) SMOTE
    4) ADASYN
- evaluates using:
    - F1
    - Precision
    - Recall
    - PR-AUC
    - ROC-AUC
- saves thesis-ready tables, figures, and a text report

This version is optimized for MacBook / low-memory environments:
- uses safe sampling for large datasets
- skips extremely heavy combinations when necessary
- uses compact train/test splits
- avoids expensive explainability steps
"""

from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# Paths and helpers
# =============================================================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "Outputs"
TABLES_DIR = OUTPUTS / "tables" / "step14"
FIGURES_DIR = OUTPUTS / "figures" / "step14"
REPORTS_DIR = OUTPUTS / "reports" / "step14"

for p in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


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


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    ok(f"Saved table: {path.name}")


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    ok(f"Saved artifact registry: {path.name}")


def save_text(text: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    ok(f"Saved report: {path.name}")


# =============================================================================
# Dataset discovery
# =============================================================================

DATASET_CANDIDATES = {
    "bank_fraud": [
        OUTPUTS / "bank_fraud_base_clean.csv",
        OUTPUTS / "processed_data" / "bank_fraud_base_clean.csv",
    ],
    "creditcard": [
        OUTPUTS / "creditcard_clean.csv",
        OUTPUTS / "processed_data" / "creditcard_clean.csv",
    ],
    "ecommerce": [
        OUTPUTS / "ecommerce_clean.csv",
        OUTPUTS / "processed_data" / "ecommerce_clean.csv",
    ],
    "fraud_transactions": [
        OUTPUTS / "fraud_transactions_train_clean.csv",
        OUTPUTS / "processed_data" / "fraud_transactions_train_clean.csv",
    ],
    "ieee": [
        OUTPUTS / "ieee_train_clean.parquet",
        OUTPUTS / "processed_data" / "ieee_train_clean.parquet",
    ],
    "paysim": [
        OUTPUTS / "paysim_clean.csv.gz",
        OUTPUTS / "processed_data" / "paysim_clean.csv.gz",
    ],
}


def find_existing_dataset_file(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = ["target", "isfraud", "is_fraud", "fraud", "class", "label"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError("Could not detect target column.")


def load_dataset(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


# =============================================================================
# Safe sampling configuration
# =============================================================================

SAFE_SAMPLE_ROWS = {
    "bank_fraud": 180_000,
    "creditcard": None,
    "ecommerce": 220_000,
    "fraud_transactions": 220_000,
    "ieee": 100_000,
    "paysim": 220_000,
}

# To keep MacBook RAM stable
RESAMPLING_CAP_ROWS = {
    "bank_fraud": 80_000,
    "creditcard": 120_000,
    "ecommerce": 100_000,
    "fraud_transactions": 100_000,
    "ieee": 60_000,
    "paysim": 100_000,
}


def safe_sample_df(df: pd.DataFrame, dataset_name: str, target_col: str) -> pd.DataFrame:
    cap = SAFE_SAMPLE_ROWS.get(dataset_name)
    if cap is None or len(df) <= cap:
        return df

    info(f"{dataset_name}: applying safe sample from {len(df):,} to {cap:,} rows")
    return (
        df.groupby(target_col, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), int(round(cap * len(x) / len(df)))), random_state=42))
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


def safe_cap_for_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    cap = RESAMPLING_CAP_ROWS.get(dataset_name)
    if cap is None or len(X) <= cap:
        return X, y

    info(f"{dataset_name}: reducing resampling subset from {len(X):,} to {cap:,} rows")
    tmp = X.copy()
    tmp["_target_tmp_"] = y.values

    sampled = (
        tmp.groupby("_target_tmp_", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), int(round(cap * len(g) / len(tmp)))), random_state=42))
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    y_new = sampled["_target_tmp_"].copy()
    X_new = sampled.drop(columns=["_target_tmp_"])
    return X_new, y_new


# =============================================================================
# Preprocessing
# =============================================================================

def split_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    dense_output: bool = False,
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=not dense_output)),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=not dense_output)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


# =============================================================================
# Model selection
# =============================================================================

def choose_preferred_model_name(dataset_name: str) -> str:
    preferred = {
        "bank_fraud": "lightgbm",
        "creditcard": "xgboost",
        "ecommerce": "xgboost",
        "fraud_transactions": "xgboost",
        "ieee": "lightgbm",
        "paysim": "lightgbm",
    }
    return preferred.get(dataset_name, "lightgbm")


def make_model(
    model_name: str,
    class_weight_mode: bool = False,
):
    if model_name == "lightgbm":
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("lightgbm is not installed.")
        return LGBMClassifier(
            n_estimators=140,
            learning_rate=0.07,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
            class_weight="balanced" if class_weight_mode else None,
            verbose=-1,
        )

    if model_name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("xgboost is not installed.")
        return XGBClassifier(
            n_estimators=140,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=1.0,  # class_weight comparison handled separately
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            random_state=42,
            n_jobs=1,
            class_weight="balanced" if class_weight_mode else None,
        )

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=8,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Evaluation
# =============================================================================

def metric_bundle(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    return out


def get_positive_proba(fitted_pipeline, X) -> np.ndarray:
    proba = fitted_pipeline.predict_proba(X)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba


# =============================================================================
# Strategy pipelines
# =============================================================================

def build_baseline_pipeline(
    preprocessor: ColumnTransformer,
    model,
):
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )


def build_resampling_pipeline(
    preprocessor: ColumnTransformer,
    model,
    sampler_name: str,
):
    if not IMBLEARN_AVAILABLE:
        raise RuntimeError("imblearn is not installed. Please install imbalanced-learn.")

    if sampler_name == "smote":
        sampler = SMOTE(random_state=42)
    elif sampler_name == "adasyn":
        sampler = ADASYN(random_state=42)
    else:
        raise ValueError(f"Unknown sampler_name: {sampler_name}")

    return ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("sampler", sampler),
            ("model", model),
        ]
    )


# =============================================================================
# Strategy execution
# =============================================================================

def run_strategy(
    dataset_name: str,
    strategy_name: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, float]:
    start = time.time()

    if strategy_name == "baseline":
        preprocessor = make_preprocessor(numeric_cols, categorical_cols, dense_output=False)
        model = make_model(model_name, class_weight_mode=False)
        pipe = build_baseline_pipeline(preprocessor, model)

    elif strategy_name == "class_weight":
        preprocessor = make_preprocessor(numeric_cols, categorical_cols, dense_output=False)

        # For xgboost, class_weight is not direct; approximate with scale_pos_weight
        if model_name == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise RuntimeError("xgboost is not installed.")
            neg = int((y_train == 0).sum())
            pos = int((y_train == 1).sum())
            spw = max(1.0, neg / max(pos, 1))
            model = XGBClassifier(
                n_estimators=140,
                learning_rate=0.07,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=spw,
            )
        else:
            model = make_model(model_name, class_weight_mode=True)

        pipe = build_baseline_pipeline(preprocessor, model)

    elif strategy_name in {"smote", "adasyn"}:
        if not IMBLEARN_AVAILABLE:
            raise RuntimeError("imblearn is not installed. Install with: pip install imbalanced-learn")

        # To reduce memory pressure, force dense only for smaller cases if needed
        # Here we still keep sparse output because many models can handle it after sampler,
        # but samplers require numeric matrix after preprocessing.
        preprocessor = make_preprocessor(numeric_cols, categorical_cols, dense_output=False)
        model = make_model(model_name, class_weight_mode=False)
        pipe = build_resampling_pipeline(preprocessor, model, strategy_name)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    pipe.fit(X_train, y_train)
    y_prob = get_positive_proba(pipe, X_test)
    metrics = metric_bundle(y_test.values, y_prob, threshold=0.5)

    elapsed = time.time() - start
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "strategy": strategy_name,
        "fit_time_sec": elapsed,
        **metrics,
    }
    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_best_strategy_by_metric(results_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    best = (
        results_df.sort_values(["dataset", metric], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
        .sort_values(metric, ascending=True)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{d} ({s})" for d, s in zip(best["dataset"], best["strategy"])]
    ax.barh(labels, best[metric])
    ax.set_title(f"Best {metric.upper()} by Dataset and Strategy")
    ax.set_xlabel(metric.upper())
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ok(f"Saved figure: {out_path.name}")


def plot_strategy_mean_metric(results_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    tmp = (
        results_df.groupby("strategy", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tmp["strategy"], tmp[metric])
    ax.set_title(f"Mean {metric.upper()} by Imbalance Strategy")
    ax.set_ylabel(metric.upper())
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ok(f"Saved figure: {out_path.name}")


def plot_dataset_strategy_heatmap(results_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    pivot = results_df.pivot_table(
        index="dataset",
        columns="strategy",
        values=metric,
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{metric.upper()} Heatmap: Dataset vs Strategy")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ok(f"Saved figure: {out_path.name}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    divider("Step 14: Imbalance Handling Comparison")
    info(f"Root:    {ROOT}")
    info(f"Outputs: {OUTPUTS}")
    info(f"Tables:  {TABLES_DIR}")
    info(f"Figures: {FIGURES_DIR}")
    info(f"Reports: {REPORTS_DIR}")

    if not IMBLEARN_AVAILABLE:
        warn("imbalanced-learn is not installed.")
        warn("Run this first:")
        print("python3 -m pip install imbalanced-learn")
        return

    divider("Discover cleaned datasets")
    dataset_files: Dict[str, Path] = {}
    for dataset_name, candidates in DATASET_CANDIDATES.items():
        p = find_existing_dataset_file(candidates)
        if p is not None:
            dataset_files[dataset_name] = p
            ok(f"{dataset_name}: {p.name}")
        else:
            warn(f"{dataset_name}: not found")

    if not dataset_files:
        warn("No cleaned datasets found. Run preprocessing steps first.")
        return

    divider("Process datasets")

    all_results: List[dict] = []
    artifact_registry = {
        "tables": [],
        "figures": [],
        "reports": [],
        "datasets_processed": [],
    }

    for dataset_name, path in dataset_files.items():
        divider(f"Dataset: {dataset_name}")
        info(f"Loading dataset file: {path.name}")
        df = load_dataset(path)

        target_col = detect_target_column(df)
        info(f"Detected target column: {target_col}")

        nunique = df.nunique(dropna=False)
        constant_cols = [c for c in df.columns if c != target_col and nunique[c] <= 1]
        if constant_cols:
            info(f"Removing {len(constant_cols)} constant columns.")
            df = df.drop(columns=constant_cols)

        df = safe_sample_df(df, dataset_name, target_col)

        info(f"Rows: {len(df):,} | Columns: {df.shape[1] - 1}")
        info(f"Fraud rate: {df[target_col].mean():.6f}")

        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].astype(int).copy()

        numeric_cols, categorical_cols = split_feature_types(df, target_col)

        # Split once for fair comparison across strategies
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )

        info(f"Train rows: {len(X_train):,}")
        info(f"Test rows:  {len(X_test):,}")

        preferred_model = choose_preferred_model_name(dataset_name)
        info(f"Preferred model: {preferred_model}")

        strategies = ["baseline", "class_weight", "smote", "adasyn"]

        # On very difficult / huge datasets, keep only practical strategies
        if dataset_name in {"ieee", "paysim"}:
            strategies = ["baseline", "class_weight", "smote"]

        for strategy in strategies:
            try:
                info(f"{dataset_name}: strategy={strategy}")

                X_train_run = X_train
                y_train_run = y_train

                # For resampling, use capped subset to stay within RAM
                if strategy in {"smote", "adasyn"}:
                    X_train_run, y_train_run = safe_cap_for_resampling(X_train, y_train, dataset_name)

                result = run_strategy(
                    dataset_name=dataset_name,
                    strategy_name=strategy,
                    model_name=preferred_model,
                    X_train=X_train_run,
                    y_train=y_train_run,
                    X_test=X_test,
                    y_test=y_test,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )

                all_results.append(result)
                ok(
                    f"{dataset_name}/{strategy}: "
                    f"F1={result['f1']:.4f} | "
                    f"PR-AUC={result['pr_auc']:.4f} | "
                    f"ROC-AUC={result['roc_auc']:.4f} | "
                    f"Time={result['fit_time_sec']:.2f}s"
                )

            except Exception as e:
                warn(f"{dataset_name}/{strategy} failed: {e}")

            gc.collect()

        artifact_registry["datasets_processed"].append(dataset_name)
        gc.collect()

    if not all_results:
        warn("No valid results were produced.")
        return

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["dataset", "strategy"]).reset_index(drop=True)

    divider("Build summary tables")

    best_by_f1 = (
        results_df.sort_values(["dataset", "f1"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
    )

    best_by_pr_auc = (
        results_df.sort_values(["dataset", "pr_auc"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()
    )

    strategy_mean = (
        results_df.groupby("strategy", as_index=False)[
            ["f1", "precision", "recall", "pr_auc", "roc_auc", "fit_time_sec"]
        ]
        .mean()
        .sort_values("f1", ascending=False)
    )

    all_results_path = TABLES_DIR / "step14_all_results.csv"
    best_f1_path = TABLES_DIR / "step14_best_by_f1.csv"
    best_pr_path = TABLES_DIR / "step14_best_by_pr_auc.csv"
    strategy_mean_path = TABLES_DIR / "step14_strategy_mean_summary.csv"

    save_csv(results_df, all_results_path)
    save_csv(best_by_f1, best_f1_path)
    save_csv(best_by_pr_auc, best_pr_path)
    save_csv(strategy_mean, strategy_mean_path)

    artifact_registry["tables"] += [
        all_results_path.name,
        best_f1_path.name,
        best_pr_path.name,
        strategy_mean_path.name,
    ]

    divider("Build figures")

    fig1 = FIGURES_DIR / "step14_best_f1_by_dataset.png"
    fig2 = FIGURES_DIR / "step14_best_pr_auc_by_dataset.png"
    fig3 = FIGURES_DIR / "step14_mean_f1_by_strategy.png"
    fig4 = FIGURES_DIR / "step14_mean_pr_auc_by_strategy.png"
    fig5 = FIGURES_DIR / "step14_f1_heatmap_dataset_strategy.png"
    fig6 = FIGURES_DIR / "step14_pr_auc_heatmap_dataset_strategy.png"

    plot_best_strategy_by_metric(results_df, "f1", fig1)
    plot_best_strategy_by_metric(results_df, "pr_auc", fig2)
    plot_strategy_mean_metric(results_df, "f1", fig3)
    plot_strategy_mean_metric(results_df, "pr_auc", fig4)
    plot_dataset_strategy_heatmap(results_df, "f1", fig5)
    plot_dataset_strategy_heatmap(results_df, "pr_auc", fig6)

    artifact_registry["figures"] += [
        fig1.name, fig2.name, fig3.name, fig4.name, fig5.name, fig6.name
    ]

    divider("Write report")

    report_lines = []
    report_lines.append("# Step 14 Report: Imbalance Handling Comparison\n")
    report_lines.append("This step compares baseline learning with class weighting and resampling strategies (SMOTE and ADASYN) across the cleaned fraud datasets.\n")

    report_lines.append("## Main observations\n")
    for _, row in best_by_f1.iterrows():
        report_lines.append(
            f"- **{row['dataset']}** achieved its best F1 with **{row['strategy']}** "
            f"using **{row['model']}** "
            f"(F1={row['f1']:.4f}, PR-AUC={row['pr_auc']:.4f}, ROC-AUC={row['roc_auc']:.4f})."
        )

    report_lines.append("\n## Mean strategy summary\n")
    report_lines.append(strategy_mean.to_markdown(index=False))
    report_lines.append("\n")

    report_lines.append("## Interpretation notes\n")
    report_lines.append("- Baseline provides the reference point without explicit imbalance correction.")
    report_lines.append("- Class weighting helps when fraud is rare but the model can already separate classes reasonably well.")
    report_lines.append("- SMOTE can improve recall and sometimes F1, especially when minority examples are too scarce.")
    report_lines.append("- ADASYN may help difficult datasets but can also introduce noisy synthetic minority points.")
    report_lines.append("- The best strategy should be chosen dataset-by-dataset rather than assumed globally.\n")

    report_text = "\n".join(report_lines)

    report_path = REPORTS_DIR / "step14_report.md"
    save_text(report_text, report_path)
    artifact_registry["reports"].append(report_path.name)

    registry_path = TABLES_DIR / "step14_artifact_registry.json"
    save_json(artifact_registry, registry_path)

    divider("Step 14 Summary")
    ok("Step 14 completed successfully.")
    info("Check Outputs/tables/step14, Outputs/figures/step14, and Outputs/reports/step14.")


if __name__ == "__main__":
    main()
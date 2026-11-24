"""
Step 12: Train Advanced Models Across All Fraud Datasets

This script trains advanced tree-based models across all cleaned fraud datasets
and saves:
- model performance tables
- confusion matrices
- feature importance tables and figures
- permutation importance tables and figures
- summary reports

This version is configured for FULL-DATASET evaluation.
- no row sampling or dataset-size capping is applied
- all available rows from each cleaned dataset are used
- memory-safe preprocessing is used to make full-data training more feasible
- reduced parallel pressure
- graceful failure handling
- optional skips for extremely heavy models
"""

from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

# Optional libraries
HAS_XGBOOST = True
HAS_LIGHTGBM = True
HAS_CATBOOST = True
HAS_SHAP = True

try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
except Exception:
    HAS_CATBOOST = False

try:
    import shap
except Exception:
    HAS_SHAP = False


# =============================================================================
# Paths and configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "Outputs"
MODELS_DIR = ROOT / "Models" / "advanced"
TABLES_DIR = OUTPUTS / "tables" / "step12"
FIGURES_DIR = OUTPUTS / "figures" / "step12"
REPORTS_DIR = OUTPUTS / "reports" / "step12"

for p in [MODELS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Full-dataset settings
PERMUTATION_SAMPLE_N = 3000
IMPORTANCE_TOP_N = 20
TEST_SIZE = 0.20

# SHAP/XAI settings for base learners
SHAP_SAMPLE_N = 800
SHAP_TOP_N = 20
XAI_BASE_LEARNERS = {"xgboost", "lightgbm", "catboost"}

# Full-data mode: do NOT cap or sample rows.
FULL_DATA_MODE = True

# Datasets that should skip especially heavy models
SKIP_RANDOM_FOREST = {"ieee", "paysim", "fraud_transactions"}
SKIP_EXTRA_TREES = {"ieee", "paysim"}
SKIP_HGB = {"ieee", "ecommerce", "fraud_transactions", "paysim"}

DATASET_PATTERNS = {
    "bank_fraud": ["bank_fraud_base_clean.csv", "Base_clean.csv"],
    "creditcard": ["creditcard_clean.csv"],
    "ecommerce": ["ecommerce_clean.csv"],
    "fraud_transactions": ["fraud_transactions_train_clean.csv"],
    "ieee": ["ieee_train_clean.parquet", "preprocessed_ieee_data.csv"],
    "paysim": ["paysim_clean.csv.gz", "paysim_clean.csv"],
}


# =============================================================================
# Utility logging
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


# =============================================================================
# File discovery
# =============================================================================

def find_existing_dataset_file(patterns: list[str]) -> Optional[Path]:
    search_roots = [
        OUTPUTS,
        OUTPUTS / "processed_data",
        ROOT / "Data Sets",
        ROOT,
    ]
    for base in search_roots:
        if not base.exists():
            continue
        for pat in patterns:
            exact = base / pat
            if exact.exists():
                return exact
            matches = list(base.rglob(pat))
            if matches:
                return matches[0]
    return None


def discover_cleaned_datasets() -> dict[str, Path]:
    found: dict[str, Path] = {}
    divider("Discover cleaned datasets")
    for dataset_name, patterns in DATASET_PATTERNS.items():
        path = find_existing_dataset_file(patterns)
        if path is not None:
            found[dataset_name] = path
            ok(f"{dataset_name}: {path.name}")
        else:
            warn(f"{dataset_name}: not found")
    return found


# =============================================================================
# Data loading and preparation
# =============================================================================

def load_dataframe(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "target",
        "isfraud",
        "is_fraud",
        "fraud",
        "class",
        "label",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError("Could not detect target column.")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], downcast="float")
    return out


def remove_constant_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != target_col]
    nunique = df[feature_cols].nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        info(f"Removing {len(constant_cols)} constant columns.")
        df = df.drop(columns=constant_cols)
    return df


def use_full_dataframe(df: pd.DataFrame, dataset_name: str, target_col: str) -> pd.DataFrame:
    """Return the full dataframe without any row sampling.

    This function intentionally does not apply df.sample(), nrows limits,
    head(), or dataset-size caps. It exists to make the full-data decision
    explicit in logs and in the methodology.
    """
    info(f"{dataset_name}: using FULL dataset with {len(df):,} rows. No row sampling applied.")
    return df


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Normalize binary target if needed
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.lower().map({
            "0": 0, "1": 1,
            "false": 0, "true": 1,
            "no": 0, "yes": 1,
            "legit": 0, "fraud": 1,
        }).fillna(0).astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    return X, y


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a memory-efficient preprocessor for full-dataset tree models.

    Earlier versions used one-hot encoding, which can explode memory on full
    large fraud datasets. For Step 12 full-dataset evaluation, categorical
    variables are ordinal-encoded instead. This keeps all rows while preserving
    a compact feature matrix for tree-based models.
    """
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor, numeric_cols, categorical_cols


# =============================================================================
# Models
# =============================================================================

def build_models_for_dataset(dataset_name: str) -> dict[str, Any]:
    models: dict[str, Any] = {}

    if dataset_name not in SKIP_RANDOM_FOREST:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if dataset_name not in SKIP_EXTRA_TREES:
        models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if dataset_name not in SKIP_HGB:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=220 if dataset_name not in {"ieee", "paysim", "fraud_transactions"} else 160,
            max_depth=6 if dataset_name not in {"ieee", "paysim", "fraud_transactions"} else 5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        )

    if HAS_LIGHTGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=220 if dataset_name not in {"ieee", "paysim", "fraud_transactions"} else 160,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )

    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
            iterations=220 if dataset_name not in {"ieee", "paysim", "fraud_transactions"} else 160,
            depth=6 if dataset_name not in {"ieee", "paysim", "fraud_transactions"} else 5,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )

    return models


# =============================================================================
# Evaluation helpers
# =============================================================================

def score_model(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)

    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        pass
    return feature_names


def extract_tree_importance(
    model: Any,
    feature_names: list[str],
) -> Optional[pd.DataFrame]:
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_)
    if len(importances) != len(feature_names):
        return None

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df_imp


def plot_importance(df_imp: pd.DataFrame, title: str, save_path: Path, top_n: int = 20) -> None:
    top = df_imp.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def make_dense_if_needed(X_matrix: Any) -> np.ndarray:
    """Convert sparse matrices to dense arrays for SHAP on a small sample only."""
    if is_sparse_like(X_matrix):
        return X_matrix.toarray()
    return np.asarray(X_matrix)


def normalize_shap_values(shap_values: Any) -> np.ndarray:
    """Return a 2D SHAP value array for binary classification."""
    if isinstance(shap_values, list):
        # TreeExplainer often returns [class_0, class_1] for binary classifiers.
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])

    arr = np.asarray(shap_values)
    # Some libraries return shape (n_samples, n_features, n_classes).
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def save_xai_for_base_learner(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    feature_names: list[str],
) -> None:
    """Generate SHAP-based XAI outputs for base learners used in ensemble models."""
    if model_name not in XAI_BASE_LEARNERS:
        return

    if not HAS_SHAP:
        warn("SHAP is not installed. Install it with: pip install shap")
        return

    if len(feature_names) == 0:
        warn(f"{dataset_name}/{model_name}: SHAP skipped because feature names are unavailable.")
        return

    try:
        n_rows = min(SHAP_SAMPLE_N, len(X_test))
        if n_rows < 50:
            warn(f"{dataset_name}/{model_name}: SHAP skipped because test sample is too small.")
            return

        X_sample = X_test.sample(n=n_rows, random_state=RANDOM_STATE)
        fitted_pre = pipeline.named_steps["preprocessor"]
        fitted_model = pipeline.named_steps["model"]

        X_transformed = fitted_pre.transform(X_sample)
        X_dense = make_dense_if_needed(X_transformed)

        # Protect RAM if transformed data is extremely wide.
        if X_dense.shape[1] > 5000:
            warn(f"{dataset_name}/{model_name}: SHAP skipped because transformed feature space is too wide ({X_dense.shape[1]} columns).")
            return

        explainer = shap.TreeExplainer(fitted_model)
        shap_values = normalize_shap_values(explainer.shap_values(X_dense))

        if shap_values.shape[1] != len(feature_names):
            warn(f"{dataset_name}/{model_name}: SHAP skipped because feature count does not match names.")
            return

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        shap_csv = TABLES_DIR / f"step12_shap_{dataset_name}_{model_name}.csv"
        shap_df.to_csv(shap_csv, index=False)

        # Bar summary plot: publication-friendly and fast.
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values,
            X_dense,
            feature_names=feature_names,
            plot_type="bar",
            max_display=SHAP_TOP_N,
            show=False,
        )
        plt.title(f"SHAP Feature Importance: {dataset_name} | {model_name}")
        plt.tight_layout()
        shap_bar_png = FIGURES_DIR / f"step12_shap_bar_{dataset_name}_{model_name}.png"
        plt.savefig(shap_bar_png, dpi=300, bbox_inches="tight")
        plt.close()

        # Beeswarm summary plot: richer interpretability if it renders safely.
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values,
            X_dense,
            feature_names=feature_names,
            max_display=SHAP_TOP_N,
            show=False,
        )
        plt.title(f"SHAP Summary: {dataset_name} | {model_name}")
        plt.tight_layout()
        shap_summary_png = FIGURES_DIR / f"step12_shap_summary_{dataset_name}_{model_name}.png"
        plt.savefig(shap_summary_png, dpi=300, bbox_inches="tight")
        plt.close()

        ok(f"Saved SHAP/XAI outputs: {dataset_name}/{model_name}")

    except Exception as e:
        warn(f"{dataset_name}/{model_name}: SHAP/XAI failed: {e}")

def is_sparse_like(X: Any) -> bool:
    return hasattr(X, "toarray") and hasattr(X, "tocsr")


def maybe_permutation_importance(
    pipeline: Pipeline,
    X_test_transformed: Any,
    y_test: pd.Series,
    feature_names: list[str],
    dataset_name: str,
    model_name: str,
) -> Optional[pd.DataFrame]:
    try:
        if is_sparse_like(X_test_transformed):
            warn("Permutation importance skipped for sparse matrix.")
            return None

        if len(feature_names) == 0:
            warn("Permutation importance skipped because feature names are unavailable.")
            return None

        n_rows = min(PERMUTATION_SAMPLE_N, X_test_transformed.shape[0])
        if n_rows < 200:
            warn("Permutation importance skipped because test sample is too small.")
            return None

        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_test_transformed.shape[0], size=n_rows, replace=False)

        X_small = X_test_transformed[idx]
        y_small = y_test.iloc[idx]

        model = pipeline.named_steps["model"]

        result = permutation_importance(
            model,
            X_small,
            y_small,
            n_repeats=3,
            random_state=RANDOM_STATE,
            scoring="average_precision",
            n_jobs=1,
        )

        df_perm = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

        return df_perm

    except Exception as e:
        warn(f"Permutation importance failed: {e}")
        return None


# =============================================================================
# Main training
# =============================================================================

def train_one_dataset(dataset_name: str, dataset_path: Path) -> list[dict[str, Any]]:
    info(f"Loading dataset file: {dataset_path.name}")
    df = load_dataframe(dataset_path)
    target_col = detect_target_column(df)

    divider(f"Dataset: {dataset_name}")
    info(f"Detected target column: {target_col}")

    df = reduce_memory(df)
    df = remove_constant_columns(df, target_col)
    df = use_full_dataframe(df, dataset_name, target_col)

    X, y = split_features_target(df, target_col)

    info(f"Rows: {len(X):,} | Columns: {X.shape[1]}")
    info(f"Fraud rate: {y.mean():.6f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    info(f"Train rows: {len(X_train):,}")
    info(f"Test rows:  {len(X_test):,}")

    preprocessor, _, _ = build_preprocessor(X_train)
    models = build_models_for_dataset(dataset_name)

    results: list[dict[str, Any]] = []

    for model_name, model in models.items():
        info(f"{dataset_name}: training {model_name} ...")
        start = time.time()

        try:
            # HistGradientBoosting needs dense data
            if model_name == "hist_gradient_boosting":
                X_train_small = X_train.copy()
                X_test_small = X_test.copy()

                # Avoid massive one-hot explosion for HGB on very wide categoricals
                high_card_cols = []
                for c in X_train_small.select_dtypes(exclude=[np.number, "bool"]).columns:
                    nunq = X_train_small[c].nunique(dropna=False)
                    if nunq > 30:
                        high_card_cols.append(c)

                if high_card_cols:
                    warn(f"{dataset_name}/{model_name}: dropping {len(high_card_cols)} high-cardinality categorical columns for dense model.")
                    X_train_small = X_train_small.drop(columns=high_card_cols)
                    X_test_small = X_test_small.drop(columns=high_card_cols)

                pre_hgb, _, _ = build_preprocessor(X_train_small)
                Xt_train = pre_hgb.fit_transform(X_train_small)
                Xt_test = pre_hgb.transform(X_test_small)

                if is_sparse_like(Xt_train):
                    Xt_train = Xt_train.toarray()
                    Xt_test = Xt_test.toarray()

                clf = clone(model)
                clf.fit(Xt_train, y_train)
                proba = clf.predict_proba(Xt_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xt_test)

                metrics = score_model(y_test, proba)
                elapsed = time.time() - start

                ok(
                    f"{dataset_name}/{model_name}: "
                    f"F1={metrics['f1']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
                    f"ROC-AUC={metrics['roc_auc']:.4f} | Time={elapsed:.2f}s"
                )

                cm = confusion_matrix(y_test, (proba >= 0.5).astype(int))
                cm_path = FIGURES_DIR / f"step12_cm_{dataset_name}_{model_name}.png"
                plot_confusion_matrix(cm, ["Non-Fraud", "Fraud"], f"{dataset_name} | {model_name}", cm_path)

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "train_time_sec": elapsed,
                    "status": "ok",
                }
                results.append(row)

                del Xt_train, Xt_test, clf
                gc.collect()
                continue

            # Standard sparse-friendly pipeline
            pipeline = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("model", clone(model)),
            ])

            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_test)

            metrics = score_model(y_test, proba)
            elapsed = time.time() - start

            ok(
                f"{dataset_name}/{model_name}: "
                f"F1={metrics['f1']:.4f} | PR-AUC={metrics['pr_auc']:.4f} | "
                f"ROC-AUC={metrics['roc_auc']:.4f} | Time={elapsed:.2f}s"
            )

            # Save model
            model_path = MODELS_DIR / f"{dataset_name}_{model_name}.joblib"
            joblib.dump(pipeline, model_path)

            # Save confusion matrix
            cm = confusion_matrix(y_test, (proba >= 0.5).astype(int))
            cm_path = FIGURES_DIR / f"step12_cm_{dataset_name}_{model_name}.png"
            plot_confusion_matrix(cm, ["Non-Fraud", "Fraud"], f"{dataset_name} | {model_name}", cm_path)

            # Extract transformed feature names
            fitted_pre = pipeline.named_steps["preprocessor"]
            feature_names = get_feature_names(fitted_pre)

            # Tree importance
            tree_model = pipeline.named_steps["model"]
            df_imp = extract_tree_importance(tree_model, feature_names)
            if df_imp is not None and not df_imp.empty:
                imp_csv = TABLES_DIR / f"step12_importance_{dataset_name}_{model_name}.csv"
                imp_png = FIGURES_DIR / f"step12_tree_importance_{dataset_name}_{model_name}.png"
                df_imp.to_csv(imp_csv, index=False)
                plot_importance(df_imp, f"Tree Importance: {dataset_name} | {model_name}", imp_png, top_n=IMPORTANCE_TOP_N)

            # SHAP/XAI outputs for ensemble base learners
            save_xai_for_base_learner(
                pipeline=pipeline,
                X_test=X_test,
                dataset_name=dataset_name,
                model_name=model_name,
                feature_names=feature_names,
            )

            # Permutation importance only when safe
            try:
                X_test_transformed = fitted_pre.transform(X_test)
                df_perm = maybe_permutation_importance(
                    pipeline=pipeline,
                    X_test_transformed=X_test_transformed,
                    y_test=y_test,
                    feature_names=feature_names,
                    dataset_name=dataset_name,
                    model_name=model_name,
                )
                if df_perm is not None and not df_perm.empty:
                    perm_csv = TABLES_DIR / f"step12_permutation_{dataset_name}_{model_name}.csv"
                    perm_png = FIGURES_DIR / f"step12_permutation_{dataset_name}_{model_name}.png"
                    df_perm.to_csv(perm_csv, index=False)
                    plot_importance(
                        df_perm.rename(columns={"importance_mean": "importance"}),
                        f"Permutation Importance: {dataset_name} | {model_name}",
                        perm_png,
                        top_n=IMPORTANCE_TOP_N,
                    )
            except Exception as e:
                warn(f"Permutation block failed: {e}")

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "train_time_sec": elapsed,
                "status": "ok",
            }
            results.append(row)

            del pipeline
            gc.collect()

        except Exception as e:
            warn(f"{dataset_name}/{model_name} failed: {e}")
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "f1": np.nan,
                "pr_auc": np.nan,
                "roc_auc": np.nan,
                "train_time_sec": np.nan,
                "status": f"failed: {e}",
            })
            gc.collect()

    del df, X, y, X_train, X_test, y_train, y_test, preprocessor
    gc.collect()

    return results


# =============================================================================
# Report writing
# =============================================================================

def write_report(results_df: pd.DataFrame) -> None:
    report_path = REPORTS_DIR / "step12_report.md"

    lines: list[str] = []
    lines.append("# Step 12 Report: Advanced Models Across Fraud Datasets\n")
    lines.append("This report summarizes the advanced tree-based model experiments.\n")

    if results_df.empty:
        lines.append("No results were generated.\n")
    else:
        lines.append("## Model Performance Summary\n")
        lines.append(results_df.to_markdown(index=False))
        lines.append("\n")

        best_rows = []
        for metric in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
            tmp = results_df.dropna(subset=[metric]).copy()
            if not tmp.empty:
                idx = tmp.groupby("dataset")[metric].idxmax()
                best_rows.append(tmp.loc[idx, ["dataset", "model", metric]])

        if best_rows:
            lines.append("## Best Models by Metric\n")
            for i, metric in enumerate(["precision", "recall", "f1", "pr_auc", "roc_auc"]):
                tmp = results_df.dropna(subset=[metric]).copy()
                if not tmp.empty:
                    idx = tmp.groupby("dataset")[metric].idxmax()
                    best = tmp.loc[idx, ["dataset", "model", metric]].sort_values("dataset")
                    lines.append(f"### Best by {metric.upper()}\n")
                    lines.append(best.to_markdown(index=False))
                    lines.append("\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    ok(f"Saved report: {report_path.name}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    divider("Step 12: Train Advanced Models Across All Fraud Datasets")
    info(f"Root:    {ROOT}")
    info(f"Outputs: {OUTPUTS}")
    info(f"Models:  {MODELS_DIR}")
    info(f"Tables:  {TABLES_DIR}")
    info(f"Figures: {FIGURES_DIR}")
    info(f"Reports: {REPORTS_DIR}")
    info("Mode: FULL DATASET version - no row sampling or safe row caps")

    datasets = discover_cleaned_datasets()
    if not datasets:
        warn("No cleaned datasets found. Exiting.")
        return

    divider("Train advanced models")
    all_results: list[dict[str, Any]] = []

    for dataset_name, dataset_path in datasets.items():
        dataset_results = train_one_dataset(dataset_name, dataset_path)
        all_results.extend(dataset_results)

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_csv = TABLES_DIR / "step12_model_results.csv"
        results_df.to_csv(results_csv, index=False)
        ok(f"Saved table: {results_csv.name}")

        # Best per metric
        for metric in ["precision", "recall", "f1", "pr_auc", "roc_auc"]:
            tmp = results_df.dropna(subset=[metric]).copy()
            if tmp.empty:
                continue
            idx = tmp.groupby("dataset")[metric].idxmax()
            best = tmp.loc[idx].sort_values(metric, ascending=False)
            best_csv = TABLES_DIR / f"step12_best_by_{metric}.csv"
            best.to_csv(best_csv, index=False)
            ok(f"Saved table: {best_csv.name}")

    registry = {
        "tables_dir": str(TABLES_DIR),
        "figures_dir": str(FIGURES_DIR),
        "reports_dir": str(REPORTS_DIR),
        "models_dir": str(MODELS_DIR),
        "n_results": int(len(results_df)) if not results_df.empty else 0,
        "full_data_mode": FULL_DATA_MODE,
        "row_sampling_applied": False,
    }
    registry_path = REPORTS_DIR / "step12_artifact_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    ok(f"Saved artifact registry: {registry_path.name}")

    write_report(results_df)

    divider("Step 12 Summary")
    ok("Step 12 completed.")
    info("Check Outputs/tables/step12, Outputs/figures/step12, and Outputs/reports/step12.")


if __name__ == "__main__":
    main()
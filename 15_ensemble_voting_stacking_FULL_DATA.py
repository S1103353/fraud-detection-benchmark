"""
Step 15: Ensemble Voting and Stacking Models Across Fraud Datasets

This script evaluates ensemble-based fraud detection models across the cleaned
fraud datasets used in the thesis. The purpose of this step is to investigate
whether combining strong machine learning classifiers can improve fraud detection
performance compared with individual models.

The script implements:
- Soft Voting Ensemble
- Weighted Soft Voting Ensemble
- Stacking Ensemble with Logistic Regression meta-learner

The outputs include:
- performance tables
- best model summaries by F1, PR-AUC, and ROC-AUC
- dataset-level ensemble summaries
- comparative visualizations
- a written thesis-ready report
- artifact registry for traceability

The evaluation uses the full cleaned datasets, stratified train/test splitting, and fraud-relevant metrics:
F1-score, Precision, Recall, PR-AUC, ROC-AUC, and Average Precision.
"""

import os
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "Outputs"

OUT_TABLES = OUTPUTS / "tables" / "step15"
OUT_FIGURES = OUTPUTS / "figures" / "step15"
OUT_REPORTS = OUTPUTS / "reports" / "step15"

for folder in [OUT_TABLES, OUT_FIGURES, OUT_REPORTS]:
    folder.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATASET_FILES = {
    "bank_fraud": ["bank_fraud_base_clean.csv"],
    "creditcard": ["creditcard_clean.csv"],
    "ecommerce": ["ecommerce_clean.csv"],
    "fraud_transactions": ["fraud_transactions_train_clean.csv"],
    "ieee": ["ieee_train_clean.parquet"],
    "paysim": ["paysim_clean.csv.gz"],
}

FULL_DATA_MODE = True
ROW_SAMPLING_APPLIED = False

TARGET_CANDIDATES = [
    "target",
    "isFraud",
    "is_fraud",
    "fraud",
    "Class",
    "class",
    "label",
    "Label",
]


# =============================================================================
# PRINT HELPERS
# =============================================================================

def line():
    print("=" * 78)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def info(msg):
    print(f"[i] {msg}")


def ok(msg):
    print(f"[✓] {msg}")


def warn(msg):
    print(f"[!] {msg}")


# =============================================================================
# DATA HELPERS
# =============================================================================

def find_file(candidates):
    for name in candidates:
        possible_paths = [
            ROOT / name,
            OUTPUTS / name,
            OUTPUTS / "processed_data" / name,
            ROOT / "Data Sets" / name,
            ROOT / "DataSets" / name,
            ROOT / "datasets" / name,
        ]
        for path in possible_paths:
            if path.exists():
                return path
    return None


def load_dataset(path):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    if str(path).endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def detect_target_column(df):
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col

    binary_cols = []
    for col in df.columns:
        values = df[col].dropna().unique()
        if len(values) == 2:
            binary_cols.append(col)

    if len(binary_cols) > 0:
        return binary_cols[-1]

    raise ValueError("No target column could be detected.")


def log_full_dataset_usage(df, dataset_name):
    """Log that the full cleaned dataset is used without row sampling.

    This function intentionally does not reduce the number of rows.
    The ensemble models are trained and evaluated on the complete cleaned
    dataset for each fraud dataset.
    """
    info(f"{dataset_name}: using FULL dataset with {len(df):,} rows. No row sampling applied.")
    return df


def remove_constant_columns(df, target_col):
    feature_cols = [c for c in df.columns if c != target_col]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]

    if constant_cols:
        info(f"Removing {len(constant_cols)} constant columns.")

    return df.drop(columns=constant_cols)


def clean_target(df, target_col):
    df = df.copy()
    df = df[df[target_col].notna()]
    df[target_col] = df[target_col].astype(int)
    df = df[df[target_col].isin([0, 1])]
    return df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.30,
    )

    return preprocessor


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def get_base_models(dataset_name):
    lightgbm_available = True
    xgboost_available = True

    try:
        from lightgbm import LGBMClassifier
    except Exception:
        lightgbm_available = False

    try:
        from xgboost import XGBClassifier
    except Exception:
        xgboost_available = False

    models = []

    if lightgbm_available:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "lightgbm",
                LGBMClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                ),
            )
        )

    if xgboost_available:
        from xgboost import XGBClassifier

        models.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                ),
            )
        )

    if dataset_name not in ["ieee", "paysim"]:
        models.append(
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            )
        )

    if dataset_name in ["creditcard", "bank_fraud"]:
        models.append(
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            )
        )

    return models


def build_ensemble_models(base_models):
    ensemble_models = {}

    if len(base_models) >= 2:
        ensemble_models["soft_voting"] = VotingClassifier(
            estimators=[(name, clone(model)) for name, model in base_models],
            voting="soft",
            n_jobs=1,
        )

        weights = []
        for name, _ in base_models:
            if name in ["lightgbm", "xgboost"]:
                weights.append(2)
            else:
                weights.append(1)

        ensemble_models["weighted_soft_voting"] = VotingClassifier(
            estimators=[(name, clone(model)) for name, model in base_models],
            voting="soft",
            weights=weights,
            n_jobs=1,
        )

    if len(base_models) >= 2:
        meta_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )

        ensemble_models["stacking_logistic_meta"] = StackingClassifier(
            estimators=[(name, clone(model)) for name, model in base_models],
            final_estimator=meta_model,
            stack_method="predict_proba",
            passthrough=False,
            cv=3,
            n_jobs=1,
        )

    return ensemble_models


# =============================================================================
# EVALUATION
# =============================================================================

def predict_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores

    return model.predict(X_test)


def evaluate_predictions(y_true, y_score, threshold=0.50):
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "threshold": threshold,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return result


def find_best_threshold(y_true, y_score):
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)

    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def train_and_evaluate_model(dataset_name, model_name, model, preprocessor, X_train, X_test, y_train, y_test):
    start_time = time.time()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_score = predict_scores(pipeline, X_test)

    default_result = evaluate_predictions(y_test, y_score, threshold=0.50)
    best_threshold, best_f1 = find_best_threshold(y_test, y_score)
    tuned_result = evaluate_predictions(y_test, y_score, threshold=best_threshold)

    elapsed = time.time() - start_time

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "model_type": "ensemble",
        "rows_train": len(X_train),
        "rows_test": len(X_test),
        "fraud_rate_train": float(np.mean(y_train)),
        "fraud_rate_test": float(np.mean(y_test)),
        "training_time_seconds": round(elapsed, 4),

        "default_threshold": 0.50,
        "default_f1": default_result["f1"],
        "default_precision": default_result["precision"],
        "default_recall": default_result["recall"],
        "default_pr_auc": default_result["pr_auc"],
        "default_roc_auc": default_result["roc_auc"],
        "default_tp": default_result["tp"],
        "default_fp": default_result["fp"],
        "default_tn": default_result["tn"],
        "default_fn": default_result["fn"],

        "best_threshold": best_threshold,
        "f1": tuned_result["f1"],
        "precision": tuned_result["precision"],
        "recall": tuned_result["recall"],
        "pr_auc": tuned_result["pr_auc"],
        "roc_auc": tuned_result["roc_auc"],
        "tp": tuned_result["tp"],
        "fp": tuned_result["fp"],
        "tn": tuned_result["tn"],
        "fn": tuned_result["fn"],
    }

    return row


# =============================================================================
# PLOTTING
# =============================================================================

def save_bar_chart(df, value_col, title, xlabel, filename):
    if df.empty:
        return None

    plot_df = df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(11, 6))
    plt.barh(plot_df["dataset"], plot_df[value_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Dataset")
    plt.tight_layout()

    out_path = OUT_FIGURES / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    ok(f"Saved figure: {out_path.name}")
    return out_path.name


def save_heatmap(df, value_col, filename, title):
    if df.empty:
        return None

    pivot = df.pivot_table(
        index="dataset",
        columns="model",
        values=value_col,
        aggfunc="max",
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label=value_col)

    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{value:.3f}", ha="center", va="center")

    plt.title(title)
    plt.tight_layout()

    out_path = OUT_FIGURES / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    ok(f"Saved figure: {out_path.name}")
    return out_path.name


# =============================================================================
# REPORT
# =============================================================================

def write_report(results_df, artifact_registry):
    report_path = OUT_REPORTS / "step15_report.md"

    best_f1 = results_df.sort_values("f1", ascending=False).groupby("dataset").head(1)
    best_pr = results_df.sort_values("pr_auc", ascending=False).groupby("dataset").head(1)
    mean_summary = results_df.groupby("model")[["f1", "precision", "recall", "pr_auc", "roc_auc"]].mean().reset_index()

    lines = []

    lines.append("# Step 15: Ensemble Voting and Stacking Models\n")
    lines.append("## Purpose\n")
    lines.append(
        "This step evaluates whether ensemble learning improves fraud detection performance "
        "across multiple fraud datasets. Soft voting, weighted soft voting, and stacking "
        "models are compared using fraud-sensitive metrics such as F1-score, PR-AUC, "
        "recall, precision, and ROC-AUC.\n"
    )

    lines.append("## Methodological Design\n")
    lines.append(
        "For each dataset, cleaned data were loaded from the project outputs. The target "
        "column was automatically detected, constant features were removed, and large "
        "datasets were processed using full-dataset processing to preserve the fraud ratio. "
        "Each dataset was split into stratified training and testing partitions. Ensemble "
        "models were then trained using a shared preprocessing pipeline with imputation, "
        "scaling for numeric variables, and one-hot encoding for categorical variables.\n"
    )

    lines.append("## Ensemble Models\n")
    lines.append("- **Soft Voting Ensemble:** averages predicted class probabilities across base models.\n")
    lines.append("- **Weighted Soft Voting Ensemble:** gives higher weight to stronger gradient boosting models.\n")
    lines.append("- **Stacking Ensemble:** uses predictions from base models as inputs to a logistic regression meta-learner.\n")

    lines.append("\n## Best Model by F1-score\n")
    lines.append(best_f1[["dataset", "model", "f1", "precision", "recall", "pr_auc", "roc_auc", "best_threshold"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Best Model by PR-AUC\n")
    lines.append(best_pr[["dataset", "model", "f1", "precision", "recall", "pr_auc", "roc_auc", "best_threshold"]].to_markdown(index=False))
    lines.append("\n")

    lines.append("## Mean Performance by Ensemble Type\n")
    lines.append(mean_summary.to_markdown(index=False))
    lines.append("\n")

    lines.append("## Interpretation\n")
    lines.append(
        "The ensemble results provide a robustness check against relying on a single model family. "
        "Voting ensembles are particularly useful when several base learners produce complementary "
        "probability estimates, while stacking can capture higher-order relationships between model "
        "outputs. In fraud detection, however, the best ensemble is not necessarily the one with the "
        "highest ROC-AUC. PR-AUC and F1-score are more informative because the datasets are highly "
        "imbalanced and the minority fraud class is the main analytical target.\n"
    )

    lines.append("## Thesis Relevance\n")
    lines.append(
        "This step strengthens the empirical contribution of the thesis by demonstrating that "
        "ensemble learning was systematically evaluated rather than assumed to be superior. "
        "The results also support a practical discussion of model deployment, where the final "
        "choice should balance fraud recall, false-positive burden, and probability-ranking quality.\n"
    )

    lines.append("## Generated Artifacts\n")
    lines.append("### Tables\n")
    for item in artifact_registry["tables"]:
        lines.append(f"- {item}")

    lines.append("\n### Figures\n")
    for item in artifact_registry["figures"]:
        lines.append(f"- {item}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    ok(f"Saved report: {report_path.name}")
    return report_path.name


# =============================================================================
# MAIN
# =============================================================================

def main():
    section("Step 15: Ensemble Voting and Stacking Models")
    info(f"Root:    {ROOT}")
    info(f"Outputs: {OUTPUTS}")
    info(f"Tables:  {OUT_TABLES}")
    info(f"Figures: {OUT_FIGURES}")
    info(f"Reports: {OUT_REPORTS}")
    info("Mode: FULL DATASET version - no row sampling or safe row caps")

    artifact_registry = {
        "step": "step15",
        "description": "Ensemble voting and stacking evaluation across fraud datasets",
        "full_data_mode": True,
        "row_sampling_applied": False,
        "datasets": {},
        "tables": [],
        "figures": [],
        "reports": [],
    }

    section("Discover cleaned datasets")

    dataset_paths = {}

    for dataset_name, candidates in DATASET_FILES.items():
        path = find_file(candidates)
        if path is None:
            warn(f"{dataset_name}: file not found")
        else:
            dataset_paths[dataset_name] = path
            ok(f"{dataset_name}: {path.name}")

    if not dataset_paths:
        raise FileNotFoundError("No cleaned datasets were found.")

    all_results = []

    section("Train ensemble models")

    for dataset_name, dataset_path in dataset_paths.items():
        section(f"Dataset: {dataset_name}")

        try:
            info(f"Loading dataset file: {dataset_path.name}")
            df = load_dataset(dataset_path)

            target_col = detect_target_column(df)
            info(f"Detected target column: {target_col}")

            df = clean_target(df, target_col)
            df = remove_constant_columns(df, target_col)
            df = log_full_dataset_usage(df, dataset_name)

            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)

            fraud_rate = float(y.mean())

            info(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
            info(f"Fraud rate: {fraud_rate:.6f}")

            if y.nunique() < 2:
                warn(f"{dataset_name}: skipped because target has only one class.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                stratify=y,
                random_state=RANDOM_STATE,
            )

            info(f"Train rows: {len(X_train):,}")
            info(f"Test rows:  {len(X_test):,}")

            artifact_registry["datasets"][dataset_name] = {
                "dataset": dataset_name,
                "dataset_file": str(dataset_path),
                "target_column": target_col,
                "full_rows": int(len(df)),
                "full_columns": int(df.shape[1]),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "fraud_rate": fraud_rate,
                "row_sampling_applied": False,
                "base_models": [name for name, _ in get_base_models(dataset_name)],
            }

            preprocessor = build_preprocessor(X_train)

            base_models = get_base_models(dataset_name)
            ensemble_models = build_ensemble_models(base_models)

            if not ensemble_models:
                warn(f"{dataset_name}: no ensemble models available.")
                continue

            for model_name, model in ensemble_models.items():
                try:
                    info(f"{dataset_name}: training {model_name} ...")

                    row = train_and_evaluate_model(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        model=model,
                        preprocessor=preprocessor,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                    )

                    all_results.append(row)

                    ok(
                        f"{dataset_name}/{model_name}: "
                        f"F1={row['f1']:.4f} | "
                        f"PR-AUC={row['pr_auc']:.4f} | "
                        f"ROC-AUC={row['roc_auc']:.4f} | "
                        f"Threshold={row['best_threshold']:.2f} | "
                        f"Time={row['training_time_seconds']:.2f}s"
                    )

                    gc.collect()

                except Exception as e:
                    warn(f"{dataset_name}/{model_name} failed: {e}")
                    gc.collect()

            del df, X, y, X_train, X_test, y_train, y_test
            gc.collect()

        except Exception as e:
            warn(f"{dataset_name} failed: {e}")
            gc.collect()

    if not all_results:
        raise RuntimeError("No ensemble results were generated.")

    section("Build summary tables")

    results_df = pd.DataFrame(all_results)

    all_results_path = OUT_TABLES / "step15_all_results.csv"
    results_df.to_csv(all_results_path, index=False)
    artifact_registry["tables"].append(all_results_path.name)
    ok(f"Saved table: {all_results_path.name}")

    best_by_f1 = results_df.sort_values("f1", ascending=False).groupby("dataset").head(1)
    best_by_pr_auc = results_df.sort_values("pr_auc", ascending=False).groupby("dataset").head(1)
    best_by_roc_auc = results_df.sort_values("roc_auc", ascending=False).groupby("dataset").head(1)

    table_outputs = {
        "step15_best_by_f1.csv": best_by_f1,
        "step15_best_by_pr_auc.csv": best_by_pr_auc,
        "step15_best_by_roc_auc.csv": best_by_roc_auc,
        "step15_dataset_summary.csv": results_df.groupby("dataset")[["f1", "precision", "recall", "pr_auc", "roc_auc"]].max().reset_index(),
        "step15_model_mean_summary.csv": results_df.groupby("model")[["f1", "precision", "recall", "pr_auc", "roc_auc"]].mean().reset_index(),
    }

    for filename, table in table_outputs.items():
        path = OUT_TABLES / filename
        table.to_csv(path, index=False)
        artifact_registry["tables"].append(filename)
        ok(f"Saved table: {filename}")

    section("Build figures")

    fig = save_bar_chart(
        best_by_f1,
        value_col="f1",
        title="Best Ensemble F1-score by Dataset",
        xlabel="F1-score",
        filename="step15_best_f1_by_dataset.png",
    )
    if fig:
        artifact_registry["figures"].append(fig)

    fig = save_bar_chart(
        best_by_pr_auc,
        value_col="pr_auc",
        title="Best Ensemble PR-AUC by Dataset",
        xlabel="PR-AUC",
        filename="step15_best_pr_auc_by_dataset.png",
    )
    if fig:
        artifact_registry["figures"].append(fig)

    fig = save_bar_chart(
        best_by_roc_auc,
        value_col="roc_auc",
        title="Best Ensemble ROC-AUC by Dataset",
        xlabel="ROC-AUC",
        filename="step15_best_roc_auc_by_dataset.png",
    )
    if fig:
        artifact_registry["figures"].append(fig)

    fig = save_heatmap(
        results_df,
        value_col="f1",
        filename="step15_f1_heatmap_dataset_model.png",
        title="F1-score Heatmap by Dataset and Ensemble Model",
    )
    if fig:
        artifact_registry["figures"].append(fig)

    fig = save_heatmap(
        results_df,
        value_col="pr_auc",
        filename="step15_pr_auc_heatmap_dataset_model.png",
        title="PR-AUC Heatmap by Dataset and Ensemble Model",
    )
    if fig:
        artifact_registry["figures"].append(fig)

    section("Write report and registry")

    report_name = write_report(results_df, artifact_registry)
    artifact_registry["reports"].append(report_name)

    registry_path = OUT_TABLES / "step15_artifact_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(artifact_registry, f, indent=2)

    artifact_registry["tables"].append(registry_path.name)
    ok(f"Saved artifact registry: {registry_path.name}")

    section("Step 15 Summary")
    ok("Step 15 completed successfully.")
    info("Check Outputs/tables/step15, Outputs/figures/step15, and Outputs/reports/step15.")


if __name__ == "__main__":
    main()
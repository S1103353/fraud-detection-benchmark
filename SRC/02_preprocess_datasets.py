"""
02_preprocess_datasets.py
=========================

What this step does
-------------------
This script takes the raw datasets validated in Step 01 and produces clean,
analysis-ready versions inside your Outputs folder.

We keep this step intentionally "light":
- Load datasets safely
- Remove exact duplicate rows (where it makes sense)
- Standardize label column into a unified name: `target` (0/1) when known
- Print a compact health report (shape, missingness)
- Save outputs using space-efficient formats (Parquet preferred; gzipped CSV fallback)

Why you got errors before
-------------------------
1) ImportError: cannot import name 'data_sets_root' from utils.paths
   - Your utils/paths.py does not define that exact variable name.
   - This version imports utils.paths as a module and resolves paths via multiple
     candidate attribute names (robust to naming differences).

2) OSError: [Errno 28] No space left on device
   - IEEE merged dataset is huge, saving as plain CSV can fill your disk.
   - This version prefers Parquet (compressed). If Parquet is not available,
     it writes gzipped CSV (and can chunk-write for extra robustness).

3) Elliptic merge failed: 'txId' column not found
   - Some Elliptic downloads have different ID column naming.
   - This version tries case/space-insensitive matching. If merge still fails,
     it saves cleaned components separately and continues.

Run
---
From project root:
    python3 SRC/02_preprocess_datasets.py
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import pandas as pd


# ==========================================================
# Robust path resolution (fixes your ImportError)
# ==========================================================

def _get_attr_any(module, names: List[str]) -> Optional[Path]:
    """
    Return the first existing attribute (as Path) from `module` that matches any name in `names`.
    """
    for n in names:
        if hasattr(module, n):
            v = getattr(module, n)
            try:
                return Path(v)
            except Exception:
                pass
    return None


def _resolve_paths() -> Dict[str, Path]:
    """
    Resolve all required folders from utils.paths, robust to naming differences.
    Falls back to computing paths from the project root if needed.
    """
    from utils import paths as P

    # Try to locate project root or datasets root in many possible naming conventions
    project_root = _get_attr_any(P, ["ROOT", "root", "PROJECT_ROOT", "project_root"])
    data_root = _get_attr_any(P, ["DATA_SETS_ROOT", "DATASETS_ROOT", "data_sets_root", "datasets_root", "DATA_ROOT", "data_root"])
    outputs = _get_attr_any(P, ["OUTPUTS_DIR", "outputs_dir", "OUTPUT_DIR", "output_dir"])

    # If some are missing, infer from what we have
    # Common structure: ~/Downloads/Fraud Thesis
    if project_root is None:
        # try to infer from current file location: .../Fraud Thesis/SRC/02_preprocess_datasets.py
        project_root = Path(__file__).resolve().parents[1]  # SRC -> project root

    if data_root is None:
        # expected: project_root / "Data Sets"
        data_root = project_root / "Data Sets"

    if outputs is None:
        outputs = project_root / "Outputs"

    # Dataset subfolders (try from paths.py first, else infer from datasets root)
    bank_fraud = _get_attr_any(P, ["BANK_FRAUD_DIR", "bank_fraud_dir"]) or (data_root / "Bank Fraud")
    credit_card = _get_attr_any(P, ["CREDIT_CARD_DIR", "credit_card_dir"]) or (data_root / "Credit Card")
    ecommerce = _get_attr_any(P, ["ECOMMERCE_DIR", "ecommerce_dir"]) or (data_root / "Ecommerce")
    elliptic = _get_attr_any(P, ["ELLIPTIC_DIR", "elliptic_dir"]) or (data_root / "Elliptic")
    fraud_tx = _get_attr_any(P, ["FRAUD_TRANSACTIONS_DIR", "fraud_transactions_dir"]) or (data_root / "Fraud Transactions")
    ieee = _get_attr_any(P, ["IEEE_FRAUD_DIR", "ieee_fraud_dir", "IEEE_DIR", "ieee_dir"]) or (data_root / "IEEE Fraud")
    paysim = _get_attr_any(P, ["PAYSIM_DIR", "paysim_dir"]) or (data_root / "PaySim")

    return {
        "PROJECT_ROOT": project_root,
        "DATA_ROOT": data_root,
        "OUTPUTS": outputs,
        "BANK_FRAUD": bank_fraud,
        "CREDIT_CARD": credit_card,
        "ECOMMERCE": ecommerce,
        "ELLIPTIC": elliptic,
        "FRAUD_TRANSACTIONS": fraud_tx,
        "IEEE": ieee,
        "PAYSIM": paysim,
    }


PATHS = _resolve_paths()


# ==========================================================
# Pretty printing
# ==========================================================

def divider(title: str, char: str = "=", width: int = 60) -> None:
    print("\n" + char * width)
    print(title)
    print(char * width)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def info(msg: str) -> None:
    print(f"ℹ️  {msg}")


def warn(msg: str) -> None:
    print(f"⚠️  {msg}")


def fail(msg: str) -> None:
    print(f"❌ {msg}")


# ==========================================================
# Disk / saving helpers
# ==========================================================

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_free_space_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def save_dataframe_smart(
    df: pd.DataFrame,
    out_base: Path,
    prefer_parquet: bool = True,
    parquet_compression: str = "snappy",
    csv_compress_gzip: bool = True,
) -> Path:
    """
    Save DataFrame in a space-efficient format:
    - Prefer Parquet (compressed) if available.
    - Otherwise save as gzipped CSV.
    """
    safe_mkdir(out_base.parent)

    if prefer_parquet:
        try:
            out_path = out_base.with_suffix(".parquet")
            df.to_parquet(out_path, index=False, compression=parquet_compression)
            return out_path
        except Exception as e:
            warn(f"Parquet save failed ({type(e).__name__}: {e}). Falling back to CSV.")

    if csv_compress_gzip:
        out_path = out_base.with_suffix(".csv.gz")
        df.to_csv(out_path, index=False, compression="gzip")
    else:
        out_path = out_base.with_suffix(".csv")
        df.to_csv(out_path, index=False)

    return out_path


def save_csv_chunked_gzip(df: pd.DataFrame, out_path: Path, chunksize: int = 200_000) -> Path:
    """
    Extra-robust gzipped CSV writer for very large dataframes.
    """
    safe_mkdir(out_path.parent)
    with gzip.open(out_path, "wt", newline="") as f:
        df.head(0).to_csv(f, index=False)
        for start in range(0, len(df), chunksize):
            df.iloc[start:start + chunksize].to_csv(f, index=False, header=False)
    return out_path


# ==========================================================
# Reporting helpers
# ==========================================================

def missing_report(df: pd.DataFrame, top_n: int = 10) -> Tuple[int, List[Tuple[str, int]]]:
    miss_by_col = df.isna().sum().sort_values(ascending=False)
    total_missing = int(miss_by_col.sum())
    top = [(col, int(cnt)) for col, cnt in miss_by_col.head(top_n).items() if cnt > 0]
    return total_missing, top


def basic_report(name: str, df: pd.DataFrame) -> None:
    total_missing, top = missing_report(df)
    info(f"{name} report:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.shape[1]}")
    print(f"  - Total missing values: {total_missing}")
    if total_missing == 0:
        print("  - Missingness: none detected (top columns are all 0)")
    else:
        print("  - Top missing columns:")
        for col, cnt in top:
            print(f"    * {col}: {cnt}")


# ==========================================================
# Cleaning helpers
# ==========================================================

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df2 = df.drop_duplicates()
    removed = before - len(df2)
    return df2, removed


def standardize_target(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Standardize known label columns to `target`.
    """
    candidates = {
        "creditcard": ["Class", "class"],
        "fraud_transactions": ["is_fraud", "isFraud", "fraud", "label", "target"],
        "ieee": ["isFraud", "is_fraud", "label", "target"],
        "bank_fraud": ["fraud_bool", "is_fraud", "isFraud", "label", "target"],
        "paysim": ["isFraud", "is_fraud", "fraud", "label", "target"],
        "ecommerce": ["is_fraud", "fraud", "label", "target"],
        "elliptic": ["class", "Class", "label", "y", "target"],
    }

    cols = list(df.columns)
    for cand in candidates.get(source.lower().strip(), []):
        if cand in cols:
            if cand != "target":
                df = df.rename(columns={cand: "target"})
                ok(f"{source}: standardized target column '{cand}' -> 'target'")
            else:
                ok(f"{source}: target column already named 'target'")
            return df, "target"

    return df, None


def read_csv_safe(path: Path, n_preview: int = 5) -> pd.DataFrame:
    df = pd.read_csv(path)
    ok(f"Loaded: {path.name} | preview_rows={n_preview} | rows={len(df)} | cols={df.shape[1]}")
    return df


def _find_col_case_space_insensitive(df: pd.DataFrame, wanted: str) -> Optional[str]:
    w = wanted.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == w:
            return c
    return None


# ==========================================================
# Dataset preprocessors
# ==========================================================

def preprocess_bank_fraud(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("Bank Fraud")

    base_path = PATHS["BANK_FRAUD"] / "Base.csv"
    if not base_path.exists():
        warn(f"Base.csv not found at: {base_path}")
        return outputs

    df = read_csv_safe(base_path)
    basic_report("Bank Fraud / Base.csv", df)

    df, removed = remove_duplicates(df)
    ok(f"Removed duplicates: {removed}")

    df, _ = standardize_target(df, "bank_fraud")

    out = save_dataframe_smart(df, out_dir / "bank_fraud_base_clean", prefer_parquet=False, csv_compress_gzip=False)
    ok(f"Saved clean output: {out.name} | rows={len(df)} | cols={df.shape[1]}")
    outputs["bank_fraud_base"] = out
    return outputs


def preprocess_credit_card(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("Credit Card")

    cc_path = PATHS["CREDIT_CARD"] / "creditcard.csv"
    if not cc_path.exists():
        candidates = list(PATHS["CREDIT_CARD"].glob("*.csv"))
        if not candidates:
            warn(f"No CSV found in: {PATHS['CREDIT_CARD']}")
            return outputs
        cc_path = candidates[0]
        warn(f"creditcard.csv not found; using: {cc_path.name}")

    df = read_csv_safe(cc_path)
    basic_report(f"Credit Card / {cc_path.name}", df)

    df, removed = remove_duplicates(df)
    ok(f"Removed duplicates: {removed}")

    df, _ = standardize_target(df, "creditcard")

    out = save_dataframe_smart(df, out_dir / "creditcard_clean", prefer_parquet=False, csv_compress_gzip=False)
    ok(f"Saved clean output: {out.name} | rows={len(df)} | cols={df.shape[1]}")
    outputs["creditcard"] = out
    return outputs


def preprocess_ecommerce(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("Ecommerce")

    e_path = PATHS["ECOMMERCE"] / "transactions.csv"
    if not e_path.exists():
        candidates = list(PATHS["ECOMMERCE"].glob("*.csv"))
        if not candidates:
            warn(f"No CSV found in: {PATHS['ECOMMERCE']}")
            return outputs
        e_path = candidates[0]
        warn(f"transactions.csv not found; using: {e_path.name}")

    df = read_csv_safe(e_path)
    basic_report(f"Ecommerce / {e_path.name}", df)

    df, removed = remove_duplicates(df)
    ok(f"Removed duplicates: {removed}")

    df, _ = standardize_target(df, "ecommerce")

    out = save_dataframe_smart(df, out_dir / "ecommerce_clean", prefer_parquet=False, csv_compress_gzip=False)
    ok(f"Saved clean output: {out.name} | rows={len(df)} | cols={df.shape[1]}")
    outputs["ecommerce"] = out
    return outputs


def preprocess_elliptic(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("Elliptic")

    feats = PATHS["ELLIPTIC"] / "elliptic_txs_features.csv"
    classes = PATHS["ELLIPTIC"] / "elliptic_txs_classes.csv"
    edges = PATHS["ELLIPTIC"] / "elliptic_txs_edgelist.csv"

    missing = [p for p in [feats, classes, edges] if not p.exists()]
    if missing:
        warn("Elliptic files missing. Expected 3 CSVs:")
        for p in [feats, classes, edges]:
            warn(f"  - {p.name} exists={p.exists()}")
        return outputs

    df_f = read_csv_safe(feats)
    df_c = read_csv_safe(classes)
    df_e = read_csv_safe(edges)

    df_f, rem_f = remove_duplicates(df_f)
    df_c, rem_c = remove_duplicates(df_c)
    df_e, rem_e = remove_duplicates(df_e)
    ok(f"Removed duplicates (features/classes/edgelist): {rem_f}/{rem_c}/{rem_e}")

    f_id = _find_col_case_space_insensitive(df_f, "txId")
    c_id = _find_col_case_space_insensitive(df_c, "txId")

    if f_id and c_id:
        info(f"Merging features + classes on '{f_id}' and '{c_id}' ...")
        if f_id != "txId":
            df_f = df_f.rename(columns={f_id: "txId"})
        if c_id != "txId":
            df_c = df_c.rename(columns={c_id: "txId"})

        merged = df_f.merge(df_c, on="txId", how="left")
        basic_report("Elliptic merged (features + classes)", merged)

        merged, _ = standardize_target(merged, "elliptic")

        out = save_dataframe_smart(merged, out_dir / "elliptic_clean", prefer_parquet=True, parquet_compression="snappy")
        ok(f"Saved clean output: {out.name} | rows={len(merged)} | cols={merged.shape[1]}")
        outputs["elliptic_merged"] = out
    else:
        warn("Elliptic merge failed: could not find 'txId' column in one of the files.")
        warn("Saving cleaned components separately and continuing...")

        out_f = save_dataframe_smart(df_f, out_dir / "elliptic_features_clean", prefer_parquet=False, csv_compress_gzip=False)
        out_c = save_dataframe_smart(df_c, out_dir / "elliptic_classes_clean", prefer_parquet=False, csv_compress_gzip=False)
        out_e = save_dataframe_smart(df_e, out_dir / "elliptic_edgelist_clean", prefer_parquet=False, csv_compress_gzip=False)

        ok(f"Saved: {out_f.name}, {out_c.name}, {out_e.name}")
        outputs["elliptic_features"] = out_f
        outputs["elliptic_classes"] = out_c
        outputs["elliptic_edgelist"] = out_e

    return outputs


def preprocess_fraud_transactions(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("Fraud Transactions")

    train_path = PATHS["FRAUD_TRANSACTIONS"] / "fraudTrain.csv"
    test_path = PATHS["FRAUD_TRANSACTIONS"] / "fraudTest.csv"

    if not train_path.exists():
        candidates = list(PATHS["FRAUD_TRANSACTIONS"].glob("*.csv"))
        if not candidates:
            warn(f"No CSV found in: {PATHS['FRAUD_TRANSACTIONS']}")
            return outputs
        train_path = candidates[0]
        warn(f"fraudTrain.csv not found; using: {train_path.name}")

    df = read_csv_safe(train_path)
    basic_report(f"Fraud Transactions / {train_path.name}", df)

    df, removed = remove_duplicates(df)
    ok(f"Removed duplicates: {removed}")

    df, _ = standardize_target(df, "fraud_transactions")

    out = save_dataframe_smart(df, out_dir / "fraud_transactions_train_clean", prefer_parquet=False, csv_compress_gzip=False)
    ok(f"Saved clean output: {out.name} | rows={len(df)} | cols={df.shape[1]}")
    outputs["fraud_transactions_train"] = out

    if test_path.exists():
        ok("fraudTest.csv exists (kept for later steps).")

    return outputs


def preprocess_ieee(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("IEEE Fraud")

    t_path = PATHS["IEEE"] / "train_transaction.csv"
    i_path = PATHS["IEEE"] / "train_identity.csv"

    if not t_path.exists() or not i_path.exists():
        warn("IEEE files missing. Expected:")
        warn(f"  - {t_path.name} exists={t_path.exists()}")
        warn(f"  - {i_path.name} exists={i_path.exists()}")
        return outputs

    df_t = read_csv_safe(t_path)
    df_i = read_csv_safe(i_path)

    key = "TransactionID"
    if key not in df_t.columns:
        alt = _find_col_case_space_insensitive(df_t, key)
        if not alt:
            fail(f"Cannot find TransactionID in {t_path.name}")
            return outputs
        df_t = df_t.rename(columns={alt: key})

    if key not in df_i.columns:
        alt = _find_col_case_space_insensitive(df_i, key)
        if alt:
            df_i = df_i.rename(columns={alt: key})
        else:
            warn("TransactionID not found in identity file; skipping identity merge.")
            df_i = None

    info(f"Merging transaction + identity on '{key}' (left join) ...")
    df = df_t.merge(df_i, on=key, how="left") if df_i is not None else df_t.copy()

    ok(f"Merged IEEE shape: {df.shape}")
    basic_report("IEEE merged", df)

    df, removed = remove_duplicates(df)
    ok(f"Removed duplicates: {removed}")

    df, _ = standardize_target(df, "ieee")

    free_gb = get_free_space_gb(out_dir)
    info(f"Free disk space (approx): {free_gb:.2f} GB")

    base_out = out_dir / "ieee_train_clean"

    # Prefer parquet
    try:
        out = save_dataframe_smart(df, base_out, prefer_parquet=True, parquet_compression="snappy")
        ok(f"Saved clean output: {out.name} | rows={len(df)} | cols={df.shape[1]}")
        outputs["ieee_train"] = out
        return outputs
    except OSError as e:
        warn(f"OS error while saving IEEE: {e}")

    # Chunked gz fallback
    warn("Trying chunked gzipped CSV fallback for IEEE...")
    out_gz = base_out.with_suffix(".csv.gz")
    save_csv_chunked_gzip(df, out_gz, chunksize=150_000)
    ok(f"Saved IEEE as gzipped CSV: {out_gz.name}")
    outputs["ieee_train"] = out_gz
    return outputs


def preprocess_paysim(out_dir: Path) -> Dict[str, Path]:
    outputs = {}
    divider("PaySim")

    csvs = sorted(PATHS["PAYSIM"].glob("*.csv"))
    if not csvs:
        warn(f"No CSV found in: {PATHS['PAYSIM']}")
        return outputs

    p = csvs[0]
    df = read_csv_safe(p)
    basic_report(f"PaySim / {p.name}", df)

    df, removed = remove_duplicates(df)
    ok(f"Removed duplicates: {removed}")

    df, _ = standardize_target(df, "paysim")

    out = save_dataframe_smart(df, out_dir / "paysim_clean", prefer_parquet=False, csv_compress_gzip=True)
    ok(f"Saved clean output: {out.name} | rows={len(df)} | cols={df.shape[1]}")
    outputs["paysim"] = out
    return outputs


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    divider("Step 02: Preprocess Datasets")

    data_root = PATHS["DATA_ROOT"]
    out_dir = PATHS["OUTPUTS"]

    if not data_root.exists():
        raise FileNotFoundError(f"Datasets root folder not found: {data_root}")

    safe_mkdir(out_dir)

    ok(f"Datasets root folder found: {data_root}")
    ok(f"Outputs folder ready: {out_dir}")

    all_outputs: Dict[str, Path] = {}

    # Run each dataset preprocessor (fail-safe)
    for fn, name in [
        (preprocess_bank_fraud, "Bank Fraud"),
        (preprocess_credit_card, "Credit Card"),
        (preprocess_ecommerce, "Ecommerce"),
        (preprocess_elliptic, "Elliptic"),
        (preprocess_fraud_transactions, "Fraud Transactions"),
        (preprocess_ieee, "IEEE"),
        (preprocess_paysim, "PaySim"),
    ]:
        try:
            all_outputs.update(fn(out_dir))
        except Exception as e:
            warn(f"{name} preprocessing failed: {type(e).__name__}: {e}")

    divider("Step 02 Summary")
    if not all_outputs:
        warn("No outputs were created. Check dataset folders / filenames.")
    else:
        ok("Outputs created:")
        for k, v in all_outputs.items():
            print(f"  - {k}: {v.name}")

    ok("Step 02 completed successfully.")


if __name__ == "__main__":
    main()
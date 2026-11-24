"""
SRC/01_load_and_validate.py

Step 01: Load & Validate Datasets

What this script does:
1) Verifies that all expected dataset folders exist under:
   <repo>/Data Sets/

2) Checks for key required files in each dataset folder (based on your current structure):
   - Bank Fraud: Base.csv + Variant I–V.csv (6 CSVs)
   - Credit Card: creditcard.csv
   - Ecommerce: transactions.csv
   - Elliptic: elliptic_txs_classes.csv, elliptic_txs_edgelist.csv, elliptic_txs_features.csv
   - Fraud Transactions: fraudTrain.csv, fraudTest.csv
   - IEEE Fraud: train_identity.csv, train_transaction.csv
   - PaySim: one CSV (whatever its long name is)

3) Runs quick pandas read checks for a few files (to ensure files are readable).
4) Prints a clean summary report and raises clear errors if something is missing.

How to run:
- From the project root folder "Fraud Thesis":
    python SRC/01_load_and_validate.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import sys

import pandas as pd

from utils.paths import (
    datasets_dir,
    bank_fraud_dir,
    credit_card_dir,
    ecommerce_dir,
    elliptic_dir,
    fraud_transactions_dir,
    ieee_fraud_dir,
    paysim_dir,
)


# -----------------------------
# Helpers
# -----------------------------
def must_exist(path: Path, label: str) -> None:
    """Raise a friendly error if a path does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing: {label}\nPath: {path}")


def find_exact(folder: Path, filename: str) -> Path:
    """Find a file with an exact name in the given folder."""
    p = folder / filename
    must_exist(p, f"File '{filename}'")
    return p


def find_glob(folder: Path, pattern: str, label: str) -> List[Path]:
    """Find files using a glob pattern and ensure at least one match."""
    matches = sorted(folder.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"❌ Missing: {label}\nPattern: {pattern}\nFolder: {folder}")
    return matches


def quick_read_csv(path: Path, nrows: int = 5) -> Tuple[int, int]:
    """
    Quick pandas read test to ensure a CSV is readable.
    Returns (nrows_read, ncols).
    """
    df = pd.read_csv(path, nrows=nrows)
    return (len(df), df.shape[1])


def print_section(title: str) -> None:
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


def warn(msg: str) -> None:
    print(f"⚠️ {msg}")


# -----------------------------
# Main validation
# -----------------------------
def validate_folders() -> None:
    print_section("Step 01: Load & Validate Datasets")

    # Root datasets folder
    must_exist(datasets_dir, "Datasets root folder 'Data Sets'")
    ok(f"Datasets root folder found: {datasets_dir}")

    # Individual dataset folders
    must_exist(bank_fraud_dir, "Bank Fraud folder")
    must_exist(credit_card_dir, "Credit Card folder")
    must_exist(ecommerce_dir, "Ecommerce folder")
    must_exist(elliptic_dir, "Elliptic folder")
    must_exist(fraud_transactions_dir, "Fraud Transactions folder")
    must_exist(ieee_fraud_dir, "IEEE Fraud folder")
    must_exist(paysim_dir, "PaySim folder")

    ok("All dataset folders exist.")


def validate_bank_fraud() -> List[Path]:
    print_section("Bank Fraud")

    # Must contain Base.csv and Variant I–V.csv
    base = find_exact(bank_fraud_dir, "Base.csv")
    variants = [
        find_exact(bank_fraud_dir, "Variant I.csv"),
        find_exact(bank_fraud_dir, "Variant II.csv"),
        find_exact(bank_fraud_dir, "Variant III.csv"),
        find_exact(bank_fraud_dir, "Variant IV.csv"),
        find_exact(bank_fraud_dir, "Variant V.csv"),
    ]

    ok(f"Base.csv found: {base.name}")
    ok(f"Variants found: {len(variants)}")

    # Quick read (just a few)
    for p in [base, variants[0], variants[1]]:
        r, c = quick_read_csv(p, nrows=5)
        ok(f"Quick read OK: {p.name} | rows={r} | cols={c}")

    return [base] + variants


def validate_credit_card() -> Path:
    print_section("Credit Card")

    # Your screenshot shows: creditcard.csv
    p = find_exact(credit_card_dir, "creditcard.csv")
    r, c = quick_read_csv(p, nrows=5)
    ok(f"Found: {p.name}")
    ok(f"Quick read OK: {p.name} | rows={r} | cols={c}")
    return p


def validate_ecommerce() -> Path:
    print_section("Ecommerce")

    # Your screenshot shows: transactions.csv
    p = find_exact(ecommerce_dir, "transactions.csv")
    r, c = quick_read_csv(p, nrows=5)
    ok(f"Found: {p.name}")
    ok(f"Quick read OK: {p.name} | rows={r} | cols={c}")
    return p


def validate_elliptic() -> List[Path]:
    print_section("Elliptic")

    classes = find_exact(elliptic_dir, "elliptic_txs_classes.csv")
    edgelist = find_exact(elliptic_dir, "elliptic_txs_edgelist.csv")
    features = find_exact(elliptic_dir, "elliptic_txs_features.csv")

    ok("Found all 3 Elliptic CSV files.")

    for p in [classes, edgelist, features]:
        r, c = quick_read_csv(p, nrows=5)
        ok(f"Quick read OK: {p.name} | rows={r} | cols={c}")

    return [classes, edgelist, features]


def validate_fraud_transactions() -> List[Path]:
    print_section("Fraud Transactions")

    # Your screenshot shows: fraudTrain.csv and fraudTest.csv
    train = find_exact(fraud_transactions_dir, "fraudTrain.csv")
    test = find_exact(fraud_transactions_dir, "fraudTest.csv")

    for p in [train, test]:
        r, c = quick_read_csv(p, nrows=5)
        ok(f"Quick read OK: {p.name} | rows={r} | cols={c}")

    return [train, test]


def validate_ieee_fraud() -> List[Path]:
    print_section("IEEE Fraud")

    # Your screenshot shows: train_identity.csv and train_transaction.csv
    identity = find_exact(ieee_fraud_dir, "train_identity.csv")
    transaction = find_exact(ieee_fraud_dir, "train_transaction.csv")

    # Quick read
    for p in [identity, transaction]:
        r, c = quick_read_csv(p, nrows=5)
        ok(f"Quick read OK: {p.name} | rows={r} | cols={c}")

    return [identity, transaction]


def validate_paysim() -> Path:
    print_section("PaySim")

    # Your PaySim file has a long name. We just ensure there is at least 1 CSV.
    csvs = find_glob(paysim_dir, "*.csv", "At least one PaySim CSV file")
    ok(f"Found {len(csvs)} CSV file(s) in PaySim folder.")
    chosen = csvs[0]

    # Quick read only for the first CSV (safe)
    try:
        r, c = quick_read_csv(chosen, nrows=5)
        ok(f"Quick read OK: {chosen.name} | rows={r} | cols={c}")
    except Exception as e:
        warn(f"CSV found but quick read failed for: {chosen.name}\nReason: {e}")

    return chosen


def main() -> None:
    try:
        validate_folders()

        bank_files = validate_bank_fraud()
        credit_file = validate_credit_card()
        ecommerce_file = validate_ecommerce()
        elliptic_files = validate_elliptic()
        ft_files = validate_fraud_transactions()
        ieee_files = validate_ieee_fraud()
        paysim_file = validate_paysim()

        print_section("Summary")
        print(f"- Data Sets root: {datasets_dir}")
        print(f"- Bank Fraud files: {len(bank_files)}")
        print(f"- Credit Card file: {credit_file.name}")
        print(f"- Ecommerce file: {ecommerce_file.name}")
        print(f"- Elliptic files: {len(elliptic_files)}")
        print(f"- Fraud Transactions files: {len(ft_files)}")
        print(f"- IEEE Fraud files: {len(ieee_files)}")
        print(f"- PaySim file (first found): {paysim_file.name}")

        print("\n✅ Step 01 completed successfully.")
    except Exception as e:
        print("\n" + "-" * 60)
        print("❌ Step 01 failed.")
        print(str(e))
        print("-" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
01_load_and_validate.py
-----------------------
Dataset presence + quick integrity validation.

What this script does:
- Locates dataset folders using SRC/utils/paths.py (robust path resolution).
- Verifies the "Bank Fraud" dataset folder exists.
- Searches for key CSV files (Base.csv, Variant*.csv) inside that folder.
- Prints a short summary (found files + quick pandas read checks) to confirm
  the pipeline can safely continue.

How to run:
- From VSCode or Terminal:
    python SRC/01_load_and_validate.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

from utils.paths import get_dataset_paths


def _find_csvs(root: Path) -> list[Path]:
    """Recursively find CSV files under a root folder."""
    return sorted(root.rglob("*.csv"))


def _quick_read_check(csv_path: Path, nrows: int = 5) -> None:
    """
    Minimal read test to detect encoding/delimiter issues early.
    Adjust sep/encoding here if you ever have a weird CSV.
    """
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"  - OK read: {csv_path.name} | columns={df.shape[1]} (preview rows={df.shape[0]})")


def main() -> None:
    print("\n=== Step 01: Load & Validate Datasets ===")

    paths = get_dataset_paths()
    bank_dir = paths.bank_fraud_dir

    print(f"\nResolved dataset folder:")
    print(f"  Bank Fraud: {bank_dir}")

    # List CSV files
    csvs = _find_csvs(bank_dir)
    print(f"\nFound {len(csvs)} CSV file(s) under Bank Fraud.")

    if not csvs:
        print("\n❌ No CSV files found inside 'Bank Fraud'.")
        print("Check that you extracted/downloaded the dataset files correctly.")
        sys.exit(1)

    # Identify key files (flexible)
    names = [p.name.lower() for p in csvs]
    base_candidates = [p for p in csvs if p.name.lower() == "base.csv"]
    variant_candidates = [p for p in csvs if p.name.lower().startswith("variant") and p.suffix.lower() == ".csv"]

    print("\nKey file checks:")
    print(f"  Base.csv found: {len(base_candidates)}")
    print(f"  Variant*.csv found: {len(variant_candidates)}")

    # If Base.csv isn't directly found, show closest suggestions
    if not base_candidates:
        print("\n⚠️ Base.csv not found by exact name.")
        print("Here are the first 15 CSV files I do see:")
        for p in csvs[:15]:
            print(f"  - {p.relative_to(bank_dir)}")

    # Quick read checks (only a few files to keep it fast)
    print("\nQuick read tests (pandas):")
    sample_paths = []
    if base_candidates:
        sample_paths.append(base_candidates[0])
    sample_paths += variant_candidates[:2]  # first 2 variants if present
    if not sample_paths:
        sample_paths = csvs[:3]

    for p in sample_paths:
        try:
            _quick_read_check(p, nrows=5)
        except Exception as e:
            print(f"  - ❌ Failed read: {p.name} | error={type(e).__name__}: {e}")
            print("Tip: If this is a delimiter/encoding issue, we can set sep/encoding explicitly.")
            sys.exit(1)

    print("\n✅ Step 01 completed successfully.\n")


if __name__ == "__main__":
    main()
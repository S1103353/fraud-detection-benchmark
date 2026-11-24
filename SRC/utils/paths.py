"""
utils/paths.py

Centralized, user-friendly path management for the Fraud Thesis project.
This file defines:
- Project root (auto-detected)
- Dataset directories (based on your Finder folder names)
- Convenience variables (bank_fraud_dir, etc.) for easy imports
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """Container for dataset directories."""
    root_dir: Path
    datasets_dir: Path

    bank_fraud_dir: Path
    credit_card_dir: Path
    ecommerce_dir: Path
    elliptic_dir: Path
    fraud_transactions_dir: Path
    ieee_fraud_dir: Path
    paysim_dir: Path


def get_project_root() -> Path:
    """
    Auto-detect the project root folder.

    Expected structure:
    Fraud Thesis/
      Data Sets/
      SRC/
        utils/
          paths.py   <-- this file
    """
    # paths.py -> utils -> SRC -> Fraud Thesis
    return Path(__file__).resolve().parents[2]


def get_dataset_paths() -> DatasetPaths:
    """
    Build dataset paths using the exact folder names in your Finder.
    """
    root = get_project_root()
    datasets = root / "Data Sets"

    return DatasetPaths(
        root_dir=root,
        datasets_dir=datasets,

        bank_fraud_dir=datasets / "Bank Fraud",
        credit_card_dir=datasets / "Credit Card",
        ecommerce_dir=datasets / "Ecommerce",
        elliptic_dir=datasets / "Elliptic",
        fraud_transactions_dir=datasets / "Fraud Transactions",
        ieee_fraud_dir=datasets / "IEEE Fraud",
        paysim_dir=datasets / "PaySim",
    )


# --- Backward-compatible convenience variables (for simple imports) ---
_paths = get_dataset_paths()

root_dir = _paths.root_dir
datasets_dir = _paths.datasets_dir

bank_fraud_dir = _paths.bank_fraud_dir
credit_card_dir = _paths.credit_card_dir
ecommerce_dir = _paths.ecommerce_dir
elliptic_dir = _paths.elliptic_dir
fraud_transactions_dir = _paths.fraud_transactions_dir
ieee_fraud_dir = _paths.ieee_fraud_dir
paysim_dir = _paths.paysim_dir
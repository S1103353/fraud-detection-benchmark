# Data Characteristics Over Model Choice: Fraud Detection Benchmark

Code and result tables for the paper

> **Data Characteristics Over Model Choice: A Six-Dataset Benchmark of Tree Ensembles, Imbalance Handling, Calibration and the Prevalence Trap in Financial Fraud Detection**
> Narges Maleki and Nhung Dinh, BI Norwegian Business School (2026).

The paper is based on our MSc thesis at BI Norwegian Business School (supervisor: Vegard Vinje).

## Main findings

* **The dataset matters more than the model.** Across tree models, 92.5% of the variance in PR-AUC lies between datasets.
* **Calibration and threshold selection give the largest gains.** Isotonic calibration of a cost-sensitive XGBoost model on a held-out calibration split raised test F1 at threshold 0.5:
  * IEEE-CIS: 0.38 to 0.60
  * PaySim: 0.42 to 0.89
* **Resampling and stacking rarely help.** SMOTE and ADASYN rarely helped a well-configured booster, and stacking was not worth its cost.
* **The prevalence trap.** Evaluating on fraud-enriched samples overstates F1 by about 0.2–0.5. The paper gives a closed-form correction to the true prevalence.

## Datasets (public, not included)

Download each dataset and place it under `Data Sets/` in the folders below:

| Dataset | Source | Folder |
|---|---|---|
| Credit Card Fraud (ULB) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `Data Sets/Credit Card` |
| E-commerce Fraud | https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset | `Data Sets/Ecommerce` |
| IEEE-CIS Fraud Detection | https://www.kaggle.com/competitions/ieee-fraud-detection | `Data Sets/IEEE Fraud` |
| Bank Account Fraud (BAF, Base) | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | `Data Sets/Bank Fraud` |
| Credit Card Transactions (Sparkov) | https://www.kaggle.com/datasets/kartik2112/fraud-detection | `Data Sets/Fraud Transactions` |
| PaySim | https://www.kaggle.com/datasets/ealaxi/paysim1 | `Data Sets/PaySim` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline (run from the repository root, in order)

| Step | Script | What it does |
|---|---|---|
| 00–01 | `SRC/00_make_folders.py`, `SRC/01_load_and_validate.py` | Create the folder structure and check the raw files |
| 02 | `SRC/02_preprocess_datasets.py` | Cleaning, deduplication, imputation and encoding. Writes `*_clean.csv` |
| 03–04 | `03_train_models_all.py`, `04_evaluate_models.py` | Logistic-regression baselines |
| 12 | `12_train_advanced_models_xai_FULL_DATA.py` | RF, Extra Trees, HistGB, XGBoost and LightGBM on full data (80/20 split), plus SHAP |
| 13 | `13_threshold_tuning_and_calibration_FULL_DATA.py` | Cost-sensitive learner, isotonic calibration and threshold selection (60/20/20 split) |
| 14 | `14_imbalance_resampling_comparison.py` | No treatment, class weights, SMOTE and ADASYN |
| 15 | `15_ensemble_voting_stacking_FULL_DATA.py` | Soft voting, weighted voting and stacking |
| 20–21 | `20_eda_feature_target_correlation.py`, `21_missing_value_summary.py` | Exploratory analysis |

The random seed is 42 everywhere. Results are written to `Outputs/`.

## Reproducing the paper's tables and figures

The result tables we obtained are in `results/`. The paper's figures, LaTeX table rows and prevalence correction can be regenerated without rerunning the models:

```bash
cd analysis
python analysis.py   # variance decomposition and prevalence correction -> projection.csv
python figs.py       # Figures 1-3
python tables.py     # LaTeX rows for Tables 2-7 -> gen_tables.txt
```

## Notes and limitations

* All results come from a single stratified split with seed 42. There are no confidence intervals.
* Thresholds tuned on the test set in step 15 are reported only as oracle upper bounds.
* In the Sparkov data, a row-index column (`Unnamed: 0`) was kept as a feature. SHAP flags it, and it should be dropped in future runs.

## Citation

```bibtex
@misc{maleki2026fraud,
  title  = {Data Characteristics Over Model Choice: A Six-Dataset Benchmark of Tree Ensembles,
            Imbalance Handling, Calibration and the Prevalence Trap in Financial Fraud Detection},
  author = {Maleki, Narges and Dinh, Nhung},
  year   = {2026}
}
```

## License

MIT

# circRNA bladder cancer: machine learning pipeline

Reproducible, leakage-free machine learning analyses accompanying the manuscript
**"Urinary circRNA Signatures for Non-Invasive Bladder Cancer Diagnosis: An
Exploratory Machine Learning Study"**.

The manuscript's machine learning sections (Methods: *Predictive Modeling of
Clinical Outcomes Using circRNA Expression Profiles*; Results 3.5; Figures 4–5)
report results produced by the four scripts in this repository. All analyses
are deterministic: every stochastic step uses a fixed random seed (42),
including the mutual-information feature selector.

## Scripts

- **`run_analysis.py`**: Main pipeline. KNN imputation (k = 5),
  standardization, and mutual-information feature selection (top 7) are fitted
  *inside* each cross-validation fold to prevent data leakage. Classifiers:
  logistic regression, random forest (300 trees), RBF-kernel SVM, and XGBoost
  (200 trees, depth 3), all with fixed hyperparameters. Evaluation: repeated
  stratified 5-fold cross-validation (20 repetitions) and leave-one-out
  cross-validation (ROC AUC, accuracy, F1). Permutation-based feature
  importance (50 permutations) for the best model. Exploratory three-tier EAU
  risk classification among cancer cases (LOOCV, hypothesis-generating).
- **`run_advanced_analysis.py`**: Elastic-net logistic regression with
  hyperparameters (C, L1 ratio) selected by nested LOOCV; bootstrap 95 %
  confidence interval (5000 resamples) for the LOOCV AUC; sample-size
  projection for a follow-up study (AUC → Cohen's d, 80 % power, α = 0.05).
- **`run_permutation_test.py`**: Label-shuffling permutation test (1000
  shuffles) for the random forest and XGBoost pipelines; empirical p-value
  with the +1 correction (Davison & Hinkley 1997). Runs in parallel across all
  CPU cores (~40 min on 20 cores).
- **`run_stability_selection.py`**: Bootstrap stability selection
  (Meinshausen & Bühlmann 2010): 1000 stratified bootstrap resamples with two
  independent selectors: elastic-net non-zero coefficients and random forest
  top-7 by importance.

## Requirements

Python 3.10+ with:

```bash
pip install numpy pandas scikit-learn xgboost scipy statsmodels joblib matplotlib
```

## Data

The analysis input is `resource/circ_edited.csv`: 15 circRNA 2^-ΔΔCt columns
plus age and gender; rows with missing EAU risk classification are dropped
(n = 41: 20 controls, 21 cancer cases). The dataset contains patient
identifiers and **is not shared in this repository**. To run the scripts, place
the file as a sibling of this folder:

```
your_repo/
├── circRNA_ML_scripts/   (or whatever this folder is named)
│   ├── run_analysis.py
│   ├── run_advanced_analysis.py
│   ├── run_permutation_test.py
│   └── run_stability_selection.py
└── resource/
    └── circ_edited.csv
```

## How to run

From a directory that contains both this folder and `resource/`:

```bash
python3 circRNA_ML_scripts/run_analysis.py
python3 circRNA_ML_scripts/run_advanced_analysis.py
python3 circRNA_ML_scripts/run_permutation_test.py
python3 circRNA_ML_scripts/run_stability_selection.py
```

Outputs are written inside the folder:

- `run_analysis.py` → `results/`, `figures/`, `logs/`
- `run_advanced_analysis.py` → `advanced_results/{results,figures,logs}`
- `run_permutation_test.py` → `permutation_results/{results,figures,logs}`
- `run_stability_selection.py` → `stability_results/{results,figures,logs}`

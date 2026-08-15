"""
circRNA ML re-analysis — feature stability selection.

A single model fit's "feature importance" (elastic-net coefficients in
run_advanced_analysis.py, RF permutation importance in run_analysis.py) is
not trustworthy at n=44 — it can flip with a different random_state or a
handful of dropped samples. Stability selection (Meinshausen & Buhlmann,
2010) fixes this the cheap way: refit on many bootstrap resamples of the
data and count how often each feature gets selected. A feature selected in
80% of resamples is a real candidate; one selected in 15% is noise that
happened to fit once.

Two independent selectors, both already used elsewhere in ML_circRNA_v2:
  - Elastic-Net logistic regression: "selected" = nonzero coefficient
    (fixed hyperparameters from run_advanced_analysis.py's nested-LOOCV
    median: C=0.359, l1_ratio=0.10 — refitting the inner CV inside every
    bootstrap would be slow for no real benefit here).
  - RandomForest: "selected" = among the top-7 features by importance
    (same k=7 used throughout ML_circRNA_v2 for SelectKBest).

Bootstraps are stratified (resample within each class) so every resample
keeps roughly the 20/21 control/cancer balance, and are run in parallel
across CPU cores.

Run from repo root: python3 ML_circRNA_v2/run_stability_selection.py
"""
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "resource" / "circ_edited.csv"
OUT = ROOT / "stability_results"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"
for _d in (RESULTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_BOOTSTRAP = 1000
K_FEATURES = 7
ELASTICNET_C = 0.359       # median from run_advanced_analysis.py's nested LOOCV
ELASTICNET_L1_RATIO = 0.10  # median from run_advanced_analysis.py's nested LOOCV
N_JOBS = -1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def load_data():
    df = pd.read_csv(DATA)
    df.columns.values[1] = "risk_score"
    df.rename(columns={"EAU RISK ": "eau"}, inplace=True)
    df["eau"] = df["eau"].str.replace("yok", "control")

    ddct_cols = [c for c in df.columns if "^" in c]
    start = df.columns.get_loc("circ0000326")
    end = df.columns.get_loc("label")
    gene_names = [c for c in df.columns[start:end] if c != "circ0000471"]
    assert len(ddct_cols) == len(gene_names)

    expr = df[ddct_cols].copy()
    expr.columns = gene_names
    expr["AGE"] = df["AGE"]
    expr["GENDER_2"] = df["GENDER_2"]

    out = pd.concat([expr, df[["eau"]]], axis=1).dropna(subset=["eau"])
    feature_cols = gene_names + ["AGE", "GENDER_2"]
    return out, feature_cols


def _bootstrap_indices(y, seed):
    # stratified resample-with-replacement: draw within each class so
    # every bootstrap keeps roughly the observed 20/21 class balance.
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y.values == cls)
        idx.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(idx)


def _one_bootstrap_elasticnet(seed, X, y, feature_cols):
    idx = _bootstrap_indices(y, seed)
    pipe = Pipeline([
        ("impute", KNNImputer(n_neighbors=5, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=ELASTICNET_C, l1_ratio=ELASTICNET_L1_RATIO, penalty="elasticnet",
            solver="saga", max_iter=5000, class_weight="balanced",
            random_state=seed,
        )),
    ])
    try:
        pipe.fit(X.iloc[idx], y.iloc[idx])
        coef = pipe.named_steps["clf"].coef_[0]
        return (np.abs(coef) > 1e-6).astype(int)
    except Exception:
        return np.zeros(len(feature_cols), dtype=int)


def _one_bootstrap_rf(seed, X, y, feature_cols):
    idx = _bootstrap_indices(y, seed)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
    rf.fit(X.iloc[idx], y.iloc[idx])
    top_k = np.argsort(rf.feature_importances_)[::-1][:K_FEATURES]
    selected = np.zeros(len(feature_cols), dtype=int)
    selected[top_k] = 1
    return selected


def run_stability(X, y, feature_cols):
    log.info("Elastic-Net stability selection: %d bootstraps (C=%.3f, l1_ratio=%.2f) ...",
              N_BOOTSTRAP, ELASTICNET_C, ELASTICNET_L1_RATIO)
    t0 = time.time()
    en_selected = Parallel(n_jobs=N_JOBS)(
        delayed(_one_bootstrap_elasticnet)(seed, X, y, feature_cols)
        for seed in range(N_BOOTSTRAP)
    )
    en_selected = np.array(en_selected)
    n_failed = int(np.sum(en_selected.sum(axis=1) == 0))
    if n_failed:
        log.warning("Elastic-Net: %d/%d bootstrap fits failed and were skipped (all-zero rows)",
                    n_failed, N_BOOTSTRAP)
    en_freq = en_selected.mean(axis=0)
    log.info("Elastic-Net stability done in %.1fs", time.time() - t0)

    log.info("RandomForest stability selection: %d bootstraps (top-%d by importance) ...",
              N_BOOTSTRAP, K_FEATURES)
    t0 = time.time()
    rf_selected = Parallel(n_jobs=N_JOBS)(
        delayed(_one_bootstrap_rf)(seed, X, y, feature_cols)
        for seed in range(N_BOOTSTRAP)
    )
    rf_freq = np.mean(rf_selected, axis=0)
    log.info("RandomForest stability done in %.1fs", time.time() - t0)

    df = pd.DataFrame({
        "Feature": feature_cols,
        "ElasticNet_selection_freq": en_freq,
        "RandomForest_top7_freq": rf_freq,
    }).sort_values("ElasticNet_selection_freq", ascending=False)
    df["mean_freq"] = df[["ElasticNet_selection_freq", "RandomForest_top7_freq"]].mean(axis=1)
    df = df.sort_values("mean_freq", ascending=False)
    df.to_csv(RESULTS / "feature_stability.csv", index=False)
    log.info("stability table:\n%s", df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = df.set_index("Feature")[["ElasticNet_selection_freq", "RandomForest_top7_freq"]].iloc[::-1]
    plot_df.plot.barh(ax=ax, width=0.75)
    ax.set_xlabel(f"Selection frequency across {N_BOOTSTRAP} bootstrap resamples")
    ax.set_xlim(0, 1)
    ax.set_title("Feature stability selection — control vs cancer")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_stability.png", dpi=300)
    plt.close(fig)
    log.info("saved figures/feature_stability.png")
    return df


def main():
    t0 = time.time()
    df, feature_cols = load_data()
    X = df[feature_cols]
    y = (df["eau"] != "control").astype(int)
    log.info("Loaded %d samples (%d cancer, %d control), %d features",
              len(y), y.sum(), (1 - y).sum(), len(feature_cols))

    stability_df = run_stability(X, y, feature_cols)

    robust = stability_df[stability_df["mean_freq"] >= 0.7]
    log.info("=" * 70)
    if len(robust):
        log.info("Robust candidates (selected in >=70%% of bootstraps by both methods' average):\n%s",
                  robust[["Feature", "mean_freq"]].to_string(index=False))
    else:
        log.info("No feature reached the >=70%% stability bar — no single circRNA "
                  "stands out as a robust candidate at this sample size.")
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()

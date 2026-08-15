"""
circRNA ML re-analysis — permutation significance test.

Answers the question the bootstrap CI in run_advanced_analysis.py can't:
is the RandomForest/XGBoost LOOCV AUC (~0.65 / ~0.62, from run_analysis.py)
actually better than chance, for THIS pipeline on THIS data? We shuffle the
class labels N times, rerun the exact same leakage-free LOOCV pipeline on
each shuffle, and see where the real (unshuffled) AUC falls in that null
distribution. Gold-standard significance check for classifier performance
at small n — a label shuffle destroys any true signal but preserves the
pipeline's own capacity to overfit, so the null is fair.

Not a GPU job: every fit here is on ~40 rows x 17 columns — GPU transfer
overhead would dominate the tiny matmuls, and RandomForest/LogReg/SVM don't
run on GPU at all in scikit-learn. The real speedup is CPU parallelism
across permutations (independent draws), via joblib.

Run from repo root: python3 ML_circRNA_v2/run_permutation_test.py
"""
import logging
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "resource" / "circ_edited.csv"
OUT = ROOT / "permutation_results"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"
for _d in (RESULTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_PERMUTATIONS = 1000
K_FEATURES = 7
N_JOBS = -1  # all cores; 20 available on this machine

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


def build_models():
    # same fixed hyperparameters as run_analysis.py's best two models — no
    # per-permutation hyperparameter search, both to keep this test honest
    # (search would need to be nested inside every permutation too) and fast.
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(clf):
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(partial(mutual_info_classif, random_state=RANDOM_STATE), k=K_FEATURES)),
        ("clf", clf),
    ])


def loocv_auc(X, y, clf):
    loo = LeaveOneOut()
    y_true, y_proba = [], []
    for train_idx, test_idx in loo.split(X, y):
        pipe = make_pipeline(clf)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_proba.append(pipe.predict_proba(X.iloc[test_idx])[0, 1])
        y_true.append(y.iloc[test_idx])
    return roc_auc_score(y_true, y_proba)


def _one_permutation(seed, X, y, model_name, clf):
    rng = np.random.default_rng(seed)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    return loocv_auc(X, y_shuffled, clf)


def run_permutation_test(X, y, model_name, clf):
    log.info("-" * 70)
    log.info("%s: observed LOOCV AUC", model_name)
    t0 = time.time()
    observed_auc = loocv_auc(X, y, clf)
    log.info("observed AUC = %.4f  (%.1fs)", observed_auc, time.time() - t0)

    log.info("%s: running %d label-shuffled permutations on %s cores ...",
              model_name, N_PERMUTATIONS, N_JOBS if N_JOBS > 0 else "all")
    t0 = time.time()
    null_aucs = Parallel(n_jobs=N_JOBS)(
        delayed(_one_permutation)(seed, X, y, model_name, clf)
        for seed in range(N_PERMUTATIONS)
    )
    null_aucs = np.array(null_aucs)
    log.info("%s: permutations done in %.1fs", model_name, time.time() - t0)

    # standard empirical p-value with +1 correction (Davison & Hinkley 1997) —
    # avoids reporting p=0 when the observed value beats every permutation.
    p_value = (1 + np.sum(null_aucs >= observed_auc)) / (N_PERMUTATIONS + 1)
    log.info("%s: null AUC mean=%.3f sd=%.3f  |  observed=%.3f  |  p=%.4f",
              model_name, null_aucs.mean(), null_aucs.std(), observed_auc, p_value)

    pd.DataFrame({"null_auc": null_aucs}).to_csv(
        RESULTS / f"null_distribution_{model_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_aucs, bins=40, color="#999999", label=f"null (n={N_PERMUTATIONS} label shuffles)")
    ax.axvline(observed_auc, color="crimson", linewidth=2,
               label=f"observed AUC={observed_auc:.2f} (p={p_value:.4f})")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1, label="chance (0.5)")
    ax.set_xlabel("LOOCV ROC AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test — {model_name} (control vs cancer)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"permutation_null_{model_name}.png", dpi=300)
    plt.close(fig)

    return {"Model": model_name, "Observed_AUC": observed_auc,
            "Null_mean": null_aucs.mean(), "Null_sd": null_aucs.std(),
            "p_value": p_value, "n_permutations": N_PERMUTATIONS}


def main():
    t0 = time.time()
    df, feature_cols = load_data()
    X = df[feature_cols]
    y = (df["eau"] != "control").astype(int)
    log.info("Loaded %d samples (%d cancer, %d control)", len(y), y.sum(), (1 - y).sum())

    rows = [run_permutation_test(X, y, name, clf) for name, clf in build_models().items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "permutation_test_summary.csv", index=False)
    log.info("=" * 70)
    log.info("SUMMARY\n%s", summary.to_string(index=False))
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()

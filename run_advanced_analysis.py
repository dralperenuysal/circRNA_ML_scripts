"""
circRNA ML re-analysis — advanced add-on.

Not "a more complex model" (n=44 punishes complexity — see run_analysis.py,
where flexible models like RandomForest/SVM/XGBoost lose most of their
single-split performance under LOOCV). Instead this script adds rigor around
the same binary target (control vs cancer):

  1. Elastic-Net logistic regression, nested inside LOOCV (inner CV picks the
     regularization strength/L1 ratio on the training fold only — no leakage,
     no separate feature-selection step needed since the L1 term does that).
  2. Bootstrap 95% CI on the LOOCV ROC AUC — a point estimate like "AUC=0.65"
     is meaningless at n=44 without an interval around it.
  3. Decision curve analysis — does using the model beat "treat everyone" /
     "treat no one" at any clinically plausible risk threshold?
  4. Sample-size projection — given the effect size actually observed here,
     how many patients would a follow-up study need for 80% power? This is
     the number to put in a grant/protocol, not a bigger model.

Run from repo root: python3 ML_circRNA_v2/run_advanced_analysis.py
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
from scipy.stats import norm
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.power import NormalIndPower

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "resource" / "circ_edited.csv"
OUT = ROOT / "advanced_results"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
LOGS = OUT / "logs"
for _d in (RESULTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_BOOTSTRAP = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# data loading (same convention as run_analysis.py)
# --------------------------------------------------------------------------
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
    log.info("Loaded %d samples, %d features", len(out), len(feature_cols))
    return out, feature_cols


def make_pipeline():
    # elastic net's L1 term does feature selection on its own — no separate
    # SelectKBest step, so there's one less leakage-prone stage to fit inside
    # each fold and one less arbitrary "k" to pick.
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf = LogisticRegressionCV(
        Cs=10,
        cv=inner_cv,
        penalty="elasticnet",
        l1_ratios=[0.1, 0.5, 0.7, 0.9, 1.0],
        solver="saga",
        max_iter=5000,
        scoring="roc_auc",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


# --------------------------------------------------------------------------
# 1+2: nested-LOOCV elastic net + bootstrap AUC CI
# --------------------------------------------------------------------------
def run_elasticnet_loocv(df, feature_cols):
    log.info("=" * 70)
    log.info("ELASTIC-NET LOGISTIC REGRESSION — nested LOOCV")
    X = df[feature_cols]
    y = (df["eau"] != "control").astype(int)

    loo = LeaveOneOut()
    y_true, y_proba = [], []
    chosen_l1_ratio, chosen_C = [], []
    for train_idx, test_idx in loo.split(X, y):
        pipe = make_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_proba.append(pipe.predict_proba(X.iloc[test_idx])[0, 1])
        y_true.append(y.iloc[test_idx].values[0])
        clf = pipe.named_steps["clf"]
        chosen_l1_ratio.append(clf.l1_ratio_[0])
        chosen_C.append(clf.C_[0])
    y_true, y_proba = np.array(y_true), np.array(y_proba)

    auc = roc_auc_score(y_true, y_proba)
    log.info("LOOCV ROC AUC = %.3f", auc)
    log.info("inner-CV hyperparameter picks: l1_ratio median=%.2f, C median=%.3g",
              np.median(chosen_l1_ratio), np.median(chosen_C))

    # bootstrap CI: resample the (y_true, y_proba) pairs from LOOCV, not the
    # raw data — refitting 5000x would mean 5000x44 model fits, unnecessary
    # given LOOCV already gives one honest out-of-fold prediction per subject.
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    boot_aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    log.info("Bootstrap 95%% CI (n=%d resamples): [%.3f, %.3f]", len(boot_aucs), ci_lo, ci_hi)

    # final model refit on all data, for coefficient inspection only
    # (interpretation, not a held-out test).
    final_pipe = make_pipeline()
    final_pipe.fit(X, y)
    coefs = pd.Series(final_pipe.named_steps["clf"].coef_[0], index=feature_cols)
    coef_df = coefs.sort_values(key=np.abs, ascending=False).rename("coefficient").to_frame()
    coef_df["nonzero"] = coef_df["coefficient"] != 0
    coef_df.to_csv(RESULTS / "elasticnet_coefficients.csv")
    log.info("non-zero coefficients (all-data refit, interpretation only):\n%s",
              coef_df[coef_df["nonzero"]].to_string())

    summary = pd.DataFrame([{
        "Model": "ElasticNet-LogReg (nested LOOCV)",
        "ROC_AUC": auc,
        "CI_2.5%": ci_lo,
        "CI_97.5%": ci_hi,
        "n_bootstrap": len(boot_aucs),
        "median_l1_ratio": np.median(chosen_l1_ratio),
    }])
    summary.to_csv(RESULTS / "elasticnet_loocv_auc_ci.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr, label=f"Elastic-Net (AUC={auc:.2f}, 95% CI {ci_lo:.2f}-{ci_hi:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Control vs cancer — Elastic-Net, nested LOOCV")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_elasticnet_loocv.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot_aucs, bins=40, color="#4c72b0")
    ax.axvline(auc, color="black", linewidth=1.5, label=f"LOOCV AUC={auc:.2f}")
    ax.axvline(ci_lo, color="gray", linestyle="--", linewidth=1)
    ax.axvline(ci_hi, color="gray", linestyle="--", linewidth=1, label="95% CI")
    ax.set_xlabel("Bootstrap ROC AUC")
    ax.set_ylabel("Count")
    ax.set_title("Bootstrap distribution of LOOCV AUC (resampled predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "bootstrap_auc_distribution.png", dpi=300)
    plt.close(fig)
    log.info("saved ROC + bootstrap distribution figures")

    return y_true, y_proba, auc, (ci_lo, ci_hi)


# --------------------------------------------------------------------------
# 3: decision curve analysis
# --------------------------------------------------------------------------
def run_decision_curve(y_true, y_proba):
    log.info("=" * 70)
    log.info("DECISION CURVE ANALYSIS")
    n = len(y_true)
    prevalence = y_true.mean()
    thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for pt in thresholds:
        pred_pos = y_proba >= pt
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        net_benefit_model = tp / n - fp / n * (pt / (1 - pt))
        net_benefit_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({"threshold": pt, "net_benefit_model": net_benefit_model,
                      "net_benefit_treat_all": net_benefit_all,
                      "net_benefit_treat_none": 0.0})
    dca_df = pd.DataFrame(rows)
    dca_df.to_csv(RESULTS / "decision_curve.csv", index=False)

    range_beats_all = dca_df[dca_df["net_benefit_model"] > dca_df["net_benefit_treat_all"]]
    if len(range_beats_all):
        log.info("model beats 'treat all' for thresholds in [%.2f, %.2f]",
                  range_beats_all["threshold"].min(), range_beats_all["threshold"].max())
    else:
        log.info("model never beats 'treat all' at any threshold tested — "
                  "no clinical net benefit over treating everyone, at this sample size.")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Elastic-Net model", linewidth=2)
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], label="Treat all", linestyle="--")
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_none"], label="Treat none", linestyle=":")
    ax.set_ylim(-0.1, max(0.5, dca_df["net_benefit_model"].max() * 1.1))
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve — control vs cancer classification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "decision_curve.png", dpi=300)
    plt.close(fig)
    log.info("saved decision curve figure")
    return dca_df


# --------------------------------------------------------------------------
# 4: sample-size projection for a follow-up study
# --------------------------------------------------------------------------
def run_power_analysis(auc, ci):
    log.info("=" * 70)
    log.info("SAMPLE-SIZE PROJECTION FOR A FOLLOW-UP STUDY")
    # AUC -> Cohen's d (Hanley/McNeil-style Gaussian equivalence):
    # d = sqrt(2) * Phi^-1(AUC)
    def auc_to_d(a):
        return np.sqrt(2) * norm.ppf(min(max(a, 0.5001), 0.9999))

    analysis = NormalIndPower()
    rows = []
    for label, a in [("observed (point estimate)", auc),
                      ("pessimistic (CI lower bound)", ci[0]),
                      ("optimistic (CI upper bound)", ci[1])]:
        if a <= 0.5:
            n_per_group, note = None, "AUC<=0.5: no detectable effect, sample size undefined"
        else:
            d = auc_to_d(a)
            n_per_group = int(np.ceil(analysis.solve_power(
                effect_size=d, alpha=0.05, power=0.8, ratio=1.0, alternative="two-sided")))
            note = ""
        rows.append({"scenario": label, "AUC": a,
                     "n_per_group_needed_80pct_power": n_per_group, "note": note})
        log.info("%-30s AUC=%.3f  n/group for 80%% power = %s  %s",
                  label, a, n_per_group if n_per_group else "N/A", note)

    power_df = pd.DataFrame(rows)
    power_df.to_csv(RESULTS / "sample_size_projection.csv", index=False)
    log.info("current study: 20 control / 21 cancer per binary target — "
              "compare against the 'pessimistic' row above for what a properly "
              "powered replication would need.")
    return power_df


def main():
    t0 = time.time()
    df, feature_cols = load_data()
    y_true, y_proba, auc, ci = run_elasticnet_loocv(df, feature_cols)
    run_decision_curve(y_true, y_proba)
    run_power_analysis(auc, ci)
    log.info("=" * 70)
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()

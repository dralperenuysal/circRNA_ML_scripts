"""
circRNA ML re-analysis (v2) — leakage-free pipeline.

Fixes vs. the original scripts/model_training_*.py:
  - imputation + feature selection are fit INSIDE each CV fold (no leakage
    from test rows into the imputer/selector, unlike edited_data_preprocessing.py
    where SelectKBest/SoftImpute were fit on the whole dataset before the split).
  - performance is estimated with repeated stratified k-fold + leave-one-out
    (n=44 is too small to trust a single 70/30 or 80/20 split, which is all
    the original scripts used).
  - one script, one data source, one log — the original had two divergent
    preprocessing/model scripts with no clear link to what the manuscript
    reports.

Targets:
  1. binary: control vs. cancer (from the "eau"/EAU-risk column; cleaner and
     better populated than the "label" column, which has 12 missing values).
  2. risk_tier: 3-class merge of EAU risk groups among cancer cases only
     (Low | Intermediate | High+VeryHigh+MuscleInvasive), evaluated with
     leave-one-out only — n=21 with 5 raw classes is too small for anything
     else, and this is exploratory, not confirmatory.

Run from repo root: python3 ML_circRNA_v2/run_analysis.py
"""
import logging
import sys
import time
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold,
                                      cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "resource" / "circ_edited.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for _d in (RESULTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 20  # repeated stratified k-fold for the binary target

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
# data loading
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA)
    df.columns.values[1] = "risk_score"
    df.rename(columns={"EAU RISK ": "eau"}, inplace=True)
    df["eau"] = df["eau"].str.replace("yok", "control")

    # 2^-ddCT columns (already delta-delta-Ct expression ratios) map 1:1 in
    # order to the raw target-circRNA Ct columns (circ0000326 ... circ0137439,
    # excluding the circ0000471 housekeeping gene used only as the Ct
    # denominator). NOTE: edited_data_preprocessing.py filtered gene names by
    # "circ" in name, which silently swapped ciRS-6 out and the housekeeping
    # gene in — the ddCT values ended up mislabeled. Using the explicit
    # column range here instead.
    ddct_cols = [c for c in df.columns if "^" in c]
    start = df.columns.get_loc("circ0000326")
    end = df.columns.get_loc("label")
    gene_names = [c for c in df.columns[start:end] if c != "circ0000471"]
    assert len(ddct_cols) == len(gene_names), "gene/ddCT column count mismatch"

    expr = df[ddct_cols].copy()
    expr.columns = gene_names
    expr["AGE"] = df["AGE"]
    expr["GENDER_2"] = df["GENDER_2"]

    df = pd.concat([expr, df[["eau", "risk_score"]]], axis=1)
    df = df.dropna(subset=["eau"])
    log.info("Loaded %d samples, %d features", *expr.shape)
    log.info("eau value counts:\n%s", df["eau"].value_counts().to_string())
    return df, gene_names + ["AGE", "GENDER_2"]


def build_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(clf, k_features):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(partial(mutual_info_classif, random_state=RANDOM_STATE), k=k_features)),
        ("clf", clf),
    ])


# --------------------------------------------------------------------------
# binary target: control vs cancer
# --------------------------------------------------------------------------
def run_binary(df, feature_cols):
    log.info("=" * 70)
    log.info("BINARY TARGET: control vs cancer")
    X = df[feature_cols]
    y = (df["eau"] != "control").astype(int)  # 1 = cancer
    log.info("class balance: %s", y.value_counts().to_dict())

    k_features = min(7, len(feature_cols))
    models = build_models()
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"}

    rows = []
    for name, clf in models.items():
        pipe = make_pipeline(clf, k_features)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {
            "Model": name,
            "ROC_AUC_mean": scores["test_roc_auc"].mean(),
            "ROC_AUC_std": scores["test_roc_auc"].std(),
            "Accuracy_mean": scores["test_accuracy"].mean(),
            "Accuracy_std": scores["test_accuracy"].std(),
            "F1_mean": scores["test_f1"].mean(),
            "F1_std": scores["test_f1"].std(),
        }
        rows.append(row)
        log.info("%-20s ROC_AUC=%.3f±%.3f  Acc=%.3f±%.3f  F1=%.3f±%.3f",
                  name, row["ROC_AUC_mean"], row["ROC_AUC_std"],
                  row["Accuracy_mean"], row["Accuracy_std"],
                  row["F1_mean"], row["F1_std"])
    cv_df = pd.DataFrame(rows).sort_values("ROC_AUC_mean", ascending=False)
    cv_df.to_csv(RESULTS / "binary_repeated_kfold_metrics.csv", index=False)

    # leave-one-out: complements the k-fold estimate, uses every sample as
    # a test point exactly once — the standard choice at n<50.
    loo = LeaveOneOut()
    loo_rows = []
    roc_data = {}
    for name, clf in models.items():
        pipe = make_pipeline(clf, k_features)
        y_true, y_proba, y_pred = [], [], []
        for train_idx, test_idx in loo.split(X, y):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = pipe.predict_proba(X.iloc[test_idx])[0, 1]
            y_proba.append(proba)
            y_pred.append(int(proba >= 0.5))
            y_true.append(y.iloc[test_idx].values[0])
        auc_score = roc_auc_score(y_true, y_proba)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        loo_rows.append({"Model": name, "ROC_AUC": auc_score, "Accuracy": acc, "F1": f1})
        roc_data[name] = (y_true, y_proba)
        log.info("[LOOCV] %-20s ROC_AUC=%.3f  Acc=%.3f  F1=%.3f",
                  name, auc_score, acc, f1)
    loo_df = pd.DataFrame(loo_rows).sort_values("ROC_AUC", ascending=False)
    loo_df.to_csv(RESULTS / "binary_loocv_metrics.csv", index=False)

    plot_roc_curves(roc_data, "Binary: control vs cancer (LOOCV)",
                     FIGURES / "roc_binary_loocv.png")

    # feature importance: fit best model on all data, permutation importance.
    best_name = loo_df.iloc[0]["Model"]
    plot_feature_importance(models[best_name], X, y, k_features, best_name,
                             FIGURES / "feature_importance_binary.png",
                             RESULTS / "feature_importance_binary.csv")
    return cv_df, loo_df


# --------------------------------------------------------------------------
# exploratory risk-tier target (cancer cases only)
# --------------------------------------------------------------------------
def run_risk_tier(df, feature_cols):
    log.info("=" * 70)
    log.info("EXPLORATORY TARGET: risk tier (cancer cases only, LOOCV)")
    cancer = df[df["eau"] != "control"].copy()
    tier_map = {
        "Low": "Low",
        "Intermediate": "Intermediate",
        "High": "High+",
        "Very high": "High+",
        "Muscle invasive": "High+",
    }
    cancer["tier"] = cancer["eau"].map(tier_map)
    log.info("risk tier counts:\n%s", cancer["tier"].value_counts().to_string())

    X = cancer[feature_cols]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(cancer["tier"]), index=cancer.index)
    k_features = min(5, len(feature_cols))  # fewer features: n=21 here
    models = build_models()

    loo = LeaveOneOut()
    rows = []
    for name, clf in models.items():
        pipe = make_pipeline(clf, k_features)
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X, y):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred.append(pipe.predict(X.iloc[test_idx])[0])
            y_true.append(y.iloc[test_idx].values[0])
        y_true = le.inverse_transform(y_true)
        y_pred = le.inverse_transform(y_pred)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        rows.append({"Model": name, "Accuracy": acc, "F1_macro": f1})
        log.info("[LOOCV] %-20s Acc=%.3f  F1_macro=%.3f", name, acc, f1)
    tier_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    tier_df.to_csv(RESULTS / "risk_tier_loocv_metrics.csv", index=False)
    log.info("NOTE: 3-class, n=21, LOOCV only — treat as hypothesis-generating, "
              "not a validated classifier.")
    return tier_df


# --------------------------------------------------------------------------
# plotting helpers
# --------------------------------------------------------------------------
def plot_roc_curves(roc_data, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (y_true, y_proba) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("saved %s", out_path)


def plot_feature_importance(clf, X, y, k_features, model_name, fig_path, csv_path):
    pipe = make_pipeline(clf, k_features)
    pipe.fit(X, y)
    selected = X.columns[pipe.named_steps["select"].get_support()]

    from sklearn.inspection import permutation_importance
    r = permutation_importance(pipe, X, y, n_repeats=50,
                                random_state=RANDOM_STATE, scoring="roc_auc")
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance_mean": r.importances_mean,
        "Importance_std": r.importances_std,
        "Selected": X.columns.isin(selected),
    }).sort_values("Importance_mean", ascending=False)
    imp_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    top = imp_df.head(10).iloc[::-1]
    ax.barh(top["Feature"], top["Importance_mean"], xerr=top["Importance_std"])
    ax.set_xlabel("Permutation importance (ROC AUC drop)")
    ax.set_title(f"Feature importance — best model: {model_name}\n"
                  "(fit on all data; refit-on-all values are for interpretation, "
                  "not a held-out test)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    log.info("saved %s", fig_path)


def main():
    t0 = time.time()
    df, feature_cols = load_data()
    cv_df, loo_df = run_binary(df, feature_cols)
    tier_df = run_risk_tier(df, feature_cols)

    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("Binary (control vs cancer) — repeated 5x20-fold CV:\n%s",
              cv_df.to_string(index=False))
    log.info("Binary (control vs cancer) — LOOCV:\n%s", loo_df.to_string(index=False))
    log.info("Risk tier (exploratory, cancer only) — LOOCV:\n%s",
              tier_df.to_string(index=False))
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()

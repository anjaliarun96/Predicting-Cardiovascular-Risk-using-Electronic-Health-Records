"""Train and evaluate CVD risk models: regularised logistic regression (baseline) and XGBoost.

Usage:
    python src/train.py --data data/cardio_train.csv
    python src/train.py --synthetic            # smoke test only, no dataset needed
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:  # only for environments without xgboost; install it via requirements.txt
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False
    print("[warn] xgboost not installed - falling back to sklearn HistGradientBoosting (install xgboost for real results)")

from data import FEATURES, TARGET, clean, load_raw, make_synthetic

SEED = 42


def build_models(fast=False):
    """Returns {name: (estimator, param_grid)}."""
    cs = [0.01, 0.1, 1.0] if fast else [0.001, 0.01, 0.1, 1.0, 10.0]
    lr = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(solver="saga", max_iter=300, **kw))
    models = {
        "LogReg (Ridge)": (lr(penalty="l2"), {"logisticregression__C": cs}),
        "LogReg (Lasso)": (lr(penalty="l1"), {"logisticregression__C": cs}),
        "LogReg (ElasticNet)": (lr(penalty="elasticnet"),
                                {"logisticregression__C": cs, "logisticregression__l1_ratio": [0.25, 0.5, 0.75]}),
    }
    xgb_grid = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "n_estimators": [200] if fast else [200, 400],
                "reg_lambda": [1, 5], "reg_alpha": [0, 1]}
    if fast:
        xgb_grid = {"max_depth": [3, 5], "learning_rate": [0.1], "n_estimators": [100], "reg_lambda": [1], "reg_alpha": [0]}
    if HAVE_XGB:
        models["XGBoost (GBM)"] = (XGBClassifier(eval_metric="auc", n_jobs=-1, random_state=SEED, tree_method="hist"), xgb_grid)
    else:
        hgb_grid = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "l2_regularization": [0.0, 1.0]}
        models["XGBoost (GBM)"] = (HistGradientBoostingClassifier(random_state=SEED), hgb_grid)
    return models


def group_auc(y, p, mask):
    if mask.sum() < 50 or len(np.unique(y[mask])) < 2:
        return np.nan
    return roc_auc_score(y[mask], p[mask])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cardio_train.csv")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test only)")
    ap.add_argument("--fast", action="store_true", help="smaller grids for quick runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = make_synthetic() if args.synthetic else load_raw(args.data)
    df = clean(raw)
    X, y = df[FEATURES], df[TARGET].values
    print(f"[data] {len(df)} patients, {len(FEATURES)} features, positive rate {y.mean():.3f}")

    json.dump({"n_raw": int(len(raw)), "n_clean": int(len(df)), "n_features": len(FEATURES),
               "positive_rate": float(y.mean()), "synthetic": bool(args.synthetic),
               "xgboost": HAVE_XGB, "cv_folds": 5 if args.fast else 10,
               "male_rate": float(df[df.gender == 2][TARGET].mean()),
               "female_rate": float(df[df.gender == 1][TARGET].mean())},
              open(f"{args.out}/data_summary.json", "w"), indent=2)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5 if args.fast else 10, shuffle=True, random_state=SEED)

    rows, probs, best_params = [], {}, {}
    for name, (est, grid) in build_models(args.fast).items():
        print(f"[train] {name} ...")
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1 if "XGB" not in name else 1, refit=True)
        gs.fit(X_tr, y_tr)
        p = gs.predict_proba(X_te)[:, 1]
        probs[name] = p
        if name.startswith("XGBoost") and hasattr(gs.best_estimator_, "feature_importances_"):
            imp = pd.Series(gs.best_estimator_.feature_importances_, index=FEATURES).sort_values()
            imp.to_csv(f"{args.out}/feature_importance.csv", header=["importance"])
            plt.figure(figsize=(5.5, 4)); imp.plot.barh(); plt.title("XGBoost feature importance")
            plt.tight_layout(); plt.savefig(f"{args.out}/feature_importance.png", dpi=200); plt.close()
        best_params[name] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in gs.best_params_.items()}
        thr = 0.5
        tn, fp, fn, tp = confusion_matrix(y_te, p >= thr).ravel()
        rows.append({"model": name, "cv_auroc": gs.best_score_, "test_auroc": roc_auc_score(y_te, p),
                     "accuracy": accuracy_score(y_te, p >= thr), "sensitivity": tp / (tp + fn),
                     "specificity": tn / (tn + fp)})
        print(f"        CV AUROC {gs.best_score_:.4f} | test AUROC {rows[-1]['test_auroc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out}/metrics.csv", index=False)
    json.dump(best_params, open(f"{args.out}/best_params.json", "w"), indent=2)

    # ROC curves
    plt.figure(figsize=(5.5, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Test-set ROC curves")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{args.out}/roc.png", dpi=200); plt.close()

    # Subgroup (fairness) analysis for the best model
    best = res.sort_values("test_auroc").iloc[-1]["model"]
    p = probs[best]
    Xt = X_te.reset_index(drop=True)
    groups = {
        "Female": Xt.gender == 1, "Male": Xt.gender == 2,
        "Age <50": Xt.age_years < 50, "Age 50-59": Xt.age_years.between(50, 59.99), "Age 60+": Xt.age_years >= 60,
    }
    sub = pd.DataFrame([{"model": best, "subgroup": g, "n": int(m.sum()), "auroc": group_auc(y_te, p, m.values)}
                        for g, m in groups.items()])
    sub.to_csv(f"{args.out}/subgroups.csv", index=False)

    print("\n=== Test results ===\n", res.round(4).to_string(index=False))
    print(f"\n=== Subgroup AUROC ({best}) ===\n", sub.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""STEP 10: the invariance versus discrimination trade-off (graduate depth analysis).

kNN on a blended distance D_lam = (1 - lam) D_FIH / m_FIH + lam D_raw / m_raw (see
models.MixedDTWkNN; m is the median training-pair distance). lam = 0 is FIH-kNN, lam = 1 is
the original kNN-DTW. For each lam we report random-CV AUC (nested, as in step 3), the same
scores ranked within each session, and cross-session AUC and balanced accuracy (as in step 6).

lam is never selected here: the curve is computed on the test sessions, so choosing lam from
it would be tuning on test data. It is an explanation of the main result, not a model.

Outputs: results/tradeoff_lambda.csv, figures/fig_tradeoff_lambda.png
Usage:   python scripts/10_invariance_tradeoff.py [--repeats 5]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import build_inputs, load_processed, make_strata
from src.config import ABLATION_REPEATS, FIGS, RESULTS, SEED
from src.evaluation import auc_rows, fit_and_threshold, nested_cv, pred_metrics
from src.models import K_GRID, MixedDTWkNN

LAMBDAS = np.round(np.arange(0, 1.01, 0.1), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=ABLATION_REPEATS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    meta, feat, wedge, hold = load_processed()
    inputs = build_inputs(meta, feat, wedge, hold)
    y = meta.y.to_numpy()
    sess = meta.session.to_numpy()
    S = list(pd.unique(sess))
    strata = make_strata(y, meta.session)
    idx = inputs["index"]

    rows = []
    for lam in LAMBDAS:
        spec = dict(estimator=MixedDTWkNN(D_inv=inputs["dtw_fih"], D_raw=inputs["dtw"], lam=float(lam)),
                    grid={"k": K_GRID}, input="index")
        res = nested_cv(spec, idx, y, strata, args.repeats, SEED, args.n_jobs, verbose=False)
        P = res["prob"]
        within = np.mean([auc_rows(y[sess == s], P[:, sess == s]).mean() for s in S])
        row = dict(lam=lam, random_cv_auc=float(auc_rows(y, P).mean()), within_session_auc=float(within))
        cross_auc, cross_ba = [], []
        for train_s in S:
            tr, te = np.flatnonzero(sess == train_s), np.flatnonzero(sess != train_s)
            model, _, taus = fit_and_threshold(spec, idx[tr], y[tr], SEED, args.n_jobs)
            p = model.predict_proba(idx[te])[:, 1]
            cross_auc.append(roc_auc_score(y[te], p))
            cross_ba.append(pred_metrics(y[te], (p >= taus["tau_ba"]).astype(int))["ba"])
            row[f"cross_auc_train_{train_s}"] = cross_auc[-1]
        row.update(cross_auc_mean=float(np.mean(cross_auc)), cross_ba_mean=float(np.mean(cross_ba)))
        rows.append(row)
        print(f"  lam={lam:.1f}  random-CV {row['random_cv_auc']:.3f}  within {within:.3f}  "
              f"cross {row['cross_auc_mean']:.3f}  cross BA {row['cross_ba_mean']:.3f}")
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "tradeoff_lambda.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(T.within_session_auc, T.cross_auc_mean, "o-", color="#1a9850")
    for _, r in T.iterrows():
        if r.lam in (0.0, 0.5, 1.0):
            ax.annotate(f"$\\lambda$={r.lam:.1f}", (r.within_session_auc, r.cross_auc_mean),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("within-session AUC (random CV)")
    ax.set_ylabel("cross-session AUC")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_tradeoff_lambda.png", dpi=200)
    plt.close(fig)
    pd.set_option("display.width", 200)
    print("\n" + T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nNext: python scripts/11_svm_platt_check.py")


if __name__ == "__main__":
    main()

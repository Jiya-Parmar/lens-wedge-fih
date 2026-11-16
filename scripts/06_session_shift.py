"""STEP 6: cross-session evaluation (deployment-risk analysis) for every model.

The data come from two sessions (7 Aug and 10 Aug) with different fixture positions,
different force levels and different success rates (45% vs 26%). Random CV mixes them,
so a model can partly score well by recognising the session. Here every model is trained
on one session (inner grid search and both thresholds included) and tested on the other,
in both directions. Inside a single test session the session identity is constant, so the
AUC measures only within-session discrimination of a model that never saw that session.

Variants
  raw              features / traces as recorded (all nine models)
  session_std      every feature z-scored within its own session (label-free, but it needs a
                   batch of trials from the new session); for kNN-DTW the per-session mean
                   trace is subtracted instead. Run for the original models only: the FIH
                   models are meant to need no data from the new session.

A paired bootstrap of cross-session AUC differences (FIH models against every other model)
resamples the test trials of each direction, scores both models on the same resample and
averages the two directions resample by resample.

Outputs
  results/session_shift.csv, session_shift_summary.csv, session_shift_paired.csv
Usage
  python scripts/06_session_shift.py
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common import build_inputs, load_processed
from src.config import DTW_BAND, N_BOOT, RESULTS, SEED
from src.dtw import pairwise_phase_dtw
from src.evaluation import (auc_rows, bootstrap_auc, bootstrap_indices, fit_and_threshold,
                            pred_metrics)
from src.models import PROPOSED_MODELS, get_model_specs

ORIGINAL = ["SingleFeature", "LogReg", "RandomForest", "SVM-RBF", "kNN-DTW"]


def session_standardize(X, session):
    Xs = X.copy()
    for s in np.unique(session):
        m = session == s
        mu = np.nanmean(X[m], 0)
        sd = np.nanstd(X[m], 0)
        sd[sd == 0] = 1.0
        Xs[m] = (X[m] - mu) / sd
    return Xs


def split_input(inputs, spec, tr, te):
    X = inputs[spec["input"]]
    if spec["input"].startswith("dtw"):
        return X[np.ix_(tr, tr)], X[np.ix_(te, tr)]
    return X[tr], X[te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORIGINAL + PROPOSED_MODELS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    meta, feat, wedge, hold = load_processed()
    y = meta.y.to_numpy()
    sess = meta.session.to_numpy()
    S = list(pd.unique(sess))
    raw = build_inputs(meta, feat, wedge, hold)
    std = dict(raw)
    std["features"] = session_standardize(raw["features"], sess)
    wc, hc = wedge.copy(), hold.copy()
    for s in S:
        m = sess == s
        wc[m] -= wc[m].mean(0)
        hc[m] -= hc[m].mean(0)
    print("  computing DTW on session-centred traces ...")
    std["dtw"] = pairwise_phase_dtw(wc, hc, DTW_BAND)
    specs = get_model_specs(args.models, raw)

    print("Session priors (fraction seated):",
          {s: round(float(y[sess == s].mean()), 3) for s in S})
    rows, scores, boots_by_dir = [], {}, {}
    for train_s in S:
        test_s = [s for s in S if s != train_s][0]
        tr = np.flatnonzero(sess == train_s)
        te = np.flatnonzero(sess == test_s)
        boots = bootstrap_indices(y[te], n_boot=1000, seed=SEED)
        boots_by_dir[train_s] = (te, bootstrap_indices(y[te], n_boot=N_BOOT, seed=SEED + 1))
        for variant, inputs in [("raw", raw), ("session_std", std)]:
            for m, spec in specs.items():
                if variant == "session_std" and m in PROPOSED_MODELS:
                    continue
                Xtr, Xte = split_input(inputs, spec, tr, te)
                model, params, taus = fit_and_threshold(spec, Xtr, y[tr], SEED, args.n_jobs)
                p = model.predict_proba(Xte)[:, 1]
                if variant == "raw":
                    scores[(train_s, m)] = p
                lo, hi, _ = bootstrap_auc(y[te], p[None, :], boots)
                mb = pred_metrics(y[te], (p >= taus["tau_ba"]).astype(int))
                ms = pred_metrics(y[te], (p >= taus["tau_safe"]).astype(int))
                rows.append(dict(train=train_s, test=test_s, variant=variant, model=m,
                                 n_train=len(tr), n_test=len(te), test_prior=y[te].mean(),
                                 roc_auc=roc_auc_score(y[te], p), ci_lo=lo, ci_hi=hi,
                                 ba_at_tau_ba=mb["ba"], false_seated_at_tau_ba=mb["false_seated_rate"],
                                 sens_at_tau_safe=ms["sens"], false_seated_at_tau_safe=ms["false_seated_rate"],
                                 tau_ba=taus["tau_ba"], tau_safe=taus["tau_safe"],
                                 mean_p_test=p.mean(), params=str(params)))
                print(f"  {train_s}->{test_s} {variant:11s} {m:13s} AUC {rows[-1]['roc_auc']:.3f} "
                      f"[{lo:.2f},{hi:.2f}]  BA {mb['ba']:.3f}  sens@safe {ms['sens']:.2f}  "
                      f"FSR@safe {ms['false_seated_rate']:.2f}")
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "session_shift.csv", index=False)

    avg = R.groupby(["variant", "model"], sort=False)[["roc_auc", "ba_at_tau_ba", "sens_at_tau_safe",
                                                       "false_seated_at_tau_safe"]].mean().reset_index()
    per_dir = R[R.variant == "raw"].pivot(index="model", columns="train", values="roc_auc")
    ref = RESULTS / "metrics_summary.csv"
    if ref.exists():
        rc = pd.read_csv(ref).set_index("model")
        avg["random_cv_auc"] = avg.model.map(rc.roc_auc_mean)
        avg["auc_drop_vs_random_cv"] = avg.random_cv_auc - avg.roc_auc
    avg.to_csv(RESULTS / "session_shift_summary.csv", index=False)

    # ---------------- paired bootstrap of cross-session AUC differences (raw variant)
    def diff_draws(a, b):
        d = []
        for train_s, (te, boots) in boots_by_dir.items():
            pa, pb, yt = scores[(train_s, a)], scores[(train_s, b)], y[te]
            d.append(np.array([auc_rows(yt[i], pa[i])[0] - auc_rows(yt[i], pb[i])[0] for i in boots]))
        return np.mean(d, axis=0)

    pairs = []
    names = list(specs)
    for a in [m for m in names if m in PROPOSED_MODELS]:
        for b in names:
            if b == a:
                continue
            d = diff_draws(a, b)
            point = float(np.mean([roc_auc_score(y[te], scores[(s, a)]) - roc_auc_score(y[te], scores[(s, b)])
                                   for s, (te, _) in boots_by_dir.items()]))
            pairs.append(dict(model_a=a, model_b=b, diff=point, lo=float(np.percentile(d, 2.5)),
                              hi=float(np.percentile(d, 97.5)), p_greater=float((d > 0).mean())))
    P = pd.DataFrame(pairs)
    P.to_csv(RESULTS / "session_shift_paired.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"
    print("\nMean over the two directions:")
    print(avg.to_string(index=False, float_format=fmt))
    print("\nCross-session AUC per direction (raw):")
    print(per_dir.to_string(float_format=fmt))
    print(f"\nPaired bootstrap of cross-session AUC differences ({N_BOOT} resamples per direction):")
    print(P.to_string(index=False, float_format=fmt))
    print("\nNext: python scripts/07_error_analysis.py")


if __name__ == "__main__":
    main()

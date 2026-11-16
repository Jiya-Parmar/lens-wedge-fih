"""STEP 8 (add-on): two diagnostics that explain the session-shift result.

  1. Within-session AUC of the step-3 (random CV) scores. Random CV pools both sessions.
     Here the same out-of-fold scores are ranked only against trials of the same session,
     so session identity cannot help. If a model still ranks well within each session, its
     random-CV signal is real; if it then fails across sessions, it needs training data
     from the same session (the features shift), rather than recognising the session.
  2. Session shift per feature group. Which groups transfer to the other session? The 14
     fixture-invariant (FIH) statistics are included as a group, so the standard learners
     can be compared on raw and on invariant inputs.

Outputs
  results/within_session_auc.csv
  results/session_shift_by_group.csv
Usage
  python scripts/08_session_diagnostics.py
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

from src.common import load_fih, load_processed
from src.config import RESULTS, SEED
from src.evaluation import auc_rows, fit_and_threshold
from src.features import select_groups
from src.models import get_model_specs

GROUPS = {"W only": ["W"], "H only": ["H"], "H+X": ["H", "X"], "MAIN W+H+X+V": ["W", "H", "X", "V"],
          "FIH": None}


def session_standardize(X, session):
    Xs = X.copy()
    for s in np.unique(session):
        m = session == s
        sd = np.nanstd(X[m], 0)
        sd[sd == 0] = 1.0
        Xs[m] = (X[m] - np.nanmean(X[m], 0)) / sd
    return Xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["SingleFeature", "LogReg", "RandomForest", "SVM-RBF"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"

    # ---------------- 1. within-session AUC of random-CV scores
    z = np.load(RESULTS / "oof_main.npz", allow_pickle=False)
    y, sess = z["y"], z["session"]
    rows = []
    for m in z["models"]:
        P = z[f"prob__{m}"]
        row = dict(model=m, pooled_auc=float(auc_rows(y, P).mean()))
        for s in np.unique(sess):
            k = sess == s
            row[f"within_{s}"] = float(auc_rows(y[k], P[:, k]).mean())
        row["within_mean"] = float(np.mean([row[f"within_{s}"] for s in np.unique(sess)]))
        rows.append(row)
    W = pd.DataFrame(rows)
    W.to_csv(RESULTS / "within_session_auc.csv", index=False)
    print("1. Random-CV scores ranked only within each session (session identity cannot help):")
    print(W.to_string(index=False, float_format=fmt))

    # ---------------- 2. session shift per feature group
    meta, feat, _, _ = load_processed()
    fih, _ = load_fih(meta)
    y = meta.y.to_numpy()
    sess = meta.session.to_numpy()
    S = list(pd.unique(sess))
    specs = get_model_specs(args.models)
    out = []
    for gname, groups in GROUPS.items():
        Xr = (fih if groups is None else select_groups(feat, groups)).to_numpy(float)
        for variant, X in [("raw", Xr), ("session_std", session_standardize(Xr, sess))]:
            for m, spec in specs.items():
                aucs = []
                for train_s in S:
                    tr, te = np.flatnonzero(sess == train_s), np.flatnonzero(sess != train_s)
                    model, _, _ = fit_and_threshold(spec, X[tr], y[tr], SEED, args.n_jobs)
                    aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
                out.append(dict(feature_set=gname, variant=variant, model=m,
                                auc_7to10=aucs[0], auc_10to7=aucs[1], auc_mean=float(np.mean(aucs))))
                print(f"   {gname:13s} {variant:11s} {m:13s} mean cross-session AUC {np.mean(aucs):.3f}")
    G = pd.DataFrame(out)
    G.to_csv(RESULTS / "session_shift_by_group.csv", index=False)
    print("\n2. Cross-session AUC (mean of both directions) by feature group:")
    print(G.pivot_table(index=["variant", "feature_set"], columns="model", values="auc_mean", sort=False)
          .to_string(float_format=fmt))


if __name__ == "__main__":
    main()

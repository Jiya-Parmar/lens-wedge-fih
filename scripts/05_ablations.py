"""STEP 5: graduate depth analysis, part 1. Ablations and sensitivity studies.

  A. Feature-group ablation   which physical evidence carries the signal (wedge, hold,
                              cross-phase, vibration, position, and the 14 invariant FIH
                              statistics)?
  B. Hold-duration study      how long must the arm hold before deciding? (cycle time vs
                              accuracy; 0 s = decide from the wedge phase alone)
  C. Label-definition study   main target (success vs failure+partial) against the paper's
                              merge (success+partial vs failure) and dropping partial trials
  D. Permutation test         is the nested-CV AUC above chance under the full pipeline?
                              (LogReg, kNN-DTW, FIH-kNN, FIH-LR; 1000 permutations by default,
                              labels shuffled within each session)

Every configuration is evaluated with the same nested CV as step 3 (fewer repeats).

Usage
  python scripts/05_ablations.py                       # LogReg + RandomForest (+ DTW and FIH models in C, D)
  python scripts/05_ablations.py --models LogReg --n-perm 50
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import build_inputs, load_fih, load_processed, make_strata
from src.config import ABLATION_REPEATS, DATA_PROC, FIGS, RESULTS, SEED
from src.data_loading import make_target
from src.evaluation import auc_rows, bootstrap_auc, bootstrap_indices, nested_cv, per_repeat_metrics
from src.features import ALL_GROUPS, MAIN_GROUPS, extract_features, select_groups
from src.models import get_model_specs

GROUP_SETS = {
    "W (wedge force)": ["W"],
    "H (hold force)": ["H"],
    "V (vibration)": ["V"],
    "P (position only)": ["P"],
    "W+V (no hold)": ["W", "V"],
    "W+H": ["W", "H"],
    "W+H+X": ["W", "H", "X"],
    "MAIN W+H+X+V": MAIN_GROUPS,
    "MAIN + P": ALL_GROUPS,
    "FIH (invariant)": None,
}
LABEL_MODELS = ["kNN-DTW", "FIH-kNN", "FIH-LR"]
PERM_MODELS = ["LogReg", "kNN-DTW", "FIH-kNN", "FIH-LR"]
HOLD_SECONDS = [0, 1, 2, 3, 4, None]   # None = full hold (~4.7 s)


def evaluate(spec, X, y, strata, n_rep, tag, n_jobs):
    res = nested_cv(spec, X, y, strata, n_rep, SEED, n_jobs, verbose=False)
    pr = pd.DataFrame(per_repeat_metrics(y, res))
    lo, hi, _ = bootstrap_auc(y, res["prob"], bootstrap_indices(y, n_boot=1000))
    n_feat = X.shape[1] if spec["input"] in ("features", "fih") else np.nan
    return dict(**tag, n=len(y), n_pos=int(y.sum()), n_features=n_feat,
                roc_auc=pr.roc_auc.mean(), roc_auc_sd=pr.roc_auc.std(ddof=1),
                roc_auc_ci_lo=lo, roc_auc_ci_hi=hi, pr_auc=pr.pr_auc.mean(),
                ba_at_tau_ba=pr.ba_at_tau_ba.mean(), sens_at_tau_safe=pr.sens_at_tau_safe.mean(),
                false_seated_at_tau_safe=pr.false_seated_at_tau_safe.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["LogReg", "RandomForest"])
    ap.add_argument("--repeats", type=int, default=ABLATION_REPEATS)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip", nargs="*", default=[], help="any of A B C D")
    ap.add_argument("--perm-models", nargs="*", default=PERM_MODELS,
                    help="which models to run the permutation test for (others keep saved rows)")
    args = ap.parse_args()

    meta, feat, wedge, hold = load_processed()
    y = meta.y.to_numpy()
    strata = make_strata(y, meta.session)
    inputs = build_inputs(meta, feat, wedge, hold)
    fih, _ = load_fih(meta)
    specs = get_model_specs(list(dict.fromkeys(args.models + LABEL_MODELS + PERM_MODELS)))
    feat_models = [m for m in args.models if specs[m]["input"] == "features"]
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"

    # ---------------- A. feature groups
    if "A" not in args.skip:
        print("A. feature-group ablation")
        rows = []
        for m in feat_models:
            for gname, groups in GROUP_SETS.items():
                t0 = time.time()
                X = (fih if groups is None else select_groups(feat, groups)).to_numpy(float)
                rows.append(evaluate(specs[m], X, y, strata, args.repeats,
                                     dict(model=m, feature_set=gname), args.n_jobs))
                print(f"   {m:12s} {gname:20s} AUC {rows[-1]['roc_auc']:.3f}  ({time.time()-t0:.0f} s)")
        A = pd.DataFrame(rows)
        A.to_csv(RESULTS / "ablation_feature_groups.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        w = 0.8 / len(feat_models)
        xs = np.arange(len(GROUP_SETS))
        for i, m in enumerate(feat_models):
            a = A[A.model == m].set_index("feature_set").loc[list(GROUP_SETS)]
            ax.bar(xs + i * w, a.roc_auc, w, label=m,
                   yerr=[a.roc_auc - a.roc_auc_ci_lo, a.roc_auc_ci_hi - a.roc_auc], capsize=2)
        ax.axhline(0.5, color="r", ls=":", lw=1)
        ax.set_xticks(xs + w * (len(feat_models) - 1) / 2, list(GROUP_SETS), rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("ROC-AUC (nested CV)")
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_ablation_feature_groups.png", dpi=200)
        plt.close(fig)

    # ---------------- B. hold duration
    if "B" not in args.skip:
        print("B. hold-duration study (features rebuilt from bursts.csv)")
        bursts = pd.read_csv(DATA_PROC / "bursts.csv", dtype={"trial_id": str})
        groups = {tid: g for tid, g in bursts.groupby("trial_id")}
        rows = []
        for T in HOLD_SECONDS:
            ft = pd.DataFrame([extract_features(groups[t], hold_max_s=T) for t in meta.trial_id])
            X = select_groups(ft, MAIN_GROUPS).to_numpy(float)
            for m in feat_models:
                rows.append(evaluate(specs[m], X, y, strata, args.repeats,
                                     dict(model=m, hold_s=("full" if T is None else T)), args.n_jobs))
                print(f"   {m:12s} hold={str(T):5s} features={X.shape[1]:3d} AUC {rows[-1]['roc_auc']:.3f}")
        B = pd.DataFrame(rows)
        B.to_csv(RESULTS / "ablation_hold_duration.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        xpos = [0, 1, 2, 3, 4, 4.7]
        for m in feat_models:
            b = B[B.model == m]
            ax.errorbar(xpos, b.roc_auc, yerr=[b.roc_auc - b.roc_auc_ci_lo, b.roc_auc_ci_hi - b.roc_auc],
                        marker="o", capsize=3, label=m)
        ax.axhline(0.5, color="r", ls=":", lw=1)
        ax.set_xticks(xpos, ["0\n(wedge)", "1", "2", "3", "4", "full"])
        ax.set_xlabel("hold time used for the decision [s]")
        ax.set_ylabel("ROC-AUC (nested CV)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_ablation_hold_duration.png", dpi=200)
        plt.close(fig)

    # ---------------- C. label definition
    if "C" not in args.skip:
        print("C. label-definition study")
        rows = []
        for scheme in ["success_vs_rest", "paper_merge", "drop_partial"]:
            yy = make_target(meta.label3, scheme).to_numpy()
            keep = ~np.isnan(yy)
            yk = yy[keep].astype(int)
            st = make_strata(yk, meta.session[keep])
            for m in feat_models + LABEL_MODELS:
                Xa = inputs[specs[m]["input"]]
                X = Xa[np.ix_(keep, keep)] if specs[m]["input"].startswith("dtw") else Xa[keep]
                rows.append(evaluate(specs[m], X, yk, st, args.repeats,
                                     dict(model=m, label_scheme=scheme), args.n_jobs))
                print(f"   {m:12s} {scheme:16s} n={keep.sum()} pos={yk.sum()} AUC {rows[-1]['roc_auc']:.3f}")
        pd.DataFrame(rows).to_csv(RESULTS / "ablation_label_scheme.csv", index=False)

    # ---------------- D. permutation test
    if "D" not in args.skip and args.n_perm > 0:
        print(f"D. label-permutation test ({args.n_perm} permutations, 1 repeat each)")
        rng = np.random.default_rng(SEED)
        out = RESULTS / "permutation_test.csv"
        prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
        rows = {r["model"]: r for r in prev.to_dict("records")
                if r.get("n_perm") == args.n_perm}
        for m in PERM_MODELS:
            # the shuffles are drawn for every model in a fixed order, so a model's null does
            # not depend on which other models are evaluated in this run
            perms = []
            for k in range(args.n_perm):
                yp = y.copy()
                for s in meta.session.unique():
                    # permute labels WITHIN session so the session/label structure is not what is tested
                    idx = np.flatnonzero((meta.session == s).to_numpy())
                    yp[idx] = rng.permutation(yp[idx])
                perms.append(yp)
            if m not in args.perm_models:
                continue
            X = inputs[specs[m]["input"]]
            obs = auc_rows(y, nested_cv(specs[m], X, y, strata, 1, SEED, args.n_jobs, verbose=False)["prob"])[0]
            t0 = time.time()
            null = np.asarray([auc_rows(yp, nested_cv(specs[m], X, yp, make_strata(yp, meta.session), 1,
                                                      SEED + k + 1, args.n_jobs, verbose=False)["prob"])[0]
                               for k, yp in enumerate(perms)])
            p = (1 + (null >= obs).sum()) / (1 + len(null))
            rows[m] = dict(model=m, observed_auc=obs, null_mean=null.mean(), null_sd=null.std(),
                           null_p95=np.percentile(null, 95), p_value=p, n_perm=len(null))
            np.save(RESULTS / f"permutation_null_{m}.npy", null)
            pd.DataFrame([rows[k] for k in PERM_MODELS if k in rows]).to_csv(out, index=False)
            print(f"   {m:10s} observed {obs:.3f}  null {null.mean():.3f}+/-{null.std():.3f}  "
                  f"p = {p:.4f}  ({time.time()-t0:.0f} s)")

    for f in ["ablation_feature_groups", "ablation_hold_duration", "ablation_label_scheme", "permutation_test"]:
        p = RESULTS / f"{f}.csv"
        if p.exists():
            print(f"\n{f}:")
            d = pd.read_csv(p)
            print(d.drop(columns=[c for c in ["n_features"] if c in d]).to_string(index=False, float_format=fmt))
    print("\nNext: python scripts/06_session_shift.py")


if __name__ == "__main__":
    main()

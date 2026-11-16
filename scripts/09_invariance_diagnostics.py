"""STEP 9: diagnostics of the fixture-invariant hold (FIH) method (graduate depth analysis).

  A. Fixture geometry      what physically changed between the sessions, read from the logged
                           rim position and wedge target (no labels used)
     Mean hold traces      burst-mean force at five hold points, by session and outcome
  B. Direction consistency every feature's AUC within each session; a feature can transfer
                           only if it points the same way on both days
  C. Exact invariance      random yaw and offset applied to every raw trial, whole per-trial
                           pipeline rerun; FIH must not change (stops with an error if it does)
  D. Stress test           models trained on one session as in step 6, the other session's raw
                           samples rotated or offset further, everything rebuilt for those
                           test trials and rescored ("what if the next fixture moves again?")
  E. Development variants  the kNN trace variants scored cross-session while the method was
                           designed (reported for honesty about forking paths)

Outputs (results/)
  fixture_geometry.csv, mean_hold_traces.csv, direction_consistency.csv,
  direction_consistency_by_group.csv, invariance_check.txt, stress_test.csv,
  development_variants.csv,
  figures/fig_direction_consistency.png
Usage
  python scripts/09_invariance_diagnostics.py
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

from src.common import build_inputs, load_fih, load_processed
from src.config import DTW_BAND, FIGS, RESULTS, SEED
from src.data_loading import find_trial_files, load_trial, to_bursts
from src.dtw import cross_dtw, cross_phase_dtw, pairwise_dtw
from src.evaluation import fit_and_threshold
from src.features import ALL_GROUPS, MAIN_GROUPS, build_series, extract_features, select_groups
from src.invariant import apply_fixture_change, fih_features, fih_signature
from src.models import get_model_specs

# Simulated further fixture changes for the stress test (edit to try others).
YAW_DEG = [15, 30, 45, 90, 135, 180]
AXIAL_OFFSET_N = [-2, -1, 1, 2]
STRESS_MODELS = ["SingleFeature", "LogReg", "RandomForest", "SVM-RBF", "kNN-DTW",
                 "FIH-kNN", "FIH-LR", "FIHV"]


def rebuild(df, main_cols):
    """Run the whole per-trial pipeline on one raw trial."""
    b = to_bursts(df)
    f = extract_features(b)
    w, h = build_series(b)
    return (np.array([f.get(c, np.nan) for c in main_cols], float), w, h,
            np.array(list(fih_features(b).values())), fih_signature(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip", nargs="*", default=[], help="any of A B C D E")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"

    meta, feat, wedge, hold = load_processed()
    fih, sig = load_fih(meta)
    y = meta.y.to_numpy()
    sess = meta.session.to_numpy()
    S = list(pd.unique(sess))
    paths = {}
    for path, cls in find_trial_files():
        paths[path.stem.split(f"wedge_{cls}_")[1]] = path
    raw = {t: load_trial(paths[t]) for t in meta.trial_id}

    # ---------------- A. fixture geometry (no labels) and mean hold traces
    if "A" not in args.skip:
        g = []
        for t, s in zip(meta.trial_id, sess):
            d = raw[t].iloc[0]
            yaw = np.degrees(np.arctan2(d.wedge_target_y - d.y, d.wedge_target_x - d.x))
            g.append(dict(session=s, rim_x=d.rim_pos_x, rim_y=d.rim_pos_y, rim_z=d.rim_pos_z, approach_yaw_deg=yaw))
        G = pd.DataFrame(g)
        geo = G.groupby("session").agg(rim_x=("rim_x", "mean"), rim_y=("rim_y", "mean"), rim_z=("rim_z", "mean"),
                                       yaw_mean=("approach_yaw_deg", "mean"), yaw_sd=("approach_yaw_deg", "std"))
        geo.to_csv(RESULTS / "fixture_geometry.csv")
        print("A. Fixture geometry by session (from metadata, no labels):")
        print(geo.to_string(float_format=lambda v: f"{v:.4f}"))
        dy = abs(geo.rim_y.iloc[0] - geo.rim_y.iloc[1]) * 100
        print(f"   rim moved {dy:.1f} cm in y; approach yaw {geo.yaw_mean.iloc[0]:+.1f} -> {geo.yaw_mean.iloc[1]:+.1f} deg")
        pts = [0, 4, 7, 10, 14]
        rows = []
        for s in S:
            for yy, nm in [(1, "seated"), (0, "not seated")]:
                k = (sess == s) & (y == yy)
                for ci, ch in [(2, "fz"), (1, "fy"), (0, "fx")]:
                    mu = hold[k, :, ci].mean(0)
                    rows.append(dict(session=s, outcome=nm, n=int(k.sum()), channel=ch,
                                     **{f"pt{p+1}": mu[p] for p in pts}))
        MT = pd.DataFrame(rows)
        MT.to_csv(RESULTS / "mean_hold_traces.csv", index=False)
        print("\n   Mean hold traces [N] at five of 15 resampled points:")
        print(MT[MT.channel.isin(["fz", "fy"])].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

    # ---------------- B. direction consistency of every feature
    if "B" not in args.skip:
        allf = pd.concat([select_groups(feat, ALL_GROUPS), fih], axis=1)
        rows = []
        for c in allf.columns:
            a = []
            for s in S:
                k = sess == s
                x = allf[c].to_numpy()[k]
                ok = ~np.isnan(x)
                a.append(roc_auc_score(y[k][ok], x[ok]) if np.nanstd(x) > 0 else 0.5)
            o12 = 0.5 + np.sign(a[0] - 0.5) * (a[1] - 0.5)
            o21 = 0.5 + np.sign(a[1] - 0.5) * (a[0] - 0.5)
            rows.append(dict(feature=c, group=c.split("__")[0], auc_s1=a[0], auc_s2=a[1],
                             same_direction=(a[0] - 0.5) * (a[1] - 0.5) > 0,
                             oriented_transfer_auc=(o12 + o21) / 2))
        DC = pd.DataFrame(rows)
        DC.to_csv(RESULTS / "direction_consistency.csv", index=False)
        order = ["W", "H", "X", "V", "P", "FIH"]
        DG = DC.groupby("group").agg(n=("feature", "size"), same_direction=("same_direction", "mean"),
                                     mean_oriented_auc=("oriented_transfer_auc", "mean"),
                                     best_oriented_auc=("oriented_transfer_auc", "max")).loc[order]
        DG.to_csv(RESULTS / "direction_consistency_by_group.csv")
        print("\nB. Direction consistency by feature group (descriptive, all labels):")
        print(DG.to_string(float_format=fmt))
        fig, ax = plt.subplots(figsize=(4.2, 4))
        cols = {"W": "#d7301f", "H": "#2b7bba", "X": "#7b3294", "V": "#999999", "P": "#f39c12", "FIH": "#1a9850"}
        for gname in order:
            d = DC[DC.group == gname]
            ax.scatter(d.auc_s1, d.auc_s2, s=14, c=cols[gname], label=gname, alpha=0.8)
        ax.axhline(0.5, color="k", lw=0.6)
        ax.axvline(0.5, color="k", lw=0.6)
        ax.set_xlabel(f"AUC within {S[0]}")
        ax.set_ylabel(f"AUC within {S[1]}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_direction_consistency.png", dpi=200)
        plt.close(fig)

    main_cols = list(select_groups(feat, MAIN_GROUPS).columns)

    # ---------------- C. exact invariance check
    if "C" not in args.skip:
        rng = np.random.default_rng(SEED)
        dF, dS, X0, X1 = [], [], [], []
        for i, t in enumerate(meta.trial_id):
            yaw = rng.uniform(0, 360)
            off = rng.normal(0, 2.0, 3)
            f0, _, _, s0, g0 = rebuild(raw[t], main_cols)
            f1, _, _, s1, g1 = rebuild(apply_fixture_change(raw[t], yaw, off), main_cols)
            dF.append(np.abs(s1 - s0).max())
            dS.append(np.abs(g1 - g0).max())
            X0.append(f0)
            X1.append(f1)
        X0, X1 = np.array(X0), np.array(X1)
        absd = np.abs(X1 - X0)
        changed = np.nanmax(absd, 0) > 1e-8
        rel = np.nanmedian(absd / (np.abs(X0) + 1e-9), 0)
        lines = ["Exact invariance check: every trial gets a random yaw ~ U(0, 360) deg and an offset",
                 "~ N(0, 2^2) N per axis applied to its raw samples; the whole per-trial pipeline",
                 "(burst averaging, features, resampling) is then rerun.",
                 f"  FIH statistics : max absolute change {max(dF):.2e}",
                 f"  FIH signatures : max absolute change {max(dS):.2e} N",
                 f"  Standard main features: {changed.sum()} of {len(main_cols)} change "
                 f"({changed.mean():.0%}); median over features of the per-feature median "
                 f"relative change {np.median(rel):.2f}"]
        txt = "\n".join(lines)
        (RESULTS / "invariance_check.txt").write_text(txt + "\n", encoding="utf-8")
        print("\nC. " + txt)
        if max(max(dF), max(dS)) > 1e-8:
            raise AssertionError("FIH representation is no longer invariant: check src/invariant.py")

    # ---------------- D. stress test
    if "D" not in args.skip:
        inputs = build_inputs(meta, feat, wedge, hold)
        specs = get_model_specs(STRESS_MODELS, inputs)
        configs = [("none", 0.0, 0.0)] + [(f"yaw {a} deg", a, 0.0) for a in YAW_DEG] + \
                  [(f"axial {o:+d} N", 0.0, o) for o in AXIAL_OFFSET_N]
        rows = []
        for train_s in S:
            tr = np.flatnonzero(sess == train_s)
            te = np.flatnonzero(sess != train_s)
            fitted = {}
            for m, spec in specs.items():
                X = inputs[spec["input"]]
                Xtr = X[np.ix_(tr, tr)] if spec["input"].startswith("dtw") else X[tr]
                fitted[m] = fit_and_threshold(spec, Xtr, y[tr], SEED, args.n_jobs)[0]
            for name, yaw, off in configs:
                mod = {k: v.copy() for k, v in inputs.items()}
                if yaw != 0 or off != 0:
                    reb = [rebuild(apply_fixture_change(raw[meta.trial_id[i]], yaw, (0, 0, off)), main_cols)
                           for i in te]
                    mod["features"][te] = np.array([r[0] for r in reb])
                    mod["fih"][te] = np.array([r[3] for r in reb])
                    cw = cross_phase_dtw(np.array([r[1] for r in reb]), np.array([r[2] for r in reb]),
                                         wedge[tr], hold[tr], DTW_BAND)
                    cf = cross_dtw(np.array([r[4] for r in reb]), sig[tr], DTW_BAND)
                    mod["dtw"][np.ix_(te, tr)] = cw
                    mod["dtw_fih"][np.ix_(te, tr)] = cf
                for m, spec in specs.items():
                    model = fitted[m]
                    if m == "FIHV":                 # index model: point it at the rebuilt matrices
                        model.D, model.F = mod["dtw_fih"], mod["fih"]
                    X = mod[spec["input"]]
                    Xte = X[np.ix_(te, tr)] if spec["input"].startswith("dtw") else X[te]
                    p = model.predict_proba(Xte)[:, 1]
                    rows.append(dict(train=train_s, change=name, model=m, roc_auc=roc_auc_score(y[te], p)))
                print(f"   trained on {train_s}, test change {name:12s} done")
        ST = pd.DataFrame(rows)
        ST.to_csv(RESULTS / "stress_test.csv", index=False)
        tab = ST.groupby(["change", "model"], sort=False).roc_auc.mean().unstack()[STRESS_MODELS]
        tab = tab.loc[[c[0] for c in configs]]
        tab.to_csv(RESULTS / "stress_test_summary.csv")
        print("\nD. Stress test: cross-session AUC (mean of both directions) under a further simulated "
              "fixture change of the test session:")
        print(tab.to_string(float_format=fmt))
    # ---------------- E. development variants (same kNN-DTW learner, cross-session)
    if "E" not in args.skip:
        spec = get_model_specs(["kNN-DTW"])["kNN-DTW"]
        rows = []
        for name, ser in [("raw hold traces (fx, fy, fz)", hold),
                          ("signed anchored change (dx, dy, dz)", hold - hold[:, :1, :]),
                          ("FIH signature (a, l)", sig)]:
            D = pairwise_dtw(ser, DTW_BAND)
            a = []
            for train_s in S:
                tr, te = np.flatnonzero(sess == train_s), np.flatnonzero(sess != train_s)
                model = fit_and_threshold(spec, D[np.ix_(tr, tr)], y[tr], SEED, args.n_jobs)[0]
                a.append(roc_auc_score(y[te], model.predict_proba(D[np.ix_(te, tr)])[:, 1]))
            rows.append(dict(variant=name, auc_train_s1=a[0], auc_train_s2=a[1], cross_auc_mean=float(np.mean(a))))
        E = pd.DataFrame(rows)
        E.to_csv(RESULTS / "development_variants.csv", index=False)
        print("\nE. Hold-trace variants tried during development (kNN-DTW, cross-session AUC):")
        print(E.to_string(index=False, float_format=fmt))
    print("\nNext: python scripts/10_invariance_tradeoff.py")


if __name__ == "__main__":
    main()

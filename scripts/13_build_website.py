"""STEP 13: build the project website (docs/index.html) from the results of steps 1 to 12.

The page is static, so GitHub Pages can serve it from the docs/ folder. Everything it shows
comes from results/ and data/processed/ and is embedded as JSON into site/template.html:
model summaries, per-trial scores (random CV averaged over repeats, cross-session per
direction), mean hold traces by session and outcome, the stress test and the trade-off.
No raw force recordings are written to the page, only aggregates and model outputs.

Usage: python scripts/13_build_website.py
"""
import base64
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import pandas as pd

from src.common import load_fih, load_processed
from src.config import RESULTS

MODELS = ["SingleFeature", "LogReg", "RandomForest", "SVM-RBF", "kNN-DTW", "FIH-kNN", "FIH-LR", "FIHV"]
SITE = ROOT / "site"
DOCS = ROOT / "docs"


def r(v, k=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), k)


def main():
    meta, _, _, hold = load_processed()
    _, sig = load_fih(meta)
    y = meta.y.to_numpy().astype(int)
    sess = meta.session.to_numpy()
    S = list(pd.unique(sess))
    D = {"models": MODELS, "sessions": S}

    # ---- trial labels
    D["trials"] = dict(y=y.tolist(), outcome=meta.label3.tolist(), session=sess.tolist())
    D["counts"] = {s: dict(n=int((sess == s).sum()), seated=int(y[sess == s].sum())) for s in S}
    D["outcome_counts"] = meta.label3.value_counts().to_dict()

    # ---- mean hold traces (aggregates only)
    tr = []
    for s in S:
        for yy in (1, 0):
            k = (sess == s) & (y == yy)
            tr.append(dict(session=s, seated=yy, n=int(k.sum()),
                           raw=[r(v, 3) for v in hold[k, :, 2].mean(0)],
                           inv=[r(v, 3) for v in sig[k, :, 0].mean(0)]))
    D["traces"] = tr

    # ---- summaries
    ms = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model")
    ws = pd.read_csv(RESULTS / "within_session_auc.csv").set_index("model")
    sh = pd.read_csv(RESULTS / "session_shift.csv", dtype={"train": str, "test": str})
    sh = sh[sh.variant == "raw"]
    th = pd.read_csv(RESULTS / "threshold_summary.csv").set_index("model")
    cal = pd.read_csv(RESULTS / "calibration.csv").set_index("model")
    summ = {}
    for m in MODELS:
        x = sh[sh.model == m].set_index("train")
        summ[m] = dict(cv_auc=r(ms.loc[m, "roc_auc_mean"]), cv_lo=r(ms.loc[m, "roc_auc_ci_lo"]),
                       cv_hi=r(ms.loc[m, "roc_auc_ci_hi"]), pr_auc=r(ms.loc[m, "pr_auc_mean"]),
                       cv_ba=r(ms.loc[m, "ba_at_tau_ba_mean"]), cv_sens=r(ms.loc[m, "sens_at_tau_safe_mean"]),
                       cv_fsr=r(ms.loc[m, "false_seated_at_tau_safe_mean"]),
                       within=r(ws.loc[m, "within_mean"]),
                       x_a=r(x.loc[S[0], "roc_auc"]), x_b=r(x.loc[S[1], "roc_auc"]),
                       x_mean=r(x.roc_auc.mean()), x_ba=r(x.ba_at_tau_ba.mean()),
                       x_sens=r(x.sens_at_tau_safe.mean()), x_fsr=r(x.false_seated_at_tau_safe.mean()),
                       tau_ba=r(th.loc[m, "nested_tau_ba_median"], 3),
                       tau_safe=r(th.loc[m, "nested_tau_safe_median"], 3),
                       brier=r(cal.loc[m, "brier"]), calib_slope=r(cal.loc[m, "calib_slope"]))
    D["summary"] = summ
    D["prior_cv_auc"] = r(ms.loc["Prior", "roc_auc_mean"])
    D["prevalence"] = r(y.mean())

    # ---- per-trial scores
    oof = np.load(RESULTS / "oof_main.npz", allow_pickle=True)
    D["cv_scores"] = {m: [r(v, 6) for v in oof[f"prob__{m}"].mean(0)] for m in MODELS}
    sc = pd.read_csv(RESULTS / "session_shift_scores.csv", dtype={"train": str, "test": str})
    xs = {}
    for a in S:
        d = sc[sc.train == a]
        first = d[d.model == MODELS[0]]
        xs[a] = dict(test=[s for s in S if s != a][0], y=first.y.astype(int).tolist(),
                     outcome=first.outcome.tolist(),
                     models={m: dict(p=[r(v, 6) for v in d[d.model == m].p_seated],
                                     tau_ba=r(d[d.model == m].tau_ba.iloc[0], 4),
                                     tau_safe=r(d[d.model == m].tau_safe.iloc[0], 4)) for m in MODELS})
    D["xs_scores"] = xs

    # ---- comparisons
    pc = pd.read_csv(RESULTS / "paired_auc_differences.csv")
    D["paired_cv"] = [dict(a=a, b=b, d=r(d), lo=r(lo), hi=r(hi)) for a, b, d, lo, hi in
                      pc[pc.model_b == "SingleFeature"][["model_a", "model_b", "diff", "lo", "hi"]].values
                      if a in MODELS]
    px = pd.read_csv(RESULTS / "session_shift_paired.csv")
    keep = [("FIH-kNN", "kNN-DTW"), ("FIH-LR", "LogReg"), ("FIH-LR", "SVM-RBF"), ("FIH-LR", "RandomForest"),
            ("FIH-LR", "SingleFeature"), ("FIHV", "kNN-DTW"), ("FIH-kNN", "SingleFeature")]
    D["paired_x"] = [dict(a=a, b=b, d=r(row["diff"]), lo=r(row.lo), hi=r(row.hi))
                     for a, b in keep for _, row in px[(px.model_a == a) & (px.model_b == b)].iterrows()]

    # ---- error analysis by outcome subtype (random CV)
    eb = pd.read_csv(RESULTS / "error_by_subtype.csv")
    D["errors"] = [dict(model=m, rule=ru, outcome=o, n=int(n), rate=r(v, 3))
                   for m, ru, o, n, v in eb[["model", "rule", "outcome", "n", "called_seated_rate"]].values
                   if m in MODELS]

    # ---- depth analyses
    st = pd.read_csv(RESULTS / "stress_test_summary.csv")
    D["stress"] = dict(changes=st.change.tolist(), auc={m: [r(v, 3) for v in st[m]] for m in MODELS})
    to = pd.read_csv(RESULTS / "tradeoff_lambda.csv")
    D["tradeoff"] = [dict(lam=r(a, 2), within=r(b, 3), cross=r(c, 3), cv=r(d, 3), ba=r(e, 3)) for a, b, c, d, e in
                     to[["lam", "within_session_auc", "cross_auc_mean", "random_cv_auc", "cross_ba_mean"]].values]
    dc = pd.read_csv(RESULTS / "direction_consistency_by_group.csv")
    D["dircons"] = [dict(group=g, n=int(n), same=r(s, 3), mean=r(m, 3)) for g, n, s, m in
                    dc[["group", "n", "same_direction", "mean_oriented_auc"]].values]
    geo = pd.read_csv(RESULTS / "fixture_geometry.csv", dtype={"session": str}).set_index("session")
    D["geo"] = dict(rim_shift_cm=r(abs(geo.rim_y.iloc[0] - geo.rim_y.iloc[1]) * 100, 1),
                    yaw={s: r(geo.loc[s, "yaw_mean"], 1) for s in geo.index})
    dv = pd.read_csv(RESULTS / "development_variants.csv")
    D["devvar"] = [dict(variant=v, auc=r(a, 3)) for v, a in dv[["variant", "cross_auc_mean"]].values]
    D["invariance"] = (RESULTS / "invariance_check.txt").read_text(encoding="utf-8").strip()
    sg = pd.read_csv(RESULTS / "session_shift_by_group.csv")
    sg = sg[(sg.variant == "raw") & (sg.feature_set == "FIH")]
    D["fih_learners"] = {m: r(a, 3) for m, a in sg[["model", "auc_mean"]].values}

    # ---- permutation test (CSV if present, else the log of step 5)
    perm = {}
    p = RESULTS / "permutation_test.csv"
    if p.exists():
        for _, row in pd.read_csv(p).iterrows():
            perm[row.model] = dict(obs=r(row.observed_auc, 3), null=r(row.null_mean, 3), p=r(row.p_value, 4))
    log = RESULTS / "log_05_ablations.txt"
    if log.exists():
        for m, o, nm, pv in re.findall(r"^\s+(\S+)\s+observed ([\d.]+)\s+null ([\d.]+)\+/-[\d.]+\s+p = ([\d.]+)",
                                       log.read_text(encoding="utf-8"), flags=re.M):
            perm.setdefault(m, dict(obs=float(o), null=float(nm), p=float(pv)))
    D["perm"] = perm

    # ---- assemble the page
    html = (SITE / "template.html").read_text(encoding="utf-8")
    img = SITE / "robot_pipeline.jpg"
    if img.exists():
        html = html.replace("__PIPELINE_IMG__", "data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode())
    html = html.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":")))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {DOCS / 'index.html'} ({len(html)/1024:.0f} kB). Open it in a browser to check.")


if __name__ == "__main__":
    main()

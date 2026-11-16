"""STEP 7: error analysis, calibration and interpretation (uses outputs of step 3).

  error_by_subtype.csv     how often each recorded outcome (success / failure / partial)
                           is called "seated", per model, at both nested thresholds
  error_by_session.csv     the same split by session
  calibration.csv          Brier score, calibration intercept and slope per model
  worst_errors_<model>.csv most confident mistakes, with the path to the trial's PNG plot
  logreg_coefficients.csv  standardised coefficients of LogReg refit on all data (descriptive)
  fih_lr_coefficients.csv  the same for FIH-LR on the 14 invariant statistics
  figures/fig_calibration.png, figures/fig_worst_errors_<model>.png

Usage
  python scripts/07_error_analysis.py                # focus = best-AUC model
  python scripts/07_error_analysis.py --model LogReg
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
from sklearn.linear_model import LogisticRegression

from src.common import load_fih, load_processed
from src.config import DATA_RAW, FIGS, RESULTS
from src.features import MAIN_GROUPS, select_groups
from src.models import get_model_specs


def calibration_line(y, p):
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(lp.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    z = np.load(RESULTS / "oof_main.npz", allow_pickle=False)
    y, label3, session, tids = z["y"], z["label3"], z["session"], z["trial_id"]
    models = [m for m in z["models"] if m != "Prior"]
    summ = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model")
    focus = args.model or summ.drop(index="Prior", errors="ignore").roc_auc_mean.idxmax()

    # ---------------- error by subtype and by session
    sub_rows, ses_rows, cal_rows = [], [], []
    for m in models:
        for rule in ["ba", "safe"]:
            pred = z[f"pred{rule}__{m}"]                 # (repeats, n)
            called_seated = pred.mean(0)                # fraction of repeats calling it seated
            for cls in ["success", "partial_contact", "failure"]:
                k = label3 == cls
                sub_rows.append(dict(model=m, rule=f"tau_{rule}", outcome=cls, n=int(k.sum()),
                                     called_seated_rate=float(called_seated[k].mean())))
            for s in np.unique(session):
                for yy, nm in [(1, "seated"), (0, "not_seated")]:
                    k = (session == s) & (y == yy)
                    ses_rows.append(dict(model=m, rule=f"tau_{rule}", session=s, truth=nm,
                                         n=int(k.sum()), called_seated_rate=float(called_seated[k].mean())))
        pbar = z[f"prob__{m}"].mean(0)
        a, b = calibration_line(y, pbar)
        cal_rows.append(dict(model=m, brier=float(np.mean((pbar - y) ** 2)),
                             brier_prior_only=float(np.mean((y.mean() - y) ** 2)),
                             calib_intercept=a, calib_slope=b, mean_pred=float(pbar.mean()),
                             observed_rate=float(y.mean())))
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(RESULTS / "error_by_subtype.csv", index=False)
    pd.DataFrame(ses_rows).to_csv(RESULTS / "error_by_session.csv", index=False)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(RESULTS / "calibration.csv", index=False)

    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"
    print("Fraction of trials called 'seated' by recorded outcome (success row = sensitivity;")
    print("failure and partial rows = false-seated rate for that subtype):")
    print(sub.pivot_table(index=["model", "rule"], columns="outcome",
                          values="called_seated_rate").to_string(float_format=fmt))
    print("\nCalibration (slope 1 and intercept 0 are perfect; slope < 1 means over-confident):")
    print(cal.to_string(index=False, float_format=fmt))

    # ---------------- reliability diagram (quantile bins, small n -> coarse)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for m in models:
        pbar = z[f"prob__{m}"].mean(0)
        q = np.quantile(pbar, np.linspace(0, 1, 6))
        bins = np.clip(np.searchsorted(q, pbar, side="right") - 1, 0, 4)
        xs = [pbar[bins == k].mean() for k in range(5) if (bins == k).any()]
        ys = [y[bins == k].mean() for k in range(5) if (bins == k).any()]
        ax.plot(xs, ys, "o-", label=m, ms=4)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("mean predicted P(seated)")
    ax.set_ylabel("observed fraction seated")
    ax.set_title("Reliability (5 quantile bins, n=75)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_calibration.png", dpi=200)
    plt.close(fig)

    # ---------------- most confident errors of the focus model
    meta, feat, wedge, hold = load_processed()
    assert list(meta.trial_id) == list(tids)
    pbar = z[f"prob__{focus}"].mean(0)
    df = pd.DataFrame(dict(trial_id=tids, outcome=label3, session=session, y=y, p_seated=pbar))
    df["plot_png"] = [str(DATA_RAW / c / "plots" / f"wedge_{c}_{t}.png") for c, t in zip(label3, tids)]
    fs = df[df.y == 0].sort_values("p_seated", ascending=False).head(8).assign(error="false_seated")
    fr = df[df.y == 1].sort_values("p_seated").head(5).assign(error="false_reject")
    worst = pd.concat([fs, fr])
    worst.to_csv(RESULTS / f"worst_errors_{focus}.csv", index=False)
    print(f"\nMost confident mistakes of {focus} (open the PNG to inspect the raw trial):")
    print(worst[["error", "trial_id", "outcome", "session", "p_seated"]].to_string(index=False, float_format=fmt))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    grid = np.linspace(0, 1, wedge.shape[1])
    for j, (arr, title) in enumerate([(wedge, "wedge"), (hold, "hold")]):
        for cls, col in [("success", "#2b7bba"), ("failure", "#d7301f")]:
            k = (meta.label3 == cls).to_numpy()
            ax[j].plot(grid, arr[k, :, 2].mean(0), color=col, lw=2.5, label=f"mean {cls}")
        for t in fs.trial_id.head(3):
            i = int(np.flatnonzero(meta.trial_id.to_numpy() == t)[0])
            ax[j].plot(grid, arr[i, :, 2], "--", lw=1, label=f"FS {t} ({meta.label3[i]})")
        ax[j].set_title(f"Fz, {title} phase", fontsize=9)
        ax[j].set_xlabel("normalised time")
    ax[0].set_ylabel("Fz burst mean [N]")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGS / f"fig_worst_errors_{focus}.png", dpi=200)
    plt.close(fig)

    # ---------------- LogReg coefficients (descriptive: refit on ALL data)
    folds = pd.read_csv(RESULTS / "folds_main.csv")
    if "LogReg" in models:
        Xdf = select_groups(feat, MAIN_GROUPS)
        C = float(folds.loc[folds.model == "LogReg", "param_clf__C"].median())
        est = get_model_specs(["LogReg"])["LogReg"]["estimator"].set_params(clf__C=C)
        est.fit(Xdf.to_numpy(float), y)
        coef = pd.DataFrame(dict(feature=Xdf.columns, coef=est.named_steps["clf"].coef_[0]))
        coef["abs"] = coef.coef.abs()
        coef = coef.sort_values("abs", ascending=False).drop(columns="abs")
        coef.to_csv(RESULTS / "logreg_coefficients.csv", index=False)
        print(f"\nLogReg (C={C}) largest standardised coefficients, positive = pushes towards 'seated':")
        print(coef.head(12).to_string(index=False, float_format=fmt))

    # ---------------- FIH-LR coefficients (descriptive: refit on ALL data)
    if "FIH-LR" in models:
        fih, _ = load_fih(meta)
        C = float(folds.loc[folds.model == "FIH-LR", "param_clf__C"].median())
        est = get_model_specs(["FIH-LR"])["FIH-LR"]["estimator"].set_params(clf__C=C)
        est.fit(fih.to_numpy(float), y)
        coef = pd.DataFrame(dict(feature=fih.columns, coef=est.named_steps["clf"].coef_[0]))
        coef["abs"] = coef.coef.abs()
        coef = coef.sort_values("abs", ascending=False).drop(columns="abs")
        coef.to_csv(RESULTS / "fih_lr_coefficients.csv", index=False)
        print(f"\nFIH-LR (C={C}) standardised coefficients, positive = pushes towards 'seated':")
        print(coef.to_string(index=False, float_format=fmt))

    # ---------------- hyperparameter stability
    pcols = [c for c in folds.columns if c.startswith("param_")]
    print("\nHyperparameters chosen across outer folds (stability check):")
    for m in models:
        f = folds[folds.model == m]
        for c in pcols:
            if f[c].notna().any():
                print(f"  {m:13s} {c:28s} {f[c].astype(str).value_counts().to_dict()}")
    print("\nAll steps finished. Share the printed outputs and the results/ folder.")


if __name__ == "__main__":
    main()

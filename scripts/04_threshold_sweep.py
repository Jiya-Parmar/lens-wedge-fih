"""STEP 4: decision-threshold sweep for every classifier.

Two different questions are answered here, and they must not be confused:

  (a) DESCRIPTIVE sweep: on the repeat-averaged out-of-fold scores, sweep tau from 0.01 to
      0.99 and report the tau that would have been best (max balanced accuracy, safety
      constraint, minimum expected cost for several cost ratios). Because tau is picked on
      the same scores it is scored on, these numbers are OPTIMISTIC.
  (b) HONEST estimate: in step 3 every outer fold picked its own tau from inner-CV scores
      only. We compare (a) with (b); the difference is the optimism of picking the
      threshold after looking at the results.

Usage
  python scripts/04_threshold_sweep.py               # detailed plots for best-AUC model
  python scripts/04_threshold_sweep.py --model LogReg

Outputs (results/)
  threshold_sweep.csv          long table: model, tau, sens, spec, BA, false-seated, costs
  threshold_summary.csv        per model: sweep-optimal tau vs nested tau, optimism gap
  figures/fig_thresh_sweep_all.png, figures/fig_thresh_<model>.png
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

from src.config import COST_RATIOS_SWEEP, FIGS, MAX_FALSE_SEATED, RESULTS
from src.evaluation import TAU_GRID, pred_metrics, thresh_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    z = np.load(RESULTS / "oof_main.npz", allow_pickle=False)
    y = z["y"]
    models = [m for m in z["models"] if m != "Prior"]
    folds = pd.read_csv(RESULTS / "folds_main.csv")
    summ = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model")
    focus = args.model or summ.drop(index="Prior", errors="ignore").roc_auc_mean.idxmax()

    long, rows = [], []
    for m in models:
        P = z[f"prob__{m}"]
        pbar = P.mean(0)
        for t in TAU_GRID:
            r = thresh_metrics(y, pbar, t)
            rec = dict(model=m, tau=t, sens=r["sens"], spec=r["spec"], ba=r["ba"],
                       false_seated_rate=r["false_seated_rate"], ppv=r["ppv"])
            for c in COST_RATIOS_SWEEP:
                rec[f"cost_ratio_{c}"] = thresh_metrics(y, pbar, t, c_fp=c, c_fn=1)["cost_per_trial"]
            long.append(rec)
        sw = pd.DataFrame([l for l in long if l["model"] == m])
        best_ba = sw.loc[sw.ba.idxmax()]
        ok = sw[sw.false_seated_rate <= MAX_FALSE_SEATED + 1e-12]
        safe = ok.iloc[0] if len(ok) else sw.iloc[-1]
        f = folds[folds.model == m]
        # honest, nested numbers (each outer fold used its own inner-CV threshold)
        nb = np.mean([pred_metrics(y, z[f"predba__{m}"][r])["ba"] for r in range(P.shape[0])])
        ns = [pred_metrics(y, z[f"predsafe__{m}"][r]) for r in range(P.shape[0])]
        row = dict(model=m,
                   sweep_tau_ba=best_ba.tau, sweep_best_ba=best_ba.ba,
                   nested_tau_ba_median=f.tau_ba.median(),
                   nested_tau_ba_iqr=f.tau_ba.quantile(0.75) - f.tau_ba.quantile(0.25),
                   nested_ba=nb, optimism_ba=best_ba.ba - nb,
                   sweep_tau_safe=safe.tau, sweep_sens_at_safe=safe.sens,
                   sweep_fsr_at_safe=safe.false_seated_rate,
                   nested_tau_safe_median=f.tau_safe.median(),
                   nested_sens_at_safe=np.mean([s["sens"] for s in ns]),
                   nested_fsr_at_safe=np.mean([s["false_seated_rate"] for s in ns]))
        for c in COST_RATIOS_SWEEP:
            k = sw[f"cost_ratio_{c}"].idxmin()
            row[f"tau_min_cost_ratio_{c}"] = sw.loc[k, "tau"]
            row[f"min_cost_ratio_{c}"] = sw.loc[k, f"cost_ratio_{c}"]
            row[f"always_reject_cost_ratio_{c}"] = y.mean()   # FN for every seated trial, cost 1
        rows.append(row)

    long = pd.DataFrame(long)
    long.to_csv(RESULTS / "threshold_sweep.csv", index=False)
    summ_t = pd.DataFrame(rows)
    summ_t.to_csv(RESULTS / "threshold_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("Threshold summary (sweep = optimistic, nested = honest):")
    print(summ_t[["model", "sweep_tau_ba", "sweep_best_ba", "nested_tau_ba_median", "nested_ba",
                  "optimism_ba", "sweep_tau_safe", "sweep_sens_at_safe", "nested_sens_at_safe",
                  "nested_fsr_at_safe"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMinimum expected cost per trial by cost ratio (false-seated : false-reject):")
    print(summ_t[["model"] + [f"tau_min_cost_ratio_{c}" for c in COST_RATIOS_SWEEP] +
                 [f"min_cost_ratio_{c}" for c in COST_RATIOS_SWEEP]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"(Reference: 'always reject' costs {y.mean():.3f} per trial at every ratio, "
          f"but it never assembles a lens; see README note on the cost model.)")

    # ---------------- figure: BA vs tau for every model
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m in models:
        sw = long[long.model == m]
        ax.plot(sw.tau, sw.ba, label=m)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"threshold $\tau$ on P(seated)")
    ax.set_ylabel("balanced accuracy")
    ax.legend(fontsize=7)
    ax.set_title("Descriptive sweep on repeat-averaged OOF scores", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_thresh_sweep_all.png", dpi=200)
    plt.close(fig)

    # ---------------- figure: detailed panel for the focus model
    sw = long[long.model == focus]
    f = folds[folds.model == focus]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(sw.tau, sw.sens, label="sensitivity (seated found)")
    ax[0].plot(sw.tau, sw.spec, label="specificity (unseated rejected)")
    ax[0].plot(sw.tau, sw.ba, "k", lw=2, label="balanced accuracy")
    ax[0].axhline(1 - MAX_FALSE_SEATED, color="r", ls="--", lw=0.8,
                  label=f"spec = {1-MAX_FALSE_SEATED:.2f} safety line")
    ymin = ax[0].get_ylim()[0]
    ax[0].plot(f.tau_ba, np.full(len(f), 0.02), "|", color="k", ms=10, label=r"nested $\tau_{BA}$ per fold")
    ax[0].plot(f.tau_safe, np.full(len(f), 0.06), "|", color="r", ms=10, label=r"nested $\tau_{safe}$ per fold")
    ax[0].set_xlabel(r"$\tau$")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=6)
    ax[0].set_title(f"{focus}: metrics vs threshold", fontsize=9)
    for c in COST_RATIOS_SWEEP:
        ax[1].plot(sw.tau, sw[f"cost_ratio_{c}"], label=f"FP:FN = {c}:1")
    ax[1].axhline(y.mean(), color="gray", ls=":", lw=1, label="always reject")
    ax[1].set_xlabel(r"$\tau$")
    ax[1].set_ylabel("expected cost per trial")
    ax[1].legend(fontsize=6)
    ax[1].set_title("Cost sensitivity", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / f"fig_thresh_{focus}.png", dpi=200)
    plt.close(fig)
    print(f"\nDetailed figure for {focus} saved. Next: python scripts/05_ablations.py")


if __name__ == "__main__":
    main()

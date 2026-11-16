"""STEP 3: main comparison. Nested, repeated, stratified cross-validation of all nine models
(six baselines and standard models, three proposed fixture-invariant models).

Usage
  python scripts/03_train_evaluate.py              # full run (5-fold x 10 repeats)
  python scripts/03_train_evaluate.py --repeats 2  # quick check

Outputs (results/)
  oof_main.npz                out-of-fold probabilities and predictions, every model and repeat
  folds_main.csv              per outer fold: chosen hyperparameters and thresholds
  metrics_per_repeat.csv      metrics for every model and repeat
  metrics_summary.csv         mean, sd and bootstrap 95% CI per model
  paired_auc_differences.csv  paired bootstrap of ROC-AUC differences between models
  single_feature_choices.csv  which feature the SingleFeature baseline picked, how often
  figures/fig_main_roc.png, figures/fig_main_auc_bars.png
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve

from src.common import build_inputs, load_processed, make_strata
from src.config import FIGS, OUTER_REPEATS, RESULTS, SEED
from src.evaluation import (bootstrap_auc, bootstrap_indices, nested_cv,
                            paired_auc_difference, per_repeat_metrics)
from src.features import MAIN_GROUPS, select_groups
from src.models import get_model_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=OUTER_REPEATS)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    meta, feat, wedge, hold = load_processed()
    y = meta.y.to_numpy()
    strata = make_strata(y, meta.session)
    Xdf = select_groups(feat, MAIN_GROUPS)
    X = Xdf.to_numpy(float)
    inputs = build_inputs(meta, feat, wedge, hold)
    print(f"n={len(y)} trials, seated={y.sum()}, not seated={(1-y).sum()}, "
          f"features={X.shape[1]} (groups {MAIN_GROUPS}), FIH statistics={inputs['fih'].shape[1]}")

    specs = get_model_specs(args.models, inputs)
    results, fold_rows, metric_rows = {}, [], []
    for name, spec in specs.items():
        t0 = time.time()
        print(f"  running {name} ...")
        inp = inputs[spec["input"]]
        res = nested_cv(spec, inp, y, strata, args.repeats, SEED, args.n_jobs, name=name)
        results[name] = res
        for r in res["folds"]:
            fold_rows.append(dict(model=name, **r))
        for r in per_repeat_metrics(y, res):
            metric_rows.append(dict(model=name, **r))
        print(f"    {name} finished in {time.time()-t0:.1f} s")

    np.savez(RESULTS / "oof_main.npz", y=y, trial_id=meta.trial_id.to_numpy().astype(str),
             label3=meta.label3.to_numpy().astype(str), session=meta.session.to_numpy().astype(str),
             models=np.array(list(results)),
             **{f"prob__{m}": r["prob"] for m, r in results.items()},
             **{f"predsafe__{m}": r["pred_safe"] for m, r in results.items()},
             **{f"predba__{m}": r["pred_ba"] for m, r in results.items()})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RESULTS / "folds_main.csv", index=False)
    per_rep = pd.DataFrame(metric_rows)
    per_rep.to_csv(RESULTS / "metrics_per_repeat.csv", index=False)

    # ---------------- summary with bootstrap CIs
    boots = bootstrap_indices(y)
    summ = []
    for name, res in results.items():
        P = res["prob"]
        g = per_rep[per_rep.model == name]
        lo, hi, _ = bootstrap_auc(y, P, boots)
        pbar = P.mean(0)
        ap_b = [average_precision_score(y[i], pbar[i]) for i in boots]
        br_b = [brier_score_loss(y[i], pbar[i]) for i in boots]
        row = dict(model=name)
        for c in ["roc_auc", "pr_auc", "brier",
                  "ba_at_tau_ba", "sens_at_tau_ba", "spec_at_tau_ba", "false_seated_at_tau_ba",
                  "ba_at_tau_safe", "sens_at_tau_safe", "spec_at_tau_safe",
                  "false_seated_at_tau_safe"]:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_sd"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
        row.update(roc_auc_ci_lo=lo, roc_auc_ci_hi=hi,
                   pr_auc_avgprob_ci_lo=np.percentile(ap_b, 2.5), pr_auc_avgprob_ci_hi=np.percentile(ap_b, 97.5),
                   brier_avgprob_ci_lo=np.percentile(br_b, 2.5), brier_avgprob_ci_hi=np.percentile(br_b, 97.5))
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS / "metrics_summary.csv", index=False)

    # ---------------- paired differences
    pairs = []
    names = list(results)
    for a in names:
        for b in names:
            if a != b:
                d = paired_auc_difference(y, results[a]["prob"], results[b]["prob"], boots)
                pairs.append(dict(model_a=a, model_b=b, **d))
    pd.DataFrame(pairs).to_csv(RESULTS / "paired_auc_differences.csv", index=False)

    # ---------------- which feature did the single-feature baseline pick?
    if "SingleFeature" in results:
        idx = folds.loc[folds.model == "SingleFeature", "single_feature_index"].astype(int)
        ch = idx.map(lambda i: Xdf.columns[i]).value_counts().rename_axis("feature").reset_index(name="n_folds")
        ch.to_csv(RESULTS / "single_feature_choices.csv", index=False)

    # ---------------- print
    pd.set_option("display.width", 200)
    show = summ[["model", "roc_auc_mean", "roc_auc_sd", "roc_auc_ci_lo", "roc_auc_ci_hi",
                 "pr_auc_mean", "brier_mean", "ba_at_tau_ba_mean", "false_seated_at_tau_ba_mean",
                 "sens_at_tau_safe_mean", "false_seated_at_tau_safe_mean"]]
    print("\nMAIN RESULTS (nested CV, mean over repeats; CI = trial bootstrap)")
    print(f"prevalence of seated = {y.mean():.3f} (this is the PR-AUC of a random scorer)")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if "SingleFeature" in results:
        print("\nSingleFeature baseline picked:")
        print(ch.to_string(index=False))
    pr = pd.DataFrame(pairs)
    print("\nPaired AUC differences vs SingleFeature baseline:")
    print(pr[pr.model_b == "SingleFeature"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for name, res in results.items():
        if name == "Prior":
            continue
        f_, t_, _ = roc_curve(y, res["prob"].mean(0))
        auc_m = summ.set_index("model").loc[name, "roc_auc_mean"]
        ax.plot(f_, t_, label=f"{name} ({auc_m:.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false-seated rate (1 - specificity)")
    ax.set_ylabel("sensitivity (seated detected)")
    ax.legend(fontsize=7)
    ax.set_title("ROC of repeat-averaged out-of-fold scores", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_main_roc.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    s2 = summ.set_index("model")
    xs = np.arange(len(s2))
    ax.bar(xs, s2.roc_auc_mean, color="#4a7ab5")
    ax.errorbar(xs, s2.roc_auc_mean, yerr=[s2.roc_auc_mean - s2.roc_auc_ci_lo,
                                           s2.roc_auc_ci_hi - s2.roc_auc_mean],
                fmt="none", ecolor="k", capsize=3)
    ax.axhline(0.5, color="r", ls=":", lw=1)
    ax.set_xticks(xs, s2.index, rotation=20, fontsize=8)
    ax.set_ylabel("ROC-AUC (nested CV)")
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_main_auc_bars.png", dpi=200)
    plt.close(fig)
    print(f"\nSaved to {RESULTS}. Next: python scripts/04_threshold_sweep.py")


if __name__ == "__main__":
    main()

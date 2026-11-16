"""STEP 12: the paper's figures, built from processed data and CSVs written by steps 3, 6, 9, 10.

  fig_paper_problem.pdf  mean hold-phase force by session and outcome: raw axial force (the
                         level moves with the session) against the invariant axial change a(t)
                         (the shape is shared across sessions)
  fig_paper_main.pdf     (a) every model under random CV and across sessions, (b) the
                         invariance trade-off, (c) the fixture stress test
Usage: python scripts/12_paper_figures.py
"""
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

from src.common import load_fih, load_processed
from src.config import FIGS, RESULTS

MODELS = ["SingleFeature", "LogReg", "RandomForest", "SVM-RBF", "kNN-DTW", "FIH-kNN", "FIH-LR", "FIHV"]
SHORT = {"SingleFeature": "Single", "LogReg": "LR", "RandomForest": "RF", "SVM-RBF": "SVM",
         "kNN-DTW": "kNN-DTW", "FIH-kNN": "FIH-kNN", "FIH-LR": "FIH-LR", "FIHV": "FIHV"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "axes.titlesize": 6.5,
                     "legend.fontsize": 5.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.1})


def problem_figure():
    meta, _, _, hold = load_processed()
    _, sig = load_fih(meta)
    y, sess = meta.y.to_numpy(), meta.session.to_numpy()
    S = list(pd.unique(sess))
    lab = {S[0]: "7 Aug", S[1]: "10 Aug"}
    t = np.linspace(0, 1, hold.shape[1])
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.85))
    style = {(1, 0): ("#2b7bba", "-"), (0, 0): ("#d7301f", "-"), (1, 1): ("#2b7bba", "--"), (0, 1): ("#d7301f", "--")}
    for si, s in enumerate(S):
        for yy in (1, 0):
            k = (sess == s) & (y == yy)
            c, ls = style[(yy, si)]
            nm = f"{lab[s]}, {'seated' if yy else 'not seated'} ({k.sum()})"
            ax[0].plot(t, hold[k, :, 2].mean(0), color=c, ls=ls, label=nm)
            ax[1].plot(t, sig[k, :, 0].mean(0), color=c, ls=ls)
    ax[0].set_title("(a) raw $f_z$ during hold [N]")
    ax[1].set_title("(b) invariant change $a(t)$ [N]")
    for a in ax:
        a.set_xlabel("normalised hold time")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               handlelength=1.8, fontsize=5.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.19, 1, 1))
    fig.savefig(FIGS / "fig_paper_problem.pdf")
    fig.savefig(FIGS / "fig_paper_problem.png", dpi=250)
    plt.close(fig)


def main_figure():
    summ = pd.read_csv(RESULTS / "metrics_summary.csv").set_index("model")
    shift = pd.read_csv(RESULTS / "session_shift_summary.csv")
    shift = shift[shift.variant == "raw"].set_index("model")
    trade = pd.read_csv(RESULTS / "tradeoff_lambda.csv")
    stress = pd.read_csv(RESULTS / "stress_test.csv")
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.8), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))

    x = np.arange(len(MODELS))
    ax[0].bar(x - 0.2, [summ.loc[m, "roc_auc_mean"] for m in MODELS], 0.4, color="#1f4e79", label="random CV (pooled)")
    ax[0].bar(x + 0.2, [shift.loc[m, "roc_auc"] for m in MODELS], 0.4, color="#c0504d", label="cross-session")
    ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[0].axvspan(4.5, 7.5, color="#1a9850", alpha=0.08, lw=0)
    ax[0].set_xticks(x, [SHORT[m] for m in MODELS], rotation=40, ha="right")
    ax[0].set_ylim(0.3, 1.0)
    ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title("(a) same models, two protocols")
    ax[0].legend(loc="upper left", frameon=False, ncol=1, handlelength=1.0)
    ax[0].text(6.0, 0.94, "proposed", ha="center", fontsize=5.5, color="#1a9850")

    ax[1].plot(trade.within_session_auc, trade.cross_auc_mean, "o-", ms=2.5, color="#1a9850")
    for _, r in trade.iterrows():
        if r.lam in (0.0, 0.5, 1.0):
            ax[1].annotate(f"$\\lambda$={r.lam:.1f}", (r.within_session_auc, r.cross_auc_mean),
                           textcoords="offset points", fontsize=5.5,
                           xytext={0.0: (3, 3), 0.5: (-27, -8), 1.0: (-27, -1)}[r.lam])
    ax[1].set_xlabel("within-session AUC")
    ax[1].set_ylabel("cross-session AUC")
    ax[1].set_title("(b) blend: invariant ($\\lambda$=0) to raw (1)")

    yaw = stress[stress.change.str.startswith("yaw") | (stress.change == "none")].copy()
    yaw["deg"] = yaw.change.map(lambda c: 0.0 if c == "none" else float(c.split()[1]))
    tab = yaw.groupby(["deg", "model"]).roc_auc.mean().unstack()
    for m, c, mk in [("SingleFeature", "#7f7f7f", "s"), ("RandomForest", "#c0504d", "^"),
                     ("kNN-DTW", "#1f4e79", "v"), ("FIH-LR", "#1a9850", "o"), ("FIHV", "#66bd63", "d")]:
        ax[2].plot(tab.index, tab[m], marker=mk, ms=2.5, color=c,
                   label={"SingleFeature": "Single", "RandomForest": "RF"}.get(m, m))
    ax[2].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[2].set_xticks([0, 45, 90, 135, 180])
    ax[2].set_ylim(0.2, 0.85)
    ax[2].set_xlabel("extra yaw of test session [deg]")
    ax[2].set_ylabel("cross-session AUC")
    ax[2].set_title("(c) simulated fixture rotation")
    ax[2].legend(loc="lower left", frameon=False, ncol=3, fontsize=5, handlelength=1.2, columnspacing=0.6)
    fig.tight_layout(pad=0.3, w_pad=0.7)
    fig.savefig(FIGS / "fig_paper_main.pdf")
    fig.savefig(FIGS / "fig_paper_main.png", dpi=250)
    plt.close(fig)


def main():
    problem_figure()
    main_figure()
    print(f"Saved {FIGS / 'fig_paper_problem.pdf'} and {FIGS / 'fig_paper_main.pdf'}")


if __name__ == "__main__":
    main()

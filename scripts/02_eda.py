"""STEP 2: exploratory analysis. Descriptive only; nothing here is used to pick models.

Outputs (results/ and results/figures/)
  eda_univariate.csv          every feature: separability for the label AND for the session
  eda_physical_scalars.csv    the obvious single-quantity proxies (depth, peak force, hold spread)
  fig_eda_bursts.png          why we aggregate bursts (sampling structure of one trial)
  fig_eda_mean_traces.png     mean Fz / |F| trace per class, wedge then hold
  fig_eda_boxplots.png        key features by the three recorded classes
  fig_eda_session_confound.png  session vs label structure
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
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from src.common import load_processed
from src.config import DATA_RAW, FIGS, GAP_S, RESULTS
from src.data_loading import load_trial

COLORS = {"success": "#2b7bba", "failure": "#d7301f", "partial_contact": "#f39c12"}


def oriented_auc(y, x):
    a = roc_auc_score(y, x)
    return max(a, 1 - a), ("higher=>seated" if a >= 0.5 else "lower=>seated")


def main():
    meta, feat, wedge, hold = load_processed()
    y = meta.y.to_numpy()
    sess = (meta.session == meta.session.unique()[1]).astype(int).to_numpy()

    # ---------------- univariate table
    rows = []
    for c in feat.columns:
        x = feat[c].to_numpy()
        ok = ~np.isnan(x)
        if np.nanstd(x) == 0:
            continue
        auc_y, direction = oriented_auc(y[ok], x[ok])
        auc_s, _ = oriented_auc(sess[ok], x[ok])
        p = mannwhitneyu(x[ok & (y == 1)], x[ok & (y == 0)]).pvalue
        rows.append(dict(feature=c, group=c.split("__")[0], auc_label=auc_y, direction=direction,
                         mw_p=p, auc_session=auc_s,
                         mean_seated=np.nanmean(x[y == 1]), mean_not=np.nanmean(x[y == 0])))
    uni = pd.DataFrame(rows).sort_values("auc_label", ascending=False)
    uni.to_csv(RESULTS / "eda_univariate.csv", index=False)
    print("Top 15 features by single-feature separability (descriptive, all data):")
    print(uni.head(15)[["feature", "auc_label", "mw_p", "auc_session"]].to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))
    print(f"\nFeatures whose session-AUC exceeds their label-AUC: "
          f"{(uni.auc_session > uni.auc_label).sum()} of {len(uni)}")

    phys = ["P__max_depth_mm", "W__fz_peakabs", "W__fmag_peakabs", "H__fz_range",
            "H__fz_std", "H__fz_last", "H__fz_resid_std"]
    ph = uni.set_index("feature").loc[[p for p in phys if p in uni.feature.values]]
    ph.to_csv(RESULTS / "eda_physical_scalars.csv")
    print("\nObvious single-quantity proxies:")
    print(ph[["auc_label", "mw_p", "auc_session"]].to_string(float_format=lambda v: f"{v:.3f}"))

    # ---------------- figure: sampling structure of one trial
    ex = meta[meta.label3 == "success"].iloc[0]
    df = load_trial(DATA_RAW / "success" / "csv" / ex.file)
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    ax[0].plot(df.time, df.fz, ".", ms=1.5, color="k")
    ax[0].set_ylabel("Fz filtered [N]")
    ax[0].set_title(f"Trial {ex.trial_id}: samples arrive in bursts (gaps > {GAP_S*1000:.0f} ms)")
    dt = np.diff(df.time.to_numpy())
    ax[1].semilogy(df.time.to_numpy()[1:], dt, ".", ms=1.5)
    ax[1].axhline(GAP_S, color="r", ls="--", lw=1)
    ax[1].set_ylabel("time step [s]")
    ax[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_eda_bursts.png", dpi=200)
    plt.close(fig)

    # ---------------- figure: mean traces per class
    grid = np.linspace(0, 2, 2 * wedge.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    for cls in ["success", "partial_contact", "failure"]:
        m = (meta.label3 == cls).to_numpy()
        tr = np.concatenate([wedge[m, :, 2], hold[m, :, 2]], axis=1)
        mu, sd = tr.mean(0), tr.std(0)
        ax.plot(grid, mu, color=COLORS[cls], label=f"{cls} (n={m.sum()})")
        ax.fill_between(grid, mu - sd, mu + sd, color=COLORS[cls], alpha=0.15)
    ax.axvline(1.0, color="gray", ls=":")
    ax.text(0.5, ax.get_ylim()[1] * 0.9, "wedge", ha="center")
    ax.text(1.5, ax.get_ylim()[1] * 0.9, "hold", ha="center")
    ax.set_xlabel("normalised phase time")
    ax.set_ylabel("Fz burst mean [N]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_eda_mean_traces.png", dpi=200)
    plt.close(fig)

    # ---------------- figure: boxplots of top features + physical proxies
    show = list(uni.feature.head(4)) + [p for p in ["P__max_depth_mm", "W__fz_peakabs"]
                                         if p not in list(uni.feature.head(4))]
    fig, axes = plt.subplots(1, len(show), figsize=(2.6 * len(show), 3.2))
    for a, c in zip(axes, show):
        data = [feat.loc[(meta.label3 == k).to_numpy(), c].dropna() for k in COLORS]
        bp = a.boxplot(data, patch_artist=True, widths=0.6)
        for patch, k in zip(bp["boxes"], COLORS):
            patch.set_facecolor(COLORS[k])
            patch.set_alpha(0.5)
        a.set_xticks([1, 2, 3], ["succ", "fail", "part"])
        auc = uni.set_index("feature").loc[c, "auc_label"]
        a.set_title(f"{c}\nAUC={auc:.2f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_eda_boxplots.png", dpi=200)
    plt.close(fig)

    # ---------------- figure: session confound
    fx, fy_ = "W__fz_peakabs", uni.feature.iloc[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    for s, mk in zip(meta.session.unique(), ["o", "^"]):
        for cls in COLORS:
            m = ((meta.session == s) & (meta.label3 == cls)).to_numpy()
            ax.scatter(feat.loc[m, fx], feat.loc[m, fy_], c=COLORS[cls], marker=mk, s=25,
                       alpha=0.8, label=f"{s} {cls}")
    ax.set_xlabel(fx)
    ax.set_ylabel(fy_)
    ax.legend(fontsize=6)
    ax.set_title("Session (marker) vs outcome (colour)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_eda_session_confound.png", dpi=200)
    plt.close(fig)
    print(f"\nFigures written to {FIGS}. Next: python scripts/03_train_evaluate.py")


if __name__ == "__main__":
    main()

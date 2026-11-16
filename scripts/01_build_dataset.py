"""STEP 1: load raw CSVs, run quality checks, aggregate bursts, build features and series.

Outputs
  data/processed/trials.csv     one row per trial: id, class, session, target, QC facts
  data/processed/bursts.csv     burst-level table (used again by the hold-duration ablation)
  data/processed/features.csv   one row per trial, engineered features
  data/processed/series.npz     resampled raw traces for the DTW model
  data/processed/fih_features.csv  the 14 fixture-invariant hold (FIH) statistics per trial
  data/processed/fih_series.npz    the two-channel FIH signature per trial (for FIH-kNN)
  results/qc_report.txt         human-readable data-quality report
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import pandas as pd

from src.config import DATA_PROC, DATA_RAW, GAP_S, LABEL_SCHEME_MAIN, RESULTS
from src.data_loading import (EXCLUDED_COLUMNS, extract_zips_if_needed, find_trial_files,
                              load_trial, make_target, qc_trial, to_bursts)
from src.features import build_series, extract_features
from src.invariant import fih_features, fih_signature


def main():
    print("[1/4] locating data")
    extract_zips_if_needed()
    files = find_trial_files()
    print(f"  found {len(files)} trial CSVs")

    print("[2/4] loading, QC and burst aggregation")
    meta_rows, burst_tables, feat_rows, wedges, holds = [], [], [], [], []
    fih_rows, sigs = [], []
    for path, cls in files:
        df = load_trial(path)
        tid = str(df.trial_id.iloc[0])
        qc = qc_trial(df, cls)
        b = to_bursts(df)
        b.insert(0, "trial_id", tid)
        burst_tables.append(b)
        meta_rows.append(dict(trial_id=tid, file=path.name, label3=cls, session=tid[:8],
                              n_bursts_wedge=int((b.phase == "wedge_attempt").sum()),
                              n_bursts_hold=int((b.phase == "hold").sum()), **qc))
        f = extract_features(b)
        f["trial_id"] = tid
        feat_rows.append(f)
        w, h = build_series(b)
        wedges.append(w)
        holds.append(h)
        fih_rows.append(dict(trial_id=tid, **fih_features(b)))
        sigs.append(fih_signature(b))

    meta = pd.DataFrame(meta_rows)
    meta["y"] = make_target(meta.label3, LABEL_SCHEME_MAIN).astype(int)
    order = np.argsort(meta.trial_id.to_numpy())
    meta = meta.iloc[order].reset_index(drop=True)
    feats = pd.DataFrame(feat_rows).set_index("trial_id").loc[meta.trial_id].reset_index()
    wedges = np.stack(wedges)[order]
    holds = np.stack(holds)[order]
    bursts = pd.concat(burst_tables, ignore_index=True)
    fih = pd.DataFrame(fih_rows).set_index("trial_id").loc[meta.trial_id].reset_index()
    sigs = np.stack(sigs)[order]

    # ---------------- QC assertions (fail loudly if the data is not what we think)
    assert meta.trial_id.is_unique, "duplicate trial ids across folders"
    assert meta.label_matches_folder.all(), "label column disagrees with folder"
    assert (meta.n_attempts == 1).all(), "a trial has more than one attempt"
    assert meta.phases_ok.all(), "unexpected phase structure"
    assert meta.n_nan.sum() == 0, "NaNs in raw data"
    assert not fih.drop(columns="trial_id").isna().any().any(), "missing FIH statistic"
    assert np.isfinite(sigs).all(), "non-finite FIH signature"

    # Outlier flag: wedge phase much longer than typical (robust z-score)
    wd = meta.wedge_dur_s
    rz = (wd - wd.median()) / (1.4826 * (wd - wd.median()).abs().median())
    meta["wedge_duration_outlier"] = rz.abs() > 5

    print("[3/4] saving processed data")
    meta.to_csv(DATA_PROC / "trials.csv", index=False)
    bursts.to_csv(DATA_PROC / "bursts.csv", index=False)
    feats.to_csv(DATA_PROC / "features.csv", index=False)
    np.savez(DATA_PROC / "series.npz", wedge=wedges, hold=holds,
             trial_id=meta.trial_id.to_numpy().astype(str))
    fih.to_csv(DATA_PROC / "fih_features.csv", index=False)
    np.savez(DATA_PROC / "fih_series.npz", sig=sigs, trial_id=meta.trial_id.to_numpy().astype(str))

    print("[4/4] writing QC report")
    lines = []
    lines.append("DATA QUALITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Trials: {len(meta)}  (one CSV = one trial = one wedge attempt)")
    lines.append("\nClass x session counts:")
    lines.append(pd.crosstab(meta.session, meta.label3, margins=True).to_string())
    lines.append("\nBinary target (1 = success/seated, 0 = failure or partial_contact):")
    lines.append(pd.crosstab(meta.session, meta.y, margins=True).to_string())
    lines.append(f"\nSamples per trial: median {meta.n_samples.median():.0f}, "
                 f"min {meta.n_samples.min()}, max {meta.n_samples.max()}")
    lines.append(f"Logging gaps > {GAP_S*1000:.0f} ms per trial: median {meta.n_gaps.median():.0f}; "
                 f"max gap median {meta.max_gap_s.median():.3f} s")
    lines.append(f"Bursts per trial: wedge median {meta.n_bursts_wedge.median():.0f}, "
                 f"hold median {meta.n_bursts_hold.median():.0f}")
    lines.append(f"Wedge duration s: median {meta.wedge_dur_s.median():.2f}, "
                 f"range [{meta.wedge_dur_s.min():.2f}, {meta.wedge_dur_s.max():.2f}]")
    lines.append(f"Hold duration s: median {meta.hold_dur_s.median():.2f}, "
                 f"range [{meta.hold_dur_s.min():.2f}, {meta.hold_dur_s.max():.2f}]")
    lines.append("Wedge duration by session (s):")
    lines.append(meta.groupby("session").wedge_dur_s.describe()[["mean", "min", "max"]].to_string())
    lines.append(f"\nWedge-duration outliers (kept, flagged): "
                 f"{meta.loc[meta.wedge_duration_outlier, ['trial_id','label3','wedge_dur_s']].to_dict('records')}")
    lines.append(f"Controller settings constant: Fhard={meta.fz_hard.unique()}, "
                 f"Fsoft={meta.fz_soft.unique()}, speed={meta.speed.unique()}")
    lines.append(f"Duplicate timestamps total: {meta.duplicate_times.sum()}")
    lines.append(f"\nFeatures built: {feats.shape[1]-1} "
                 f"(groups: " + ", ".join(f"{g}={sum(c.startswith(g+'__') for c in feats.columns)}"
                                          for g in 'WHXVP') + ")")
    lines.append(f"FIH invariant statistics: {fih.shape[1]-1}; FIH signature shape per trial: "
                 f"{sigs.shape[1]} points x {sigs.shape[2]} channels (a, l)")
    lines.append(f"Feature cells that are NaN: {int(feats.drop(columns='trial_id').isna().sum().sum())}")
    lines.append("\nColumns excluded from modelling and why:")
    for k, v in EXCLUDED_COLUMNS.items():
        lines.append(f"  {k:18s} {v}")
    txt = "\n".join(lines)
    (RESULTS / "qc_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print("\nDone. Next: python scripts/02_eda.py")


if __name__ == "__main__":
    main()

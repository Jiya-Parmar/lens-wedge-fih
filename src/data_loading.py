"""Loading raw trial CSVs, quality checks, and burst aggregation.

Each CSV in data/raw/<class>/csv/ is ONE wedge trial with ONE attempt.
Rows are force/pose samples. The logger records in bursts (~55 samples at ~1 kHz)
separated by ~0.27 s gaps, so we aggregate every burst into one row before
computing features. Within-burst raw variation is kept as a separate
"vibration" measurement.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, DATA_RAW, GAP_S, MIN_BURST_SAMPLES_FOR_VIB

FORCE_F = ["fx", "fy", "fz", "fmag"]          # filtered, baseline-removed (logged)
FORCE_RAW = ["fx_raw", "fy_raw", "fz_raw"]    # raw joint-torque wrench estimate
KIN = ["distance", "dist_to_target"]

# Columns that must NEVER be used as model inputs, and why.
EXCLUDED_COLUMNS = {
    "label": "the target itself (direct leakage)",
    "wedge_succeeded": "constant True in every trial, carries no information",
    "trial_id": "timestamp, identifies session",
    "experiment": "constant",
    "attempt": "always 1",
    "rim_pos_x": "constant within a session, identifies session perfectly",
    "rim_pos_y": "constant within a session, identifies session perfectly",
    "rim_pos_z": "constant within a session, identifies session perfectly",
    "wedge_target_x": "constant within a session, identifies session perfectly",
    "wedge_target_y": "constant within a session, identifies session perfectly",
    "wedge_target_z": "constant within a session, identifies session perfectly",
    "fz_hard_limit": "constant controller setting",
    "fz_soft_limit": "constant controller setting",
    "speed_wedge_mm_s": "constant controller setting",
    "x": "absolute pose, fixture/session coupled",
    "y": "absolute pose, fixture/session coupled",
    "z": "absolute pose, fixture/session coupled",
}


def extract_zips_if_needed(raw_dir: Path = DATA_RAW) -> None:
    """If the three class zips are sitting in data/raw, unzip them once."""
    for cls in CLASSES:
        target = raw_dir / cls / "csv"
        zp = raw_dir / f"{cls}.zip"
        if not target.exists() and zp.exists():
            print(f"  extracting {zp.name} ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(raw_dir)


def find_trial_files(raw_dir: Path = DATA_RAW):
    files = []
    for cls in CLASSES:
        d = raw_dir / cls / "csv"
        if not d.exists():
            raise FileNotFoundError(
                f"Missing folder {d}. Put success.zip, failure.zip and "
                f"partial_contact.zip (or their extracted folders) in {raw_dir}."
            )
        for f in sorted(d.glob("*.csv")):
            files.append((f, cls))
    return files


def load_trial(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    return df


def qc_trial(df: pd.DataFrame, folder_label: str) -> dict:
    """Return quality-control facts for one trial."""
    t = df["time"].to_numpy()
    dt = np.diff(t)
    phase_changes = int((df["phase"] != df["phase"].shift()).sum()) - 1
    w = df[df.phase == "wedge_attempt"]
    h = df[df.phase == "hold"]
    return {
        "n_samples": len(df),
        "n_nan": int(df.isna().sum().sum()),
        "label_matches_folder": bool((df["label"] == folder_label).all()),
        "n_attempts": int(df["attempt"].nunique()),
        "phases_ok": bool(set(df.phase.unique()) == {"wedge_attempt", "hold"}
                          and phase_changes == 1
                          and df.phase.iloc[0] == "wedge_attempt"),
        "duplicate_times": int((dt == 0).sum()),
        "n_gaps": int((dt > GAP_S).sum()),
        "max_gap_s": float(dt.max()),
        "wedge_dur_s": float(w.time.max() - w.time.min()),
        "hold_dur_s": float(h.time.max() - h.time.min()),
        "fz_hard": float(df.fz_hard_limit.iloc[0]),
        "fz_soft": float(df.fz_soft_limit.iloc[0]),
        "speed": float(df.speed_wedge_mm_s.iloc[0]),
    }


def _detrended_std(t: np.ndarray, v: np.ndarray) -> float:
    if len(v) < MIN_BURST_SAMPLES_FOR_VIB:
        return np.nan
    tc = t - t.mean()
    denom = (tc ** 2).sum()
    slope = (tc * (v - v.mean())).sum() / denom if denom > 0 else 0.0
    resid = v - v.mean() - slope * tc
    return float(resid.std(ddof=1))


def to_bursts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a trial into bursts. A new burst starts at a gap > GAP_S or at a phase change."""
    t = df["time"].to_numpy()
    new = np.r_[True, (np.diff(t) > GAP_S) | (df["phase"].to_numpy()[1:] != df["phase"].to_numpy()[:-1])]
    bid = np.cumsum(new) - 1
    out = []
    for b, g in df.groupby(bid, sort=True):
        tt = g["time"].to_numpy()
        row = {"burst": int(b), "phase": g["phase"].iloc[0], "t": float(tt.mean()),
               "n": len(g)}
        for c in FORCE_F + KIN:
            row[c] = float(g[c].mean())
        for c in FORCE_RAW:
            row[f"vib_{c[:2]}"] = _detrended_std(tt, g[c].to_numpy())
        out.append(row)
    return pd.DataFrame(out)


def make_target(label3: pd.Series, scheme: str) -> pd.Series:
    """Map the three recorded outcomes to a binary target (1 = seated).

    success_vs_rest : success=1, failure=0, partial_contact=0   (MAIN, as requested)
    paper_merge     : success=1, partial_contact=1, failure=0  (merge used in the internship paper)
    drop_partial    : success=1, failure=0, partial rows -> NaN (removed)
    """
    if scheme == "success_vs_rest":
        return (label3 == "success").astype(float)
    if scheme == "paper_merge":
        return label3.isin(["success", "partial_contact"]).astype(float)
    if scheme == "drop_partial":
        y = (label3 == "success").astype(float)
        y[label3 == "partial_contact"] = np.nan
        return y
    raise ValueError(scheme)

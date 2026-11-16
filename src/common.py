"""Helpers shared by the scripts: loading processed data, cached DTW matrices, model inputs."""
import numpy as np
import pandas as pd

from .config import DATA_PROC, DTW_BAND
from .dtw import pairwise_dtw, pairwise_phase_dtw
from .features import MAIN_GROUPS, select_groups


def load_processed():
    meta_path = DATA_PROC / "trials.csv"
    if not meta_path.exists():
        raise FileNotFoundError("Run scripts/01_build_dataset.py first.")
    meta = pd.read_csv(meta_path, dtype={"trial_id": str, "session": str})
    feat = pd.read_csv(DATA_PROC / "features.csv", dtype={"trial_id": str}).set_index("trial_id")
    feat = feat.loc[meta.trial_id]
    s = np.load(DATA_PROC / "series.npz", allow_pickle=False)
    assert list(s["trial_id"]) == list(meta.trial_id), "series order mismatch"
    return meta, feat, s["wedge"], s["hold"]


def load_fih(meta):
    """The 14 FIH statistics (DataFrame) and the FIH signatures (n, L, 2), in meta order."""
    path = DATA_PROC / "fih_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/01_build_dataset.py first.")
    fih = pd.read_csv(path, dtype={"trial_id": str}).set_index("trial_id").loc[meta.trial_id]
    s = np.load(DATA_PROC / "fih_series.npz", allow_pickle=False)
    assert list(s["trial_id"]) == list(meta.trial_id), "FIH series order mismatch"
    return fih, s["sig"]


def _cached(path, meta, compute):
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if list(z["trial_id"]) == list(meta.trial_id) and int(z["band"]) == DTW_BAND:
            return z["D"]
    print(f"  computing {path.name} (one time) ...")
    D = compute()
    np.savez(path, D=D, trial_id=meta.trial_id.to_numpy().astype(str), band=DTW_BAND)
    return D


def get_dtw_matrix(meta, wedge, hold):
    return _cached(DATA_PROC / "dtw_matrix.npz", meta,
                   lambda: pairwise_phase_dtw(wedge, hold, DTW_BAND))


def get_fih_dtw_matrix(meta, sig):
    # label-free and no learned scaling, so one matrix over all trials does not leak
    return _cached(DATA_PROC / "dtw_fih_matrix.npz", meta, lambda: pairwise_dtw(sig, DTW_BAND))


def build_inputs(meta, feat, wedge, hold):
    """Every model input in one dictionary; each model spec names the key it uses."""
    fih, sig = load_fih(meta)
    return dict(
        features=select_groups(feat, MAIN_GROUPS).to_numpy(float),
        fih=fih.to_numpy(float),
        dtw=get_dtw_matrix(meta, wedge, hold),
        dtw_fih=get_fih_dtw_matrix(meta, sig),
        index=np.arange(len(meta)).reshape(-1, 1),
    )


def make_strata(y, session):
    """Outer CV is stratified on label x session so every fold sees both sessions."""
    codes = pd.factorize(session)[0]
    return (np.asarray(y).astype(int) * 10 + codes).astype(int)

"""Physics-motivated, sampling-aware features computed on burst means.

Feature names are "<GROUP>__<name>" so groups can be ablated by prefix:
  W : wedge-phase force         (what happens while the arm pushes)
  H : hold-phase force          (does the contact settle afterwards?)
  X : cross-phase force change  (hold relative to end of wedge)
  V : within-burst vibration    (raw 1 kHz content inside each burst)
  P : position / kinematics     (insertion depth, creep; fixture coupled, NOT in main set)

Durations and absolute poses are deliberately not used: they encode the
controller's behaviour and the session, not the physics of seating.
"""
import numpy as np
import pandas as pd

from .config import N_POINTS_PER_PHASE

MAIN_GROUPS = ["W", "H", "X", "V"]
ALL_GROUPS = ["W", "H", "X", "V", "P"]
EPS = 1e-3


def _slope(t, v):
    if len(v) < 2 or np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, v, 1)[0])


def _resid_std(t, v):
    if len(v) < 3 or np.ptp(t) == 0:
        return np.nan
    p = np.polyfit(t, v, 1)
    return float(np.std(v - np.polyval(p, t), ddof=1))


def _std(v):
    return float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan


def _hold_channel_stats(t, v, prefix):
    f = {}
    n = len(v)
    if n == 0:
        return f
    half = n // 2
    f[f"{prefix}_mean"] = float(v.mean())
    f[f"{prefix}_std"] = _std(v)
    f[f"{prefix}_range"] = float(np.ptp(v))
    f[f"{prefix}_first"] = float(v[0])
    f[f"{prefix}_last"] = float(v[-2:].mean())
    f[f"{prefix}_delta"] = float(v[-2:].mean() - v[0])
    f[f"{prefix}_slope"] = _slope(t, v)                      # N/s drift during hold
    f[f"{prefix}_resid_std"] = _resid_std(t, v)              # unsettledness after removing drift
    dur = np.ptp(t)
    f[f"{prefix}_tv_rate"] = float(np.abs(np.diff(v)).sum() / dur) if dur > 0 else np.nan
    f[f"{prefix}_late_slope"] = _slope(t[half:], v[half:]) if n - half >= 2 else np.nan
    if half >= 2 and n - half >= 2:
        f[f"{prefix}_settle_logratio"] = float(np.log((_std(v[half:]) + EPS) / (_std(v[:half]) + EPS)))
    else:
        f[f"{prefix}_settle_logratio"] = np.nan
    return f


def extract_features(b: pd.DataFrame, hold_max_s=None) -> dict:
    """Features for one trial from its burst table.

    hold_max_s: if given, only the first hold_max_s seconds of the hold are used
                (0 means wedge phase only). Used for the time-to-decision ablation.
    """
    w = b[b.phase == "wedge_attempt"].sort_values("t")
    h = b[b.phase == "hold"].sort_values("t")
    if len(h):
        h = h.assign(trel=h.t - h.t.iloc[0])
        if hold_max_s is not None:
            h = h[h.trel <= hold_max_s + 1e-9]
    f = {}

    # ---- W: wedge phase ----
    tw = (w.t - w.t.iloc[0]).to_numpy()
    for c in ["fx", "fy", "fz", "fmag"]:
        v = w[c].to_numpy()
        f[f"W__{c}_peakabs"] = float(np.abs(v).max())
        f[f"W__{c}_mean"] = float(v.mean())
        f[f"W__{c}_std"] = _std(v)
        f[f"W__{c}_last"] = float(v[-2:].mean())
    d = w.distance.to_numpy() * 1000.0                       # mm
    f["W__stiff_fz_per_mm"] = _slope(d, w.fz.to_numpy())     # contact stiffness proxy
    f["W__work_Nmm"] = float(np.trapezoid(w.fmag.to_numpy(), d)) if hasattr(np, "trapezoid") \
        else float(np.trapz(w.fmag.to_numpy(), d))
    lat_w = np.hypot(w.fx, w.fy).to_numpy()
    f["W__lat_ratio"] = float(lat_w.mean() / (np.abs(w.fz).mean() + 0.1))
    f["W__peak_time_frac"] = float(tw[np.argmax(np.abs(w.fz.to_numpy()))] / tw[-1]) if tw[-1] > 0 else np.nan

    # ---- H: hold phase ----
    have_hold = len(h) >= 2
    if hold_max_s is None or hold_max_s > 0:
        if have_hold:
            th = h.trel.to_numpy()
            for c in ["fx", "fy", "fz", "fmag"]:
                f.update({f"H__{k}": v for k, v in _hold_channel_stats(th, h[c].to_numpy(), c).items()})
            lat_h = np.hypot(h.fx, h.fy).to_numpy()
            f["H__lat_ratio"] = float(lat_h.mean() / (np.abs(h.fz).mean() + 0.1))
            a0 = np.arctan2(h.fy.iloc[0], h.fx.iloc[0])
            a1 = np.arctan2(h.fy.iloc[-1], h.fx.iloc[-1])
            f["H__lat_angle_change"] = float(np.abs(np.angle(np.exp(1j * (a1 - a0)))))
            dh = h.distance.to_numpy() * 1000.0
            f["H__stiff_fz_per_mm"] = _slope(dh, h.fz.to_numpy()) if np.ptp(dh) > 0.05 else np.nan
            # ---- X: cross-phase ----
            f["X__fz_hold_minus_wedge_end"] = float(h.fz.mean() - w.fz.iloc[-2:].mean())
            f["X__fmag_hold_over_wedge_peak"] = float(h.fmag.mean() / (np.abs(w.fmag).max() + 0.1))
            f["X__lat_ratio_change"] = f["H__lat_ratio"] - f["W__lat_ratio"]
            f["X__fz_std_hold_over_wedge"] = float(np.log((_std(h.fz.to_numpy()) + EPS) /
                                                          (_std(w.fz.to_numpy()) + EPS)))

    # ---- V: within-burst vibration (raw signal, drift-free by detrending) ----
    for ph, g in (("w", w), ("h", h)):
        if ph == "h" and (hold_max_s == 0 or not have_hold):
            continue
        for c in ["fx", "fy", "fz"]:
            f[f"V__{ph}_{c}_vib"] = float(np.nanmedian(g[f"vib_{c}"])) if g[f"vib_{c}"].notna().any() else np.nan
    if "V__h_fz_vib" in f:
        f["V__fz_vib_logratio_h_w"] = float(np.log((f["V__h_fz_vib"] + EPS) / (f["V__w_fz_vib"] + EPS)))

    # ---- P: position / kinematics (ablation only) ----
    f["P__wedge_depth_mm"] = float(w.distance.max() * 1000)
    f["P__wedge_end_dist_to_target_mm"] = float(w.dist_to_target.iloc[-1] * 1000)
    if have_hold and (hold_max_s is None or hold_max_s > 0):
        f["P__max_depth_mm"] = float(max(w.distance.max(), h.distance.max()) * 1000)
        f["P__hold_creep_mm"] = float((h.distance.iloc[-1] - h.distance.iloc[0]) * 1000)
        f["P__final_dist_to_target_mm"] = float(h.dist_to_target.iloc[-1] * 1000)
    return f


def resample_phase(g: pd.DataFrame, cols, n_points: int) -> np.ndarray:
    """Linear interpolation of burst means onto n_points over normalised phase time [0, 1]."""
    t = g.t.to_numpy()
    tn = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else np.linspace(0, 1, len(t))
    grid = np.linspace(0, 1, n_points)
    return np.stack([np.interp(grid, tn, g[c].to_numpy()) for c in cols], axis=1)


def build_series(b: pd.DataFrame, cols=("fx", "fy", "fz")):
    """Return (wedge_series, hold_series), each (N_POINTS_PER_PHASE, len(cols)), in Newtons."""
    w = b[b.phase == "wedge_attempt"].sort_values("t")
    h = b[b.phase == "hold"].sort_values("t")
    return (resample_phase(w, list(cols), N_POINTS_PER_PHASE),
            resample_phase(h, list(cols), N_POINTS_PER_PHASE))


def select_groups(feat: pd.DataFrame, groups) -> pd.DataFrame:
    cols = [c for c in feat.columns if c.split("__")[0] in groups]
    return feat[cols]

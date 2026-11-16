"""Fixture-Invariant Hold (FIH) representation: the method contributed by this project.

Between sessions the fixture was moved (rim 9.2 cm sideways, approach direction rotated
about 30 degrees in the horizontal plane). We model such a change as a per-trial nuisance
transform of every force sample

    f -> R_theta f + b,

where R_theta rotates (fx, fy) about the vertical axis and b is a constant wrench offset
(pose-dependent bias of the joint-torque force estimate, different preload). The FIH
representation is built so that it is exactly unchanged by any such transform:

  1. Anchor:  subtract the first hold burst, d(t) = f(t) - f(t0)   (removes b)
  2. Axial:   a(t) = d_z(t)                                        (R_theta leaves z alone)
  3. Lateral: l(t) = || (d_x(t), d_y(t)) ||                        (rotation keeps norms)

Burst averaging and linear time resampling are affine combinations of samples, so they
commute with the transform and the invariance holds through the whole pipeline. The wedge
phase is not used: it is driven by the compliant controller and its relation to seating
reversed between sessions. scripts/09_invariance_diagnostics.py checks the invariance
numerically.
"""
import numpy as np
import pandas as pd

from .config import N_POINTS_PER_PHASE
from .features import resample_phase


def _slope(t, v):
    return float(np.polyfit(t, v, 1)[0])


def _anchored_hold(h: pd.DataFrame):
    """Return hold time from onset, axial change a(t) and lateral change (dx, dy)."""
    t = (h.t - h.t.iloc[0]).to_numpy()
    a = h.fz.to_numpy() - h.fz.iloc[0]
    dx = h.fx.to_numpy() - h.fx.iloc[0]
    dy = h.fy.to_numpy() - h.fy.iloc[0]
    return t, a, dx, dy


def fih_features(b: pd.DataFrame) -> dict:
    """Fourteen offset- and yaw-invariant statistics of the hold phase of one trial.

    Axial (a = fz change from hold onset; fz < 0 is compression):
      a_end, a_min            final change and largest compression reached during the hold
      a_recover               a_end - a_min: how much of the extra load is released again
      a_slope, a_late_slope   drift over the whole hold and over its second half (N/s)
      a_range, a_resid, a_tv  spread, detrended spread and total variation rate
    Lateral (d = (fx, fy) change from hold onset, l = |d|):
      l_max, l_end, l_mean    size of the sideways excursion
      l_drift                 norm of the 2-D linear drift vector (N/s)
      l_resid                 sqrt(trace of detrended 2-D covariance), invariant to rotation
      l_path                  sideways path length per second
    """
    h = b[b.phase == "hold"].sort_values("t")
    t, a, dx, dy = _anchored_hold(h)
    l = np.hypot(dx, dy)
    half = len(a) // 2
    px, py, pa = np.polyfit(t, dx, 1), np.polyfit(t, dy, 1), np.polyfit(t, a, 1)
    rx, ry = dx - np.polyval(px, t), dy - np.polyval(py, t)
    dur = float(np.ptp(t))
    return {
        "FIH__a_end": float(a[-2:].mean()),
        "FIH__a_min": float(a.min()),
        "FIH__a_recover": float(a[-2:].mean() - a.min()),
        "FIH__a_slope": float(pa[0]),
        "FIH__a_late_slope": _slope(t[half:], a[half:]),
        "FIH__a_range": float(np.ptp(a)),
        "FIH__a_resid": float(np.std(a - np.polyval(pa, t), ddof=1)),
        "FIH__a_tv": float(np.abs(np.diff(a)).sum() / dur),
        "FIH__l_max": float(l.max()),
        "FIH__l_end": float(l[-2:].mean()),
        "FIH__l_mean": float(l.mean()),
        "FIH__l_drift": float(np.hypot(px[0], py[0])),
        "FIH__l_resid": float(np.sqrt(rx.var(ddof=1) + ry.var(ddof=1))),
        "FIH__l_path": float(np.hypot(np.diff(dx), np.diff(dy)).sum() / dur),
    }


def fih_signature(b: pd.DataFrame) -> np.ndarray:
    """Two-channel invariant hold trace (a, l), resampled to N_POINTS_PER_PHASE points."""
    h = b[b.phase == "hold"].sort_values("t")
    s = resample_phase(h, ["fx", "fy", "fz"], N_POINTS_PER_PHASE)
    d = s - s[:1]
    return np.stack([d[:, 2], np.hypot(d[:, 0], d[:, 1])], axis=1)


def apply_fixture_change(df: pd.DataFrame, yaw_deg: float = 0.0,
                         offset_n=(0.0, 0.0, 0.0)) -> pd.DataFrame:
    """Simulate a fixture change on a raw trial: rotate (fx, fy) about z and add an offset.

    Applied to both the filtered and the raw force columns; magnitudes are recomputed
    because the logged fmag columns are exact Euclidean norms of their components.
    """
    th = np.deg2rad(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    out = df.copy()
    for suf in ("", "_raw"):
        fx, fy, fz = df[f"fx{suf}"], df[f"fy{suf}"], df[f"fz{suf}"]
        out[f"fx{suf}"] = c * fx - s * fy + offset_n[0]
        out[f"fy{suf}"] = s * fx + c * fy + offset_n[1]
        out[f"fz{suf}"] = fz + offset_n[2]
        out[f"fmag{suf}"] = np.sqrt(out[f"fx{suf}"] ** 2 + out[f"fy{suf}"] ** 2 + out[f"fz{suf}"] ** 2)
    return out

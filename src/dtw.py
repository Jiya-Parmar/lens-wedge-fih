"""Multivariate (dependent) dynamic time warping with a Sakoe-Chiba band.

Distances are computed per phase and summed, so warping can never move a
wedge-phase point into the hold phase. All channels are in Newtons, so no
learned scaling is needed and the full distance matrix is label-free. This
means it can be computed once on all trials without leaking test labels.
"""
import numpy as np


def dtw(a: np.ndarray, b: np.ndarray, band: int) -> float:
    """a, b: (L, C) arrays. Returns sqrt of minimal cumulative squared Euclidean cost."""
    la, lb = len(a), len(b)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    D = np.full((la + 1, lb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, la + 1):
        j0, j1 = max(1, i - band), min(lb, i + band)
        for j in range(j0, j1 + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[la, lb]))


def pairwise_phase_dtw(wedge: np.ndarray, hold: np.ndarray, band: int) -> np.ndarray:
    """wedge, hold: (n_trials, L, C). Returns symmetric (n, n) distance matrix."""
    n = len(wedge)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(wedge[i], wedge[j], band) + dtw(hold[i], hold[j], band)
            D[i, j] = D[j, i] = d
    return D


def pairwise_dtw(series: np.ndarray, band: int) -> np.ndarray:
    """series: (n_trials, L, C), one phase. Returns symmetric (n, n) distance matrix."""
    n = len(series)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw(series[i], series[j], band)
    return D


def cross_dtw(A: np.ndarray, B: np.ndarray, band: int) -> np.ndarray:
    """Rectangular (len(A), len(B)) DTW distances between two sets of one-phase series."""
    return np.array([[dtw(a, b, band) for b in B] for a in A])


def cross_phase_dtw(wA, hA, wB, hB, band: int) -> np.ndarray:
    """Rectangular version of pairwise_phase_dtw (wedge distance plus hold distance)."""
    return cross_dtw(wA, wB, band) + cross_dtw(hA, hB, band)

"""Nested cross-validation, honest threshold selection, metrics and bootstrap CIs.

Outer loop  : repeated stratified K-fold (stratified on label x session). The outer
              test fold is used ONLY to score the final model of that fold.
Inner loop  : on the outer-training data only
              (1) GridSearchCV picks hyperparameters by ROC-AUC,
              (2) inner out-of-fold probabilities of the chosen model pick two decision
                  thresholds: tau_ba (maximises balanced accuracy) and tau_safe (the
                  lowest threshold whose false-seated rate is <= MAX_FALSE_SEATED,
                  i.e. the most sensitive rule that respects the safety limit).
So hyperparameters AND thresholds never see the outer test fold.
"""
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)

from .config import (COST_FALSE_REJECT, COST_FALSE_SEATED, INNER_SPLITS, MAX_FALSE_SEATED,
                     N_BOOT, OUTER_SPLITS, SEED, THRESH_GRID_STEP)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
TAU_GRID = np.round(np.arange(THRESH_GRID_STEP, 1.0, THRESH_GRID_STEP), 4)


# ---------------------------------------------------------------- metrics
def confusion_at(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fn, fp, tn


def thresh_metrics(y, p, tau, c_fp=COST_FALSE_SEATED, c_fn=COST_FALSE_REJECT):
    tp, fn, fp, tn = confusion_at(y, p, tau)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(sens=sens, spec=spec, ba=(sens + spec) / 2,
                false_seated_rate=1 - spec,
                ppv=tp / (tp + fp) if tp + fp else np.nan,
                cost_per_trial=(c_fp * fp + c_fn * fn) / len(y),
                tp=tp, fn=fn, fp=fp, tn=tn)


def pred_metrics(y, pred, c_fp=COST_FALSE_SEATED, c_fn=COST_FALSE_REJECT):
    """Same as thresh_metrics but from hard predictions (each fold used its own threshold)."""
    return thresh_metrics(y, pred.astype(float), 0.5, c_fp, c_fn)


def _pick_middle(mask):
    idx = np.flatnonzero(mask)
    return float(TAU_GRID[idx[len(idx) // 2]])


def safe_threshold(y, p, max_fsr=MAX_FALSE_SEATED):
    """Lowest grid threshold whose false-seated rate (1 - specificity) is <= max_fsr."""
    for t in TAU_GRID:
        if thresh_metrics(y, p, t)["false_seated_rate"] <= max_fsr + 1e-12:
            return float(t)
    return float(TAU_GRID[-1])


def choose_thresholds(y, p, max_fsr=MAX_FALSE_SEATED):
    """BA-optimal threshold (ties -> middle of optimal range) and the safety-constrained threshold."""
    bas = np.nan_to_num(np.asarray([thresh_metrics(y, p, t)["ba"] for t in TAU_GRID]))
    return dict(tau_ba=_pick_middle(np.isclose(bas, bas.max())),
                tau_safe=safe_threshold(y, p, max_fsr))


def auc_rows(y, P):
    """Rank AUC for each row of P (R, n). Handles ties."""
    P = np.atleast_2d(P)
    r = rankdata(P, axis=1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------- nested CV
def _slice(X, input_type, tr, te):
    """Distance matrices (any input named dtw*) are sliced by block, everything else by row."""
    if input_type.startswith("dtw"):
        return X[np.ix_(tr, tr)], X[np.ix_(te, tr)]
    return X[tr], X[te]


def fit_and_threshold(spec, Xtr, ytr, seed, n_jobs=-1):
    """Tune on training data, then pick thresholds from inner OOF probabilities."""
    inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
    if spec["grid"]:
        gs = GridSearchCV(clone(spec["estimator"]), spec["grid"], scoring="roc_auc",
                          cv=inner, n_jobs=n_jobs, refit=True, error_score="raise")
        gs.fit(Xtr, ytr)
        model, params = gs.best_estimator_, gs.best_params_
    else:
        model, params = clone(spec["estimator"]).fit(Xtr, ytr), {}
    p_inner = cross_val_predict(clone(model), Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
    taus = choose_thresholds(ytr, p_inner)
    return model, params, taus


def nested_cv(spec, X, y, strata, n_repeats, seed=SEED, n_jobs=-1, verbose=True, name=""):
    n = len(y)
    rskf = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=n_repeats, random_state=seed)
    prob = np.full((n_repeats, n), np.nan)
    pred_safe = np.full((n_repeats, n), -1)
    pred_ba = np.full((n_repeats, n), -1)
    folds = []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(n), strata)):
        r = k // OUTER_SPLITS
        Xtr, Xte = _slice(X, spec["input"], tr, te)
        model, params, taus = fit_and_threshold(spec, Xtr, y[tr], seed + k, n_jobs)
        p = model.predict_proba(Xte)[:, 1]
        prob[r, te] = p
        pred_safe[r, te] = (p >= taus["tau_safe"]).astype(int)
        pred_ba[r, te] = (p >= taus["tau_ba"]).astype(int)
        rec = dict(repeat=r, fold=k % OUTER_SPLITS, **taus,
                   **{f"param_{a}": b for a, b in params.items()})
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "best_index_"):
            rec["single_feature_index"] = clf.best_index_
        folds.append(rec)
        if verbose and (k + 1) % OUTER_SPLITS == 0:
            print(f"    {name:<13s} repeat {r + 1}/{n_repeats} done")
    assert not np.isnan(prob).any() and (pred_safe >= 0).all()
    return dict(prob=prob, pred_safe=pred_safe, pred_ba=pred_ba, folds=folds)


def per_repeat_metrics(y, res):
    rows = []
    for r in range(res["prob"].shape[0]):
        p = res["prob"][r]
        ms = pred_metrics(y, res["pred_safe"][r])
        mb = pred_metrics(y, res["pred_ba"][r])
        rows.append(dict(
            repeat=r,
            roc_auc=roc_auc_score(y, p),
            pr_auc=average_precision_score(y, p),
            brier=brier_score_loss(y, p),
            ba_at_tau_ba=mb["ba"], sens_at_tau_ba=mb["sens"], spec_at_tau_ba=mb["spec"],
            false_seated_at_tau_ba=mb["false_seated_rate"],
            ba_at_tau_safe=ms["ba"], sens_at_tau_safe=ms["sens"], spec_at_tau_safe=ms["spec"],
            false_seated_at_tau_safe=ms["false_seated_rate"]))
    return rows


# ---------------------------------------------------------------- bootstrap
def bootstrap_indices(y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            out.append(idx)
    return out


def bootstrap_auc(y, P, boots):
    """CI for the mean-over-repeats AUC. P is (R, n). Trials are resampled; repeats kept."""
    vals = np.array([auc_rows(y[i], P[:, i]).mean() for i in boots])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def paired_auc_difference(y, PA, PB, boots):
    a = np.array([auc_rows(y[i], PA[:, i]).mean() for i in boots])
    b = np.array([auc_rows(y[i], PB[:, i]).mean() for i in boots])
    d = a - b
    point = auc_rows(y, PA).mean() - auc_rows(y, PB).mean()
    return dict(diff=float(point), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_greater=float((d > 0).mean()))

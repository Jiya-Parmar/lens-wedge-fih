"""Baselines and models compared in this project.

  Prior            : predicts the training-fold class prior (AUC = 0.5 by construction)
  SingleFeature    : picks the ONE feature with the best training-fold AUC, then a 1-D
                     logistic regression on it. This is the "well-chosen threshold" a
                     conventional pipeline would use, chosen without seeing test data.
  LogReg           : L2-regularised logistic regression on standardised features
  RandomForest     : bagged decision trees (bagging, not boosting)
  SVM-RBF          : kernel support vector machine; probabilities from a monotone Platt map
                     fitted on internal CV decision values (see PlattSVC for why)
  kNN-DTW          : k-nearest neighbours on the raw burst-mean traces, DTW distance

Proposed (fixture-invariant hold, FIH; see src/invariant.py):
  FIH-kNN          : the same kNN-DTW learner on the two-channel invariant hold signature
  FIH-LR           : the same logistic regression on the 14 invariant hold statistics
  FIHV             : Fixture-Invariant Hold Verifier, the average of the two probabilities
Instrument for the trade-off study (not a proposed model):
  MixedDTWkNN      : kNN on a blend of the invariant and the raw DTW distance
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED


class BestSingleFeature(ClassifierMixin, BaseEstimator):
    """Select the single most discriminative column on the training data, then fit a 1-D logistic model."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        aucs = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.nanstd(col) == 0:
                aucs.append(0.5)
                continue
            aucs.append(roc_auc_score(y, col))
        aucs = np.asarray(aucs)
        self.best_index_ = int(np.argmax(np.abs(aucs - 0.5)))
        self.train_auc_ = float(max(aucs[self.best_index_], 1 - aucs[self.best_index_]))
        self.lr_ = LogisticRegression(class_weight="balanced", max_iter=5000)
        self.lr_.fit(X[:, [self.best_index_]], y)
        self.classes_ = self.lr_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        return self.lr_.predict_proba(X[:, [self.best_index_]])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class PlattSVC(ClassifierMixin, BaseEstimator):
    """RBF-SVM whose probabilities can never invert the decision-function ranking.

    sklearn's SVC(probability=True) fits libsvm's internal Platt scaling. On small data that
    fit can come out with the wrong sign, so predict_proba ranks trials in the OPPOSITE order
    to decision_function (we observed AUC 0.29 vs 0.66 for the same model at C=0.1).
    Here the Platt map is a 1-D logistic regression on 5-fold out-of-fold decision values of
    the training data, and its slope is forced to be non-negative.
    """

    def __init__(self, C=1.0, gamma="scale", class_weight="balanced", random_state=0):
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    def _svc(self):
        return SVC(kernel="rbf", C=self.C, gamma=self.gamma,
                   class_weight=self.class_weight, random_state=self.random_state)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.svc_ = self._svc().fit(X, y)
        n_min = np.bincount(y.astype(int)).min()
        k = int(min(5, n_min))
        if k >= 2:
            cv = StratifiedKFold(k, shuffle=True, random_state=self.random_state)
            d = cross_val_predict(self._svc(), X, y, cv=cv, method="decision_function")
        else:
            d = self.svc_.decision_function(X)
        self.platt_ = LogisticRegression(C=1e4, max_iter=5000).fit(d.reshape(-1, 1), y)
        self.platt_flipped_ = bool(self.platt_.coef_[0, 0] <= 0)
        self.classes_ = self.svc_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(np.asarray(X, float))

    def predict_proba(self, X):
        d = self.decision_function(X)
        if self.platt_flipped_:
            p1 = 1.0 / (1.0 + np.exp(-d))          # monotone fallback, uncalibrated
        else:
            p1 = self.platt_.predict_proba(d.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def _prep(scale=True):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


class FIHVerifier(ClassifierMixin, BaseEstimator):
    """Late fusion of FIH-kNN (shape of the hold response) and FIH-LR (its summary statistics).

    p(seated) = w * p_kNN + (1 - w) * p_LR. Both parts see only invariant inputs, so the fused
    score is invariant too. w is fixed at 0.5 (no evidence to weight one part more);
    k and C are tuned by the inner CV like any other hyperparameter.
    The "features" passed to fit/predict are trial indices into the label-free matrices D
    (FIH DTW distances) and F (FIH statistics), so the model runs through the same nested-CV
    code as every other model.
    """

    def __init__(self, D=None, F=None, k=7, C=0.1, w=0.5):
        self.D = D
        self.F = F
        self.k = k
        self.C = C
        self.w = w

    def fit(self, X, y):
        idx = np.asarray(X)[:, 0].astype(int)
        self.train_idx_ = idx
        self.knn_ = KNeighborsClassifier(min(self.k, len(idx) - 1), metric="precomputed",
                                         weights="distance").fit(self.D[np.ix_(idx, idx)], y)
        self.lr_ = Pipeline(_prep() + [("clf", LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=10000))]).fit(self.F[idx], y)
        self.classes_ = self.knn_.classes_
        self.n_features_in_ = 1
        return self

    def predict_proba(self, X):
        idx = np.asarray(X)[:, 0].astype(int)
        p_knn = self.knn_.predict_proba(self.D[np.ix_(idx, self.train_idx_)])[:, 1]
        p_lr = self.lr_.predict_proba(self.F[idx])[:, 1]
        p = self.w * p_knn + (1 - self.w) * p_lr
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class MixedDTWkNN(ClassifierMixin, BaseEstimator):
    """kNN on a blend of the invariant and the raw DTW distance (invariance trade-off study).

    D_lam = (1 - lam) * D_inv / m_inv + lam * D_raw / m_raw, where m is the median pairwise
    distance among the TRAINING trials, so the two scales are comparable and no test trial
    influences the blend. lam = 0 is FIH-kNN, lam = 1 is the original kNN-DTW.
    """

    def __init__(self, D_inv=None, D_raw=None, lam=0.0, k=7):
        self.D_inv = D_inv
        self.D_raw = D_raw
        self.lam = lam
        self.k = k

    def _blend(self, rows, cols):
        return ((1 - self.lam) * self.D_inv[np.ix_(rows, cols)] / self.m_inv_
                + self.lam * self.D_raw[np.ix_(rows, cols)] / self.m_raw_)

    def fit(self, X, y):
        idx = np.asarray(X)[:, 0].astype(int)
        self.train_idx_ = idx
        off = ~np.eye(len(idx), dtype=bool)
        self.m_inv_ = float(np.median(self.D_inv[np.ix_(idx, idx)][off]))
        self.m_raw_ = float(np.median(self.D_raw[np.ix_(idx, idx)][off]))
        self.knn_ = KNeighborsClassifier(min(self.k, len(idx) - 1), metric="precomputed",
                                         weights="distance").fit(self._blend(idx, idx), y)
        self.classes_ = self.knn_.classes_
        self.n_features_in_ = 1
        return self

    def predict_proba(self, X):
        idx = np.asarray(X)[:, 0].astype(int)
        return self.knn_.predict_proba(self._blend(idx, self.train_idx_))

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


K_GRID = [3, 5, 7, 9, 11]
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BASELINE_MODELS = ["Prior", "SingleFeature", "LogReg", "RandomForest", "SVM-RBF", "kNN-DTW"]
PROPOSED_MODELS = ["FIH-kNN", "FIH-LR", "FIHV"]
ALL_MODELS = BASELINE_MODELS + PROPOSED_MODELS


def get_model_specs(include=None, inputs=None):
    """Return {name: dict(estimator, grid, input)}; input names a key of common.build_inputs.

    inputs (the dictionary from common.build_inputs) is needed only by FIHV, whose estimator
    carries the label-free FIH distance matrix and statistics table.
    """
    names = ALL_MODELS if include is None else list(include)
    unknown = [n for n in names if n not in ALL_MODELS]
    if unknown:
        raise ValueError(f"unknown model name(s): {unknown}; choose from {ALL_MODELS}")
    specs = {
        "Prior": dict(
            estimator=DummyClassifier(strategy="prior"),
            grid=None, input="features"),
        "SingleFeature": dict(
            estimator=Pipeline(_prep() + [("clf", BestSingleFeature())]),
            grid=None, input="features"),
        "LogReg": dict(
            estimator=Pipeline(_prep() + [("clf", LogisticRegression(
                class_weight="balanced", max_iter=10000))]),
            grid={"clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
            input="features"),
        "RandomForest": dict(
            estimator=Pipeline(_prep(scale=False) + [("clf", RandomForestClassifier(
                n_estimators=150, max_features="sqrt", class_weight="balanced_subsample",
                random_state=SEED, n_jobs=1))]),
            grid={"clf__max_depth": [3, None],
                  "clf__min_samples_leaf": [1, 4]},
            input="features"),
        "SVM-RBF": dict(
            estimator=Pipeline(_prep() + [("clf", PlattSVC(random_state=SEED))]),
            grid={"clf__C": [0.1, 1.0, 10.0],
                  "clf__gamma": ["scale", 0.003, 0.01, 0.03]},
            input="features"),
        "kNN-DTW": dict(
            estimator=KNeighborsClassifier(metric="precomputed", weights="distance"),
            grid={"n_neighbors": K_GRID},
            input="dtw"),
        "FIH-kNN": dict(
            estimator=KNeighborsClassifier(metric="precomputed", weights="distance"),
            grid={"n_neighbors": K_GRID},
            input="dtw_fih"),
        "FIH-LR": dict(
            estimator=Pipeline(_prep() + [("clf", LogisticRegression(
                class_weight="balanced", max_iter=10000))]),
            grid={"clf__C": C_GRID},
            input="fih"),
    }
    if "FIHV" in names:
        if inputs is None:
            raise ValueError("FIHV needs inputs=common.build_inputs(...)")
        specs["FIHV"] = dict(
            estimator=FIHVerifier(D=inputs["dtw_fih"], F=inputs["fih"]),
            grid={"k": K_GRID, "C": C_GRID},
            input="index")
    return {n: specs[n] for n in names}

"""STEP 11 (diagnostic cited in the paper, Methods): libsvm Platt scaling can invert ranking.

For a plain SVC(probability=True), compares the out-of-fold ROC-AUC of predict_proba with
that of decision_function on the main features. At small C the two disagree in sign
(e.g. ~0.29 vs ~0.66 at C=0.1), which is why models.PlattSVC is used instead.

Usage: python scripts/11_svm_platt_check.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common import load_processed, make_strata
from src.config import SEED
from src.features import MAIN_GROUPS, select_groups

warnings.filterwarnings("ignore", category=FutureWarning)   # SVC(probability=True) is deprecated in new sklearn


def main():
    meta, feat, _, _ = load_processed()
    y = meta.y.to_numpy()
    X = select_groups(feat, MAIN_GROUPS).to_numpy(float)
    strata = make_strata(y, meta.session)
    print(" C      gamma   AUC(predict_proba)  AUC(decision_function)")
    for C in [0.1, 1.0, 10.0]:
        for g in ["scale", 0.003, 0.01]:
            pp, dd = np.zeros(len(y)), np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strata):
                m = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                              ("clf", SVC(kernel="rbf", C=C, gamma=g, probability=True,
                                          class_weight="balanced", random_state=SEED))]).fit(X[tr], y[tr])
                pp[te] = m.predict_proba(X[te])[:, 1]
                dd[te] = m.decision_function(X[te])
            print(f" {C:<6} {str(g):<7} {roc_auc_score(y, pp):18.3f}  {roc_auc_score(y, dd):22.3f}")


if __name__ == "__main__":
    main()

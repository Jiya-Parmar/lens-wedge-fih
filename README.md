# Fixture-invariant lens-seating verification from robot force signals

ES 678 (Machine Learning, IIT Gandhinagar) course project of Jiya Udaybhai Parmar (26250019).
Paper: *Invariant Time-Series Classification under Distribution Shift: Verifying Lens Seating
from Robot Force Signals*.

**Question.** A robot arm pushes a spectacle lens into the groove of a frame (the *wedge*), holds
still for about 5 s (the *hold*) and only then presses hard to snap it in. From the force signal of
the wedge and the hold, can a classifier tell whether the lens is fully seated, and does that
answer survive in a new recording session where the fixture was placed differently?

One row = one trial (one CSV). Target: 1 = success (seated), 0 = failure or partial contact.
The prediction is made after the hold and before the snap-fit.

**Main result.** Under nested random cross-validation the force signal separates seated from
unseated lenses well (kNN with dynamic time warping, ROC-AUC 0.855), but trained on one session
and tested on the other the same model falls to chance (0.479). The fixture moved and rotated
between sessions. The proposed fixture-invariant hold (FIH) representation is provably unchanged
by any constant force offset and any rotation about the vertical axis; with the learner held
fixed it raises cross-session AUC (kNN 0.479 to 0.698, logistic regression 0.604 to 0.760).
With only two sessions this is evidence, not proof, of transfer.

## Folder layout

```
lens_wedge_ml/
  README.md, HOW_TO_RUN.md, AI_USE.md, requirements.txt
  run_all.sh (Linux/Mac), run_all.bat (Windows)    run every step in order
  src/
    config.py          every setting (CV sizes, seeds, safety limit)
    data_loading.py    CSV loading, quality checks, burst aggregation, label schemes
    features.py        83 physics-motivated features (groups W, H, X, V, P) and traces
    invariant.py       the FIH representation (the method of this project)
    dtw.py             dynamic time warping distances
    models.py          baselines, standard models, FIH-kNN, FIH-LR, FIHV
    evaluation.py      nested CV, thresholds, metrics, bootstrap
    common.py          loading processed data, cached distance matrices, model inputs
  scripts/             01 ... 12, run in order (see HOW_TO_RUN.md)
  data/raw/            success.zip, failure.zip, partial_contact.zip (not in the public repo)
  data/processed/      written by step 01
  results/             every table (.csv), log (log_*.txt) and figure (figures/)
```

## Models

| name | input | idea |
|---|---|---|
| Prior | none | predicts the training prior (sanity floor) |
| SingleFeature | 78 features | best single feature chosen inside the training fold, 1-D logistic |
| LogReg | 78 features | L2 logistic regression, C tuned |
| RandomForest | 78 features | 150 bagged trees, depth and leaf size tuned |
| SVM-RBF | 78 features | kernel SVM, C and gamma tuned, monotone Platt map |
| kNN-DTW | raw wedge and hold traces | k nearest neighbours under per-phase DTW, k tuned |
| FIH-kNN (proposed) | invariant hold signature | same kNN-DTW learner on the 2-channel invariant trace |
| FIH-LR (proposed) | 14 invariant statistics | same logistic regression on invariant statistics |
| FIHV (proposed) | both | average of FIH-kNN and FIH-LR probabilities |

## Evaluation

Outer loop: stratified 5-fold (label x session), repeated 10 times. Inner loop (training fold
only): 5-fold grid search on ROC-AUC, then inner out-of-fold scores choose two thresholds,
`tau_ba` (best balanced accuracy) and `tau_safe` (most sensitive threshold with false-seated
rate at most 10%). Imputation, scaling and feature selection are fitted inside each training
fold. Cross-session protocol: train on one session (tuning and thresholds included), test on
the other, both directions. Uncertainty: trial bootstrap (2000 resamples), paired for model
differences; within-session label-permutation test (1000 permutations).

## Reproduce

See HOW_TO_RUN.md. In short: Python 3.10+, `pip install -r requirements.txt`, put the three data
zips in `data/raw/`, then `bash run_all.sh` (Windows: `run_all.bat`). Every number in the paper is
printed by a script and saved in `results/`.

## Data access

The trial recordings come from an internship project at a robotics lab and are not public, so
`data/raw/` and `data/processed/` are excluded from the repository by `.gitignore`. The course
instructor and graders can obtain the three zips from the author on request. All derived tables,
logs and figures are included in `results/`.

## AI use

See AI_USE.md.

## Notes on the reported run

All numbers in the paper come from the run stored in `results/`. Step 5 (ablations and the
1000-permutation test) is slow on a single CPU, so in the stored run the permutation test was run
in parts (`--perm-models`; each model's shuffles are drawn in a fixed order, so the result does not
depend on how the run is split), and the ablation tables A to C are from the earlier run of the
original six models. A full `bash run_all.sh` regenerates every file, including FIH-LR's
permutation p-value and the FIH rows of the ablation tables.

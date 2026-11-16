# How to run the project, step by step

This guide assumes nothing beyond a computer with Python. Follow the steps in order.
Every command is typed in a terminal (Linux or Mac) or in Command Prompt / PowerShell (Windows).

## 0. What you need

* Python 3.10 or newer. Check with `python3 --version` (Linux, Mac) or `python --version` (Windows).
* About 200 MB of free disk space.
* Time: about 10 minutes for everything except the permutation test, plus about
  30 to 60 minutes for the permutation test in step 5 (it is the slow part).

Tested with (the reported numbers are identical on all of them):
| | Python | numpy | pandas | scipy | scikit-learn | matplotlib |
|---|---|---|---|---|---|---|
| Your laptop (Ubuntu) | 3.10.12 | 2.2.6 | 2.3.3 | 1.15.3 | 1.7.2 | 3.10.9 |
| Second check (Ubuntu) | 3.12.3 | 2.4.4 | 3.0.2 | 1.17.1 | 1.8.0 | 3.10.8 |
| Third check (Ubuntu) | 3.12.3 | 2.5.4 | 3.0.6 | 1.18.1 | 1.9.1 | 3.11.2 |

## 1. Unzip

Clone the repository (or unzip the downloaded copy) anywhere, for example into your home folder.
You get a folder that contains `src/`, `scripts/`, `data/`, `README.md` and this file.

Open a terminal **inside** that folder:

```
cd ~/lens-wedge-fih         # Linux / Mac (adjust the path)
cd C:\Users\you\lens-wedge-fih  # Windows (adjust the path)
```

Every command below must be run from this folder (the one that contains `src/` and `scripts/`).
The code is written so that the folder name does not matter.

## 2. Create a virtual environment (once)

A virtual environment keeps the project's packages separate from the rest of your computer.

Linux / Mac:
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

Your prompt now starts with `(.venv)`. Do this `activate` line again every time you open a new terminal.

## 3. Install the packages (once)

```
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Check the data is in place

The three data zips must be in `data/raw/`:

```
data/raw/success.zip
data/raw/failure.zip
data/raw/partial_contact.zip
```

The recordings are private (internship data), so they are not in the public repository. Course
staff can get the three zips from the author; copy them into `data/raw/`. Step 1 extracts them
automatically. Without them, the tables, logs and figures of the reported run are still in `results/`.

## 5. Run everything with one command

Linux / Mac:
```
bash run_all.sh
```

Windows:
```
run_all.bat
```

This runs the thirteen scripts in order, prints their output, and saves each script's log to
`results/log_<script>.txt`. It stops at the first error and tells you which script failed.

## 6. Or run the scripts one at a time

Useful if you want to watch each step or rerun only one part. Run them in this order,
because each step uses files written by the earlier ones.

| step | command | what it does | time |
|---|---|---|---|
| 1 | `python scripts/01_build_dataset.py` | unzips data, quality checks, burst averaging, all features (standard and invariant) | 5 s |
| 2 | `python scripts/02_eda.py` | descriptive tables and figures | 5 s |
| 3 | `python scripts/03_train_evaluate.py` | main nested-CV comparison of all nine models | 3 to 10 min |
| 4 | `python scripts/04_threshold_sweep.py` | decision-threshold sweep | 5 s |
| 5 | `python scripts/05_ablations.py` | ablations and the 1000-permutation test | 30 to 90 min |
| 6 | `python scripts/06_session_shift.py` | train on one day, test on the other | 1 to 2 min |
| 7 | `python scripts/07_error_analysis.py` | errors, calibration, coefficients | 10 s |
| 8 | `python scripts/08_session_diagnostics.py` | within-session AUC; transfer by feature group | 1 min |
| 9 | `python scripts/09_invariance_diagnostics.py` | invariance proof check and fixture-change stress test | 1 to 2 min |
| 10 | `python scripts/10_invariance_tradeoff.py` | invariance versus discrimination trade-off | 1 min |
| 11 | `python scripts/11_svm_platt_check.py` | the SVM probability bug check | 10 s |
| 12 | `python scripts/12_paper_figures.py` | the paper's figures | 5 s |
| 13 | `python scripts/13_build_website.py` | rebuilds the website `docs/index.html` from `results/` | 5 s |

Faster versions for a quick check (numbers will differ slightly from the paper):
```
python scripts/03_train_evaluate.py --repeats 2
python scripts/05_ablations.py --n-perm 50
python scripts/10_invariance_tradeoff.py --repeats 1
```

## 7. Where the results are

* `results/*.csv`  every table (the paper's numbers come from these)
* `results/log_*.txt`  everything each script printed
* `results/figures/`  every figure; `fig_paper_main.pdf` is Figure 1 of the paper

The most important files:

| file | contents |
|---|---|
| `metrics_summary.csv` | random-CV ROC-AUC (with 95% CI), PR-AUC, Brier, decisions, every model |
| `session_shift_summary.csv` | cross-session AUC and decision metrics, every model |
| `session_shift_paired.csv` | paired bootstrap of cross-session AUC differences |
| `within_session_auc.csv` | random-CV scores ranked within each session |
| `direction_consistency_by_group.csv` | which feature groups point the same way on both days |
| `invariance_check.txt` | numerical proof that the FIH representation is invariant |
| `stress_test.csv` | AUC under simulated fixture rotations and offsets |
| `tradeoff_lambda.csv` | the invariance versus discrimination curve |
| `permutation_test.csv` | p-values against the within-session label-shuffle null |

## 8. If something goes wrong

| message | fix |
|---|---|
| `python: command not found` | use `python3` instead of `python` (Linux / Mac) |
| `No module named 'sklearn'` (or numpy, pandas) | the virtual environment is not active: run the `activate` line from step 2 |
| `No module named 'src'` | you are not in the project folder: `cd` into it |
| `Missing folder data/raw/success/csv` | the three zips are not in `data/raw/` |
| `Run scripts/01_build_dataset.py first` | run the steps in order |
| `bash: run_all.sh: Permission denied` | use `bash run_all.sh`, not `./run_all.sh` |
| step 5 seems frozen | it is the permutation test; it prints a line per model when done |

## 9. Making a small change (if asked in the viva)

All settings are in `src/config.py`. For example:

* stricter safety limit: `MAX_FALSE_SEATED = 0.05`, then rerun steps 3 and 4
* more DTW warping: `DTW_BAND = 5`, delete `data/processed/dtw_*.npz`, rerun from step 3
* fewer repeats for speed: `OUTER_REPEATS = 3`

The invariant representation is in `src/invariant.py`. To test it on a new fixture change,
edit `YAW_DEG` or `AXIAL_OFFSET_N` at the top of `scripts/09_invariance_diagnostics.py`.

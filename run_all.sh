#!/usr/bin/env bash
# Run every step in order; stop at the first error (pipefail keeps tee from hiding it).
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p results
for s in 01_build_dataset 02_eda 03_train_evaluate 04_threshold_sweep 05_ablations \
         06_session_shift 07_error_analysis 08_session_diagnostics 09_invariance_diagnostics \
         10_invariance_tradeoff 11_svm_platt_check 12_paper_figures 13_build_website; do
  echo "================ $s ================"
  python scripts/$s.py 2>&1 | tee results/log_$s.txt || { echo "FAILED at $s, see results/log_$s.txt"; exit 1; }
done
echo "All thirteen steps finished. Results are in results/, figures in results/figures/."

@echo off
cd /d %~dp0
if not exist results mkdir results
for %%s in (01_build_dataset 02_eda 03_train_evaluate 04_threshold_sweep 05_ablations 06_session_shift 07_error_analysis 08_session_diagnostics 09_invariance_diagnostics 10_invariance_tradeoff 11_svm_platt_check 12_paper_figures 13_build_website) do (
  echo ================ %%s ================
  python scripts\%%s.py > results\log_%%s.txt 2>&1
  if errorlevel 1 (
    echo FAILED at %%s, see results\log_%%s.txt
    exit /b 1
  )
  type results\log_%%s.txt
)
echo All thirteen steps finished.

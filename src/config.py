"""Central configuration. Every script imports from here so settings stay consistent."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

CLASSES = ["success", "failure", "partial_contact"]

# Main target: 1 = seated (success), 0 = not seated (failure OR partial_contact)
LABEL_SCHEME_MAIN = "success_vs_rest"

# Logging is bursty: ~55 samples at ~1 kHz, then a ~0.27 s gap.
# Any time step larger than this starts a new burst.
GAP_S = 0.02
MIN_BURST_SAMPLES_FOR_VIB = 8      # need enough samples to detrend inside a burst

# Raw-trace model: each phase resampled to this many points (time-normalised)
N_POINTS_PER_PHASE = 15
DTW_BAND = 3                       # Sakoe-Chiba band, in resampled points

SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 10                 # main evaluation: 5-fold x 10 repeats
ABLATION_REPEATS = 5
INNER_SPLITS = 5
N_BOOT = 2000

# Safety constraint used for the deployment threshold: at most this fraction of unseated
# lenses may be called "seated" (those would be snap-fitted at 20 N).
MAX_FALSE_SEATED = 0.10

# Deployment cost (used only in the threshold-sweep sensitivity analysis): a false "seated" verdict sends the arm into a 20 N snap-fit on an
# unseated lens (possible crack/scrap). A false "not seated" costs one extra attempt.
COST_FALSE_SEATED = 3.0            # cost of FP (positive = seated)
COST_FALSE_REJECT = 1.0            # cost of FN
COST_RATIOS_SWEEP = [1, 2, 3, 5, 10]

THRESH_GRID_STEP = 0.01

for p in (DATA_PROC, RESULTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

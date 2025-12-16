from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# parent folder that contains the session folder
DATA_PARENT = REPO_ROOT / "ec013.18"

# session folder name (in your screenshot: ec013.205 is a folder)
BASE = "ec013.205"
SESSION_DIR = DATA_PARENT / BASE

OUT = REPO_ROOT / "pipeline_2" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# sampling rates (confirmed by diagnostics)
WHL_FS   = 39.0625
EEG_FS   = 1250.0
SPIKE_FS = 20000.0

# discretization
DELTA = 0.02   # 20 ms
KPOS  = 50     # position bins (x-axis)

# unit filtering
MIN_SPIKES = 50

import numpy as np
from pathlib import Path
from config import SESSION_DIR, BASE, OUT

# ----------------------------
# Known hc-3 constants (typical)
# ----------------------------
WHL_FS = 39.0625      # position sampling
EEG_FS = 1250.0       # eeg sampling (you already saw this earlier)
DEFAULT_SPIKE_FS = 20000.0  # res timestamps usually at 20 kHz

# ----------------------------
# Helpers
# ----------------------------
def read_par_preview(par_path: Path, nlines=20):
    lines = par_path.read_text(errors="ignore").splitlines()
    return lines[:nlines], lines

def try_infer_spike_fs_from_par(all_lines):
    # very forgiving: look for common values present in text
    txt = " ".join(all_lines)
    candidates = [20000.0, 32000.0, 24000.0]
    for c in candidates:
        if str(int(c)) in txt:
            return c
    return None

def file_int16_count(path: Path):
    return path.stat().st_size // 2  # bytes -> int16 count

# ----------------------------
# WHL
# ----------------------------
whl_path = SESSION_DIR / f"{BASE}.whl"
whl = np.loadtxt(whl_path)
T_whl = len(whl)
dur_whl = T_whl / WHL_FS

# ----------------------------
# PAR (preview + maybe infer spike_fs)
# ----------------------------
par_path = SESSION_DIR / f"{BASE}.par"
par_preview, par_all = read_par_preview(par_path)
spike_fs = try_infer_spike_fs_from_par(par_all) or DEFAULT_SPIKE_FS

# ----------------------------
# EEG duration from file size (infer nchan)
# ----------------------------
eeg_path = SESSION_DIR / f"{BASE}.eeg"
n_int16 = file_int16_count(eeg_path)

# Use WHL duration as a strong hint: if EEG is same epoch, samples ~ dur_whl*EEG_FS
Ns_guess = int(round(dur_whl * EEG_FS))
# infer channel count
nchan_guess = n_int16 / max(Ns_guess, 1)

# Also compute duration assuming nchan_guess rounded to nearest int
nchan_round = int(round(nchan_guess)) if nchan_guess > 0 else None
if nchan_round and nchan_round > 0:
    Ns = n_int16 // nchan_round
    dur_eeg = Ns / EEG_FS
else:
    Ns = None
    dur_eeg = None

# ----------------------------
# RES min/max per tetrode
# ----------------------------
res_files = sorted(SESSION_DIR.glob(f"{BASE}.res.*"))

per_tet = []
gmin = None
gmax = None

for rf in res_files:
    tet = int(rf.name.split(".")[-1])
    r = np.loadtxt(rf, dtype=np.int64)
    if r.size == 0:
        continue
    rmin = int(r.min())
    rmax = int(r.max())
    if gmin is None or rmin < gmin: gmin = rmin
    if gmax is None or rmax > gmax: gmax = rmax
    per_tet.append((tet, rmin, rmax))

# ----------------------------
# Print report
# ----------------------------
report_lines = []
report_lines.append(f"SESSION_DIR = {SESSION_DIR}")
report_lines.append(f"BASE        = {BASE}")
report_lines.append("")
report_lines.append("=== WHL ===")
report_lines.append(f"frames = {T_whl}")
report_lines.append(f"fs     = {WHL_FS}")
report_lines.append(f"dur    = {dur_whl:.6f} s")
report_lines.append("")
report_lines.append("=== PAR (preview first 20 lines) ===")
for line in par_preview:
    report_lines.append(line)
report_lines.append("")
report_lines.append("=== SPIKE_FS (assumed/inferred) ===")
report_lines.append(f"SPIKE_FS = {spike_fs}")
report_lines.append("")
report_lines.append("=== EEG (from file size) ===")
report_lines.append(f"eeg file bytes  = {eeg_path.stat().st_size}")
report_lines.append(f"int16 count     = {n_int16}")
report_lines.append(f"Ns_guess (WHLdur*EEG_FS) = {Ns_guess}")
report_lines.append(f"nchan_guess = int16_count / Ns_guess = {nchan_guess:.4f}")
report_lines.append(f"nchan_round = {nchan_round}")
if Ns is not None:
    report_lines.append(f"Ns (int16_count/nchan_round) = {Ns}")
if dur_eeg is not None:
    report_lines.append(f"EEG_FS = {EEG_FS}")
    report_lines.append(f"dur_eeg = {dur_eeg:.6f} s")
report_lines.append("")
report_lines.append("=== RES time ranges ===")
for tet, rmin, rmax in per_tet:
    report_lines.append(
        f"tet {tet:02d}: res_min={rmin} ({rmin/spike_fs:.6f}s), "
        f"res_max={rmax} ({rmax/spike_fs:.6f}s), "
        f"span={(rmax-rmin)/spike_fs:.3f}s"
    )

report_lines.append("")
report_lines.append("=== GLOBAL RES ===")
report_lines.append(f"res_min_global = {gmin} ({gmin/spike_fs:.6f}s)")
report_lines.append(f"res_max_global = {gmax} ({gmax/spike_fs:.6f}s)")
report_lines.append(f"session span   = {(gmax-gmin)/spike_fs:.3f}s = {(gmax-gmin)/spike_fs/60:.3f} min")

report = "\n".join(report_lines)
print(report)

out_path = OUT / "diag_01_timespans.txt"
out_path.write_text(report, encoding="utf-8")
print(f"\nSaved -> {out_path}")

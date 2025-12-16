import numpy as np
from config import SESSION_DIR, BASE, OUT

EEG_FS   = 1250.0
SPIKE_FS = 20000.0
PAD_S    = 0.02  # 20 ms padding

swr_path = OUT / "swr_events.npy"
events = np.load(swr_path)
events = np.asarray(events)

print("Loaded swr_events:", events.shape, events.dtype)
print("First 10 rows:\n", events[:10])

# --- infer which units events are in ---
mx = float(np.max(events))
mn = float(np.min(events))
print(f"\nEvent value range: min={mn}, max={mx}")

# Heuristics:
#  - if max ~ 200k: likely EEG samples (1250 Hz * 187s ~ 234k)
#  - if max ~ 180: likely seconds
#  - if max ~ 9000: likely 20ms bins (187/0.02 ~ 9360)
mode = None
if mx > 1e4:
    if mx < 5e5:
        mode = "eeg_samples"
    else:
        mode = "unknown_big"
elif mx > 500:
    mode = "bins_20ms_or_whl_frames"
else:
    mode = "seconds"

print("Inferred mode:", mode)

def events_to_spike_samples(ev):
    ev = ev.astype(float)
    if mode == "seconds":
        t0 = ev[:,0]; t1 = ev[:,1]
    elif mode == "eeg_samples":
        t0 = ev[:,0] / EEG_FS
        t1 = ev[:,1] / EEG_FS
    elif mode == "bins_20ms_or_whl_frames":
        # try interpreting as 20ms bins first
        t0 = ev[:,0] * 0.02
        t1 = ev[:,1] * 0.02
    else:
        raise ValueError("Cannot infer event unit mode safely.")
    # pad + convert to spike samples
    t0 = np.maximum(0.0, t0 - PAD_S)
    t1 = np.maximum(t0, t1 + PAD_S)
    s0 = np.floor(t0 * SPIKE_FS).astype(np.int64)
    s1 = np.ceil (t1 * SPIKE_FS).astype(np.int64)
    return s0, s1

s0, s1 = events_to_spike_samples(events)

# --- load spikes for all tets, excluding clu==0 (noise) ---
totals = []
for tet in range(1, 9):
    res = np.loadtxt(SESSION_DIR / f"{BASE}.res.{tet}", dtype=np.int64)
    clu = np.loadtxt(SESSION_DIR / f"{BASE}.clu.{tet}", dtype=np.int64)
    labels = clu[1:]              # first line is header
    keep = labels != 0            # drop noise cluster 0
    res = res[keep]
    totals.append(res)

all_spikes = np.concatenate(totals)
all_spikes.sort()
print("\nAll spikes (non-noise) total:", all_spikes.size)
print("All spikes span samples:", int(all_spikes[0]), "to", int(all_spikes[-1]),
      f"({all_spikes[-1]/SPIKE_FS:.3f} s)")

# --- count spikes per event ---
counts = np.zeros(len(s0), dtype=int)
j0 = 0
j1 = 0
n = len(all_spikes)
for i, (a, b) in enumerate(zip(s0, s1)):
    while j0 < n and all_spikes[j0] < a:
        j0 += 1
    j1 = max(j1, j0)
    while j1 < n and all_spikes[j1] < b:
        j1 += 1
    counts[i] = j1 - j0

print("\nSWR spike counts summary:")
print("events:", len(counts))
print("total spikes in all events:", int(counts.sum()))
print("max spikes in one event:", int(counts.max()))
print("nonzero events:", int(np.sum(counts > 0)))
print("first 20 counts:", counts[:20].tolist())

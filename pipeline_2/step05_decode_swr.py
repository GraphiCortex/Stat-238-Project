import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUT, SESSION_DIR, BASE, DELTA, EEG_FS, SPIKE_FS

EPS = 1e-12
PAD_S = 0.02  # pad SWR windows by 20ms

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + EPS), axis=axis)

def forward_backward_log(E, logA, logpi):
    # E: (T,K) log emission
    T, K = E.shape
    log_alpha = np.full((T, K), -np.inf)
    log_beta  = np.full((T, K), -np.inf)

    log_alpha[0] = logpi + E[0]
    for t in range(1, T):
        log_alpha[t] = E[t] + logsumexp(log_alpha[t-1][:, None] + logA, axis=0)

    log_beta[T-1] = 0.0
    for t in range(T-2, -1, -1):
        log_beta[t] = logsumexp(logA + E[t+1][None, :] + log_beta[t+1][None, :], axis=1)

    log_post = log_alpha + log_beta
    log_post = log_post - logsumexp(log_post, axis=1)[:, None]
    post = np.exp(log_post)
    return post

def load_unit_spikes_from_units_json(units_json_path):
    units = json.loads(Path(units_json_path).read_text(encoding="utf-8"))
    spikes = []
    for u in units:
        tet = int(u["tet"])
        clu_id = int(u["clu"])
        res = np.loadtxt(SESSION_DIR / f"{BASE}.res.{tet}", dtype=np.int64)
        clu = np.loadtxt(SESSION_DIR / f"{BASE}.clu.{tet}", dtype=np.int64)
        labels = clu[1:]
        keep = labels == clu_id
        spikes.append(res[keep])  # spike samples @ 20k
    return spikes

def bin_spikes_in_window(spike_samples_list, t0_s, t1_s, delta_s):
    # returns (Tbin, N)
    dur = max(0.0, t1_s - t0_s)
    Tbin = max(1, int(np.ceil(dur / delta_s)))
    edges = t0_s + np.arange(Tbin + 1) * delta_s

    Y = np.zeros((Tbin, len(spike_samples_list)), dtype=np.int32)
    for n, sp in enumerate(spike_samples_list):
        ts = sp / SPIKE_FS
        m = (ts >= t0_s) & (ts < t1_s)
        if np.any(m):
            Y[:, n], _ = np.histogram(ts[m], bins=edges)
    return Y

def poisson_loglik_matrix(Y, lam):
    # lam: (K,N) expected counts per bin
    Y = Y.astype(float)
    loglam = np.log(lam + EPS)
    term1 = Y @ loglam.T               # (T,K)
    term2 = np.sum(lam, axis=1)[None]  # (1,K)
    return term1 - term2

# -------- load artifacts ----------
pf = np.load(OUT / "placefields.npz")
lam = pf["lam"]           # (K,N) counts per bin
K, N = lam.shape

hmm = np.load(OUT / "hmm_full_params.npz")
A = hmm["A"]
pi = hmm["pi"]

logA = np.log(A + EPS)
logpi = np.log(pi + EPS)

events = np.load(OUT / "swr_events.npy")  # EEG samples (start,end)
events = np.asarray(events)

spike_samples_list = load_unit_spikes_from_units_json(OUT / "units.json")

print("Loaded:")
print("  lam:", lam.shape)
print("  events:", events.shape)
print("  units spikes:", len(spike_samples_list))

# -------- decode each SWR ----------
metrics = []

for i, (a_eeg, b_eeg) in enumerate(events):
    t0 = float(a_eeg) / EEG_FS
    t1 = float(b_eeg) / EEG_FS
    t0 = max(0.0, t0 - PAD_S)
    t1 = t1 + PAD_S

    Yev = bin_spikes_in_window(spike_samples_list, t0, t1, DELTA)  # (Tbin,N)
    E = poisson_loglik_matrix(Yev, lam)                             # (Tbin,K)
    post = forward_backward_log(E, logA, logpi)                     # (Tbin,K)
    zmap = np.argmax(post, axis=1)

    total_spikes = int(Yev.sum())
    dur_ms = 1000.0 * (t1 - t0)

    # simple “replay-ish” score: correlation of MAP position with time
    t_idx = np.arange(len(zmap))
    if len(zmap) >= 2:
        r = np.corrcoef(t_idx, zmap)[0, 1]
        r = float(r) if np.isfinite(r) else 0.0
    else:
        r = 0.0

    metrics.append({
        "event": int(i),
        "a_eeg": int(a_eeg),
        "b_eeg": int(b_eeg),
        "t0_s": float(t0),
        "t1_s": float(t1),
        "dur_ms": float(dur_ms),
        "Tbins": int(len(zmap)),
        "total_spikes": int(total_spikes),
        "corr_time_mapZ": float(r),
        "mapZ_start": int(zmap[0]),
        "mapZ_end": int(zmap[-1]),
    })

    # plot posterior heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(post.T, aspect="auto", origin="lower")
    plt.plot(zmap, color="white", linewidth=2, label="MAP")
    plt.xlabel("Time bin (20ms)")
    plt.ylabel("Position bin")
    plt.title(f"SWR {i:02d}: {dur_ms:.1f} ms, spikes={total_spikes}, r={r:.2f}")
    plt.tight_layout()
    plt.savefig(FIG / f"swr_decode_event_{i:02d}.png", dpi=200)
    plt.close()

# save metrics
out_json = OUT / "swr_decode_metrics.json"
out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Saved ->", out_json)
print("Saved event figures ->", FIG / "swr_decode_event_*.png")

# summary plot: corr distribution
rs = np.array([m["corr_time_mapZ"] for m in metrics], dtype=float)
plt.figure()
plt.hist(rs, bins=10)
plt.xlabel("corr(time, MAP position)")
plt.ylabel("count")
plt.title("SWR decoded trajectory correlation (heuristic)")
plt.tight_layout()
plt.savefig(FIG / "swr_replay_corr_hist.png", dpi=200)
plt.close()

print("Saved ->", FIG / "swr_replay_corr_hist.png")

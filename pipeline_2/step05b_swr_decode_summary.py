import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUT, SESSION_DIR, BASE, DELTA, EEG_FS, SPIKE_FS

EPS = 1e-12
PAD_S = 0.02  # 20 ms padding

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + EPS), axis=axis)

def forward_backward_log(E, logA, logpi):
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
    return np.exp(log_post)

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
        spikes.append(res[keep])
    return spikes

def bin_spikes_in_window(spike_samples_list, t0_s, t1_s, delta_s):
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

def counts_in_window(spike_samples_list, t0_s, t1_s):
    y = np.zeros(len(spike_samples_list), dtype=np.int32)
    for n, sp in enumerate(spike_samples_list):
        ts = sp / SPIKE_FS
        y[n] = int(np.sum((ts >= t0_s) & (ts < t1_s)))
    return y

def poisson_loglik_matrix(Y, lam):
    Y = Y.astype(float)
    loglam = np.log(lam + EPS)
    term1 = Y @ loglam.T
    term2 = np.sum(lam, axis=1)[None]
    return term1 - term2

def poisson_loglik_single(y, lam):
    y = y.astype(float)
    loglam = np.log(lam + EPS)
    return (y[None, :] * loglam).sum(axis=1) - np.sum(lam, axis=1)

# -------- load artifacts ----------
pf = np.load(OUT / "placefields.npz")
lam = pf["lam"]  # (K,N)
K, N = lam.shape

hmm = np.load(OUT / "hmm_full_params.npz")
A = hmm["A"]
pi = hmm["pi"]
logA = np.log(A + EPS)
logpi = np.log(pi + EPS)

events = np.load(OUT / "swr_events.npy")
spike_samples_list = load_unit_spikes_from_units_json(OUT / "units.json")

metrics = []
single_map = []

for i, (a_eeg, b_eeg) in enumerate(events):
    t0 = float(a_eeg) / EEG_FS
    t1 = float(b_eeg) / EEG_FS
    t0 = max(0.0, t0 - PAD_S)
    t1 = t1 + PAD_S
    dur_ms = 1000.0 * (t1 - t0)

    # time-binned decode
    Yev = bin_spikes_in_window(spike_samples_list, t0, t1, DELTA)
    E = poisson_loglik_matrix(Yev, lam)
    post = forward_backward_log(E, logA, logpi)
    zmap = np.argmax(post, axis=1)

    # improved metrics
    if len(zmap) >= 2:
        path_len = int(np.sum(np.abs(np.diff(zmap))))
        net_disp = int(abs(int(zmap[-1]) - int(zmap[0])))
    else:
        path_len = 0
        net_disp = 0

    peak = float(np.mean(np.max(post, axis=1)))
    total_spikes = int(Yev.sum())
    Tbins = int(len(zmap))

    # single-bin decode (whole SWR as one observation)
    y = counts_in_window(spike_samples_list, t0, t1)
    ll = poisson_loglik_single(y, lam)
    ll = ll - logsumexp(ll)  # normalize in log-space
    p_single = np.exp(ll)
    z_single = int(np.argmax(p_single))
    single_map.append(z_single)

    metrics.append({
        "event": int(i),
        "dur_ms": float(dur_ms),
        "Tbins": int(Tbins),
        "total_spikes": int(total_spikes),
        "path_len": int(path_len),
        "net_disp": int(net_disp),
        "post_peak_mean": float(peak),
        "map_start": int(zmap[0]),
        "map_end": int(zmap[-1]),
        "map_single": int(z_single),
    })

# save metrics
out_json = OUT / "swr_decode_metrics_v2.json"
out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("Saved ->", out_json)

# ---------- plots ----------
dur = np.array([m["dur_ms"] for m in metrics])
spk = np.array([m["total_spikes"] for m in metrics])
pl  = np.array([m["path_len"] for m in metrics])
nd  = np.array([m["net_disp"] for m in metrics])
pk  = np.array([m["post_peak_mean"] for m in metrics])
zs  = np.array([m["map_single"] for m in metrics])

plt.figure()
plt.scatter(dur, spk)
plt.xlabel("SWR duration (ms)")
plt.ylabel("Total spikes in event")
plt.title("SWR spikes vs duration")
plt.tight_layout()
plt.savefig(FIG / "swr_spikes_vs_duration.png", dpi=200)
plt.close()

plt.figure()
plt.hist(pl, bins=15)
plt.xlabel("MAP path length (sum |Δz|)")
plt.ylabel("count")
plt.title("SWR decoded path length")
plt.tight_layout()
plt.savefig(FIG / "swr_pathlen_hist.png", dpi=200)
plt.close()

plt.figure()
plt.hist(zs, bins=K)
plt.xlabel("Decoded position bin (single-bin MAP)")
plt.ylabel("count")
plt.title("SWR decoded content (single-bin)")
plt.tight_layout()
plt.savefig(FIG / "swr_singlebin_pos_hist.png", dpi=200)
plt.close()

plt.figure()
plt.hist(pk, bins=10)
plt.xlabel("mean_t max_k posterior(t,k)")
plt.ylabel("count")
plt.title("Posterior sharpness across SWRs")
plt.tight_layout()
plt.savefig(FIG / "swr_posterior_peak_hist.png", dpi=200)
plt.close()

print("Saved figures ->", FIG)

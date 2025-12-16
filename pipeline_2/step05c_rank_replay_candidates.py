import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUT, SESSION_DIR, BASE, DELTA, EEG_FS, SPIKE_FS

EPS = 1e-12
PAD_S = 0.02
TOPK = 8
BANDW = 2  # posterior mass band half-width around fitted line

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

def load_unit_spikes(units_json_path):
    units = json.loads(Path(units_json_path).read_text(encoding="utf-8"))
    spikes = []
    for u in units:
        tet = int(u["tet"])
        clu_id = int(u["clu"])
        res = np.loadtxt(SESSION_DIR / f"{BASE}.res.{tet}", dtype=np.int64)
        clu = np.loadtxt(SESSION_DIR / f"{BASE}.clu.{tet}", dtype=np.int64)
        labels = clu[1:]
        spikes.append(res[labels == clu_id])
    return spikes

def bin_spikes(spike_samples_list, t0_s, t1_s, delta_s):
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

def poisson_loglik(Y, lam):
    Y = Y.astype(float)
    loglam = np.log(lam + EPS)
    return (Y @ loglam.T) - np.sum(lam, axis=1)[None, :]

def line_score_from_posterior(post, bandw=2):
    # Use posterior mean position per time, fit a line, score mass near that line.
    T, K = post.shape
    kgrid = np.arange(K)[None, :]
    zbar = (post * kgrid).sum(axis=1)              # (T,)
    t = np.arange(T, dtype=float)

    # fit zbar ~ a t + b
    if T < 2:
        return dict(slope=0.0, r2=0.0, band_mass=float(np.max(post))), zbar

    a, b = np.polyfit(t, zbar, 1)
    zhat = a * t + b

    # R^2 on zbar (not perfect, but useful)
    ss_res = float(np.sum((zbar - zhat) ** 2))
    ss_tot = float(np.sum((zbar - np.mean(zbar)) ** 2) + EPS)
    r2 = 1.0 - ss_res / ss_tot

    # posterior mass in a band around the line
    band = 0.0
    for ti in range(T):
        center = int(round(zhat[ti]))
        lo = max(0, center - bandw)
        hi = min(K - 1, center + bandw)
        band += float(np.sum(post[ti, lo:hi+1]))
    band_mass = band / T

    return dict(slope=float(a), r2=float(r2), band_mass=float(band_mass)), zbar

# ---- load artifacts ----
pf = np.load(OUT / "placefields.npz")
lam = pf["lam"]          # (K,N)
K, N = lam.shape

hmm = np.load(OUT / "hmm_full_params.npz")
A = hmm["A"]
pi = hmm["pi"]
logA = np.log(A + EPS)
logpi = np.log(pi + EPS)

events = np.load(OUT / "swr_events.npy")
spikes = load_unit_spikes(OUT / "units.json")

rows = []

for i, (a_eeg, b_eeg) in enumerate(events):
    t0 = float(a_eeg) / EEG_FS
    t1 = float(b_eeg) / EEG_FS
    t0 = max(0.0, t0 - PAD_S)
    t1 = t1 + PAD_S

    Yev = bin_spikes(spikes, t0, t1, DELTA)
    E = poisson_loglik(Yev, lam)
    post = forward_backward_log(E, logA, logpi)
    zmap = np.argmax(post, axis=1)

    total_spikes = int(Yev.sum())
    dur_ms = 1000.0 * (t1 - t0)
    peak = float(np.mean(np.max(post, axis=1)))

    if len(zmap) >= 2:
        path_len = int(np.sum(np.abs(np.diff(zmap))))
        net_disp = int(abs(int(zmap[-1]) - int(zmap[0])))
    else:
        path_len = 0
        net_disp = 0

    ls, zbar = line_score_from_posterior(post, bandw=BANDW)

    # combined replay-ish score (simple heuristic)
    score = (ls["band_mass"] * max(0.0, ls["r2"])) * (1.0 + 0.05 * net_disp) * (0.5 + 0.5 * peak)

    rows.append({
        "event": int(i),
        "dur_ms": float(dur_ms),
        "Tbins": int(len(zmap)),
        "total_spikes": int(total_spikes),
        "path_len": int(path_len),
        "net_disp": int(net_disp),
        "post_peak_mean": float(peak),
        "slope": ls["slope"],
        "r2": ls["r2"],
        "band_mass": ls["band_mass"],
        "score": float(score),
    })

# rank and save
rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
out = OUT / "swr_replay_ranked.json"
out.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
print("Saved ->", out)

# save topK event plots
for rank, r in enumerate(rows_sorted[:TOPK]):
    i = r["event"]
    a_eeg, b_eeg = events[i]
    t0 = max(0.0, float(a_eeg)/EEG_FS - PAD_S)
    t1 = float(b_eeg)/EEG_FS + PAD_S

    Yev = bin_spikes(spikes, t0, t1, DELTA)
    E = poisson_loglik(Yev, lam)
    post = forward_backward_log(E, logA, logpi)
    zmap = np.argmax(post, axis=1)

    plt.figure(figsize=(8,4))
    plt.imshow(post.T, aspect="auto", origin="lower")
    plt.plot(zmap, color="white", linewidth=2)
    plt.title(f"rank {rank} SWR {i:02d}: score={r['score']:.3f}, net={r['net_disp']}, path={r['path_len']}")
    plt.xlabel("time bin (20ms)")
    plt.ylabel("position bin")
    plt.tight_layout()
    plt.savefig(FIG / f"swr_top_rank_{rank:02d}_event_{i:02d}.png", dpi=200)
    plt.close()

print("Saved top-ranked plots ->", FIG / "swr_top_rank_*.png")

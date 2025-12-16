import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUT, SESSION_DIR, BASE, EEG_FS, SPIKE_FS, KPOS

EPS = 1e-12

DELTA_SWR = 0.005
PAD_S     = 0.02
TOPK      = 8
NSHUFFLE  = 200
BANDW     = 2

# --- SWR prior knobs (the important part) ---
BAND_PRIOR = 6      # allow i±6 per 5ms step (tunable)
ETA        = 0.50   # mix weight toward fast banded prior (tunable)
SIGMA      = 2.0    # controls how quickly probs decay with distance

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

def line_band_score(post, bandw=2):
    T, K = post.shape
    kgrid = np.arange(K)[None, :]
    zbar = (post * kgrid).sum(axis=1)
    t = np.arange(T, dtype=float)
    if T < 2:
        return dict(slope=0.0, r2=0.0, band_mass=float(np.max(post))), zbar

    a, b = np.polyfit(t, zbar, 1)
    zhat = a * t + b

    ss_res = float(np.sum((zbar - zhat) ** 2))
    ss_tot = float(np.sum((zbar - np.mean(zbar)) ** 2) + EPS)
    r2 = 1.0 - ss_res / ss_tot

    band = 0.0
    for ti in range(T):
        center = int(round(zhat[ti]))
        lo = max(0, center - bandw)
        hi = min(K - 1, center + bandw)
        band += float(np.sum(post[ti, lo:hi + 1]))
    band_mass = band / T

    return dict(slope=float(a), r2=float(r2), band_mass=float(band_mass)), zbar

def replay_score(post, zmap):
    ls, _ = line_band_score(post, bandw=BANDW)
    if len(zmap) >= 2:
        path_len = int(np.sum(np.abs(np.diff(zmap))))
        net_disp = int(abs(int(zmap[-1]) - int(zmap[0])))
    else:
        path_len = 0
        net_disp = 0
    T = post.shape[0]
    peak = float(np.mean(np.max(post, axis=1)))

    motion_gate = 1.0 if net_disp >= 3 else 0.25
    traj_strength = ls["band_mass"] * max(0.0, ls["r2"]) * abs(ls["slope"])
    score = traj_strength * motion_gate * (0.5 + 0.5 * peak) * np.sqrt(max(1, T))
    return score, ls, path_len, net_disp, peak

def shuffle_placefields(lam):
    K, N = lam.shape
    lam_sh = np.empty_like(lam)
    shifts = np.random.randint(0, K, size=N)
    for n in range(N):
        lam_sh[:, n] = np.roll(lam[:, n], shifts[n])
    return lam_sh

def make_fast_banded_prior(K, band=6, sigma=2.0):
    A = np.zeros((K, K), dtype=float)
    for i in range(K):
        js = np.arange(max(0, i-band), min(K, i+band+1))
        w = np.exp(-0.5*((js - i)/sigma)**2)
        w = w / w.sum()
        A[i, js] = w
    return A

# ---- load artifacts ----
pf = np.load(OUT / "placefields.npz")
lam_20 = pf["lam"]             # expected spikes / 20ms
K, N = lam_20.shape
assert K == KPOS

lam = lam_20 * (DELTA_SWR / 0.02)

hmm = np.load(OUT / "hmm_full_params.npz")
A_beh = hmm["A"]
pi = hmm["pi"]

# build SWR prior: mix behavior A with a faster banded kernel
A_fast = make_fast_banded_prior(K, band=BAND_PRIOR, sigma=SIGMA)
A_swr = (1.0 - ETA) * A_beh + ETA * A_fast
A_swr = A_swr / np.maximum(A_swr.sum(axis=1, keepdims=True), EPS)

logA = np.log(A_swr + EPS)
logpi = np.log(pi + EPS)

events = np.load(OUT / "swr_events.npy")
spikes = load_unit_spikes(OUT / "units.json")

print("Using SWR prior:")
print("  BAND_PRIOR =", BAND_PRIOR, "SIGMA =", SIGMA, "ETA =", ETA)
print("  mean diag(A_beh) =", float(np.mean(np.diag(A_beh))))
print("  mean diag(A_swr) =", float(np.mean(np.diag(A_swr))))

rank_rows = []
pval_rows = []

for i, (a_eeg, b_eeg) in enumerate(events):
    t0 = max(0.0, float(a_eeg) / EEG_FS - PAD_S)
    t1 = float(b_eeg) / EEG_FS + PAD_S

    Yev = bin_spikes(spikes, t0, t1, DELTA_SWR)
    E = poisson_loglik(Yev, lam)
    post = forward_backward_log(E, logA, logpi)
    zmap = np.argmax(post, axis=1)

    score_true, ls, path_len, net_disp, peak = replay_score(post, zmap)
    total_spikes = int(Yev.sum())
    dur_ms = 1000.0 * (t1 - t0)

    sh_scores = []
    for _ in range(NSHUFFLE):
        lam_sh = shuffle_placefields(lam)
        Esh = poisson_loglik(Yev, lam_sh)
        post_sh = forward_backward_log(Esh, logA, logpi)
        zmap_sh = np.argmax(post_sh, axis=1)
        s, _, _, _, _ = replay_score(post_sh, zmap_sh)
        sh_scores.append(float(s))
    sh_scores = np.array(sh_scores, dtype=float)
    pval = float((np.sum(sh_scores >= score_true) + 1.0) / (NSHUFFLE + 1.0))

    rank_rows.append({
        "event": int(i),
        "score": float(score_true),
        "pval": pval,
        "dur_ms": float(dur_ms),
        "Tbins": int(len(zmap)),
        "total_spikes": int(total_spikes),
        "path_len": int(path_len),
        "net_disp": int(net_disp),
        "post_peak_mean": float(peak),
        "slope": float(ls["slope"]),
        "r2": float(ls["r2"]),
        "band_mass": float(ls["band_mass"]),
    })

    pval_rows.append({
        "event": int(i),
        "score": float(score_true),
        "pval": pval,
        "shuffle_mean": float(np.mean(sh_scores)),
        "shuffle_std": float(np.std(sh_scores)),
        "shuffle_max": float(np.max(sh_scores)),
    })

    print(f"event {i:02d}: score={score_true:.4f}, p={pval:.3f}, net={net_disp}, path={path_len}, spikes={total_spikes}, T={len(zmap)}")

rank_sorted = sorted(rank_rows, key=lambda r: r["score"], reverse=True)

out_rank = OUT / "swr_highres_ranked_swrprior.json"
out_pval = OUT / "swr_highres_pvals_swrprior.json"
out_rank.write_text(json.dumps(rank_sorted, indent=2), encoding="utf-8")
out_pval.write_text(json.dumps(pval_rows, indent=2), encoding="utf-8")

print("Saved ->", out_rank)
print("Saved ->", out_pval)

for rank, r in enumerate(rank_sorted[:TOPK]):
    i = r["event"]
    a_eeg, b_eeg = events[i]
    t0 = max(0.0, float(a_eeg) / EEG_FS - PAD_S)
    t1 = float(b_eeg) / EEG_FS + PAD_S

    Yev = bin_spikes(spikes, t0, t1, DELTA_SWR)
    E = poisson_loglik(Yev, lam)
    post = forward_backward_log(E, logA, logpi)
    zmap = np.argmax(post, axis=1)

    plt.figure(figsize=(9,4))
    plt.imshow(post.T, aspect="auto", origin="lower")
    plt.plot(zmap, color="white", linewidth=2)
    plt.title(f"SWR-prior rank {rank} SWR {i:02d}: score={r['score']:.3f}, p={r['pval']:.3f}, net={r['net_disp']}, path={r['path_len']}")
    plt.xlabel(f"time bin ({int(1000*DELTA_SWR)}ms)")
    plt.ylabel("position bin")
    plt.tight_layout()
    plt.savefig(FIG / f"swrprior_highres_top_rank_{rank:02d}.png", dpi=200)
    plt.close()

print("Saved figures ->", FIG / "swrprior_highres_top_rank_*.png")

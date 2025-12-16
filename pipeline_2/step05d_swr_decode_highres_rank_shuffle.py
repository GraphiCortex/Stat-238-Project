# pipeline_2/step05d_swr_decode_highres_rank_shuffle.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUT, SESSION_DIR, BASE, EEG_FS, SPIKE_FS, KPOS

EPS = 1e-12

# High-res SWR binning
DELTA_SWR = 0.005   # 5 ms bins
PAD_S     = 0.02    # 20 ms padding around SWR
TOPK      = 8
NSHUFFLE  = 200
BANDW     = 2       # posterior band half-width around fitted line

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
    # lam: expected spikes per bin, shape (K,N)
    Y = Y.astype(float)
    loglam = np.log(lam + EPS)
    return (Y @ loglam.T) - np.sum(lam, axis=1)[None, :]


def line_band_score(post, bandw=2):
    """
    Compute:
      - zbar(t) = posterior mean position
      - fit zbar ~ a t + b
      - r2 on zbar
      - band_mass = mean posterior mass within +/- bandw of fitted line
    """
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
    """
    Replay-first score: must reward movement.
    """
    ls, _ = line_band_score(post, bandw=BANDW)

    if len(zmap) >= 2:
        path_len = int(np.sum(np.abs(np.diff(zmap))))
        net_disp = int(abs(int(zmap[-1]) - int(zmap[0])))
    else:
        path_len = 0
        net_disp = 0

    T = post.shape[0]
    peak = float(np.mean(np.max(post, axis=1)))

    # motion gate (prevents stationary, ultra-confident lines from winning)
    motion_gate = 1.0 if net_disp >= 3 else 0.25

    # main trajectory strength: line-likeness * slope magnitude
    traj_strength = ls["band_mass"] * max(0.0, ls["r2"]) * abs(ls["slope"])

    # stabilize for short events (reward more bins a bit)
    score = traj_strength * motion_gate * (0.5 + 0.5 * peak) * np.sqrt(max(1, T))

    return score, ls, path_len, net_disp, peak


def shuffle_placefields(lam_20ms):
    """
    Circularly shift each neuron's place field along position (destroys
    consistent spatial code but preserves tuning shape & firing rate).
    """
    K, N = lam_20ms.shape
    lam_sh = np.empty_like(lam_20ms)
    shifts = np.random.randint(0, K, size=N)
    for n in range(N):
        lam_sh[:, n] = np.roll(lam_20ms[:, n], shifts[n])
    return lam_sh


# ---------------- load artifacts ----------------
pf = np.load(OUT / "placefields.npz")
lam_20 = pf["lam"]                # (K,N) expected spikes / 20ms
K, N = lam_20.shape
assert K == KPOS, f"placefields K={K} but KPOS={KPOS}"

# scale to 5ms bins
lam = lam_20 * (DELTA_SWR / 0.02)

hmm = np.load(OUT / "hmm_full_params.npz")
A = hmm["A"]
pi = hmm["pi"]
logA = np.log(A + EPS)
logpi = np.log(pi + EPS)

events = np.load(OUT / "swr_events.npy")
spikes = load_unit_spikes(OUT / "units.json")

rank_rows = []
pval_rows = []

print(f"Decoding {len(events)} SWRs at DELTA_SWR={DELTA_SWR}s  (lam scaled by {DELTA_SWR/0.02:.3f})")

for i, (a_eeg, b_eeg) in enumerate(events):
    t0 = max(0.0, float(a_eeg) / EEG_FS - PAD_S)
    t1 = float(b_eeg) / EEG_FS + PAD_S

    Yev = bin_spikes(spikes, t0, t1, DELTA_SWR)
    E = poisson_loglik(Yev, lam)
    post = forward_backward_log(E, logA, logpi)

    # MAP sequence (THIS is what step06 needs)
    zmap = np.argmax(post, axis=1).astype(int)
    zmap_list = [int(z) for z in zmap.tolist()]

    score_true, ls, path_len, net_disp, peak = replay_score(post, zmap)
    total_spikes = int(Yev.sum())
    dur_ms = 1000.0 * (t1 - t0)

    # shuffle test
    sh_scores = []
    for _ in range(NSHUFFLE):
        lam_sh = shuffle_placefields(lam)   # shuffle in the 5ms-rate domain
        Esh = poisson_loglik(Yev, lam_sh)
        post_sh = forward_backward_log(Esh, logA, logpi)
        zmap_sh = np.argmax(post_sh, axis=1)
        s, _, _, _, _ = replay_score(post_sh, zmap_sh)
        sh_scores.append(float(s))
    sh_scores = np.array(sh_scores, dtype=float)

    pval = float((np.sum(sh_scores >= score_true) + 1.0) / (NSHUFFLE + 1.0))

    # NOTE: we write BOTH event_idx and event, to stay compatible with older scripts.
    rank_rows.append({
        "event_idx": int(i),
        "event": int(i),

        # core replay outputs
        "score": float(score_true),
        "pval": pval,
        "dur_ms": float(dur_ms),
        "T_bins": int(len(zmap)),
        "Tbins": int(len(zmap)),  # keep old key too

        "total_spikes": int(total_spikes),
        "path_len": int(path_len),
        "net_disp": int(net_disp),
        "post_peak_mean": float(peak),

        "slope": float(ls["slope"]),
        "r2": float(ls["r2"]),
        "band_mass": float(ls["band_mass"]),

        # timing info (useful for reporting / sanity checks)
        "t0_s": float(t0),
        "t1_s": float(t1),
        "a_eeg": int(a_eeg),
        "b_eeg": int(b_eeg),
        "delta_swr_s": float(DELTA_SWR),
        "pad_s": float(PAD_S),

        # **the missing ingredient for Step 6**
        "z_map": zmap_list,
        "z_start": int(zmap_list[0]) if len(zmap_list) else None,
        "z_end": int(zmap_list[-1]) if len(zmap_list) else None,
    })

    pval_rows.append({
        "event_idx": int(i),
        "event": int(i),
        "score": float(score_true),
        "pval": pval,
        "shuffle_mean": float(np.mean(sh_scores)),
        "shuffle_std": float(np.std(sh_scores)),
        "shuffle_max": float(np.max(sh_scores)),
    })

    print(f"event {i:02d}: score={score_true:.4f}, p={pval:.3f}, net={net_disp}, path={path_len}, spikes={total_spikes}, T={len(zmap)}")

# rank
rank_sorted = sorted(rank_rows, key=lambda r: r["score"], reverse=True)

out_rank = OUT / "swr_highres_ranked.json"
out_pval = OUT / "swr_highres_pvals.json"
out_rank.write_text(json.dumps(rank_sorted, indent=2), encoding="utf-8")
out_pval.write_text(json.dumps(pval_rows, indent=2), encoding="utf-8")

print("Saved ->", out_rank)
print("Saved ->", out_pval)

# save top plots
for rank, r in enumerate(rank_sorted[:TOPK]):
    i = int(r["event_idx"])
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
    plt.title(
        f"highres rank {rank} SWR {i:02d}: score={r['score']:.3f}, p={r['pval']:.3f}, "
        f"net={r['net_disp']}, path={r['path_len']}"
    )
    plt.xlabel(f"time bin ({int(1000*DELTA_SWR)}ms)")
    plt.ylabel("position bin")
    plt.tight_layout()
    plt.savefig(FIG / f"swr_highres_top_rank_{rank:02d}.png", dpi=200)
    plt.close()

print("Saved figures ->", FIG / "swr_highres_top_rank_*.png")

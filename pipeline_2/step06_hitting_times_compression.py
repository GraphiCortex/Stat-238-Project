# pipeline_2/step06_hitting_times_compression.py
import json
from pathlib import Path

import numpy as np

from config import OUT, KPOS, DELTA

# High-res SWR bin (used in your step05d)
DELTA_SWR = 0.005  # seconds (5 ms)

# Inputs
Z_BEH_PATH = OUT / "Z.npy"
MOVE_MASK_PATH = OUT / "moving_mask.npy"
SWR_RANKED_PATH = OUT / "swr_highres_ranked.json"

# Outputs
OUT_JSON = OUT / "mathq2_hitting_times_summary.json"
OUT_NPZ  = OUT / "mathq2_hitting_times_mats.npz"


# ----------------------------
# Utilities: sequences -> segments
# ----------------------------
def to_segments(z: np.ndarray):
    """Return list of (state, run_length) for consecutive-constant segments."""
    z = np.asarray(z, dtype=int)
    if z.size == 0:
        return []
    segs = []
    cur = z[0]
    run = 1
    for t in range(1, z.size):
        if z[t] == cur:
            run += 1
        else:
            segs.append((int(cur), int(run)))
            cur = z[t]
            run = 1
    segs.append((int(cur), int(run)))
    return segs


def load_swr_paths(path: Path):
    """
    Extract decoded MAP paths from swr_highres_ranked.json.
    We expect step05d to store it as "z_map" (list[int]).
    Returns list of 1D int arrays.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    paths = []

    candidate_keys = [
        "z_map", "zMAP", "map_path", "decoded_path", "z_hat", "zhat", "path_z",
        "z", "Z", "path"
    ]

    for d in data:
        seq = None
        for k in candidate_keys:
            if k in d and isinstance(d[k], (list, tuple)):
                seq = d[k]
                break

        # Sometimes nested (e.g. d["decode"]["z_map"])
        if seq is None and "decode" in d and isinstance(d["decode"], dict):
            for k in candidate_keys:
                if k in d["decode"] and isinstance(d["decode"][k], (list, tuple)):
                    seq = d["decode"][k]
                    break

        if seq is None:
            continue

        z = np.array(seq, dtype=int).ravel()
        z = z[(0 <= z) & (z < KPOS)]
        if z.size >= 2:
            paths.append(z)

    return paths


# ----------------------------
# Build reduced state space (visited states)
# ----------------------------
def make_state_map(z: np.ndarray, K: int, min_count: int):
    """
    Return:
      states: sorted array of kept original states
      s2i: dict original_state -> compact_index
      counts: bincount
    """
    z = np.asarray(z, dtype=int)
    counts = np.bincount(z[(0 <= z) & (z < K)], minlength=K).astype(int)
    states = np.where(counts >= min_count)[0]
    if states.size == 0:
        # fallback: keep everything that appears at least once
        states = np.where(counts > 0)[0]
    s2i = {int(s): int(i) for i, s in enumerate(states.tolist())}
    return states, s2i, counts


def remap_segments(segs, s2i):
    """Keep only segments whose state is in s2i, and remap state labels to compact indices."""
    out = []
    for s, L in segs:
        if s in s2i and L > 0:
            out.append((s2i[s], int(L)))
    return out


# ----------------------------
# Estimate Q + dwell means (HSMM) and geometric dwell (HMM)
# ----------------------------
def estimate_Q_and_m_hsmm_from_segments(segs_compact, Kc, alpha=0.25):
    """
    Embedded Q from segment boundaries + empirical mean dwell.
    Smoothing:
      - transition counts get alpha added to avoid zero rows / singularities.
      - missing dwell -> filled with global mean dwell.
    """
    C = np.full((Kc, Kc), alpha, dtype=float)  # light smoothing everywhere
    dwell = [[] for _ in range(Kc)]

    for (s, L) in segs_compact:
        dwell[s].append(L)

    for k in range(len(segs_compact) - 1):
        i = segs_compact[k][0]
        j = segs_compact[k + 1][0]
        C[i, j] += 1.0

    # row normalize
    Q = C / C.sum(axis=1, keepdims=True)

    m = np.full(Kc, np.nan, dtype=float)
    for i in range(Kc):
        if len(dwell[i]) > 0:
            m[i] = float(np.mean(dwell[i]))

    # fill missing dwell with global mean (of observed)
    finite = np.isfinite(m)
    if np.any(finite):
        global_mean = float(np.mean(m[finite]))
        m[~finite] = global_mean
    else:
        m[:] = 1.0

    return Q, m


def estimate_m_hmm_geometric(z_compact: np.ndarray, Kc: int, beta=0.5):
    """
    Estimate geometric mean dwell m_i = 1/(1-p_stay(i)).
    Uses pseudocount beta to avoid p_stay=1 exactly.
    """
    z = np.asarray(z_compact, dtype=int)
    if z.size < 2:
        return np.full(Kc, np.nan, dtype=float)

    n_i = np.zeros(Kc, dtype=float)
    n_stay = np.zeros(Kc, dtype=float)

    for t in range(z.size - 1):
        i = z[t]
        j = z[t + 1]
        if 0 <= i < Kc:
            n_i[i] += 1.0
            if j == i:
                n_stay[i] += 1.0

    m_geo = np.full(Kc, np.nan, dtype=float)
    for i in range(Kc):
        if n_i[i] > 0:
            # smoothed p_stay
            p_stay = (n_stay[i] + beta) / (n_i[i] + 2.0 * beta)
            p_stay = float(np.clip(p_stay, 0.0, 1.0 - 1e-6))
            p_leave = 1.0 - p_stay
            m_geo[i] = 1.0 / p_leave

    # fill missing with global mean
    finite = np.isfinite(m_geo)
    if np.any(finite):
        m_geo[~finite] = float(np.mean(m_geo[finite]))
    else:
        m_geo[:] = 1.0

    return m_geo


# ----------------------------
# Expected hitting times for semi-Markov with mean holding m and embedded Q
# ----------------------------
def expected_hitting_times_all_pairs(Q: np.ndarray, m: np.ndarray, ridge=1e-10):
    """
    H[i,j] = E[time bins to hit j starting from i]
    for semi-Markov with embedded Q and mean holding m.

    For target j:
      h_j = 0
      h_i = m_i + sum_k Q[i,k] h_k,  i != j
    => (I - Q_NN) h_N = m_N

    ridge helps when A is near-singular.
    """
    K = Q.shape[0]
    H = np.full((K, K), np.nan, dtype=float)

    m = np.asarray(m, dtype=float)
    Q = np.asarray(Q, dtype=float)

    for j in range(K):
        idxN = [i for i in range(K) if i != j]
        if len(idxN) == 0:
            H[j, j] = 0.0
            continue

        Q_NN = Q[np.ix_(idxN, idxN)]
        m_N  = m[idxN]

        if not np.all(np.isfinite(m_N)):
            continue

        A = np.eye(len(idxN)) - Q_NN
        A = A + ridge * np.eye(A.shape[0])

        try:
            h_N = np.linalg.solve(A, m_N)
        except np.linalg.LinAlgError:
            # fallback: least squares
            h_N, *_ = np.linalg.lstsq(A, m_N, rcond=None)

        for ii, i in enumerate(idxN):
            H[i, j] = float(h_N[ii])
        H[j, j] = 0.0

    return H


def summarize_H(H: np.ndarray):
    K = H.shape[0]
    mask = np.isfinite(H) & (~np.eye(K, dtype=bool))
    vals = H[mask]
    if vals.size == 0:
        return dict(mean=np.nan, median=np.nan, n=0)
    return dict(
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        n=int(vals.size),
    )


def main():
    # ----------------------------
    # Behavior sequence (movement only)
    # ----------------------------
    Z = np.load(Z_BEH_PATH).astype(int)
    move = np.load(MOVE_MASK_PATH).astype(bool)
    Zm = Z[move]
    Zm = Zm[(0 <= Zm) & (Zm < KPOS)]

    # pick support states (behavior is long, so we can require decent counts)
    beh_states, beh_s2i, beh_counts = make_state_map(Zm, KPOS, min_count=50)
    Zm_c = np.array([beh_s2i[int(s)] for s in Zm if int(s) in beh_s2i], dtype=int)

    beh_segs = remap_segments(to_segments(Zm), beh_s2i)

    Q_beh, m_beh_hsmm = estimate_Q_and_m_hsmm_from_segments(beh_segs, len(beh_states))
    m_beh_hmm = estimate_m_hmm_geometric(Zm_c, len(beh_states))
    # same embedded jumps, different dwell model
    Q_beh_hmm = Q_beh.copy()
    Q_beh_hsmm = Q_beh.copy()

    # ----------------------------
    # Replay sequences from SWR decoded paths
    # ----------------------------
    if not SWR_RANKED_PATH.exists():
        raise FileNotFoundError(f"Missing: {SWR_RANKED_PATH}")

    swr_paths = load_swr_paths(SWR_RANKED_PATH)
    if len(swr_paths) == 0:
        raise RuntimeError(
            "No SWR MAP paths found in swr_highres_ranked.json. "
            "Re-run step05d after adding 'z_map' to each event."
        )

    Zrep = np.concatenate(swr_paths)
    Zrep = Zrep[(0 <= Zrep) & (Zrep < KPOS)]

    # replay is short, use a smaller threshold
    rep_states, rep_s2i, rep_counts = make_state_map(Zrep, KPOS, min_count=5)
    Zrep_c = np.array([rep_s2i[int(s)] for s in Zrep if int(s) in rep_s2i], dtype=int)

    rep_segs = remap_segments(to_segments(Zrep), rep_s2i)

    Q_rep, m_rep_hsmm = estimate_Q_and_m_hsmm_from_segments(rep_segs, len(rep_states))
    m_rep_hmm = estimate_m_hmm_geometric(Zrep_c, len(rep_states))
    Q_rep_hmm = Q_rep.copy()
    Q_rep_hsmm = Q_rep.copy()

    # ----------------------------
    # Expected hitting times (all pairs, on reduced state sets)
    # ----------------------------
    H_beh_hmm_c  = expected_hitting_times_all_pairs(Q_beh_hmm,  m_beh_hmm)
    H_beh_hsmm_c = expected_hitting_times_all_pairs(Q_beh_hsmm, m_beh_hsmm)

    H_rep_hmm_c  = expected_hitting_times_all_pairs(Q_rep_hmm,  m_rep_hmm)
    H_rep_hsmm_c = expected_hitting_times_all_pairs(Q_rep_hsmm, m_rep_hsmm)

    sum_beh_hmm  = summarize_H(H_beh_hmm_c)
    sum_beh_hsmm = summarize_H(H_beh_hsmm_c)
    sum_rep_hmm  = summarize_H(H_rep_hmm_c)
    sum_rep_hsmm = summarize_H(H_rep_hsmm_c)

    # Convert to seconds
    def to_sec(summary, delta):
        return {k: (summary[k] * delta if k in ("mean", "median") else summary[k]) for k in summary}

    beh_hmm_sec  = to_sec(sum_beh_hmm,  DELTA)
    beh_hsmm_sec = to_sec(sum_beh_hsmm, DELTA)
    rep_hmm_sec  = to_sec(sum_rep_hmm,  DELTA_SWR)
    rep_hsmm_sec = to_sec(sum_rep_hsmm, DELTA_SWR)

    # Compression factors (behavior time / replay time), based on mean hitting time
    C_hmm  = np.nan
    C_hsmm = np.nan
    if np.isfinite(beh_hmm_sec["mean"]) and np.isfinite(rep_hmm_sec["mean"]) and rep_hmm_sec["mean"] > 0:
        C_hmm = beh_hmm_sec["mean"] / rep_hmm_sec["mean"]
    if np.isfinite(beh_hsmm_sec["mean"]) and np.isfinite(rep_hsmm_sec["mean"]) and rep_hsmm_sec["mean"] > 0:
        C_hsmm = beh_hsmm_sec["mean"] / rep_hsmm_sec["mean"]

    summary = dict(
        notes=dict(
            behavior_sequence="Z restricted to moving bins (moving_mask.npy), then restricted to sufficiently-visited states",
            replay_sequence="concatenated decoded MAP paths from swr_highres_ranked.json (z_map), then restricted to sufficiently-visited states",
            HMM="geometric dwell (estimated p_stay), embedded chain from segment-boundary jumps",
            HSMM="state-dependent non-geometric dwell from empirical run lengths, embedded chain from segment-boundary jumps",
            hitting_time_definition="expected time (in bins) to hit target state j from start i, averaged over i!=j",
            smoothing="light transition pseudocount + dwell fill for unobserved states; small ridge added to linear solves",
        ),
        supports=dict(
            KPOS=int(KPOS),
            behavior_states_used=int(len(beh_states)),
            replay_states_used=int(len(rep_states)),
            behavior_min_count=int(50),
            replay_min_count=int(5),
        ),
        behavior=dict(
            hmm_bins=sum_beh_hmm,
            hsmm_bins=sum_beh_hsmm,
            hmm_seconds=beh_hmm_sec,
            hsmm_seconds=beh_hsmm_sec,
        ),
        replay=dict(
            hmm_bins=sum_rep_hmm,
            hsmm_bins=sum_rep_hsmm,
            hmm_seconds=rep_hmm_sec,
            hsmm_seconds=rep_hsmm_sec,
        ),
        compression_factors=dict(
            C_HMM=float(C_hmm),
            C_HSMM=float(C_hsmm),
            definition="C = (mean pairwise hitting time in behavior seconds) / (mean pairwise hitting time in replay seconds)",
            delta_behavior=float(DELTA),
            delta_replay=float(DELTA_SWR),
        )
    )

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        OUT_NPZ,
        # reduced matrices + maps
        beh_states=beh_states,
        rep_states=rep_states,
        Q_beh=Q_beh,
        Q_rep=Q_rep,
        m_beh_hmm=m_beh_hmm,
        m_beh_hsmm=m_beh_hsmm,
        m_rep_hmm=m_rep_hmm,
        m_rep_hsmm=m_rep_hsmm,
        H_beh_hmm=H_beh_hmm_c,
        H_beh_hsmm=H_beh_hsmm_c,
        H_rep_hmm=H_rep_hmm_c,
        H_rep_hsmm=H_rep_hsmm_c,
    )

    print("Saved:")
    print(" ", OUT_JSON)
    print(" ", OUT_NPZ)
    print("\nKey outputs (seconds, mean pairwise hitting time):")
    print(" Behavior HMM :", beh_hmm_sec["mean"])
    print(" Behavior HSMM:", beh_hsmm_sec["mean"])
    print(" Replay   HMM :", rep_hmm_sec["mean"])
    print(" Replay   HSMM:", rep_hsmm_sec["mean"])
    print("\nCompression factors (behavior/replay):")
    print(" C(HMM) :", C_hmm)
    print(" C(HSMM):", C_hsmm)


if __name__ == "__main__":
    main()

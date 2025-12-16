import numpy as np
import matplotlib.pyplot as plt
from config import OUT
from model_core import make_folds, fit_poisson_means, fit_transition, fit_pi

EPS = 1e-12

# hyperparams (match decoding scripts)
N_FOLDS = 5
BAND = 1
ALPHA_A = 1.0
ALPHA_MU = 0.1

# duration settings
D_MAX = 80        # bins (80*20ms = 1.6s)
ALPHA_D = 1.0     # HSMM duration smoothing
MIN_RUNS_PER_STATE = 5  # only evaluate KL if enough runs

def run_lengths(Z):
    states = []
    lengths = []
    cur = Z[0]
    L = 1
    for z in Z[1:]:
        if z == cur:
            L += 1
        else:
            states.append(cur)
            lengths.append(L)
            cur = z
            L = 1
    states.append(cur); lengths.append(L)
    return np.array(states, dtype=int), np.array(lengths, dtype=int)

def fit_durations(Z, K, Dmax, alpha=1.0):
    s, L = run_lengths(Z)
    G = np.zeros((K, Dmax), dtype=float)  # G[k,d-1]
    for st, ln in zip(s, L):
        d = min(int(ln), Dmax)
        G[st, d-1] += 1.0
    G = G + alpha
    G = G / np.sum(G, axis=1, keepdims=True)
    return G

def empirical_duration_hist(Z, K, Dmax):
    s, L = run_lengths(Z)
    H = np.zeros((K, Dmax), dtype=float)
    runs = np.zeros(K, dtype=int)
    for st, ln in zip(s, L):
        d = min(int(ln), Dmax)
        H[st, d-1] += 1.0
        runs[st] += 1
    # normalize per state where possible
    for k in range(K):
        if H[k].sum() > 0:
            H[k] /= H[k].sum()
    return H, runs

def hmm_geometric_duration(A, Dmax):
    # For each state i: P(D=d) = (A_ii)^(d-1) * (1-A_ii), truncated/renormalized to 1..Dmax
    K = A.shape[0]
    G = np.zeros((K, Dmax), dtype=float)
    for i in range(K):
        pii = float(A[i, i])
        pii = min(max(pii, EPS), 1.0 - EPS)
        p_exit = 1.0 - pii
        d = np.arange(1, Dmax + 1, dtype=float)
        g = (pii ** (d - 1.0)) * p_exit
        g = g / max(g.sum(), EPS)
        G[i] = g
    return G

def KL(p, q):
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum(p * np.log(p / q)))

# ---- load data ----
Y = np.load(OUT / "Y.npy")
Z = np.load(OUT / "Z.npy").astype(int)

T, N = Y.shape
K = int(Z.max() + 1)
folds = make_folds(T, N_FOLDS)

kl_hmm_folds = []
kl_hsmm_folds = []

# optional: store overlays for a few states
overlay_states = [10, 25, 40]  # change if you want

for f, (a, b) in enumerate(folds):
    test_idx = np.zeros(T, dtype=bool); test_idx[a:b] = True
    train_idx = ~test_idx

    Ztr = Z[train_idx]
    Zte = Z[test_idx]

    # fit transitions on training labels
    A = fit_transition(Ztr, K, band=BAND, alpha=ALPHA_A, allow_self=True)

    # HMM implied durations
    G_hmm = hmm_geometric_duration(A, D_MAX)

    # HSMM learned durations (from Ztr)
    G_hsmm = fit_durations(Ztr, K, D_MAX, alpha=ALPHA_D)

    # empirical from test
    H_emp, runs = empirical_duration_hist(Zte, K, D_MAX)

    kl_hmm_states = []
    kl_hsmm_states = []

    for k in range(K):
        if runs[k] < MIN_RUNS_PER_STATE:
            continue
        p = H_emp[k]
        kl_hmm_states.append(KL(p, G_hmm[k]))
        kl_hsmm_states.append(KL(p, G_hsmm[k]))

    kl_hmm_fold = float(np.mean(kl_hmm_states)) if len(kl_hmm_states) else float("nan")
    kl_hsmm_fold = float(np.mean(kl_hsmm_states)) if len(kl_hsmm_states) else float("nan")

    kl_hmm_folds.append(kl_hmm_fold)
    kl_hsmm_folds.append(kl_hsmm_fold)

    print(f"fold {f}: KL mean over states (emp||model)  HMM={kl_hmm_fold:.4f}  HSMM={kl_hsmm_fold:.4f}  states_used={len(kl_hmm_states)}")

    # overlays (first fold only to keep output light)
    if f == 0:
        d = np.arange(1, D_MAX + 1)
        for k in overlay_states:
            if k < 0 or k >= K or runs[k] < MIN_RUNS_PER_STATE:
                continue
            plt.figure()
            plt.plot(d, H_emp[k], label="Empirical (test)")
            plt.plot(d, G_hmm[k], label="HMM geometric")
            plt.plot(d, G_hsmm[k], label="HSMM learned")
            plt.xlabel("Duration (bins)")
            plt.ylabel("Probability")
            plt.title(f"Duration overlay: state {k}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT / f"duration_overlay_state_{k}.png", dpi=200)
            plt.close()

kl_hmm_folds = np.array(kl_hmm_folds, dtype=float)
kl_hsmm_folds = np.array(kl_hsmm_folds, dtype=float)

np.savez(OUT / "duration_KL_results.npz", kl_hmm=kl_hmm_folds, kl_hsmm=kl_hsmm_folds)

plt.figure()
plt.boxplot([kl_hmm_folds[~np.isnan(kl_hmm_folds)], kl_hsmm_folds[~np.isnan(kl_hsmm_folds)]],
            tick_labels=["HMM", "HSMM"])
plt.ylabel("Mean KL(empirical || model) over states")
plt.title("Duration model fit (CV)")
plt.tight_layout()
plt.savefig(OUT / "duration_KL_bar.png", dpi=200)
plt.close()

print("Saved:",
      OUT / "duration_KL_results.npz",
      OUT / "duration_KL_bar.png")

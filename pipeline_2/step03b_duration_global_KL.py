import numpy as np
import matplotlib.pyplot as plt
from config import OUT
from model_core import make_folds, fit_transition

EPS = 1e-12

# Match earlier settings
N_FOLDS = 5
BAND = 1
ALPHA_A = 1.0

D_MAX = 80      # bins (80*20ms = 1.6s)
ALPHA_D = 1.0   # HSMM duration smoothing

def run_lengths(Z):
    states = []
    lengths = []
    cur = int(Z[0])
    L = 1
    for z in Z[1:]:
        z = int(z)
        if z == cur:
            L += 1
        else:
            states.append(cur)
            lengths.append(L)
            cur = z
            L = 1
    states.append(cur)
    lengths.append(L)
    return np.array(states, dtype=int), np.array(lengths, dtype=int)

def empirical_global_duration_hist(Z, Dmax):
    _, L = run_lengths(Z)
    H = np.zeros(Dmax, dtype=float)
    for ln in L:
        d = min(int(ln), Dmax)
        H[d - 1] += 1.0
    H = H / max(H.sum(), EPS)
    return H

def empirical_state_weights(Z, K):
    s, _ = run_lengths(Z)
    w = np.zeros(K, dtype=float)
    for st in s:
        w[int(st)] += 1.0
    if w.sum() > 0:
        w /= w.sum()
    return w

def fit_hsmm_durations_from_Z(Z, K, Dmax, alpha=1.0):
    s, L = run_lengths(Z)
    G = np.zeros((K, Dmax), dtype=float)
    for st, ln in zip(s, L):
        d = min(int(ln), Dmax)
        G[int(st), d - 1] += 1.0
    G = G + alpha
    G = G / np.sum(G, axis=1, keepdims=True)
    return G

def hmm_geometric_duration(A, Dmax):
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

# --------------------
# Load data
# --------------------
Z = np.load(OUT / "Z.npy").astype(int)
T = len(Z)
K = int(Z.max() + 1)

folds = make_folds(T, N_FOLDS)

kl_hmm = []
kl_hsmm = []

for f, (a, b) in enumerate(folds):
    test_idx = np.zeros(T, dtype=bool); test_idx[a:b] = True
    train_idx = ~test_idx

    Ztr = Z[train_idx]
    Zte = Z[test_idx]

    # Empirical pooled durations from test
    p_emp = empirical_global_duration_hist(Zte, D_MAX)

    # State weights from test (mixture weights)
    w = empirical_state_weights(Zte, K)

    # HMM model durations from train (geometric implied by A_ii)
    A = fit_transition(Ztr, K, band=BAND, alpha=ALPHA_A, allow_self=True)
    G_hmm = hmm_geometric_duration(A, D_MAX)
    q_hmm = (w[:, None] * G_hmm).sum(axis=0)
    q_hmm = q_hmm / max(q_hmm.sum(), EPS)

    # HSMM model durations from train (learned histogram per state)
    G_hsmm = fit_hsmm_durations_from_Z(Ztr, K, D_MAX, alpha=ALPHA_D)
    q_hsmm = (w[:, None] * G_hsmm).sum(axis=0)
    q_hsmm = q_hsmm / max(q_hsmm.sum(), EPS)

    kl1 = KL(p_emp, q_hmm)
    kl2 = KL(p_emp, q_hsmm)

    kl_hmm.append(kl1)
    kl_hsmm.append(kl2)

    print(f"fold {f}: global KL(emp||model)  HMM={kl1:.4f}  HSMM={kl2:.4f}")

    # Overlay for fold 0
    if f == 0:
        d = np.arange(1, D_MAX + 1)
        plt.figure()
        plt.plot(d, p_emp, label="Empirical (test, pooled)")
        plt.plot(d, q_hmm, label="HMM pooled geometric")
        plt.plot(d, q_hsmm, label="HSMM pooled learned")
        plt.xlabel("Duration (bins)")
        plt.ylabel("Probability")
        plt.title("Global duration distribution (fold 0)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "duration_global_overlay_fold0.png", dpi=200)
        plt.close()

kl_hmm = np.array(kl_hmm, dtype=float)
kl_hsmm = np.array(kl_hsmm, dtype=float)

np.savez(OUT / "duration_global_KL_results.npz", kl_hmm=kl_hmm, kl_hsmm=kl_hsmm)

plt.figure()
plt.boxplot([kl_hmm, kl_hsmm], tick_labels=["HMM", "HSMM"])
plt.ylabel("KL(empirical || model) [pooled durations]")
plt.title("Global duration model fit (CV)")
plt.tight_layout()
plt.savefig(OUT / "duration_global_KL_boxplot.png", dpi=200)
plt.close()

print("Saved:",
      OUT / "duration_global_KL_results.npz",
      OUT / "duration_global_KL_boxplot.png",
      OUT / "duration_global_overlay_fold0.png")

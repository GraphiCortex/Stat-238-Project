import numpy as np
import json
import matplotlib.pyplot as plt
from config import OUT
from model_core import make_folds, fit_poisson_means, fit_pi, poisson_loglik_matrix, mean_abs_bin_error, fit_transition

EPS = 1e-12

# hyperparams
N_FOLDS = 5
BAND = 1
D_MAX = 80      # max duration in bins (80*20ms = 1.6s). adjust if you want.
ALPHA_D = 1.0   # duration smoothing
ALPHA_MU = 0.1  # emission smoothing
ALPHA_A = 1.0   # segment-transition smoothing

def run_lengths(Z):
    # returns (states, lengths)
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

def viterbi_hsmm(Y, pi, Aseg, mu, G):
    # HSMM MAP decoding via segment DP
    T, N = Y.shape
    K = mu.shape[0]
    Dmax = G.shape[1]

    logpi = np.log(pi + EPS)
    logA = np.log(Aseg + EPS)
    logG = np.log(G + EPS)

    E = poisson_loglik_matrix(Y, mu)  # (T,K)
    C = np.zeros((K, T + 1), dtype=float)  # cumulative for segment sums
    for k in range(K):
        C[k, 1:] = np.cumsum(E[:, k])

    delta = np.full((T, K), -np.inf)
    prev_state = np.full((T, K), -1, dtype=int)
    prev_dur = np.full((T, K), -1, dtype=int)

    for t in range(T):
        maxd = min(Dmax, t + 1)
        for k in range(K):
            best_val = -np.inf
            best_j = -1
            best_d = -1

            for d in range(1, maxd + 1):
                s = t - d + 1
                seg_em = C[k, t + 1] - C[k, s]
                val = seg_em + logG[k, d - 1]

                if s == 0:
                    # first segment
                    val += logpi[k]
                    j = -1
                else:
                    pt = s - 1
                    # prev segment must end at pt and transition j->k (no self at segment level is typical)
                    # banded constraint already in Aseg; we just take max over j
                    tmp = delta[pt] + logA[:, k]
                    j = int(np.argmax(tmp))
                    val += tmp[j]

                if val > best_val:
                    best_val = val
                    best_j = j
                    best_d = d

            delta[t, k] = best_val
            prev_state[t, k] = best_j
            prev_dur[t, k] = best_d

    # backtrack segments
    zhat = np.zeros(T, dtype=int)
    t = T - 1
    k = int(np.argmax(delta[t]))
    while t >= 0:
        d = int(prev_dur[t, k])
        s = t - d + 1
        zhat[s:t+1] = k
        j = int(prev_state[t, k])
        t = s - 1
        k = j if j != -1 else 0
        if t >= 0 and j == -1:
            break
    return zhat

Y = np.load(OUT / "Y.npy")
Z = np.load(OUT / "Z.npy").astype(int)
moving = np.load(OUT / "moving_mask.npy").astype(bool)

T, N = Y.shape
K = int(Z.max() + 1)

folds = make_folds(T, N_FOLDS)
fold_losses = []

for f, (a, b) in enumerate(folds):
    test_idx = np.zeros(T, dtype=bool)
    test_idx[a:b] = True
    train_idx = ~test_idx

    Ytr, Ztr = Y[train_idx], Z[train_idx]
    Yte, Zte = Y[test_idx], Z[test_idx]
    mte = moving[test_idx]

    mu = fit_poisson_means(Ytr, Ztr, K, alpha=ALPHA_MU)
    pi = fit_pi(Ztr, K, alpha=1.0)

    # segment transitions: estimate on Z but forbid self-loops at segment level
    Aseg = fit_transition(Ztr, K, band=BAND, alpha=ALPHA_A, allow_self=True)
    G = fit_durations(Ztr, K, D_MAX, alpha=ALPHA_D)

    Zhat = viterbi_hsmm(Yte, pi, Aseg, mu, G)

    loss = mean_abs_bin_error(Zte, Zhat, mte)
    fold_losses.append(loss)

    print(f"fold {f}: test [{a},{b})  moving_bins={int(mte.sum())}  loss={loss}")

fold_losses = np.array(fold_losses, dtype=float)
np.savez(OUT / "hsmm_cv_results.npz", fold_losses=fold_losses)

(OUT / "hsmm_cv_summary.json").write_text(
    json.dumps({"D_MAX": int(D_MAX),
                "fold_losses": fold_losses.tolist(),
                "mean": float(np.nanmean(fold_losses)),
                "std": float(np.nanstd(fold_losses))}, indent=2),
    encoding="utf-8"
)

plt.figure()
plt.boxplot(fold_losses[~np.isnan(fold_losses)])
plt.ylabel("Mean |Z - Zhat| on moving bins")
plt.title("HSMM decoding loss (CV)")
plt.tight_layout()
plt.savefig(OUT / "hsmm_loss_boxplot.png", dpi=200)
plt.close()

print("Saved:",
      OUT / "hsmm_cv_results.npz",
      OUT / "hsmm_loss_boxplot.png")

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

EPS = 1e-12

@dataclass
class HMMParams:
    pi: np.ndarray         # (K,)
    A: np.ndarray          # (K,K)
    mu: np.ndarray         # (K,N) mean spike count per DELTA-bin

def make_folds(T: int, n_folds: int = 5) -> List[Tuple[int,int]]:
    # contiguous folds
    edges = np.linspace(0, T, n_folds + 1).astype(int)
    return [(int(edges[i]), int(edges[i+1])) for i in range(n_folds)]

def fit_poisson_means(Y: np.ndarray, Z: np.ndarray, K: int, alpha: float = 0.1) -> np.ndarray:
    # mu[k,n] = average count in bins where Z==k (with smoothing)
    T, N = Y.shape
    mu = np.zeros((K, N), dtype=float)
    for k in range(K):
        idx = (Z == k)
        nk = int(idx.sum())
        if nk == 0:
            mu[k] = alpha  # tiny fallback
        else:
            mu[k] = (Y[idx].sum(axis=0) + alpha) / (nk + alpha)
    mu = np.clip(mu, 1e-6, None)
    return mu

def fit_transition(Z: np.ndarray, K: int, band: int = 1, alpha: float = 1.0, allow_self: bool = True) -> np.ndarray:
    # counts
    C = np.zeros((K, K), dtype=float)
    for a, b in zip(Z[:-1], Z[1:]):
        C[a, b] += 1.0

    # band constraint
    M = np.zeros((K, K), dtype=bool)
    for i in range(K):
        for j in range(max(0, i - band), min(K, i + band + 1)):
            M[i, j] = True
    if not allow_self:
        np.fill_diagonal(M, False)

    A = np.zeros((K, K), dtype=float)
    for i in range(K):
        allowed = M[i]
        if not np.any(allowed):
            # shouldn't happen, but just in case
            allowed[i] = True
        row = C[i].copy()
        row[~allowed] = 0.0
        row = row + alpha * allowed.astype(float)
        A[i] = row / max(row.sum(), EPS)
    return A

def fit_pi(Z: np.ndarray, K: int, alpha: float = 1.0) -> np.ndarray:
    pi = np.zeros(K, dtype=float)
    pi[Z[0]] += 1.0
    pi = pi + alpha
    pi = pi / pi.sum()
    return pi

def poisson_loglik_matrix(Y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    # returns log p(Y_t | state=k) up to constants independent of k (drop log(y!))
    # log p = sum_n [ y log mu - mu ]
    Y = Y.astype(float)
    logmu = np.log(mu + 1e-12)  # (K,N)
    # compute E[t,k]
    # (T,N) @ (N,K) -> (T,K)
    term1 = Y @ logmu.T
    term2 = np.sum(mu, axis=1)[None, :]  # (1,K)
    return term1 - term2

def viterbi_hmm(Y: np.ndarray, params: HMMParams) -> np.ndarray:
    T, N = Y.shape
    K = params.A.shape[0]
    logA = np.log(params.A + EPS)
    logpi = np.log(params.pi + EPS)

    E = poisson_loglik_matrix(Y, params.mu)  # (T,K)

    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = logpi + E[0]
    psi[0] = 0

    for t in range(1, T):
        # max over previous states
        prev = delta[t-1][:, None] + logA  # (K,K)
        psi[t] = np.argmax(prev, axis=0)
        delta[t] = prev[psi[t], np.arange(K)] + E[t]

    # backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path

def mean_abs_bin_error(Z_true: np.ndarray, Z_hat: np.ndarray, mask: np.ndarray) -> float:
    idx = mask.astype(bool)
    if idx.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(Z_true[idx] - Z_hat[idx])))

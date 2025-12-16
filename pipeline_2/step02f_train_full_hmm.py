import json
import numpy as np
from config import OUT
from model_core import fit_poisson_means, fit_transition, fit_pi

# match your CV hyperparams
BAND = 1
ALPHA_A = 1.0
ALPHA_MU = 0.1

Y = np.load(OUT / "Y.npy")
Z = np.load(OUT / "Z.npy").astype(int)

T, N = Y.shape
K = int(Z.max() + 1)

mu = fit_poisson_means(Y, Z, K, alpha=ALPHA_MU)             # (K,N)
A  = fit_transition(Z, K, band=BAND, alpha=ALPHA_A, allow_self=True)  # (K,K)
pi = fit_pi(Z, K, alpha=1.0)                                # (K,)

np.savez(OUT / "hmm_full_params.npz", pi=pi, A=A, mu=mu)

summary = {
    "T": int(T),
    "N_units": int(N),
    "K_states": int(K),
    "BAND": int(BAND),
    "ALPHA_A": float(ALPHA_A),
    "ALPHA_MU": float(ALPHA_MU),
    "A_diag_mean": float(np.mean(np.diag(A))),
    "A_diag_min": float(np.min(np.diag(A))),
    "A_diag_max": float(np.max(np.diag(A))),
    "mu_mean": float(np.mean(mu)),
    "mu_min": float(np.min(mu)),
    "mu_max": float(np.max(mu)),
}

(OUT / "hmm_full_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Saved:")
print(" ", OUT / "hmm_full_params.npz")
print(" ", OUT / "hmm_full_summary.json")
print("Summary:", summary)

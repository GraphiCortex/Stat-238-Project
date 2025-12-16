import numpy as np
import json
import matplotlib.pyplot as plt
from config import OUT
from model_core import HMMParams, make_folds, fit_poisson_means, fit_transition, fit_pi, viterbi_hmm, mean_abs_bin_error

# hyperparams
N_FOLDS = 5
BAND = 1        # allow i-1,i,i+1 transitions
ALPHA_A = 1.0   # transition smoothing
ALPHA_MU = 0.1  # emission smoothing

Y = np.load(OUT / "Y.npy")
Z = np.load(OUT / "Z.npy").astype(int)
moving = np.load(OUT / "moving_mask.npy").astype(bool)

T, N = Y.shape
K = int(Z.max() + 1)

folds = make_folds(T, N_FOLDS)
fold_losses = []
fold_paths = []

for f, (a, b) in enumerate(folds):
    test_idx = np.zeros(T, dtype=bool)
    test_idx[a:b] = True
    train_idx = ~test_idx

    Ytr, Ztr = Y[train_idx], Z[train_idx]
    Yte, Zte = Y[test_idx], Z[test_idx]
    mte = moving[test_idx]

    mu = fit_poisson_means(Ytr, Ztr, K, alpha=ALPHA_MU)
    A  = fit_transition(Ztr, K, band=BAND, alpha=ALPHA_A, allow_self=True)
    pi = fit_pi(Ztr, K, alpha=1.0)

    params = HMMParams(pi=pi, A=A, mu=mu)
    Zhat = viterbi_hmm(Yte, params)

    loss = mean_abs_bin_error(Zte, Zhat, mte)
    fold_losses.append(loss)
    fold_paths.append((Zte, Zhat))

    print(f"fold {f}: test [{a},{b})  moving_bins={int(mte.sum())}  loss={loss}")

fold_losses = np.array(fold_losses, dtype=float)
np.savez(OUT / "hmm_cv_results.npz", fold_losses=fold_losses)

(OUT / "hmm_cv_summary.json").write_text(
    json.dumps({"fold_losses": fold_losses.tolist(),
                "mean": float(np.nanmean(fold_losses)),
                "std": float(np.nanstd(fold_losses))}, indent=2),
    encoding="utf-8"
)

# simple plot: losses
plt.figure()
plt.boxplot(fold_losses[~np.isnan(fold_losses)])
plt.ylabel("Mean |Z - Zhat| on moving bins")
plt.title("HMM decoding loss (CV)")
plt.tight_layout()
plt.savefig(OUT / "hmm_loss_boxplot.png", dpi=200)
plt.close()

# overlay plot for fold 0
Z0, H0 = fold_paths[0]
plt.figure()
plt.plot(Z0, label="true Z")
plt.plot(H0, label="HMM decoded")
plt.legend()
plt.title("HMM decode overlay (fold 0)")
plt.tight_layout()
plt.savefig(OUT / "hmm_decode_overlay_fold0.png", dpi=200)
plt.close()

print("Saved:",
      OUT / "hmm_cv_results.npz",
      OUT / "hmm_loss_boxplot.png",
      OUT / "hmm_decode_overlay_fold0.png")

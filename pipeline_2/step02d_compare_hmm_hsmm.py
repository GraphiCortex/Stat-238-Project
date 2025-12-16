import numpy as np
import matplotlib.pyplot as plt
from config import OUT

hmm = np.load(OUT / "hmm_cv_results.npz")["fold_losses"]
hsmm = np.load(OUT / "hsmm_cv_results.npz")["fold_losses"]

plt.figure()
plt.boxplot([hmm[~np.isnan(hmm)], hsmm[~np.isnan(hsmm)]], tick_labels=["HMM", "HSMM"])
plt.ylabel("Mean |Z - Zhat| on moving bins")
plt.title("Decoding risk comparison (CV)")
plt.tight_layout()
plt.savefig(OUT / "compare_hmm_hsmm_boxplot.png", dpi=200)
plt.close()

print("HMM mean:", float(np.nanmean(hmm)))
print("HSMM mean:", float(np.nanmean(hsmm)))
print("Saved ->", OUT / "compare_hmm_hsmm_boxplot.png")

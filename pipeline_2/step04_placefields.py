import json
import numpy as np
import matplotlib.pyplot as plt
from config import OUT, DELTA, KPOS

EPS = 1e-9

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

Y = np.load(OUT / "Y.npy")                  # (T, N)
Z = np.load(OUT / "Z.npy").astype(int)      # (T,)
moving = np.load(OUT / "moving_mask.npy").astype(bool)

T, N = Y.shape
K = int(Z.max() + 1)

# Use moving bins for place fields (standard). You can flip to all bins if you want.
mask = moving.copy()
if mask.sum() < 10:
    mask[:] = True

occ_bins = np.zeros(K, dtype=int)
spk_sum = np.zeros((K, N), dtype=float)

for k in range(K):
    idx = (Z == k) & mask
    occ_bins[k] = int(idx.sum())
    if occ_bins[k] > 0:
        spk_sum[k] = Y[idx].sum(axis=0)

# lambda_per_bin[k,n] = E[count in one 20ms bin | Z=k]
lam = spk_sum / np.maximum(occ_bins[:, None], 1)
lam = np.clip(lam, EPS, None)

# occupancy in seconds + occupancy prob
occ_s = occ_bins * DELTA
p_occ = occ_bins / max(occ_bins.sum(), 1)

# Spatial info (bits/spike) using common formula (works with per-bin lam too)
lam_bar = (p_occ[:, None] * lam).sum(axis=0)  # (N,)
ratio = lam / np.clip(lam_bar[None, :], EPS, None)
spatial_info = (p_occ[:, None] * ratio * np.log2(np.clip(ratio, EPS, None))).sum(axis=0)

order = np.argsort(-spatial_info)

np.savez(
    OUT / "placefields.npz",
    lam=lam,
    occ_bins=occ_bins,
    occ_s=occ_s,
    p_occ=p_occ,
    spatial_info=spatial_info,
    order=order,
    used_mask="moving",
)

print("Saved ->", OUT / "placefields.npz")
print("mask used:", "moving", "mask bins:", int(mask.sum()), "of", T)
print("lam shape:", lam.shape)

# --------- plots ----------
# occupancy
plt.figure()
plt.plot(occ_s)
plt.xlabel("Position bin")
plt.ylabel("Occupancy (s)")
plt.title("Occupancy (moving bins)")
plt.tight_layout()
plt.savefig(FIG / "pf_occupancy.png", dpi=200)
plt.close()

# place field heatmap (sorted)
plt.figure(figsize=(10, 5))
plt.imshow(lam[:, order].T, aspect="auto", origin="lower")
plt.xlabel("Position bin")
plt.ylabel("Neuron (sorted by spatial info)")
plt.title("Place fields (lambda per 20ms bin)")
plt.tight_layout()
plt.savefig(FIG / "pf_heatmap_sorted.png", dpi=200)
plt.close()

# top10 spatial info
top = order[:10]
plt.figure()
plt.bar(np.arange(len(top)), spatial_info[top])
plt.xticks(np.arange(len(top)), [str(i) for i in top], rotation=0)
plt.xlabel("Neuron index")
plt.ylabel("Spatial info (bits/spike)")
plt.title("Top 10 spatial information")
plt.tight_layout()
plt.savefig(FIG / "pf_spatial_info_top10.png", dpi=200)
plt.close()

# top3 tuning curves
plt.figure()
for j in order[:3]:
    plt.plot(lam[:, j], label=f"unit {j}")
plt.xlabel("Position bin")
plt.ylabel("lambda (counts / 20ms)")
plt.title("Top 3 tuning curves")
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "pf_top3_tuning.png", dpi=200)
plt.close()

print("Saved figures ->", FIG)

import numpy as np
from config import OUT

Z = np.load(OUT / "Z.npy").astype(int)
K = int(Z.max() + 1)

dz = np.abs(np.diff(Z))
wrap_jumps = np.sum(dz > K//2)     # big jumps suggest wrap-around
print("K =", K)
print("max |ΔZ| =", int(dz.max()))
print("count(|ΔZ| > K/2) =", int(wrap_jumps), "out of", len(dz))
print("example big jumps indices:", np.where(dz > K//2)[0][:20])

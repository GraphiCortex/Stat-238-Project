import numpy as np
from config import OUT

Z = np.load(OUT / "Z.npy").astype(int)

# movement = position bin changes
moving = np.zeros_like(Z, dtype=bool)
moving[1:] = (Z[1:] != Z[:-1])

np.save(OUT / "moving_mask.npy", moving)

print("Saved moving_mask.npy")
print("moving fraction:", moving.mean())
print("n moving bins:", moving.sum(), "out of", len(moving))

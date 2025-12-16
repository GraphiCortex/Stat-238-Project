import numpy as np
from config import SESSION_DIR, BASE

res_files = sorted(SESSION_DIR.glob(f"{BASE}.res.*"))
clu_files = sorted(SESSION_DIR.glob(f"{BASE}.clu.*"))

print("Session:", BASE)
print("DATA_ROOT:", SESSION_DIR)
print("Found res files:", len(res_files))
print("Found clu files:", len(clu_files))

def tet_id(path):
    return int(path.name.split(".")[-1])

res_tets = set(map(tet_id, res_files))
clu_tets = set(map(tet_id, clu_files))
both = sorted(res_tets.intersection(clu_tets))

print("Tetrodes with both:", len(both))
print("First 20 tets:", both[:20])

def read_clu_units(clu_path):
    x = np.loadtxt(clu_path, dtype=int)
    nclusters = int(x[0])
    labels = x[1:]
    uniq = set(labels.tolist())
    nunits_all = len(uniq)
    nunits_no0 = nunits_all - (1 if 0 in uniq else 0)
    return nclusters, nunits_all, nunits_no0, len(labels)

print("\nPer-tetrode summary (first 15):")
for t in both[:15]:
    clu_path = SESSION_DIR / f"{BASE}.clu.{t}"
    nclusters, u_all, u_no0, nspk = read_clu_units(clu_path)
    print(f"tet {t:02d}: clu_header={nclusters:3d}, unique={u_all:3d}, unique_no0={u_no0:3d}, spikes={nspk}")

import json
import numpy as np
from pathlib import Path
from config import SESSION_DIR, BASE, OUT, WHL_FS, SPIKE_FS, DELTA, KPOS, MIN_SPIKES

def load_whl_xy(whl_path: Path):
    whl = np.loadtxt(whl_path)
    # hc-3 .whl is typically columns with x,y pairs; sometimes -1 indicates missing
    x = whl[:, 0].copy()
    y = whl[:, 1].copy() if whl.shape[1] > 1 else np.zeros_like(x)
    return x, y

def fill_missing_linear(x):
    x = x.astype(float)
    bad = (x < 0) | ~np.isfinite(x)
    if np.all(bad):
        raise RuntimeError("All WHL positions are missing/invalid.")
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x

def discretize_to_bins(x, K):
    # map x to {0,...,K-1} using global min/max
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        raise RuntimeError("WHL x has zero range.")
    edges = np.linspace(xmin, xmax, K + 1)
    z = np.digitize(x, edges) - 1
    z = np.clip(z, 0, K - 1)
    return z, edges

def read_res_clu_units(tet: int):
    res = np.loadtxt(SESSION_DIR / f"{BASE}.res.{tet}", dtype=np.int64)
    clu = np.loadtxt(SESSION_DIR / f"{BASE}.clu.{tet}", dtype=np.int64)
    labels = clu[1:]  # first entry is header
    if len(labels) != len(res):
        raise RuntimeError(f"Length mismatch tet {tet}: res={len(res)} clu={len(labels)}")
    # exclude cluster 0 (noise)
    keep = labels != 0
    return res[keep], labels[keep]

def bin_spikes_for_unit(res_samples, t0_s, t1_s, delta_s):
    # convert spike samples to seconds
    ts = res_samples / SPIKE_FS
    # restrict to [t0,t1)
    mask = (ts >= t0_s) & (ts < t1_s)
    ts = ts[mask]
    # histogram into delta bins
    nbins = int(np.floor((t1_s - t0_s) / delta_s))
    edges = t0_s + np.arange(nbins + 1) * delta_s
    counts, _ = np.histogram(ts, bins=edges)
    return counts

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- time horizon from WHL (master) ----
    whl_path = SESSION_DIR / f"{BASE}.whl"
    x, y = load_whl_xy(whl_path)
    x = fill_missing_linear(x)

    T_whl = len(x)
    dur_s = T_whl / WHL_FS  # ~187.62s in this session :contentReference[oaicite:3]{index=3}
    t0_s, t1_s = 0.0, dur_s

    # ---- resample WHL x onto DELTA grid ----
    nbins = int(np.floor((t1_s - t0_s) / DELTA))
    t_bins = t0_s + (np.arange(nbins) + 0.5) * DELTA  # bin centers
    whl_t = np.arange(T_whl) / WHL_FS
    x_bin = np.interp(t_bins, whl_t, x)

    Z, x_edges = discretize_to_bins(x_bin, KPOS)

    # ---- build unit list + Y ----
    Y_cols = []
    units = []  # list of dicts: {tet, clu, nspikes_in_session}

    for tet in range(1, 9):  # you have 8 tets
        res_keep, labels_keep = read_res_clu_units(tet)
        for clu_id in sorted(set(labels_keep.tolist())):
            mask = labels_keep == clu_id
            res_unit = res_keep[mask]
            # count spikes inside [0,dur)
            n_in = int(np.sum((res_unit / SPIKE_FS >= t0_s) & (res_unit / SPIKE_FS < t1_s)))
            if n_in < MIN_SPIKES:
                continue
            counts = bin_spikes_for_unit(res_unit, t0_s, t1_s, DELTA)
            # ensure matches nbins
            counts = counts[:nbins]
            Y_cols.append(counts)
            units.append({"tet": tet, "clu": int(clu_id), "spikes_in_epoch": n_in})

    if len(Y_cols) == 0:
        raise RuntimeError("No units passed MIN_SPIKES within the WHL epoch.")

    Y = np.stack(Y_cols, axis=1).astype(np.int32)  # (T, N)
    Z = Z.astype(np.int32)

    # ---- save ----
    np.save(OUT / "Y.npy", Y)
    np.save(OUT / "Z.npy", Z)
    (OUT / "units.json").write_text(json.dumps(units, indent=2), encoding="utf-8")

    meta = {
        "BASE": BASE,
        "t0_s": t0_s,
        "t1_s": t1_s,
        "dur_s": dur_s,
        "DELTA": DELTA,
        "KPOS": KPOS,
        "WHL_FS": WHL_FS,
        "SPIKE_FS": SPIKE_FS,
        "N_units": int(Y.shape[1]),
        "T_bins": int(Y.shape[0]),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:")
    print("  Y:", Y.shape, "->", OUT / "Y.npy")
    print("  Z:", Z.shape, "->", OUT / "Z.npy")
    print("  units:", len(units), "->", OUT / "units.json")
    print("  meta ->", OUT / "meta.json")

if __name__ == "__main__":
    main()

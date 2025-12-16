# pipeline_2/step05e_finalize_replay.py
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import OUT

FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Inputs (produced earlier)
# -----------------------------
RANKED_PATH = OUT / "swr_highres_ranked.json"

# optional pval files (use whichever exists)
PVALS_BEHAVIOR_PRIOR = OUT / "swr_highres_pvals.json"
PVALS_SWR_PRIOR      = OUT / "swr_highres_pvals_swrprior.json"

# -----------------------------
# Outputs
# -----------------------------
MERGED_JSON = OUT / "swr_highres_merged_table.json"
MERGED_CSV  = OUT / "swr_highres_merged_table.csv"

CANDS_JSON  = OUT / "swr_highres_candidates.json"
CANDS_CSV   = OUT / "swr_highres_candidates.csv"

TOP10_TXT   = OUT / "swr_highres_top10.txt"

# -----------------------------
# Candidate rule (edit freely)
# -----------------------------
MIN_SPIKES = 40          # total spikes in SWR
MIN_PATH   = 3           # sum |ΔZ| across decoded path
MAX_P      = 0.20        # shuffle p-value threshold
MAX_Q      = 0.50        # BH-FDR q-value threshold (loose, dataset is tiny)

# -----------------------------
# Helpers
# -----------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values aligned with pvals."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out

def write_csv(rows, path: Path):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)

def _find_first_key(d: dict, keys):
    for k in keys:
        if k in d:
            return k
    return None

def build_pvals_map(p_obj):
    """
    Accepts pval json in several formats and returns {event_idx: pval}.

    Supported:
      1) list[dict] where dict has event id key among:
         event_idx/event/event_id/idx/index/swr_idx/swr
         and p-value key among: pval/p/p_value/pvalue

      2) dict with {"pvals": [...]} -> assumed aligned by event index 0..N-1

      3) dict with {"events": [...]} where each element is dict (like format 1)

      4) dict mapping event_idx (as string/int) -> pval (float)
    """
    idx_keys = ["event_idx", "event", "event_id", "idx", "index", "swr_idx", "swr"]
    p_keys   = ["pval", "p", "p_value", "pvalue"]

    out = {}

    # (4) dict mapping directly: {"0":0.12,"1":0.9,...}
    if isinstance(p_obj, dict):
        # (2)
        if "pvals" in p_obj and isinstance(p_obj["pvals"], list):
            arr = p_obj["pvals"]
            for i, pv in enumerate(arr):
                out[int(i)] = to_float(pv)
            return out

        # (3)
        if "events" in p_obj and isinstance(p_obj["events"], list):
            p_obj = p_obj["events"]  # fall through to list[dict] handling

        else:
            # try direct mapping
            ok = True
            tmp = {}
            for k, v in p_obj.items():
                try:
                    ei = int(k)
                    tmp[ei] = to_float(v)
                except Exception:
                    ok = False
                    break
            if ok and tmp:
                return tmp
            # unknown dict schema
            return {}

    # (1) list[dict]
    if not isinstance(p_obj, list):
        return {}

    for d in p_obj:
        if not isinstance(d, dict):
            continue

        k_evt = _find_first_key(d, idx_keys)
        if k_evt is None:
            continue
        try:
            evt = int(d[k_evt])
        except Exception:
            continue

        k_p = _find_first_key(d, p_keys)
        pv = to_float(d.get(k_p, np.nan)) if k_p is not None else np.nan
        out[evt] = pv

    return out

def main():
    if not RANKED_PATH.exists():
        raise FileNotFoundError(f"Missing ranked file: {RANKED_PATH}")

    ranked = load_json(RANKED_PATH)  # list[dict] (expected)

    # attach p-values if separate file exists (some versions store pval inside ranked already)
    p_source = None
    pvals_map = {}

    if PVALS_SWR_PRIOR.exists():
        p_source = "swr_prior"
        p_obj = load_json(PVALS_SWR_PRIOR)
        pvals_map = build_pvals_map(p_obj)

    elif PVALS_BEHAVIOR_PRIOR.exists():
        p_source = "behavior_prior"
        p_obj = load_json(PVALS_BEHAVIOR_PRIOR)
        pvals_map = build_pvals_map(p_obj)

    # standardize fields + ensure we have pval
    merged = []
    for d in ranked:
        # robust event id extraction
        evt = d.get("event_idx", d.get("event", d.get("idx", d.get("event_id", -1))))
        try:
            evt = int(evt)
        except Exception:
            evt = -1

        out = dict(d)
        out["event_idx"] = evt

        # prefer pval inside ranked, else from pvals file
        if "pval" in out and np.isfinite(to_float(out["pval"], np.nan)):
            out["pval"] = to_float(out["pval"])
        else:
            out["pval"] = to_float(pvals_map.get(evt, np.nan))

        out["p_source"] = p_source if p_source is not None else "embedded"

        # helpful extras
        out["neglog10_p"] = float(-math.log10(max(out["pval"], 1e-300))) if np.isfinite(out["pval"]) else float("nan")
        out["score"] = to_float(out.get("score", np.nan))
        out["path_len"] = int(out.get("path_len", out.get("path", -1)))
        out["net_disp"] = int(out.get("net_disp", out.get("net", -1)))
        out["total_spikes"] = int(out.get("total_spikes", out.get("spikes", -1)))
        out["T_bins"] = int(out.get("T_bins", out.get("T", -1)))

        merged.append(out)

    # compute q-values on finite pvals
    pvals = np.array([m["pval"] for m in merged], dtype=float)
    good = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan)
    if good.any():
        qvals[good] = bh_fdr(pvals[good])
    for m, q in zip(merged, qvals):
        m["qval_bh"] = to_float(q)

    # sort by score desc
    merged = sorted(merged, key=lambda x: to_float(x.get("score", -np.inf)), reverse=True)

    # save merged tables
    MERGED_JSON.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    write_csv(merged, MERGED_CSV)

    # select candidates
    cands = []
    for m in merged:
        if not np.isfinite(m["pval"]):
            continue
        ok = (
            (m["total_spikes"] >= MIN_SPIKES) and
            (m["path_len"] >= MIN_PATH) and
            (m["pval"] <= MAX_P) and
            (np.isfinite(m["qval_bh"]) and m["qval_bh"] <= MAX_Q)
        )
        if ok:
            cands.append(m)

    CANDS_JSON.write_text(json.dumps(cands, indent=2), encoding="utf-8")
    write_csv(cands, CANDS_CSV)

    # top10 text (for quick report writing)
    top_lines = []
    for m in merged[:10]:
        score = m["score"]
        p = m["pval"]
        q = m["qval_bh"]
        top_lines.append(
            f"evt={m['event_idx']:02d} score={score:.3f} "
            f"p={p:.3f} q={q:.3f} "
            f"net={m['net_disp']} path={m['path_len']} spikes={m['total_spikes']} T={m['T_bins']}"
        )
    TOP10_TXT.write_text("\n".join(top_lines) + "\n", encoding="utf-8")

    # -----------------------------
    # Summary plots
    # -----------------------------
    scores = np.array([m["score"] for m in merged], dtype=float)
    pvals  = np.array([m["pval"] for m in merged], dtype=float)
    spikes = np.array([m["total_spikes"] for m in merged], dtype=float)
    pathl  = np.array([m["path_len"] for m in merged], dtype=float)
    netd   = np.array([m["net_disp"] for m in merged], dtype=float)

    # score vs -log10(p)
    plt.figure()
    mask = np.isfinite(scores) & np.isfinite(pvals)
    plt.scatter(scores[mask], -np.log10(np.maximum(pvals[mask], 1e-300)))
    plt.xlabel("replay score")
    plt.ylabel("-log10(p) (shuffle)")
    plt.title("High-res SWR replay: score vs significance")
    plt.tight_layout()
    plt.savefig(FIGDIR / "swr_highres_score_vs_p.png", dpi=200)
    plt.close()

    # p histogram
    plt.figure()
    plt.hist(pvals[np.isfinite(pvals)], bins=10)
    plt.xlabel("p-value (shuffle)")
    plt.ylabel("count")
    plt.title("High-res SWR replay: p-value histogram")
    plt.tight_layout()
    plt.savefig(FIGDIR / "swr_highres_pval_hist.png", dpi=200)
    plt.close()

    # path length histogram
    plt.figure()
    plt.hist(pathl[pathl >= 0], bins=10)
    plt.xlabel("MAP path length (sum |ΔZ|)")
    plt.ylabel("count")
    plt.title("High-res SWR decoded path length")
    plt.tight_layout()
    plt.savefig(FIGDIR / "swr_highres_pathlen_hist.png", dpi=200)
    plt.close()

    # net vs path
    plt.figure()
    plt.scatter(netd[netd >= 0], pathl[pathl >= 0])
    plt.xlabel("net displacement |Z_end - Z_start|")
    plt.ylabel("path length sum|ΔZ|")
    plt.title("Trajectory geometry: net vs path")
    plt.tight_layout()
    plt.savefig(FIGDIR / "swr_highres_net_vs_path.png", dpi=200)
    plt.close()

    # score vs spikes
    plt.figure()
    plt.scatter(spikes[spikes >= 0], scores[spikes >= 0])
    plt.xlabel("total spikes in SWR")
    plt.ylabel("replay score")
    plt.title("Replay score vs spike count")
    plt.tight_layout()
    plt.savefig(FIGDIR / "swr_highres_score_vs_spikes.png", dpi=200)
    plt.close()

    print("Saved merged tables:")
    print(" ", MERGED_JSON)
    print(" ", MERGED_CSV)
    print("Saved candidates:")
    print(" ", CANDS_JSON)
    print(" ", CANDS_CSV)
    print("Saved plots in:", FIGDIR)
    print("Top10:", TOP10_TXT)
    print(f"P-values source: {p_source if p_source else 'embedded/none'}")
    print(f"Loaded pvals_map: {len(pvals_map)} entries")
    print(f"Candidate rule: spikes>={MIN_SPIKES}, path>={MIN_PATH}, p<={MAX_P}, q<={MAX_Q}")

if __name__ == "__main__":
    main()

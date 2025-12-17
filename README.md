# STAT 238 — Decoding hippocampal dynamics & replay (HMM vs HSMM)

**Jad Sawan**  
Course project (STAT 238) — hippocampal place-cell decoding + SWR replay  
Dataset: **CRCNS hc-3**, session **ec013.205**

---

## 1) What this project is about (in one minute)

This project asks two linked questions:

1. **Modeling question (behavior):**  
   Does an **HSMM** (explicit state-duration model) better describe hippocampal position dynamics than a standard **HMM** (geometric dwell time) during movement?

2. **Replay question (SWRs):**  
   During sharp-wave ripples (SWRs), can we decode a latent position sequence (“replay trajectory”), and how does its **timescale** compare to behavior (compression)?

The pipeline:
- learns **place fields** from behavior,
- fits / evaluates **HMM vs HSMM** decoding on movement,
- decodes **SWR content** at high temporal resolution,
- ranks events with simple metrics + a shuffle test (guardrail),
- computes a **hitting-time-based compression** statistic.

---

## 2) Quick navigation for the professor (recommended order)

### Key outputs (figures + JSON summaries)
All outputs are written to:
- `pipeline_2/out/`

Most important artifacts:
- **Behavior decoding comparison (HMM vs HSMM)**  
  CV risk plots, decoded overlays, duration-fit plots
- **SWR decoding examples**  
  posterior heatmaps + MAP paths
- **Compression via hitting times**  
  `pipeline_2/out/mathq2_hitting_times_summary.json`

If you only want headline numbers, open:
- `pipeline_2/out/mathq2_hitting_times_summary.json`

Example (one run):
- Mean pairwise hitting time (behavior) ≈ **1.22 s**
- Mean pairwise hitting time (replay) ≈ **0.59 s**
- Compression ≈ **2.06×**

---

## 3) Repository structure

```
.
├── pipeline_2/
│   ├── step01_*.py
│   ├── step02_*.py
│   ├── step03_*.py
│   ├── step04_*.py
│   ├── step05_*.py
│   ├── step06_hitting_times_compression.py
│   ├── out/
│   │   ├── *.png
│   │   ├── *.json
│   │   ├── *.npz
│   │   └── ...
│   └── config.py
├── data/            # not tracked
├── notebooks/       # optional
└── README.md
```

---

## 4) Pipeline overview (conceptual)

### Behavior (20 ms bins)
- **Input:** position + spikes  
- **Output:** place fields λ(z,n) and decoded sequences Ẑ_t  
- **Models:** HMM vs HSMM

### SWR replay (5 ms bins)
- **Input:** SWR windows + spikes  
- **Output:** posterior over position + MAP path  
- **Metrics:** path length, displacement, shuffle p-values

### Compression analysis
- **Input:** decoded sequences (behavior vs replay)  
- **Output:** hitting-time scale (seconds) + compression factor

---

## 5) Methods snapshot (math-first)

### Observation model (Poisson place-field decoder)

For position bin z and neuron n, the learned place field is λ(z,n).  
Given spike counts y(t,n) in bin width Δt:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%7Bp(Y_t%20%7C%20Z_t=z)%20=%20%5Cprod_%7Bn=1%7D%5EN%20%5Cmathrm%7BPoisson%7D(y_%7Bt,n%7D;%5Clambda_%7Bz,n%7D%5CDelta%20t)%7D" />
</p>

Place fields are estimated from moving behavioral bins using
occupancy-normalized spike counts.

---

### State dynamics prior (movement model)

<p align="center">
<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%7Bp(Z_t%20%7C%20Z_%7Bt-1%7D)=A_%7BZ_%7Bt-1%7D,Z_t%7D%7D" />
</p>

For behavior, A is learned empirically from movement.  
For SWRs, the same prior acts as a *no-teleportation* constraint.

---

### HMM vs HSMM (duration modeling)

- **HMM:** implicit geometric dwell time (memoryless)
- **HSMM:** explicit learned dwell-time distribution (semi-Markov)

---

### Compression via hitting times (step06)

<p align="center">
<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%7Bh_j=0,%5Cquad%20h_i=m_i+%5Csum_kQ_%7Bik%7Dh_k%5C;(i%5Cneq%20j)%7D" />
</p>

Compression is defined as:

<p align="center">
<img src="https://latex.codecogs.com/svg.image?%5Ccolor%7Bwhite%7D%7BC=%5Cfrac%7B%5Ctext%7Bbehavior%20hitting-time%20scale%7D%7D%7B%5Ctext%7Breplay%20hitting-time%20scale%7D%7D%7D" />
</p>

---

## 6) How to run

### Requirements
- Python 3.10+
- numpy, scipy, matplotlib
- pandas, tqdm (optional)

```bash
python -m venv .venv
pip install -r requirements.txt
```

### Run pipeline

```bash
python pipeline_2/step01_*.py
python pipeline_2/step02_*.py
python pipeline_2/step03_*.py
python pipeline_2/step04_*.py
python pipeline_2/step05_*.py
python pipeline_2/step06_hitting_times_compression.py
```

---

## 7) Notes

- Short session → noisy duration estimates
- Compression is a coarse timescale summary
- Not a claim of definitive replay significance

---

## 8) References

- Rabiner (1989) — HMM tutorial
- Yu (2010) — HSMMs
- Zhang et al. (1998) — place-cell decoding
- Buzsáki (2015) — sharp-wave ripples

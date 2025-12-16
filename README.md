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

### A) Final report / slides
- Report (PDF / LaTeX, if included): `./report/`
- Slides (if included): `./slides/`

### B) Key outputs (figures + JSON summaries)
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
- Compression ≈ **2.06×** (similar under HSMM)

---

## 3) Repository structure

```text
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
├── data/            # not tracked; see below
├── notebooks/       # optional
└── README.md
```

---

## 4) Pipeline overview (conceptual)

### Behavior (20 ms bins)
- **Input:** position + spikes  
- **Output:** place fields `λ(z,n)` and decoded sequences `\hat Z_t`  
- **Models:** HMM vs HSMM

### SWR replay (5 ms bins within SWR windows)
- **Input:** SWR windows + spikes  
- **Output:** posterior over position + MAP path per SWR  
- **Metrics:** path length, net displacement, shuffle p-values (guardrail)

### Compression analysis
- **Input:** decoded sequences (behavior vs replay)  
- **Output:** mean all-pairs hitting-time scale (seconds) + compression factor

---

## 5) Methods snapshot (math-first)

### Observation model (Poisson place-field decoder)
For position bin \(z\) and neuron \(n\), the learned place field is \(\lambda_{z,n}\).  
Given spike counts \(y_{t,n}\) in bin width \(\Delta t\):

\[
p(Y_t \mid Z_t=z)
= \prod_{n=1}^N \text{Poisson}\!\left(y_{t,n};\lambda_{z,n}\Delta t\right).
\]

Place fields \(\lambda_{z,n}\) are estimated from **moving behavioral bins** using occupancy-normalized spike counts.

### State dynamics prior (movement model)
A transition prior enforces temporal continuity:

\[
p(Z_t \mid Z_{t-1}) = A_{Z_{t-1},Z_t}.
\]

For behavior, \(A\) is learned empirically from movement.  
For SWRs, the same prior acts as a “no teleportation” constraint while decoding.

### HMM vs HSMM (duration modeling)
- **HMM:** implicit geometric dwell time (memoryless)
- **HSMM:** explicit learned dwell-time distribution (semi-Markov)

### Compression via hitting times (step06)
We compute expected hitting times using an embedded jump chain \(Q\) and mean dwell times \(m_i\). For target state \(j\):

\[
h_j = 0,\quad
h_i = m_i + \sum_k Q_{ik} h_k \;\;\; (i\neq j).
\]

We summarize with the mean over all \(i \neq j\), convert bins → seconds, and define:

\[
C = \frac{\text{behavior hitting-time scale}}{\text{replay hitting-time scale}}.
\]

---

## 6) How to run (minimal reproducibility)

### Requirements
- Python 3.10+ recommended
- `numpy`, `scipy`, `matplotlib`, `pandas` (if used), `tqdm` (optional)

If you use a virtual environment:

```bash
python -m venv .venv
# activate it (Windows: .venv\Scripts\activate, macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

### Data layout (not tracked)
Download hc-3 and place the session files under `data/`. One workable layout:

```text
data/
└── ec013.205/
    ├── ec013.205.whl
    ├── ec013.205.eeg
    ├── ec013.205.res.*
    ├── ec013.205.clu.*
    └── ...
```

### Run the pipeline
Run scripts in order (filenames may include additional descriptors):

```bash
python pipeline_2/step01_*.py
python pipeline_2/step02_*.py
python pipeline_2/step03_*.py
python pipeline_2/step04_*.py
python pipeline_2/step05_*.py
python pipeline_2/step06_hitting_times_compression.py
```

Outputs will appear in `pipeline_2/out/`.

---

## 7) Where the key code is

If you want the “core math” parts:

### Poisson likelihood + decoding
- Look in the **behavior decoding** and **SWR decoding** steps (step0X scripts)
- Key objects: place fields `lambda[z,n]`, posterior over `Z_t`, MAP / decoded paths

### HSMM vs HMM logic
- The HSMM portions (explicit durations) live in the behavior modeling step(s)
- Compare:
  - decoding risk metrics (CV)
  - duration distribution fit (KL)

### Compression analysis (hitting times)
- `pipeline_2/step06_hitting_times_compression.py`
  - builds segments / dwell statistics
  - estimates embedded chain `Q`
  - solves linear systems for hitting times
  - reports compression `C`

---

## 8) Notes / limitations (honest, short)
- This run uses a relatively short clip (~188 s), so **duration estimates and replay significance are noisy**.
- Some SWR events look trajectory-like, but shuffle p-values are often \(\gtrsim 0.1\) in this session (suggestive, not definitive).
- The hitting-time compression result is meant as a **coarse timescale summary**, not a claim of strong replay significance.

---

## 9) References (minimal)
- Rabiner (1989) — HMM tutorial  
- Yu (2010) — HSMMs  
- Buzsáki (2015) — sharp-wave ripples  
- Zhang et al. (1998) — place-cell decoding framework

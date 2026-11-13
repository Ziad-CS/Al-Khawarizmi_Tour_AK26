# Al-Khawarizmi_Tour_AK26
# HaT @ AKT26 · Team Transformers

**Air-quality data science + a Julia HPC speed run, with every claim backed by a number.**

Our work for the Al-Khawarizmi Tour 2026 online hackathon (High Performance Computing and AI, Oct 3 – Nov 7, 2026).
Pre-selected: top 41 of 236 teams. Not selected for the final 20 on Demo Day.

> **TL;DR** A week of careful data cleaning improved our PM2.5 model by only **0.77%** (test RMSE 37.03 → 36.75).
> We kept the result, because the number is the point.

---

## What's inside

| Notebook | What it does |
|---|---|
| `L1_Air_Quality_Data_Preprocessing_Notebook_Transformers` | Data-quality audit, cleaning, imputation study, outliers, leak-free pipeline, naive-vs-cleaned impact study |
| `L2_Air_Quality_AI_Methods_Notebook_Transformers` | Clustering of daily pollution profiles (K-means, hierarchical, DBSCAN) |
| `L2_Cholesky_Performance_Julia_Notebook` | Making a Cholesky solver fast in Julia: threads, tiling, precision, LAPACK, roofline |

---

## 1 · Air quality (12 Beijing stations)

**Data:** hourly multi-site air-quality data from 12 monitoring stations.
Training years: 1 Mar 2013 – 29 Feb 2016, delivered as a deliberately messy merged export.
Test year: 1 Mar 2016 – 28 Feb 2017, untouched until the final comparison.

**Goal:** a "virtual PM2.5 sensor" that predicts PM2.5 from the other pollutants and weather.
PM10 is excluded on purpose: it comes from the same instrument and goes down at the same time.

### Key decisions

- **Error or real episode?** An outlier is not automatically an error.
  A sensor frozen at 354 µg/m³ for 67 hours (while neighbours kept moving) is an error.
  A spike across all 12 stations on Spring Festival night is real.
  We set **1,104** isolated spikes to NaN and kept **7,691** real episodes.
- **Imputation depends on gap length.**

  | Gap | Best imputer | RMSE (PM2.5) |
  |---|---|---|
  | 1 hour | Linear time interpolation | 13.95 |
  | 24 hours | KNN across stations | 20.17 (interpolation: 53.96) |
  | > 24 hours | Left as NaN (a fill would be invented data) | – |

- **Leak-free evaluation.** Time-ordered cross-validation (`TimeSeriesSplit`), with every fitted step inside the pipeline.
- **A textbook fix that hurt.** `log1p` reduced PM2.5 skewness from 1.96 to -0.33,
  but made time-series CV worse (RMSE 43.3 → 47.3), so we dropped it.

### Final result (same model, unseen test year)

Model: `HistGradientBoostingRegressor`, 300 iterations.

| | RMSE | MAE | R² |
|---|---|---|---|
| Naive (merged export as-is) | 37.033 | 22.211 | 0.798 |
| Cleaned | **36.748** | **22.127** | **0.801** |
| Change | -0.77% | -0.38% | +0.003 |

Cleaning helped at 8 of 12 stations and slightly hurt at 4 (Changping, Dongsi, Gucheng, Shunyi).
Re-running may shift the last digits.

### Clustering pollution days

K-means on daily PM2.5 profiles. k=2 had a slightly higher silhouette score,
but we chose **k=3** because the split was clearer and closer to reality.

| Regime | Share of station-days | Typical day |
|---|---|---|
| R0 · Clean | 32% | Low all day |
| R1 · Moderate | 19% | High at night, falls through the morning |
| R2 · Heavily polluted | 49% | High all day, slower wind (1.44 m/s vs about 2 m/s) |

- Weekday vs weekend effect: Cramér's V = 0.055 (negligible).
- DBSCAN flagged 44 anomalous days; the most frequent were 2016-02-08 and 2015-11-30.
- Hierarchical clustering of the 12 stations by daily profile splits them into two groups.

---

## 2 · Julia Cholesky speed run

Goal: make a Cholesky solver fast without breaking its accuracy (validated with the normwise backward error).

**Hardware:** Intel Xeon @ 2.20 GHz, 2 logical cores, 2 Julia threads, OpenBLAS.
Numbers below are for this machine and will differ on yours.

| Version | Time (N=1024) | Speed-up vs naive |
|---|---|---|
| Naive nested loops (baseline) | 0.942 s | 1x |
| Left-looking, column-oriented | 0.077 s | 12.1x |
| Best of our own versions (optimized tiled) | 0.070 s | 13.4x |

**What didn't work:**
- Threaded version (one barrier per column): **0.49x** with 2 threads, slower than a single thread.
- Float32 with our loops: only **1.01x** faster (8.7 digits lost).
  Mixed-precision iterative refinement recovered Float64 accuracy in 4 steps (backward error 7.2e-16).

**Fastest implementation per size (Float64):**

| N | Fastest | Time | GFlop/s |
|---|---|---|---|
| 256 | tiled + BLAS | 0.0003 s | 17.5 |
| 512 | LAPACK | 0.0020 s | 22.0 |
| 1024 | LAPACK | 0.0156 s | 23.0 |
| 2048 | LAPACK | 0.1092 s | 26.2 |
| 4096 | LAPACK | 0.8171 s | 28.0 |

**Roofline:** measured peak 33.3 GFlop/s, memory bandwidth 12.6 GB/s, ridge point 2.64 flop/byte.
LAPACK reaches 62% of the roof; tiled + BLAS reaches 84%.

**Recommendation for N ≈ 4000:** deploy LAPACK (`potrf`) in Float64.
It is one library call, fully accurate, and far inside an hourly deadline.

---

## Lessons

1. **Effort is not evidence.** A week of cleaning moved the score by 0.77%. Only the held-out test year could tell us that.
2. **Outliers need context.** The same spike is an error at one station and a real event at twelve.
3. **Benchmark before you believe.** Our threads were slower than one thread; loop order alone gave 12x.
4. **New topics sound harder than they are.** The only real barrier is starting.

---

## How to run

**Python** (Air-quality notebooks): Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `joblib`.

```bash
pip install pandas numpy scikit-learn scipy matplotlib joblib jupyter
jupyter notebook
```

**Julia** (Cholesky notebook): Julia 1.9+ with the IJulia kernel.
Start Julia with several threads (`julia -t auto`) to reproduce the threading experiments.

Run each notebook top to bottom (Kernel → Restart & Run All).

---

## Team

**Transformers** · Abdelrahman Zayan, Ibrahim Elhofy, Huda Ashraf, Ziad Saleh

Thank you to **Al-Khawarizmi Tour** for organizing such a challenging, well-built hackathon.

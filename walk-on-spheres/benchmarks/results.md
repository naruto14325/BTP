# Benchmarks

Machine: arm, 10 threads. Times are wall-clock seconds.

## 1. Speed: baseline (NumPy, brute force) vs fast (Numba + BVH, multithreaded)

| domain | segments | baseline walks/s | fast walks/s | speedup |
|---|---|---|---|---|
| square | 200 | 13,409 | 3,523,761 | **263x** |
| square | 800 | 3,334 | 3,054,520 | **916x** |
| Koch snowflake L4 | 768 | 2,811 | 1,981,226 | **705x** |
| Koch snowflake L5 | 3072 | 798 | 1,073,684 | **1,346x** |

## 2. Poisson source term: uniform sampling (baseline) vs importance sampling

Per-walk standard deviation of the estimate (lower is better); Δu = -4 on the unit disk.

| point | uniform σ | importance σ | variance reduction |
|---|---|---|---|
| (0.3, 0.1) | 0.803 | 0.572 | **2.0x** |
| (0.6, -0.2) | 0.731 | 0.562 | **1.7x** |
| (-0.2, 0.8) | 0.599 | 0.516 | **1.3x** |
| (0.0, -0.9) | 0.443 | 0.399 | **1.2x** |

## 3. Gradient estimation: control variate

Variance of ∇u at (0.5, 0.5) with 64 walks, repeated 400 times. u = Re(z³) + c; adding c does not change ∇u.

| offset c | no control variate | with control variate | reduction |
|---|---|---|---|
| 0 | 0.106 | 0.092 | **1.2x** |
| 1 | 0.223 | 0.092 | **2.4x** |
| 10 | 23.052 | 0.092 | **251.2x** |

## 4. Adaptive sampling (Koch snowflake L5, 2000 random points)

Target: standard error < 0.02 at every point.

| strategy | total walks | time |
|---|---|---|
| same count everywhere (1,474 per point) | 2,948,000 | 1.7 s (est.) |
| adaptive (64 minimum, batches of 32) | 1,689,504 | 1.0 s |

**1.7x fewer walks**; max stderr achieved 0.0200.

## 5. Silhouette queries: normal-cone pruning (Walk on Stars)

Two insulated circles in a box, 400 points x 512 walks. Results are bit-identical.

| segments per circle | without cones | with cones | speedup |
|---|---|---|---|
| 256 | 0.32 s | 0.32 s | **1.0x** |
| 4096 | 1.79 s | 0.70 s | **2.6x** |
| 16384 | 5.59 s | 0.92 s | **6.1x** |

## 6. Localized sources: several source samples per walk step

Heat demo scene (two narrow Gaussian heaters, mixed boundaries), 5 points x 8192 walks. Efficiency = 1 / (variance x time), relative to K = 1.

| samples per step K | time | mean per-walk σ | relative efficiency |
|---|---|---|---|
| 1 | 0.13 s | 2.24 | **1.00x** |
| 2 | 0.21 s | 1.72 | **1.11x** |
| 4 | 0.28 s | 1.44 | **1.18x** |
| 8 | 0.36 s | 1.21 | **1.29x** |
| 16 | 0.55 s | 1.12 | **0.98x** |

## 7. Boundary value caching vs direct walks (full image)

Potential flow past an insulated cylinder (analytic solution known), evaluated on a 200 x 200 grid. Errors are against the exact u and ∇u.

| method | walks | time | u RMS error | ∇u median error | ∇u 95th pct error |
|---|---|---|---|---|---|
| direct, 256 walks/point | 9,734,144 | 15.1 s | 0.0638 | 0.2506 | 1.181 |
| direct, 1024 walks/point | 38,936,576 | 62.6 s | 0.0322 | 0.1260 | 0.611 |
| **cache**, M=2000, 512 walks/cache pt (93% cached) | 1,752,064 | 3.4 s (2.3 build) | 0.0231 | 0.0285 | 0.666 |
| **cache**, M=4000, 1024 walks/cache pt (95% cached) | 4,596,736 | 10.2 s (9.2 build) | 0.0175 | 0.0170 | 0.292 |

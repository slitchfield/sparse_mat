# DC power flow on every MATPOWER case, 2026-10-31

The Phase 1.5 baseline: `powerflow`'s DC power flow on all 78 cases in MATPOWER 8.1, before
Phase 2's fixed-pattern matrices and Phase 3's ordered LU. The survey's output, with more
columns than the table below, is in [`dc-power-flow-2026-10-31.csv`](dc-power-flow-2026-10-31.csv).

## Setup

| | |
|---|---|
| Code | `powerflow` at `0e31586`, plus the survey tools added with this record, which don't change the library |
| Machine | Intel Core i7-14700K (8 performance and 12 efficiency cores, 28 threads), 15 GiB of memory visible to WSL2 |
| System | Ubuntu 24.04.3 under WSL2 (Linux 6.6.87.2), Rust 1.97.1, release profile |
| Cases | MATPOWER 8.1's 78 cases, written as literals by `scripts/normalize-cases.sh` |
| Reference | MATPOWER's `rundcpf` under Octave 8.4.0, run locally by `scripts/generate-reference.sh`; its results aren't committed |
| Run | `scripts/benchmark-dc.sh` with its defaults, from 09:17 to 09:35 EDT: each case in its own process, limited to 1 hour and 10 GiB of address space |

Each step is timed by its median over repeated runs, for up to 1 s or 101 runs, so the larger
cases ran once, without warm-up. Expect a few percent of noise: case2383wp's solve took
334–360 ms over six runs of the survey tool, and 317–330 ms in two runs of the criterion
benchmark, which warms up first. Peak memory is the whole process's, including a second
factorization after the timed runs that counts the factor's entries; a run without that step
peaked up to 50 MiB lower (case13659pegase: 384 MiB).

## Results

- **All 78 cases solve, and agree with MATPOWER.** Relative to the largest value of each
  quantity in its case, no difference exceeds 1.1e-10 (case_SyntheticUSA's angles). The
  largest absolute differences are 7.6e-8° (case13659pegase) and 1.7e-6 MW (case_SyntheticUSA,
  whose flows reach 16,148 MW). Every case of up to 300 buses is within the 1e-9 that the
  committed reference tests allow.
- **59 cases solve by Cholesky and 19 by LU.** All 19 have in-service branches with a negative
  series susceptance, which can make B′ indefinite; case145 has 24 and stays positive definite.
- **The solve dominates.** Parsing and building the model take at most 168 ms and 34 ms
  (case_SyntheticUSA). The solve takes up to 89 s (case13659pegase), and 522 s over all 78
  cases.
- **Run time follows fill-in, not size.** Without a fill-reducing ordering, a factor holds up to
  109 times the entries it would without fill-in (case6468rte). How the buses are numbered
  decides it: case_ACTIVSg10k's LU factor fills 17× and takes 1.2 s, while case9241pegase, of
  similar size, fills 99× and takes 74 s.
- **Memory stays modest.** The largest peak is 928 MiB (case_SyntheticUSA).

## For Phase 3

- **Order before factoring.** The PEGASE and RTE cases above 6,000 buses fill 70–109×, and even
  the large synthetic cases (case_ACTIVSg25k, case_ACTIVSg70k, case_SyntheticUSA) fill 34–41×.
  A fill-reducing ordering attacks the main cost.
- **Make the LU fallback cheaper.** At similar size and fill, LU costs about four times as much
  as Cholesky: case2848rte (LU, 42×) solves in 2.1 s and case2869pegase (Cholesky, 35×) in
  0.57 s. LU's pivot search also visits every remaining row at every step, however sparse the
  matrix: 3.4 billion row lookups for case_SyntheticUSA. Phase 3's Gilbert–Peierls LU looks only
  at a column's nonzeros.
- **The failed Cholesky attempt costs nothing.** In every LU case it stops at an early pivot
  within 13 ms (measured once, with a scratch program), so an LU case's time is the LU's.

## Columns

- **Solve** is `DcModel::solve`: building B′, factoring it, the triangular solves, the flows and
  the slack. For an LU case it includes the failed Cholesky attempt.
- **Factor entries** are those of L for Cholesky, or of L and U, with their diagonal counted
  once, for LU. **Fill** divides them by the entries the factor would hold without fill-in: B′'s
  lower triangle for Cholesky, or all of B′ for LU.
- **Angle**, **Flow** and **Output** are the largest absolute differences from MATPOWER's bus
  angles (degrees), branch flows (MW) and generator outputs (MW).

## All cases

Sorted by bus count.

| Case | Buses | Branches | Solver | Parse (ms) | Model (ms) | Solve (ms) | Peak (MiB) | Factor entries | Fill | Angle (°) | Flow (MW) | Output (MW) |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| case4_dist | 4 | 3 | Cholesky | 0.006 | 0.001 | 0.002 | 3 | 4 | 1.0 | 5.6e-17 | 1.1e-16 | 0 |
| case4gs | 4 | 4 | Cholesky | 0.007 | 0.001 | 0.002 | 3 | 6 | 1.2 | 4.4e-16 | 2.8e-14 | 2.8e-14 |
| case5 | 5 | 6 | Cholesky | 0.010 | 0.001 | 0.003 | 3 | 8 | 1.1 | 8.9e-16 | 2.8e-14 | 6.7e-14 |
| case6ww | 6 | 11 | Cholesky | 0.011 | 0.002 | 0.004 | 3 | 15 | 1.2 | 4.4e-16 | 7.1e-15 | 0 |
| case9 | 9 | 9 | Cholesky | 0.012 | 0.002 | 0.005 | 3 | 19 | 1.2 | 7.1e-15 | 2.8e-14 | 2.8e-14 |
| case9Q | 9 | 9 | Cholesky | 0.012 | 0.002 | 0.004 | 3 | 19 | 1.2 | 7.1e-15 | 2.8e-14 | 2.8e-14 |
| case9target | 9 | 9 | Cholesky | 0.012 | 0.002 | 0.005 | 3 | 19 | 1.2 | 1.1e-14 | 9.2e-14 | 1.1e-13 |
| case10ba | 10 | 9 | Cholesky | 0.011 | 0.002 | 0.005 | 3 | 17 | 1.0 | 7.1e-15 | 1.6e-14 | 1.2e-14 |
| case12da | 12 | 11 | Cholesky | 0.013 | 0.002 | 0.006 | 3 | 21 | 1.0 | 3.9e-15 | 1.3e-15 | 1.1e-15 |
| case14 | 14 | 20 | Cholesky | 0.022 | 0.003 | 0.010 | 3 | 50 | 1.6 | 1.4e-14 | 1.7e-13 | 2.8e-13 |
| case15da | 15 | 14 | Cholesky | 0.015 | 0.003 | 0.011 | 3 | 63 | 2.3 | 2.2e-15 | 1.8e-15 | 1.8e-15 |
| case15nbr | 15 | 14 | Cholesky | 0.014 | 0.002 | 0.011 | 3 | 63 | 2.3 | 3.6e-15 | 3.6e-15 | 3.6e-15 |
| case16am | 15 | 14 | Cholesky | 0.014 | 0.002 | 0.009 | 3 | 46 | 1.7 | 8.9e-16 | 1.3e-14 | 3.6e-15 |
| case16ci | 16 | 16 | Cholesky | 0.016 | 0.003 | 0.008 | 3 | 29 | 1.3 | 1.6e-15 | 2.1e-14 | 2.1e-14 |
| case17me | 17 | 16 | Cholesky | 0.016 | 0.003 | 0.009 | 3 | 42 | 1.4 | 1.3e-14 | 5.3e-14 | 5.3e-15 |
| case18 | 18 | 17 | Cholesky | 0.017 | 0.003 | 0.012 | 3 | 64 | 1.9 | 7.3e-14 | 1.2e-13 | 1.2e-13 |
| case18nbr | 18 | 17 | Cholesky | 0.017 | 0.003 | 0.013 | 3 | 79 | 2.4 | 1.8e-15 | 2.8e-15 | 4.4e-16 |
| case22 | 22 | 21 | Cholesky | 0.021 | 0.004 | 0.012 | 3 | 52 | 1.3 | 1.8e-14 | 2.0e-14 | 1.8e-14 |
| case24_ieee_rts | 24 | 38 | Cholesky | 0.050 | 0.006 | 0.017 | 3 | 93 | 1.7 | 2.8e-14 | 7.4e-13 | 7.0e-13 |
| case28da | 28 | 27 | Cholesky | 0.023 | 0.004 | 0.019 | 3 | 111 | 2.1 | 9.1e-14 | 2.9e-14 | 2.8e-14 |
| case30 | 30 | 41 | Cholesky | 0.033 | 0.006 | 0.024 | 3 | 140 | 2.1 | 1.2e-14 | 5.5e-14 | 5.0e-14 |
| case30Q | 30 | 41 | Cholesky | 0.033 | 0.006 | 0.024 | 3 | 140 | 2.1 | 1.2e-14 | 5.5e-14 | 5.0e-14 |
| case30pwl | 30 | 41 | Cholesky | 0.033 | 0.006 | 0.024 | 3 | 140 | 2.1 | 1.2e-14 | 5.5e-14 | 5.0e-14 |
| case_ieee30 | 30 | 41 | Cholesky | 0.040 | 0.006 | 0.023 | 3 | 133 | 2.0 | 3.2e-14 | 1.8e-13 | 2.3e-13 |
| case33bw | 33 | 37 | Cholesky | 0.032 | 0.005 | 0.021 | 3 | 117 | 1.9 | 1.3e-14 | 1.4e-14 | 7.5e-15 |
| case33mg | 33 | 37 | Cholesky | 0.032 | 0.005 | 0.021 | 3 | 117 | 1.9 | 1.9e-14 | 2.8e-14 | 1.2e-14 |
| case34sa | 34 | 33 | Cholesky | 0.031 | 0.006 | 0.022 | 3 | 124 | 1.9 | 1.1e-14 | 7.3e-14 | 7.3e-14 |
| case38si | 38 | 37 | Cholesky | 0.033 | 0.006 | 0.033 | 3 | 230 | 3.2 | 6.3e-14 | 5.7e-14 | 5.6e-14 |
| case39 | 39 | 46 | Cholesky | 0.051 | 0.007 | 0.028 | 3 | 157 | 1.9 | 8.3e-14 | 3.3e-12 | 3.3e-12 |
| case51ga | 51 | 50 | Cholesky | 0.045 | 0.008 | 0.034 | 3 | 221 | 2.2 | 2.4e-14 | 1.3e-14 | 1.2e-14 |
| case51he | 51 | 50 | Cholesky | 0.043 | 0.008 | 0.063 | 3 | 449 | 4.5 | 8.9e-15 | 2.5e-14 | 3.1e-15 |
| case57 | 57 | 80 | Cholesky | 0.071 | 0.012 | 0.077 | 3 | 522 | 4.0 | 1.2e-13 | 4.9e-13 | 7.4e-13 |
| case59 | 59 | 138 | Cholesky | 0.090 | 0.020 | 0.040 | 3 | 207 | 1.6 | 5.2e-13 | 1.1e-11 | 5.7e-13 |
| case60nordic | 60 | 88 | LU | 0.083 | 0.013 | 0.058 | 3 | 327 | 1.6 | 4.2e-13 | 9.3e-12 | 9.3e-12 |
| case69 | 69 | 68 | Cholesky | 0.058 | 0.011 | 0.057 | 3 | 427 | 3.2 | 2.1e-13 | 4.9e-13 | 4.9e-13 |
| case70da | 70 | 76 | Cholesky | 0.063 | 0.011 | 0.040 | 3 | 184 | 1.4 | 2.0e-14 | 8.2e-15 | 1.1e-14 |
| case_RTS_GMLC | 73 | 120 | Cholesky | 0.256 | 0.020 | 0.064 | 3 | 375 | 2.1 | 1.4e-13 | 1.1e-12 | 3.3e-12 |
| case74ds | 74 | 73 | Cholesky | 0.065 | 0.011 | 0.035 | 3 | 143 | 1.0 | 1.4e-14 | 8.5e-14 | 5.3e-15 |
| case85 | 85 | 84 | Cholesky | 0.071 | 0.013 | 0.083 | 3 | 603 | 3.6 | 3.0e-13 | 2.8e-13 | 2.8e-13 |
| case89pegase | 89 | 210 | Cholesky | 0.146 | 0.028 | 0.246 | 3 | 1,162 | 4.0 | 7.1e-13 | 4.9e-11 | 9.5e-11 |
| case94pi | 94 | 93 | Cholesky | 0.079 | 0.014 | 0.204 | 3 | 1,347 | 7.3 | 7.9e-13 | 4.5e-13 | 3.9e-13 |
| case118 | 118 | 186 | Cholesky | 0.182 | 0.026 | 0.216 | 3 | 1,276 | 4.4 | 2.3e-13 | 3.6e-12 | 2.4e-12 |
| case118zh | 118 | 132 | Cholesky | 0.106 | 0.018 | 0.067 | 3 | 377 | 1.6 | 9.3e-14 | 8.9e-14 | 1.7e-13 |
| case136ma | 136 | 156 | Cholesky | 0.126 | 0.021 | 0.070 | 3 | 336 | 1.3 | 6.0e-14 | 1.6e-13 | 2.2e-13 |
| case141 | 141 | 140 | Cholesky | 0.120 | 0.021 | 0.381 | 3 | 2,248 | 8.1 | 7.9e-11 | 2.9e-10 | 2.2e-10 |
| case145 | 145 | 453 | Cholesky | 0.313 | 0.059 | 0.500 | 3 | 2,166 | 3.9 | 1.8e-12 | 1.4e-10 | 1.8e-11 |
| case_ACTIVSg200 | 200 | 245 | Cholesky | 0.386 | 0.037 | 0.260 | 3 | 1,374 | 3.1 | 3.9e-13 | 4.0e-12 | 4.0e-12 |
| case300 | 300 | 411 | LU | 0.383 | 0.057 | 2.904 | 3 | 7,887 | 7.1 | 3.0e-13 | 7.1e-12 | 4.3e-12 |
| case_ACTIVSg500 | 500 | 597 | Cholesky | 0.916 | 0.087 | 1.250 | 3 | 5,805 | 5.4 | 3.3e-13 | 7.5e-12 | 1.1e-12 |
| case533mt_hi | 533 | 577 | Cholesky | 0.521 | 0.079 | 1.090 | 3 | 6,765 | 6.4 | 8.7e-14 | 6.4e-14 | 1.6e-14 |
| case533mt_lo | 533 | 577 | Cholesky | 0.533 | 0.079 | 1.091 | 3 | 6,765 | 6.4 | 4.2e-14 | 1.2e-13 | 1.1e-13 |
| case1197 | 1,197 | 1,196 | Cholesky | 0.973 | 0.166 | 3.091 | 5 | 16,876 | 7.1 | 2.0e-12 | 5.8e-13 | 5.8e-13 |
| case1354pegase | 1,354 | 1,991 | Cholesky | 1.816 | 0.315 | 94.3 | 13 | 74,834 | 24.5 | 7.8e-12 | 1.5e-10 | 6.8e-10 |
| case1888rte | 1,888 | 2,531 | LU | 2.449 | 0.399 | 323.0 | 17 | 149,940 | 23.1 | 3.7e-11 | 1.6e-09 | 3.0e-09 |
| case1951rte | 1,951 | 2,596 | LU | 2.469 | 0.411 | 414.7 | 18 | 170,090 | 25.5 | 5.1e-12 | 1.4e-10 | 6.6e-10 |
| case_ACTIVSg2000 | 2,000 | 3,206 | Cholesky | 4.523 | 0.522 | 33.8 | 15 | 59,868 | 12.8 | 1.7e-12 | 5.4e-11 | 5.4e-11 |
| case2383wp | 2,383 | 2,896 | Cholesky | 2.879 | 0.480 | 333.7 | 27 | 209,361 | 39.8 | 7.0e-10 | 9.4e-09 | 3.1e-08 |
| case2736sp | 2,736 | 3,504 | Cholesky | 5.381 | 0.549 | 290.3 | 26 | 211,223 | 35.2 | 3.6e-11 | 1.8e-09 | 2.8e-09 |
| case2737sop | 2,737 | 3,506 | Cholesky | 5.222 | 0.535 | 238.4 | 27 | 194,979 | 32.5 | 1.3e-11 | 4.6e-10 | 9.1e-10 |
| case2746wop | 2,746 | 3,514 | Cholesky | 4.995 | 0.535 | 371.9 | 43 | 237,376 | 39.3 | 5.7e-11 | 2.3e-09 | 4.5e-09 |
| case2746wp | 2,746 | 3,514 | Cholesky | 5.071 | 0.536 | 329.6 | 26 | 226,454 | 37.7 | 5.5e-11 | 1.0e-09 | 1.9e-09 |
| case2848rte | 2,848 | 3,776 | LU | 3.619 | 0.630 | 2,149 | 35 | 409,914 | 42.2 | 3.2e-12 | 1.1e-10 | 4.8e-11 |
| case2868rte | 2,868 | 3,808 | LU | 3.675 | 0.642 | 2,136 | 36 | 409,669 | 41.9 | 1.6e-11 | 1.4e-10 | 7.3e-10 |
| case2869pegase | 2,869 | 4,582 | Cholesky | 4.041 | 0.750 | 574.1 | 43 | 240,089 | 35.1 | 4.1e-11 | 6.4e-10 | 9.1e-10 |
| case3012wp | 3,012 | 3,572 | LU | 3.477 | 0.598 | 2,136 | 64 | 539,432 | 53.2 | 8.5e-12 | 4.5e-10 | 7.3e-11 |
| case3120sp | 3,120 | 3,693 | LU | 3.572 | 0.625 | 2,069 | 65 | 546,023 | 52.1 | 4.0e-11 | 1.3e-09 | 2.6e-09 |
| case3375wp | 3,374 | 4,161 | LU | 4.239 | 0.710 | 2,188 | 54 | 568,958 | 49.5 | 1.9e-11 | 6.4e-10 | 5.6e-10 |
| case6468rte | 6,468 | 9,000 | LU | 9.019 | 1.585 | 43,922 | 230 | 2,458,503 | 108.9 | 1.2e-11 | 1.8e-10 | 7.8e-10 |
| case6470rte | 6,470 | 9,005 | LU | 8.807 | 1.604 | 41,692 | 227 | 2,350,095 | 104.1 | 3.8e-11 | 3.6e-10 | 2.0e-09 |
| case6495rte | 6,495 | 9,019 | LU | 8.283 | 1.587 | 43,492 | 227 | 2,380,132 | 105.1 | 8.1e-11 | 7.8e-10 | 4.0e-09 |
| case6515rte | 6,515 | 9,037 | LU | 8.512 | 1.610 | 44,097 | 228 | 2,413,707 | 106.3 | 1.5e-10 | 1.5e-09 | 5.9e-09 |
| case8387pegase | 8,387 | 14,561 | Cholesky | 14.7 | 2.675 | 16,495 | 162 | 1,497,459 | 70.1 | 4.6e-10 | 8.7e-09 | 3.7e-08 |
| case9241pegase | 9,241 | 16,049 | LU | 14.8 | 2.913 | 73,532 | 421 | 3,710,049 | 98.6 | 5.2e-10 | 7.3e-09 | 3.1e-08 |
| case_ACTIVSg10k | 10,000 | 12,706 | LU | 20.3 | 2.306 | 1,191 | 73 | 600,482 | 17.4 | 1.7e-11 | 1.8e-09 | 1.8e-09 |
| case13659pegase | 13,659 | 20,467 | LU | 22.3 | 4.279 | 89,034 | 434 | 4,069,190 | 79.9 | 7.6e-08 | 8.5e-07 | 8.5e-07 |
| case_ACTIVSg25k | 25,000 | 32,230 | LU | 52.1 | 8.412 | 24,825 | 289 | 3,509,551 | 41.2 | 6.0e-10 | 2.2e-07 | 2.4e-07 |
| case_ACTIVSg70k | 70,000 | 88,207 | LU | 156.6 | 28.1 | 61,148 | 883 | 8,826,607 | 37.3 | 3.9e-08 | 1.5e-06 | 1.5e-06 |
| case_SyntheticUSA | 82,000 | 104,121 | LU | 167.9 | 34.1 | 69,268 | 928 | 9,548,931 | 34.3 | 4.4e-08 | 1.7e-06 | 1.7e-06 |

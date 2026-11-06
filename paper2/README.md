# The *t*-spacing test for a spiked tensor: code for every number

This folder recomputes every number of the paper *The t-spacing test for a spiked tensor* (Azaïs, Dalmao, and
De Castro, 2026) that comes from a computation: tables, simulated powers and deficiencies, constants printed
with digits. Statement numbers and labels refer to the version of 2026-10-26 (30 pages).

## Layout

| Folder | Content | Time |
|---|---|---|
| `numerics/` | Closed forms, quadratures, exact laws and small Monte Carlo runs. `../run_all.sh` runs them all and saves each printed output as `<script>.log`. | about 11 minutes on one core |
| `sim/` | Monte Carlo of the *t*-spacing test from the Karhunen–Loève expansion of the field: two maxima, Hessian and noise estimate computed on each realised path. The JSON lines of the runs used in the paper are kept; the raw per-replicate files (about 370 MB) are not, and the scripts regenerate them. | hours (see below) |

## Where each number comes from

| Paper | Numbers | Script (output) |
|---|---|---|
| Section 1, *Why the t-test* | the level map moves by 12 to 17% at (3,5), 3% at (3,8), about 1% at (5,6) | `numerics/level_map_draw.py` |
| Proposition 2 (`prop:closed`) | closed forms of G_r and H_r | `numerics/closed_form_check.py` |
| Remark 9 (`rem:tuple`), Section 7 *The power bound* | Assumption 3 at the tuple, u† = 60.97, u₊ = 109.7, p₋ = 75, P̄ = 3.5e-8, q_η = 5.2e-9, q_τ = 1.3e-5, ℰ = 6.85e-3, 99.4%, δ = 0.99876, power bound 0.991, ρu₊ = 95.0 | `numerics/remark9_power_bound.py`, `numerics/kacrice_mass_and_sandwich.py` |
| Remark 10 (`rem:Shigh`) | q_η against P(χ²_κ > 4κ) | `numerics/remark9_power_bound.py` |
| Remark after Corollary 7 | 0.0472, 0.01290, 0.003305, 0.000831; 88.54% to 99.79% | `numerics/remark9_power_bound.py` |
| Section 4 (`sec:high`) | balance exponents 0.032 and 0.086, 6.4% and 17.3% of 1/2 | `numerics/exponent_A_Estar.py`, `numerics/numbers_sections_3_to_6.py` |
| Theorem 16 (`thm:upper`), Lemma 18 (`lem:T2`) | α^{-1/κ}, √κ/ρ | `numerics/exponent_A_Estar.py` |
| Remark 21 (`rem:Estar`), Lemma 22 (`lem:Phibounds`) | I(1), D_A, c_far, c_near, the values at c_far and c_near (smallest margin 0.176 against 0.122, at (7,5)), w₁ | `numerics/numbers_sections_3_to_6.py` |
| Theorem 23 (`thm:lower`), Remark 24 (`rem:bracket`) | the table of A, E*, A_low, E_low, balance and S at the six pairs; 0.121; 10 to 41% | `numerics/exponent_constants.py`, `numerics/exponent_A_Estar.py` |
| Section 6 (`sec:low`), quadratic test | μ_j, n_j, t_α = 2.022 and 2.732, the row of the quadratic test, the crossings | `numerics/quadratic_test.py` |
| Section 6, maximum test | the row of the maximum test, the crossings at λ ≈ 1.7 and 1.46 | `numerics/maximum_test.py` (run with `4000000 2000000`) |
| Section 6, rows **S** and **T** | the two rows, KS p = 0.28 and 0.99 | `sim/lowsnr_ST.py` (`sim/results_lowsnr.jsonl`, `sim/lowsnr_ST_summary.txt`) |
| Section 6, *F*-test | 1.54e-2, 42% and 52%, powers 0.100 and 0.191 | `numerics/ftest.py`, `numerics/numbers_sections_3_to_6.py` |
| Sections 6 and 7, local efficiency | u_α, t_α, C_max, C_opt, efficiencies 0.810 and 0.777 | `numerics/local_efficiency.py` |
| Section 7, *The studentised second maximum* | T₂ ≤ √κ/ρ on every replicate, the medians of T₂/(√κ/ρ) and of λ_max(Ω)/λ₂ | `sim/studentised_T2.py` (needs the raw files) |
| Section 7, *The true exponent, measured* | the deficiency table, the KS p-values 0.18, 0.31, 0.56, 0.88, the slopes 0.065, 0.115, 0.124, 0.15, the local slopes and the fit with a linear term (quadratic coefficients 0.065, 0.151, 0.226, 0.153; linear coefficients 0.00, -0.60, -1.31, -0.42) | `sim/production_2026-09-27.sh`, `sim/production_rows_33_34.sh`, then `sim/analyze.py` (`sim/results.jsonl`, `sim/results_rows_33_34.jsonl`) |
| Section 7, comparison with the bracket | 5% and 7% above E_low, the factors 1.7 and 1.2 | `numerics/exponent_constants.py` (reads `sim/*.jsonl`) |
| Section 7, *The estimation sandwich* | the four rows, 2.6% and 0.40% | `numerics/kacrice_mass_and_sandwich.py`, `numerics/sandwich_refine.py` |
| Section 7, *The price of knowing σ* | the level-map table, its typicality over 3000 draws, 2.8% | `numerics/level_map_draw.py`, `numerics/level_map_typicality.py`, `numerics/level_map.py` |

Most scripts print the value of the paper next to the recomputed one.

## The field simulation

`sim/run_point.py k d lambda nrep --seed 777` runs one point in parallel and appends one JSON line of counts to a
results file. Each chunk of 1000 replicates has its own seed, `SeedSequence([777, k, d, round(1000 lambda),
chunk])`, so the counts do not depend on the number of processes. Both production scripts take `PY`, `PROCS`
and `RAW` (folder of the raw files) from the environment.

- `sim/production_2026-09-27.sh`: the rows (3,5) and (5,3), their null calibrations and the row (3,3)
  (8.5e6 replicates; 1 h 40 min on 10 processes). Output `sim/results.jsonl`. The line of the point (3,3),
  λ = 12 was not written to that file; the next script runs that point again.
- `sim/production_rows_33_34.sh`: the null calibrations at (3,3) and (3,4), the row (3,4) and the point (3,3),
  λ = 12 (6.5e6 replicates; 1 h 25 min on 5 processes). Output `sim/results_rows_33_34.jsonl`.
- `sim/lowsnr_ST.py`: the rows **S** and **T** of Section 6, seed 20261019 (1e6 null, 5e5 per λ; about
  25 min on 10 processes).
- `sim/validate.py`: checks of every formula of `spike_lib.py` (tensor, regression, helix, Hessian, statistic),
  a few seconds; run by `run_all.sh`.

Checks of reproducibility made on 2026-10-25 with Python 3.14.3, numpy 2.4.2, scipy 1.17.1: the point (3,3),
λ = 8, 1e5 replicates, and the point (3,3), λ = 12, 1e6 replicates, give the same counts as the runs of
2026-09-27 (Python 3.13); the point λ = 2 of `sim/lowsnr_ST.py` gives the same counts as its run of
2026-10-19; `numerics/maximum_test.py 4000000 2000000` prints the same output as its run of 2026-09-27.

## Requirements

`numpy`, `scipy`, `mpmath` (installed with `sympy`). From the repository root:

```bash
./paper2/run_all.sh                                   # numerics/, then sim/analyze.py on the saved results
PROCS=5 RAW=/tmp/raw ./paper2/sim/production_rows_33_34.sh   # one of the field simulations
```

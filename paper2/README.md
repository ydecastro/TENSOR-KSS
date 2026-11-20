# The *t*-spacing test for a spiked tensor: code for every number

This folder recomputes every number of the paper *The t-spacing test for a spiked tensor* (Azaïs, Dalmao, and
De Castro, 2026) that comes from a computation: tables, simulated powers and deficiencies, constants printed
with digits, thresholds, Figure 1. Statement numbers, labels and section names refer to the version of 2026-11-18
(23 pages, and 12 pages of Supplementary Material). Statements of the Supplementary Material are numbered A to F
(for example Lemma C.2).

## Layout

| Folder | Content | Time |
|---|---|---|
| `numerics/` | Closed forms, quadratures, exact laws and small Monte Carlo runs. `../run_all.sh` runs them and saves each printed output as `<script>.log`. | about 25 minutes on one core, of which 10 for `maximum_test.py` and 10 for `deficiency_exponent_constants.py` |
| `sim/` | Monte Carlo of the *t*-spacing test from the Karhunen–Loève expansion of the field: two maxima, Hessian and noise estimate computed on each realised path. The JSON lines of the runs used in the paper are kept; the raw per-replicate files (about 370 MB) are not, and the scripts regenerate them. | hours (see below) |
| `figure1/` | Figure 1 and the paragraph "The figure of the introduction" of Section 6: every competing test and estimator on the same replicates of the field simulation. The raw per-replicate files (about 630 MB) are not kept; `figure1_data.json` and `figure1_null_ecdf.npz` are their reduction, and `make_figure.py` draws Figure 1 from them. | hours for the simulation, seconds for the figure |

## Where each number comes from

In the order of the paper. "Main" is the main paper, "Supplement" the Supplementary Material.

| Paper | Numbers | Script (output) |
|---|---|---|
| Main, Figure 1 (`fig:intro`), caption | 10^5 replicates per point; the rejection probability 0.097 at nominal level 0.05; at most 0.08 below the most powerful invariant test; within 10% of one at λ = 10 | `figure1/make_figure.py` (`fig_intro.pdf`), `figure1/figure_numbers.py` (both read `figure1_data.json`, `figure1_null_ecdf.npz`) |
| Main, Section 1, *Contributions*, item 3 | at (3,3) the quadratic test is above the maximum test only where both have power below 0.19 | `numerics/maximum_test.py` (crossing, common power 0.186) |
| Main, Section 1, *Growing dimension* | sup X/√d tends to about 1.657; C₃ = 7.70; 1.12; 1.21 | `numerics/literature_constants.py`, `numerics/growing_dimension_constants.py` |
| Main, Section 3.1, after Theorem 1 (`thm:upper`) | κ = 7 and α^{-1/κ} = 1.534 at (3,3); κ = 121 and 1.025 at (5,5) | `numerics/model_constants.py` |
| Main, Section 3.2, after Lemma 5 (`lem:T2sharp`) | T_max = 3.06, 6.32, 13.9; never exceeded on the simulations, approached on the replicates where the test fails | `numerics/model_constants.py`; `sim/studentised_T2.py` (needs the raw files; `sim/studentised_T2.log`) |
| Main, Section 3.2, after Corollary 3 (`cor:closed`) | the bound (`eq:closed-nct`) is below 0.05 from λ = 11.0 at (3,5) and 11.4 at (3,3) | `numerics/power_bound_student.py` |
| Main, Section 3.2, after Corollary 4 (`cor:Sclosed`) | c_α = 12.6 and 18.3; the bound (`eq:Sclosed-nct`) is below 0.05 from λ = 6.8 at (3,3) and 10.0 at (3,5); k/(2(3k−2)) | `numerics/model_constants.py`, `numerics/power_bound_student.py` |
| Main, Remark 1 (`rem:bracket`), Section 3.3 | the maximiser (0.995, 0.103) T_max at (3,3); the upper bound exceeds the column A by 1.3·10⁻³, 4.6·10⁻⁴, 1.9·10⁻⁶, 3.6·10⁻⁸; the search over all spectra; the table of κ, A, E and the column **S** at the six pairs | `numerics/deficiency_exponent_constants.py` (`deficiency_exponent_constants.json`), `numerics/model_constants.py` |
| Main, Remark 1, last sentence | the measured slopes lie between 10% below and 7% above E | `numerics/deficiency_vs_exponent.py` |
| Main, Section 3.5, after Theorem 3 (`thm:sqrtd`) | Υ₃ = 2.214, C₃ = 7.70, C₅ = 21.74, C₇ = 47.06 | `numerics/growing_dimension_constants.py` |
| Main, Remark 3 (`rem:dim`) | 1.12 for k = 3 | `numerics/literature_constants.py` |
| Main, Section 4.2, after Theorem 4 (`thm:estim`) | λ₀(k,d) = 28.83, 33.11, 29.50, 38.52, 34.52 | `numerics/lambda0.py` |
| Main, Remark after Corollary 6 (`cor:minimax`) | 0.0472, 0.01291, 0.003306, 0.000832; 88.54%, 96.80%, 99.18%, 99.79% | `numerics/estimation_lower_bound.py` |
| Main, Section 5 (`sec:low`), local efficiency | T_q = (1/5)χ²₃ + (2/35)χ²₇; efficiencies 0.810 and 0.777 | `numerics/quadratic_test.py`, `numerics/local_efficiency.py` (`local_efficiency.out`), `numerics/maximum_test.py` |
| Main, Section 5, *F*-test | 1.54·10⁻², 42% and 52%, powers 0.100 and 0.191 | `numerics/ftest.py`, `numerics/numbers_sections_3_to_6.py` (part 4) |
| Main, Section 5, table of the four tests | the row of the quadratic test | `numerics/quadratic_test.py` |
| | the row of the maximum test, the crossings at λ ≈ 1.7 and 1.46 with common power 0.19 and 0.045 | `numerics/maximum_test.py` (run with `4000000 2000000`) |
| | the rows **S** and **T**, 5·10⁵ replicates, standard error below 10⁻³, KS p = 0.28 and 0.99, power 0.65 of **T** at λ = 5 | `sim/lowsnr_ST.py` (`sim/results_lowsnr.jsonl`, `sim/lowsnr_ST_summary.txt`) |
| | **S** within 10⁻³ of the quadratic test at λ = 5 | `numerics/numbers_sections_3_to_6.py` (part 5) |
| Main, Section 6 (`sec:num`), opening | the closed forms checked against quadrature to ten significant digits | `numerics/closed_form_check.py` |
| Main, Section 6, *The model constants* | m, κ, ν at (3,3), (3,5), (3,8), (5,6) | `numerics/model_constants.py` |
| Main, Section 6, *The price of knowing σ* | the level-map table; over 3000 draws the median extra level 0.505, this draw above 90% of them, the draws at (3,8) and (5,6) close to the median | `numerics/level_map_draw.py`, `numerics/level_map_typicality.py` |
| | the 0.95-quantile of the reference Student law 3.4% below the Gaussian one at (3,5), its tail heavier only beyond u = 2.7 | `numerics/reference_student_quantile.py` (also `numerics/level_map.py`) |
| | the extra level falls below 0.1 when m is a few hundred | `numerics/level_map_draw.py` (the row (5,6), m = 252) |
| Main, Section 6, *The power bound* | 1.2873 ≤ A ≤ 1.2878 at (3,5); λ = 11.0 and 12.0 for (`eq:closed-nct`), 13.1 and 14.1 for (`eq:closed-chernoff`); the simulated deficiency 0.05 near λ = 6.2 and 0.01 near 7.2; the factor 1.7 to 1.8 | `numerics/deficiency_exponent_constants.py`, `numerics/power_bound_student.py` (reads `sim/*.jsonl`) |
| Main, Section 6, *The true exponent, measured* | the recheck of 2% of the replicates (λ₂ rose 3 times in 279 000, no decision changed); 2·10⁴ to 2·10⁶ replicates per point, 1.0·10⁷ in all, at least 78 failures per point | `sim/replicate_counts.py` |
| | the KS p-values 0.18, 0.31, 0.56, 0.88; the order relation (`eq:order`) in every replicate under the alternatives | `sim/analyze.py` |
| | the deficiency table | `sim/production_2026-09-27.sh`, `sim/production_rows_33_34.sh`, then `sim/analyze.py` (`sim/results.jsonl`, `sim/results_rows_33_34.jsonl`) |
| | the slopes 0.065 ± 0.002 (flat to within 4% over [6,12]), 0.115 ± 0.012, 0.124 ± 0.003; at (3,5) the weighted fits 0.136, 0.143, 0.153 and 0.14 ± 0.01, the local slopes 0.106, 0.128, 0.134, 0.162 and 0.12 ± 0.02, the deficiency 3.9·10⁻⁵ at λ = 9.5 | `sim/analyze.py` (`sim/analyze.log`) |
| | the table "measured against E"; 6% above E at (3,3), 7% above at (5,3) (2.7 times the uncertainty), within the error at (3,4), 10% below at (3,5); −λ⁻² log of the deficiency below E at every λ; the factors 1.7 at λ = 6 and 1.1 at λ = 12 at (3,3) | `numerics/deficiency_vs_exponent.py` (reads `sim/*.jsonl` and `numerics/deficiency_exponent_constants.json`) |
| Main, Section 6, *The estimation sandwich* | the four rows of the table: the lower bound (i), the Kac–Rice column and its ratio | `numerics/estimation_lower_bound.py`, `numerics/estimation_sandwich.py`; the Kac–Rice column also by `numerics/kacrice_mass_and_sandwich.py` |
| | the right-hand side of (ii): 4.261·10⁻³, 2.695·10⁻², 8.934·10⁻⁴, 9.244·10⁻⁴, ratios 1.278, 8.09, 1.072, 1.109 | `numerics/estimation_sandwich.py` (`estimation_lib.py`) |
| | within a factor 2 of the target from λ ≈ 11, 13, 13, 16 and within 10% from λ ≈ 26, 28, 34, 45; 4 to 46 times the target at λ = 10; the Kac–Rice bound within 10% of the lower bound from λ ≈ 9 (10.4 at (3,8)) | `numerics/estimation_thresholds.py`, which reads `estimation_range_scan.jsonl` (from `estimation_range_scan.py`) and `estimation_kacrice_scan.jsonl` (from `estimation_kacrice_scan.py`) |
| | the bracket within 2.6% at λ = 20 and 0.40% at λ = 40; λ − u = 4 costs 3.8% | `numerics/estimation_sandwich.py`; also `numerics/sandwich_refine.py` and `numerics/kacrice_mass_and_sandwich.py` |
| Main, Section 6, *The figure of the introduction* | κ = 112 at (3,8); the null levels 0.097 and 0.026 of the studentised maximum with the quantile of sup Z/σ; the power 0.20 at λ = 4 and 0.94 at λ = 12 of the locally optimal invariant test; the unfolding test above **T** by at most 0.011; the most powerful invariant test above the studentised maximum by at most 0.011; **T** below both by at most 0.08 at λ = 5; the tests with σ known above **T** by at most 0.16; the risk ratios at λ = 10 (within 10%, and 7 to 24% for the unfolding estimator) and at λ = 20 at (3,8) (2.5% and 5.5%) | `figure1/figure_numbers.py` (reads `figure1_data.json`) |
| | 2000 points of importance sampling per replicate; on 3000 replicates a cloud four times larger changes the power by at most 0.002 | `figure1/run_tests.py` (default cloud of four components of 500 points), `figure1/check_cloud_size.py` |
| Main, Section 6, *Local efficiency at vanishing signal* | the table: q_α, t_α, C_max, C_opt, efficiency at α = 0.05 and 0.01 | `numerics/local_efficiency.py` (`local_efficiency.out`), `numerics/quadratic_test.py` |
| Supplement, Appendix C, before Lemma C.2 (`lem:Aupper`) | 2.29 against 2.36 at q = 8, the ratio decreasing in q | `numerics/goe_norm_moment.py` |
| Supplement, Appendix C, definition of λ₀ (`eq:lam0`) | the condition that binds | `numerics/lambda0.py` |
| Supplement, Appendix E (`app:norm`) | 1.5847 and 1.1205 √d; 1.2066 √d | `numerics/literature_constants.py` |
| Supplement, Appendix F, Proposition F.1 (`prop:closed`) | the closed forms of G_r and H_r | `numerics/closed_form_check.py` |

The inequalities with explicit decimals inside the proofs of Appendix C (3.43, 2.86, 2.55, 3.49, 0.67, 0.73, 1.92,
1.41, 0.47, 0.76, 0.19, 0.142) are elementary arithmetic and have no script.

Most scripts print the value of the paper next to the recomputed one.

Some scripts were written for the version of 2026-10-26 and print the statement numbers of that version.
`numbers_sections_3_to_6.py`, `kacrice_mass_and_sandwich.py` and `sandwich_refine.py` serve the current version
through the parts named in the table; their other parts concern statements that are no longer in the paper. The
lines "(ii) optimised" of `kacrice_mass_and_sandwich.py` and the explicit bound of `sandwich_refine.py` use the
additive form 1 + β*/(ρu) + r_N + ε_det of 1 + ε_A. The paper uses the product form of Lemma C.2,
(1 + β*/(ρu) + ε_det)(1 + r_N), computed by `estimation_sandwich.py`; the two forms differ in the fourth digit
(4.257·10⁻³ against 4.261·10⁻³ in the first row).

## The constant A and the exponent E

`numerics/deficiency_exponent_constants.py` computes A of (`eq:Adef`) as a maximum over the spectra of the
normalised Hessian. For d = 3 the maximum is in one variable. For d ≥ 4 it gives the maximum over the spectra
(a, b, …, b), which is the column A of Remark 1, an upper bound from Lemma 6 (`lem:mono`), and the result of a
search over the whole sphere of spectra (seed 20261116 + 100k + d). H_r is evaluated in closed form at 50 digits and
checked by a quadrature at 80 digits. The script writes `deficiency_exponent_constants.json`, which
`power_bound_student.py` and `deficiency_vs_exponent.py` read. It takes about 10 minutes for the six pairs; it also runs
pair by pair (`python3 deficiency_exponent_constants.py 3 5`).

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
- `sim/lowsnr_ST.py`: the rows **S** and **T** of Section 5, seed 20261019 (1e6 null, 5e5 per λ; about
  25 min on 10 processes).
- `sim/validate.py`: checks of every formula of `spike_lib.py` (tensor, regression, helix, Hessian, statistic),
  a few seconds; run by `run_all.sh`.

Checks of reproducibility made on 2026-10-25 with Python 3.14.3, numpy 2.4.2, scipy 1.17.1: the point (3,3),
λ = 8, 1e5 replicates, and the point (3,3), λ = 12, 1e6 replicates, give the same counts as the runs of
2026-09-27 (Python 3.13); the point λ = 2 of `sim/lowsnr_ST.py` gives the same counts as its run of
2026-10-19; `numerics/maximum_test.py 4000000 2000000` prints the same output as its run of 2026-09-27.

## Figure 1

`figure1/run_tests.py k d lambda nrep` runs one point: every test and estimator of `figure1/tests_lib.py` on the
same replicates of the field simulation of `sim/` (the *t*-spacing test **T**, the spacing test **S** with σ known,
the maximum test, the studentised maximum, the locally optimal invariant test, the unfolding test and estimator,
the most powerful invariant test by importance sampling, and the argmax). Each chunk of 1000 replicates has its own
seed, `SeedSequence([seed, k, d, round(1000 lambda), chunk])`. `figure1/production.sh` runs every point of the
figure (1e5 replicates per point; 1e6 null replicates of seed 777 calibrate the tests without a closed form; the null
panel uses seed 778), then `figure1/reduce_raw.py`, which writes `figure1_data.json` (rejection rates, powers and
risks) and `figure1_null_ecdf.npz` (for panel (a), the sorted null p-values of each test at the ranks 1, 50, 100,
…, 100 000). The simulation ran on 2026-11-07 and 2026-11-09 (about 3 h 45 min on 4 processes). It was not run again
for this release. `figure1/make_figure.py` draws `fig_intro.pdf`, the Figure 1 of the paper, from the two reduced
files, and `figure1/figure_numbers.py` prints the numbers of the caption and of Section 6. `figure1/check_tests_lib.py`
checks the projectors, the quadratures and the samplers of `tests_lib.py`; `figure1/check_cloud_size.py` checks the
size of the importance-sampling cloud.

On 2026-11-18 the figure drawn from the reduced files was compared with the figure drawn from the raw files by the
script that produced the figure of the paper: panels (b) and (c) plot the same arrays to every digit; in panel (a)
each plotted point is one of the 100 000 points of the full curve, and the two curves are less than 2·10⁻⁴ apart
vertically (the axis runs from 0 to 1). The pages of the two PDF files, rendered at the same resolution, differ in
0.04% of the pixels.

## Requirements

`numpy`, `scipy`, `mpmath` (installed with `sympy`), `matplotlib` (for Figure 1). From the repository root:

```bash
./paper2/run_all.sh                                   # numerics/, figure1/, then sim/analyze.py on the saved results
PROCS=5 RAW=/tmp/raw ./paper2/sim/production_rows_33_34.sh   # one of the field simulations
PROCS=4 RAW=/tmp/raw1 ./paper2/figure1/production.sh          # the simulation of Figure 1
```

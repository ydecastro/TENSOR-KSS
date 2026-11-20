#!/bin/zsh
# Paper 2 (The t-spacing test for a spiked tensor): recompute every number that does not need a field simulation,
# draw Figure 1 from its reduced data, then the tables of the field simulation from the saved JSON lines.
# Each script prints its results; the printed output is saved as <script>.log next to it.
# Not run here (see README.md): the field simulations sim/production_2026-09-27.sh, sim/production_rows_33_34.sh,
# sim/lowsnr_ST.py and figure1/production.sh (hours), sim/studentised_T2.py (needs their raw files), and the scans
# numerics/estimation_range_scan.py and numerics/estimation_kacrice_scan.py (about an hour; their JSON lines are kept).
PY=${PY:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
run() {   # run <folder> <script> [args]: print the time and the name, save the output as <script>.log
  local dir=$1 s=$2; shift 2
  echo "$(date '+%H:%M:%S') $dir/$s"
  ( cd "$ROOT/$dir" && $PY $s.py "$@" > $s.log 2>&1 && perl -pi -e "s|\Q$ROOT/$dir/\E||g" $s.log ) \
    || echo "  FAILED: see $dir/$s.log"
}
ROOT=${0:A:h}
# numerics/: closed forms, quadratures, exact laws, small Monte Carlo runs
for s in model_constants closed_form_check level_map level_map_draw level_map_typicality reference_student_quantile \
         quadratic_test ftest local_efficiency numbers_sections_3_to_6 growing_dimension_constants \
         literature_constants estimation_lower_bound estimation_sandwich estimation_thresholds lambda0 \
         kacrice_mass_and_sandwich sandwich_refine goe_norm_moment; do
  run numerics $s
done
run numerics maximum_test 4000000 2000000                 # replicate counts of the paper (about 10 min)
run numerics deficiency_exponent_constants                 # A and E at the six pairs (about 10 min)
run numerics power_bound_student                           # reads deficiency_exponent_constants.json and sim/*.jsonl
run numerics deficiency_vs_exponent                        # reads deficiency_exponent_constants.json and sim/*.jsonl
# figure1/: Figure 1 and its numbers, from figure1_data.json and figure1_null_ecdf.npz
run figure1 make_figure
run figure1 figure_numbers
# sim/: the tables of the field simulation from the saved JSON lines, and the checks of spike_lib.py
run sim analyze results.jsonl results_rows_33_34.jsonl
run sim replicate_counts
run sim validate
echo "$(date '+%H:%M:%S') done"

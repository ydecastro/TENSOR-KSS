#!/bin/zsh
# Paper 2 (The t-spacing test for a spiked tensor): recompute every number that does not need the field
# simulation (a few minutes in all), then the tables of the field simulation from the saved JSON lines.
# Each script prints its results; the printed output is saved as <script>.log next to it.
# The field simulations themselves are sim/production_2026-09-27.sh, sim/production_rows_33_34.sh and
# sim/lowsnr_ST.py (hours; see README.md).
PY=${PY:-python3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1} \
       OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1} VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
cd "${0:A:h}/numerics"
for s in closed_form_check level_map level_map_draw level_map_typicality quadratic_test maximum_test ftest \
         local_efficiency exponent_constants exponent_A_Estar numbers_sections_3_to_6 remark9_power_bound \
         kacrice_mass_and_sandwich sandwich_refine; do
  echo "$(date '+%H:%M:%S') $s"
  args=(); [[ $s == maximum_test ]] && args=(4000000 2000000)   # replicate counts of the paper (about 10 min)
  $PY $s.py $args > $s.log 2>&1 || echo "  FAILED: see numerics/$s.log"
  perl -pi -e "s|\Q$PWD/\E||g" $s.log    # no absolute path in the saved output
done
cd ../sim
echo "$(date '+%H:%M:%S') analyze"
$PY analyze.py results.jsonl results_rows_33_34.jsonl > analyze.log 2>&1 || echo "  FAILED: see sim/analyze.log"
echo "$(date '+%H:%M:%S') validate"
$PY validate.py > validate.log 2>&1 || echo "  FAILED: see sim/validate.log"
echo "$(date '+%H:%M:%S') done"

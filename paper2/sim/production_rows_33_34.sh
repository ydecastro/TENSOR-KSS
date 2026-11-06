#!/bin/zsh
# Rows (3,3) and (3,4) of the deficiency table: null calibration at (3,3) and (3,4), and the (3,4) row.
# Seed 777, as production_2026-09-27.sh; each point appends one JSON line to results_rows_33_34.jsonl.
# The (3,3) row is in results.jsonl (production_2026-09-27.sh, parts (a) and (e)), except lambda = 12, whose
# line was not written in 2026-09-27 and is run again here.
PY=${PY:-python3}
PROCS=${PROCS:-5}
RAW=${RAW:-raw}
OUT=${OUT:-results_rows_33_34.jsonl}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "${0:A:h}"
run() { echo "$(date '+%H:%M:%S') start k=$1 d=$2 lam=$3 n=$4"; $PY run_point.py $1 $2 $3 $4 --check-frac 0.02 --seed 777 --procs $PROCS --out $OUT --raw-dir $RAW > /dev/null; echo "$(date '+%H:%M:%S') done  k=$1 d=$2 lam=$3"; }
run 3 3 0 1000000
run 3 4 0 1000000
run 3 4 5 50000
run 3 4 6 50000
run 3 4 7 100000
run 3 4 8 300000
run 3 4 9 1000000
run 3 4 10 2000000
run 3 3 12 1000000
echo "$(date '+%H:%M:%S') ALL DONE"

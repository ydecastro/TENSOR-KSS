#!/bin/zsh
# Field simulation of Figure 1 and of the paragraph "The figure of the introduction" of Section 6: every test and
# estimator of tests_lib.py on the same replicates, 1e5 replicates per point (1e6 for the null calibration).
# Each chunk of 1000 replicates has its own seed SeedSequence([seed, k, d, round(1000 lambda), chunk]).
# The outputs (one .npz per point and results.jsonl, about 630 MB) go to $RAW; reduce_raw.py then writes the two
# small files of this folder. Takes PY, PROCS and RAW from the environment.
# Run on 2026-11-07 and 2026-11-09 with 4 processes: about 2 h 20 min (block A), 55 min (block B), 30 min (block C).
set -u
PY=${PY:-python3}
PROCS=${PROCS:-4}
HERE=${0:A:h}
RAW=${RAW:-$HERE/raw}
mkdir -p "$RAW"
LOG="$RAW/production.log"
run() { echo "[$(date '+%F %T')] run_tests.py $*" >> "$LOG"; $PY "$HERE/run_tests.py" "$@" --procs $PROCS --out "$RAW" >> "$LOG" 2>&1; }
ORA="1,2,3,4,5,6,7,8,9,10,11,12"
# Block A: (3,8) null calibration (1e6, seed 777) and null panel (1e5, seed 778), most powerful invariant test at every
# lambda of the grid; then (3,8) power and estimation, this test at its own lambda up to 12; estimation at the other pairs
run 3 8 0 1000000 --seed 777 --oracle "$ORA" --tag calib_k3d8_null_1e6
run 3 8 0 100000  --seed 778 --oracle "$ORA" --tag panel_k3d8_null_1e5
for lam in 1 2 3 4 5 6 7 8 9 10 11 12; do run 3 8 $lam 100000 --seed 777 --oracle self --tag "k3d8_lam$lam"; done
for lam in 15 20 30 40; do run 3 8 $lam 100000 --seed 777 --tag "k3d8_lam$lam"; done
for kd in "3 3" "3 5" "5 3"; do
  set -- ${=kd}
  for lam in 2 3 4 5 6 8 10 12 15 20; do run $1 $2 $lam 100000 --seed 777 --tag "k$1d$2_lam$lam"; done
done
# Block B: panel (b) on lambda^2 = 2, 4, ..., 16: lambda = sqrt(2), sqrt(6), ..., sqrt(14) (lambda = 2 and 4 exist),
# then the most powerful invariant test at these six lambda on the same 1e6 null replicates (seed 777, same chunks)
LAMS=(1.4142135624 2.4494897428 2.8284271247 3.1622776602 3.4641016151 3.7416573868)
SQ=(2 6 8 10 12 14)
for i in 1 2 3 4 5 6; do run 3 8 ${LAMS[$i]} 100000 --seed 777 --oracle self --tag "k3d8_lamsq${SQ[$i]}"; done
ORA2=$(IFS=,; echo "${LAMS[*]}")
run 3 8 0 1000000 --seed 777 --oracle "$ORA2" --tag calib_k3d8_null_1e6_b
# Block C: panel (c) at small lambda for the four pairs ((3,8) already has lambda = 1)
for kd in "3 3" "3 5" "5 3"; do
  set -- ${=kd}
  for lam in 0.125 0.25 0.5 1 1.5; do run $1 $2 $lam 100000 --seed 777 --tag "k$1d$2_lam${lam}"; done
done
for lam in 0.125 0.25 0.5 1.5; do run 3 8 $lam 100000 --seed 777 --tag "k3d8_lam${lam}"; done
echo "[$(date '+%F %T')] done" >> "$LOG"
$PY "$HERE/reduce_raw.py" "$RAW"

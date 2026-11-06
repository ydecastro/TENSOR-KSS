#!/bin/zsh
# Production runs of 2026-09-27 (seed 777). Each point appends one JSON line to results.jsonl.
PY=${PY:-python3}
cd "${0:A:h}"
run() { echo "$(date '+%H:%M:%S') start k=$1 d=$2 lam=$3 n=$4"; $PY run_point.py $1 $2 $3 $4 --check-frac 0.02 --seed 777 --out results.jsonl > /dev/null; echo "$(date '+%H:%M:%S') done  k=$1 d=$2 lam=$3"; }
# (a) pipeline cross-check at (3,3)
run 3 3 8 100000
run 3 3 10 500000
# (b) null calibration
run 3 5 0 1000000
run 5 3 0 1000000
# (c) measurement grids
run 3 5 5 50000
run 3 5 6 50000
run 3 5 7 100000
run 3 5 8 300000
run 5 3 5 50000
run 5 3 6 50000
run 5 3 7 100000
run 5 3 8 200000
run 3 5 9 1000000
run 5 3 9 800000
# (d) deepest point at (3,5)
run 3 5 9.5 2000000
# (e) rest of the (3,3) row
run 3 3 6 20000
run 3 3 7 50000
run 3 3 9 100000
run 3 3 12 1000000
echo "$(date '+%H:%M:%S') ALL DONE"

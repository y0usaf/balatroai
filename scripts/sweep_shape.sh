#!/usr/bin/env bash
# Throughput sweep over rollout shape (workers x envs x groups).
# Each variant runs ~8 iterations from the same checkpoint; we report the
# median steps/sec of the last 5 (first iterations include warmup).
#
#   nix develop -c bash scripts/sweep_shape.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-runs/pointer_v3/latest.pt}
ROLLOUT=128
ITERS=8

run() {
  local w=$1 e=$2 g=$3
  local per_iter=$((w * e * ROLLOUT))
  local base=4020224
  local total=$((base + per_iter * ITERS))
  local log=/tmp/sweep_${w}x${e}_g${g}.log
  timeout 900 .venv/bin/python -u scripts/train_fast_v3.py \
    --resume "$CKPT" --total-steps "$total" \
    --workers "$w" --envs-per-worker "$e" --groups "$g" \
    --log-dir "runs/sweep/w${w}e${e}g${g}" > "$log" 2>&1
  local sps
  sps=$(rg "^step" "$log" | tail -5 | rg -o "sps\s+[0-9,]+" | tr -d ', ' | sed 's/sps//' \
        | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  local cu
  cu=$(rg "^step" "$log" | tail -1 | rg -o "c/u [0-9.]+/[0-9.]+s")
  printf '%-18s %-8s envs=%-4s sps=%-8s %s\n' "w${w} e${e} g${g}" "" "$((w * e))" "${sps:-FAIL}" "$cu"
}

echo "config             envs      sps       collect/update"
run 26 8 3
run 26 16 3
run 26 16 4
run 30 12 3
run 26 32 4

#!/usr/bin/env bash
# Keep a training run alive unattended.
#
# The trainer has no internal recovery: a CUDA fault, an OOM or a worker dying
# ends the run, and an overnight run that dies at 2am wastes the whole night.
# This restarts it, with a cap and a backoff so a genuinely broken
# configuration fails loudly instead of thrashing.
#
#   setsid nohup nix develop -c bash scripts/supervise.sh runs/pointer_v7 \
#       > runs/pointer_v7/supervisor.log 2>&1 < /dev/null &
#
# With no trainer command after the run directory, runs the v3 pointer
# trainer (resuming from latest.pt when present).  Otherwise everything
# after the run directory is the trainer command, executed as-is — it must
# handle its own resume from its own checkpoint (balatroai train does).
set -uo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=${1:?usage: supervise.sh RUN_DIR [trainer command + args...]}
shift

MAX_RESTARTS=${MAX_RESTARTS:-20}
BACKOFF=${BACKOFF:-60}
LOG="$RUN_DIR/train.log"
mkdir -p "$RUN_DIR"

restarts=0
while :; do
  if [ $# -eq 0 ]; then
    resume=()
    if [ -f "$RUN_DIR/latest.pt" ]; then
      resume=(--resume "$RUN_DIR/latest.pt")
      echo "[supervisor] resuming from $RUN_DIR/latest.pt" | tee -a "$LOG"
    fi

    echo "[supervisor] $(date -Is) starting v3 trainer (restart $restarts)" | tee -a "$LOG"
    .venv/bin/python -u scripts/train_fast_v3.py --log-dir "$RUN_DIR" \
      "${resume[@]}" >> "$LOG" 2>&1
  else
    resume=()
    if [ -f "$RUN_DIR/latest.pt" ]; then
      resume=(--resume "$RUN_DIR/latest.pt")
      echo "[supervisor] resuming from $RUN_DIR/latest.pt" | tee -a "$LOG"
    fi
    echo "[supervisor] $(date -Is) starting trainer (restart $restarts)" | tee -a "$LOG"
    "$@" "${resume[@]}" >> "$LOG" 2>&1
  fi
  code=$?

  if [ $code -eq 0 ]; then
    echo "[supervisor] $(date -Is) trainer finished cleanly" | tee -a "$LOG"
    break
  fi

  restarts=$((restarts + 1))
  echo "[supervisor] $(date -Is) trainer exited $code (restart $restarts/$MAX_RESTARTS)" \
    | tee -a "$LOG"
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "[supervisor] giving up after $MAX_RESTARTS restarts" | tee -a "$LOG"
    break
  fi
  sleep "$BACKOFF"
done

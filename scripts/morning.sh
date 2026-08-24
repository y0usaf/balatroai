#!/usr/bin/env bash
# Morning report for an overnight training run.
#
#   nix develop -c bash scripts/morning.sh
#
# Prints, in order: whether the run is still alive, the learning curve as a
# coarse table, any supervisor restarts, and a benchmark of the best checkpoint
# against the bar that matters -- the heuristic bot's 2.9 mean ante.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN=${RUN:-runs/pointer_v8}
GAMES=${GAMES:-60}

echo "=============================================================="
echo " balatroAI overnight report — $(date -Is)"
echo "=============================================================="
echo

if pgrep -f "scripts/train_fast_v3.py" > /dev/null; then
  echo "run:        ALIVE"
else
  echo "run:        STOPPED (finished, crashed, or gave up — see restarts below)"
fi

.venv/bin/python scripts/train_status.py "$RUN" 2>/dev/null || true

echo
echo "--- curve (fresh-start mean ante) ---"
rg "^step" "$RUN/train.log" 2>/dev/null | awk 'NR % 40 == 1' | tail -15 \
  | sed -E 's/  +/  /g; s/(sps [0-9,]+).*(ante [0-9.]+)/\1  \2/'

echo
echo "--- supervisor restarts ---"
rg "^\[supervisor\]" "$RUN/train.log" 2>/dev/null | tail -6 || echo "none"

best=$(ls -t "$RUN"/ckpt_*.pt 2>/dev/null | head -1)
if [ -z "$best" ]; then
  echo
  echo "no checkpoint yet — nothing to evaluate"
  exit 0
fi

echo
echo "--- evaluating $best over $GAMES games ---"
echo "    (bar to beat: heuristic bot = 2.9 mean ante, random = 1.0)"
.venv/bin/python scripts/eval_fast_v3.py "$best" --episodes "$GAMES" 2>&1 | tail -12

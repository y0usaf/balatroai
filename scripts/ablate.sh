#!/usr/bin/env bash
# Which of the four changes cost us the learning curve?
#
# Four probes, cumulative, one change at a time, from the old baseline config
# up to the current one. Same seed, same rollout shape, same step budget, run
# sequentially so none of them compete for CPU.
#
#   nix develop -c bash scripts/ablate.sh
#
# Reference points from earlier runs, at ~3M steps, fresh-start mean ante:
#   old d=128 baseline  1.68
#   v5 (all but curriculum, d=256)  1.01
#   v6 (everything, d=256)          1.00
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-5000000}
OUT=runs/ablate
mkdir -p "$OUT"

common=(--workers 30 --envs-per-worker 12 --groups 3 --compile cudagraph
        --total-steps "$STEPS" --seed 0 --checkpoint-every 100000000)

run() {
  local name=$1; shift
  local log="$OUT/$name.log"
  if [ -s "$log" ] && rg -q "^step" "$log"; then
    echo "skip $name (already has results)"
    return
  fi
  echo "=== $name ==="
  timeout 3600 .venv/bin/python -u scripts/train_fast_v3.py \
    "${common[@]}" "$@" --log-dir "$OUT/$name" > "$log" 2>&1
}

# A: the old configuration -- small net, no embedding, no shaping, no curriculum.
run baseline       --d-model 128 --n-heads 4 --n-layers 2 --no-center-emb --shaping-coef 0

# B: + joker/shop identity embeddings.
run emb            --d-model 128 --n-heads 4 --n-layers 2 --shaping-coef 0

# C: + potential-based shaping.
run emb_shape      --d-model 128 --n-heads 4 --n-layers 2 --shaping-coef 1.0

# D: + curriculum injection.
run emb_shape_curr --d-model 128 --n-heads 4 --n-layers 2 --shaping-coef 1.0 \
                   --curriculum-pool runs/curriculum/pool.pkl --curriculum-frac 0.35

# E: everything, but at the larger net -- isolates capacity from the rest.
run big_all        --d-model 256 --n-heads 8 --n-layers 3 --shaping-coef 1.0 \
                   --curriculum-pool runs/curriculum/pool.pkl --curriculum-frac 0.35

echo
echo "probe            fresh ante (last 10 iters)   steps/episode   sps"
for name in baseline emb emb_shape emb_shape_curr big_all; do
  log="$OUT/$name.log"
  [ -s "$log" ] || continue
  .venv/bin/python - "$log" "$name" <<'PY'
import re, sys
log, name = sys.argv[1], sys.argv[2]
rows = []
for line in open(log, errors="ignore"):
    m = re.search(r"step\s+([\d,]+).*?sps\s+([\d,]+).*?ante\s+([\d.]+).*?eps\s+(\d+)", line)
    if m:
        rows.append((int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")),
                     float(m.group(3)), int(m.group(4))))
if not rows:
    print(f"{name:<16} no data"); raise SystemExit
last = rows[-10:]
ante = sum(r[2] for r in last) / len(last)
sps = sum(r[1] for r in last) / len(last)
per_iter = rows[-1][0] - rows[-2][0] if len(rows) > 1 else rows[-1][0]
spe = per_iter / max(sum(r[3] for r in last) / len(last), 1)
print(f"{name:<16} {ante:>6.2f} @ {rows[-1][0]:>10,}      {spe:>6.1f}        {sps:>6,.0f}")
PY
done

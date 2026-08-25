#!/usr/bin/env bash
# Joint-sweep transfer selection (0824_g19_replicates follow-up) +
# cross-character transfer question (Bradley: "would one that is good
# vs mewtwo also be good vs yoshi and ICs?").
#
# Fox is SATURATED at the ceiling (~439/min) — among near-ceiling
# epochs (>=434 in the existing fox sweep tables), gate stand-MEWTWO,
# stand-YOSHI, and stand-POPO(ICs). Selection = best transfer among
# ceiling epochs; the cross-character correlation answers whether
# transfer is one opponent-invariance axis or per-character luck.
# Run inside devenv shell. NO-MIX while running.
set -uo pipefail
cd "$(dirname "$0")/../.."

export DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
export ISO="$HOME/isos/melee.iso"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_SKIP_NIF_COMPILE=1
export EXPHIL_STARVATION_FATAL=1

BASE=eval_runs/0824_g19_replicates
OUT=eval_runs/0824_joint_sweep
mkdir -p "$OUT"
TABLE="$OUT/joint_table.txt"
: > "$TABLE"

gate() {
  local snap="$1" dir="$2" char="$3"
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh \
    "$snap" "$dir" --runs 1 --dummy stand --runner sync \
    -- --frame-delay 3 --delay-id-override 3 --dummy-character "$char" \
       --headless --emulation-speed 0 --blocking-input --slippi-port 51442 \
    > "$dir.log" 2>&1
  local line
  line=$(grep -aE "^\[[0-9:]+\] r1 " "$dir.log" | tail -1 || true)
  if [ -n "$line" ]; then
    echo "$line" | awk '{print $(NF-1)" (chain "$NF")"}'
  else
    echo "GATE FAILED"
  fi
}

for i in 1 2 3; do
  eps=$(grep -aE "^ep[0-9]+: " "$BASE/r$i/sweep/sweep_table.txt" \
    | awk -F'[: /]' '{gsub("ep","",$1); if ($3+0 >= 434) print $1}')
  for ep in $eps; do
    snap="checkpoints/ms_g19r${i}_ep${ep}.bin"
    [ -f "$snap" ] || { echo "r$i ep$ep MISSING SNAPSHOT" | tee -a "$TABLE"; continue; }
    fox=$(grep -aE "^ep${ep}: " "$BASE/r$i/sweep/sweep_table.txt" | head -1)
    row="r$i ep$ep [$fox]"
    for char in mewtwo yoshi popo; do
      d="$OUT/r${i}_ep${ep}_${char}"
      mkdir -p "$d"
      score=$(gate "$snap" "$d" "$char")
      row="$row ${char}=${score}"
    done
    echo "$row" | tee -a "$TABLE"
  done
done

echo "=== JOINT SWEEP DONE $(date +%H:%M:%S)" | tee -a "$TABLE"

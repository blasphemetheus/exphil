#!/usr/bin/env bash
# ONE-knob + latency-probe smoke (INVARIANTS item 12 structural form).
#  1 sync  --reaction-delay 4        -> probe: latency 5 measured (expected 5) aligned; chains
#  2 async --reaction-delay 4        -> same on the async runner
#  3 async --reaction-delay 4 --local-delay 1 -> probe MUST report a mismatch (latency 6)
#  4 suite --reaction-delay 2        -> control chains (rd 2)
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 DOLPHIN_DIR="$HOME/.local/share/slippi/exi-ai/dolphin-emu-headless"
O=eval_runs/0912_sync_rung
P=checkpoints/ms_g23a_ep57.bin
for cell in "sync 4 " "async 4 " "async 4 --local-delay 1"; do
  set -- $cell; runner=$1; k=$2; shift 2; extra="$*"
  echo "=== $runner --reaction-delay $k $extra ==="
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh $P "$O/smoke2_${runner}_k${k}${extra// /_}" --runs 1 --seconds 45 --dummy stand --runner $runner --temperature 1.0 -- --reaction-delay $k $extra --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "latency|Reaction delay|^\[[0-9:]+\] r1 |delay_id" | tail -5
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done
echo "=== suite --reaction-delay 2 ==="
mix run scripts/scenario_suite.exs --driver policy --policy $P --reaction-delay 2 --character fox --manifest scenarios/ms_midchain_control.json --runs 1 --temperature 1.0 --out $O/smoke2_suite.json --run-dir $O/smoke2_suite --quiet 2>&1 | grep -aE "reaction delay|delay_id|multishine_reentry runs=|error" | tail -4
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== DONE ==="

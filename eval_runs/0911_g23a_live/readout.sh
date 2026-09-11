#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12
echo "=== Bradley's session replays: chains ==="
R=$(ls eval_runs/0911_g23a_live/**/*.slp eval_runs/0911_g23a_live/*.slp 2>/dev/null | tr '\n' ' ')
[ -n "$R" ] && EXPHIL_GPU_MEMORY_FRACTION=0.25 mix run scripts/analyze_shine_source.exs $R 2>&1 | grep -aE "^\[[0-9:]+\] (replay|Game|r[0-9]|2026)|slp" | head -12
for d in 2 3 4; do
  echo "=== ASYNC runner, stand dummy, frame-delay $d, id 2 ==="
  EXLA_TARGET=host EXPHIL_GPU_MEMORY_FRACTION=0.25 bash scripts/eval_live_protocol.sh checkpoints/ms_g23a_ep57.bin eval_runs/0911_g23a_live/async_d${d} --runs 1 --dummy stand --runner async --temperature 1.0 -- --frame-delay $d --delay-id-override 2 --headless --emulation-speed 0 --blocking-input --slippi-port 51442 2>&1 | grep -aE "^\[[0-9:]+\] r1 "
  pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
done

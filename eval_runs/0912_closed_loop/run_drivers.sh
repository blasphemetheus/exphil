#!/usr/bin/env bash
# Closed-loop correction validation (Astra's note, HANDOFF_2026-09-12 §3):
# same break moments, three drivers at the handoff, multishine_reentry score.
set -uo pipefail
cd "$(dirname "$0")/../.."
export XLA_TARGET_EVAL=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25
bash scripts/x11_socket_fix.sh
ONLY="${ONLY:-0,1,2,3,4,5,6,7,8,9,10,11}"
M=scenarios/ms_breaks_manifest.json
O=eval_runs/0912_closed_loop
echo "=== driver: teacher ==="
mix run scripts/scenario_suite.exs --driver teacher --character fox --manifest $M --only "$ONLY" --runs 1 --out $O/scores_teacher.json --run-dir $O/runs_teacher --quiet 2>&1 | grep -aE "pass|PASS|score|Scenario|drift|error" | tail -20
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== driver: neutral ==="
mix run scripts/scenario_suite.exs --driver neutral --character fox --manifest $M --only "$ONLY" --runs 1 --out $O/scores_neutral.json --run-dir $O/runs_neutral --quiet 2>&1 | grep -aE "pass|PASS|score|Scenario|drift|error" | tail -20
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== driver: policy ms_g24a_ep55 (T=1.0, 2 runs) ==="
mix run scripts/scenario_suite.exs --driver policy --policy checkpoints/ms_g24a_ep55.bin --character fox --manifest $M --only "$ONLY" --runs 2 --temperature 1.0 --out $O/scores_policy.json --run-dir $O/runs_policy --quiet 2>&1 | grep -aE "pass|PASS|score|Scenario|drift|error" | tail -20
pkill -9 -f "squashfs-root/usr/bin/dolphin-emu" 2>/dev/null || true
echo "=== DONE ==="

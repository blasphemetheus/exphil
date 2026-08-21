#!/usr/bin/env bash
# Peak-science queue driver (2026-08-21): waits for the F3 arms
# pipeline to exit, then runs P1 (anneal-at-peak) and P2 (SS arms)
# sequentially. P3 (freeze probes) is deliberately NOT here — it needs
# a dagger_drill edit that must not happen while F3's loop can still
# re-read the script (round-4 mid-loop-edit lesson).
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "=== QUEUE: waiting for F3 arms to finish..."
while pgrep -f "[r]un_f3_arms.sh" >/dev/null; do sleep 60; done
echo "=== QUEUE: F3 done at $(date +%H:%M:%S) — launching P1"

bash eval_runs/0821_peak_p1_anneal/run_p1.sh || echo "=== P1 FAILED (continuing to P2)"
echo "=== QUEUE: P1 done at $(date +%H:%M:%S) — launching P2"
bash eval_runs/0821_peak_p2_ss/run_p2.sh || echo "=== P2 FAILED"
echo "=== QUEUE DONE $(date +%H:%M:%S)"
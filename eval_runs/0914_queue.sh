#!/usr/bin/env bash
# 2026-09-14 queue: A = reaction-4 proof (~25 min), then B = g26 hold vs drop (~3.5 h).
cd "$(dirname "$0")/.."
bash eval_runs/0914_delay4_proof/run.sh 2>&1 | tail -3
bash eval_runs/0914_g26_hold_vs_drop/run.sh 2>&1 | tail -3

#!/usr/bin/env bash
# Instrument 1 at the TRAINING shift on the g25 pool sources (shifts 0,3,4,5).
set -uo pipefail
cd "$(dirname "$0")/../.."
ROLL="eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp,eval_runs/0911_g23a_live/2026-09-Mainline/*.slp,eval_runs/0911_g23a_live/async_d3_cpu/r*.slp,eval_runs/0911_g23a_live/cpu_rollouts/r*.slp"
OPEN='eval_runs/0730_farm9_*/r*.slp,eval_runs/d3_div_*/r*.slp'
SNIP=eval_runs/0910_snippets_human_causal_d1/snippets.frames
mix run scripts/audit_ms_pool_labels.exs --rollouts "$ROLL" --openers "$OPEN" --snippets "$SNIP" --shifts "0,3,4,5"
echo "=== exit $?"

#!/usr/bin/env bash
# Coverage round, stage 1b: SECOND-GENERATION source games. Stage-1 rollouts
# vs live CPUs cannot be replayed (CPU AI sets stick floats no controller byte
# reproduces: 0.0039 recorded -> 0.0062 replayed; the error compounds and the
# first hit differs by frame ~112). So each rollout is re-played by the suite:
# the policy (reaction 4, T=1.0) from frame 30 for 5000 frames vs a GHOST of
# the CPU's recorded inputs on the correct body (opponent character now read
# from the replay; it was hard-coded fox). Both ports are then bridge-driven
# and quantization-stable: 4/4 mined handoffs replayed with zero drift in the
# smoke (frames 321..3193). Sources: 10 CPU rollouts (5 bodies) + Bradley's
# 09-13 session (human ghost) + 2 old Fox-CPU rollouts.
set -uo pipefail
cd "$(dirname "$0")/../.."
if pgrep -x beam.smp >/dev/null; then echo "A BEAM is live; wait." >&2; exit 1; fi
OUT=eval_runs/0914_coverage_round
POLICY=eval_runs/0914_delay4_proof/round21/candidate.bin
P=$OUT/progress.log
say() { echo "[$(date +%T)] $*" | tee -a "$P"; }
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.15
say "=== stage 1b: gen2 input-driven games (policy vs ghost, frame 30, 5000 f)"
SRC=$(command ls $OUT/rollouts/*/r*.slp eval_runs/local_multishine_20260913_224806/2026-09-Mainline/*.slp eval_runs/0911_g23a_live/cpu_rollouts/r{1,2}.slp)
jq -n --args '{entries: [$ARGS.positional[] | {slp: ., frame: 30, type: "multishine_reentry", note: ("gen2 source " + .)}]}' $SRC > $OUT/gen2_manifest.json
rm -rf $OUT/gen2 $OUT/gen2.json
mix run scripts/scenario_suite.exs --driver policy --policy "$POLICY" \
  --reaction-delay 4 --temperature 1.0 --character fox --prefix-history committed \
  --manifest $OUT/gen2_manifest.json --runs 1 --window 5000 --response-opponent replay --live-af \
  --no-orphan-sweep --quiet \
  --out $OUT/gen2.json --run-dir $OUT/gen2 > $OUT/gen2.log 2>&1
say "  gen2 exit $?: errored=$(jq .errored_runs $OUT/gen2.json) diverged=$(jq .diverged_runs $OUT/gen2.json) invalid_timing=$(jq .invalid_timing_runs $OUT/gen2.json) runs=$(jq '.runs|length' $OUT/gen2.json)"
say "  bodies/chains: $(jq -c '[.runs[] | [.opponent_character, .details.max_chain, .frames_observed]]' $OUT/gen2.json)"
say "stage 1b done"

#!/usr/bin/env bash
# Evaluate the Mewtwo PPO arm, with the selection/test separation the standing
# rule requires. Waits for the training unit to exit first, so it can be armed
# while training is still running.
#
#   scripts/launch_unit.sh exphil-mewtwo-eval 'bash scripts/mewtwo_eval_chain.sh'
#
# Structure, and why:
#   SELECTION  candidate heads, 60 games each, seed 8100. Picks the head.
#   TEST       the winner, 200 games, seed 9200 — games selection never saw.
#   CONTROL    the untouched prior against itself, 200 games, same seed 9200.
# The control is not optional: a win rate against one frozen opponent has no
# scale without it. Fox's control came back 49.0 %, which is what made its
# 91.0 % believable.
set -euo pipefail

REPO=/home/blewf/git/exphil
cd "$REPO"

PRIOR=checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin
RUN=eval_runs/0924_mewtwo_ppo
ARM=$RUN/v1
EVAL=$RUN/eval

export EXLA_TARGET=cuda
export EXPHIL_GPU_MEMORY_FRACTION=0.80
export EXPHIL_EXLA_PRECISION=highest

stamp() { date '+%H:%M:%S'; }

echo "=== $(stamp) waiting for exphil-mewtwo-ppo to finish"
while systemctl --user is-active --quiet exphil-mewtwo-ppo; do sleep 30; done
echo "=== $(stamp) training unit is done; starting evaluation"

mkdir -p "$EVAL"

# The prior's own head, for the control arm.
PRIOR_HEAD=$RUN/prior_head.bin
if [ ! -f "$PRIOR_HEAD" ]; then
  mix run scripts/ppo_make_prior_head.exs \
    --policy "$PRIOR" --character mewtwo --out "$PRIOR_HEAD"
fi

# --- SELECTION: which checkpoint, decided on its own games.
# Take the last saved head plus two earlier ones, whatever actually exists.
mapfile -t HEADS < <(ls -1 "$ARM"/head_iter*.bin 2>/dev/null \
  | sed 's/.*head_iter\([0-9]*\)\.bin/\1 &/' | sort -n | awk '{print $2}')
if [ ${#HEADS[@]} -eq 0 ]; then echo "no heads in $ARM; aborting"; exit 1; fi

LAST=${HEADS[-1]}
MID=${HEADS[$(( ${#HEADS[@]} / 2 ))]}
EARLY=${HEADS[$(( ${#HEADS[@]} / 4 ))]}
CANDIDATES=$(printf '%s\n' "$EARLY" "$MID" "$LAST" | awk '!seen[$0]++')

echo "=== $(stamp) SELECTION over: $(echo $CANDIDATES | tr '\n' ' ')"
for h in $CANDIDATES; do
  tag=$(basename "$h" .bin)
  echo "--- $(stamp) selection $tag"
  mix run scripts/ppo_r3_eval.exs \
    --policy "$PRIOR" --head "$h" \
    --games 60 --envs 60 --cap 28800 --fingerprint-games 8 \
    --seed 8100 --out "$EVAL/sel_$tag"
done

# Pick the candidate with the highest win rate on the SELECTION games.
BEST=$(python3 - "$EVAL" <<'PY'
import json, sys, glob, os
best, best_wr = None, -1.0
for p in sorted(glob.glob(os.path.join(sys.argv[1], "sel_*", "summary.json"))):
    d = json.load(open(p))
    wr = d.get("win_rate", 0.0)
    if wr > best_wr:
        best_wr, best = wr, d.get("head")
print(best or "")
PY
)
if [ -z "$BEST" ]; then echo "selection produced no winner; aborting"; exit 1; fi
echo "=== $(stamp) SELECTION winner: $BEST"

# --- TEST: fresh games the selection never touched.
echo "=== $(stamp) TEST: $BEST, 200 games, seed 9200"
mix run scripts/ppo_r3_eval.exs \
  --policy "$PRIOR" --head "$BEST" \
  --games 200 --envs 100 --cap 28800 --fingerprint-games 16 \
  --seed 9200 --out "$EVAL/test"

# --- CONTROL: the untouched prior against itself, same seed as the test.
echo "=== $(stamp) CONTROL: prior vs prior, 200 games, seed 9200"
mix run scripts/ppo_r3_eval.exs \
  --policy "$PRIOR" --head "$PRIOR_HEAD" \
  --games 200 --envs 100 --cap 28800 --fingerprint-games 16 \
  --seed 9200 --out "$EVAL/control"

echo "=== $(stamp) EVAL COMPLETE"
python3 scripts/degeneracy_check.py --baseline "$EVAL/control" \
  --candidate "$EVAL"/sel_head_iter* "$EVAL/test" \
  --out "$EVAL/degeneracy_report.json"
# Report-only: retrospective heuristics require review, not automatic promotion.
# Keep the test result separate from the selection decision above.
python3 - "$EVAL" <<'PY'
import json, glob, os, sys
root = sys.argv[1]
print(f"\n{'arm':<28}{'W':>5}{'L':>5}{'D':>5}{'win %':>9}{'  95% CI':>18}")
for p in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
    d = json.load(open(p))
    ci = d.get("wilson95") or ["?", "?"]
    ci = f"{ci[0]:.3f}-{ci[1]:.3f}" if isinstance(ci[0], float) else str(ci)
    print(f"{os.path.basename(os.path.dirname(p)):<28}{d.get('wins',0):>5}"
          f"{d.get('losses',0):>5}{d.get('draws',0):>5}"
          f"{d.get('win_rate',0)*100:>8.1f}%{ci:>18}")
print("\nThe control must land near 50 %. If it does not, the test number is an "
      "artifact of the evaluator and the gate is NOT passed.")
PY

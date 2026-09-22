#!/usr/bin/env bash
# Launch scripts/coach_review.exs as a systemd user unit (survives the Bash tool's
# ~10-min SIGTERM). The unit runs from $HOME by default, so we cd explicitly.
#
#   scripts/launch_coach_unit.sh UNIT REPLAY.slp OUT_DIR [extra coach_review args...]
#
# Log: logs/<UNIT>.log. Waits for any existing exphil-coach-* unit to finish first.
set -euo pipefail
unit=$1; replay=$2; out=$3; shift 3
repo=/home/blewf/git/exphil
policy=${COACH_POLICY:-checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin}

while systemctl --user list-units --state=active 'exphil-coach-*' --no-legend | grep -q .; do sleep 10; done

systemd-run --user --unit "$unit" --collect bash -lc \
  "cd $repo && devenv shell -- mix run scripts/coach_review.exs '$replay' --policy $policy --out $out $* > logs/$unit.log 2>&1"
sleep 8
systemctl --user is-active "$unit"
tail -c 300 "$repo/logs/$unit.log"

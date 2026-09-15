#!/usr/bin/env bash
set -euo pipefail
cd /home/blewf/git/exphil
while systemctl --user is-active --quiet exphil-float-coverage02.service; do sleep 2; done
if pgrep -x beam.smp >/dev/null; then echo 'Another BEAM is live'; exit 1; fi
export EXLA_TARGET=host
mix format lib/exphil/data/peppi.ex lib/exphil/eval/replay_prefix_audit.ex lib/exphil/eval/float_input_build.ex test/exphil/eval/float_input_build_test.exs scripts/install_float_dolphin.exs
mix test test/exphil/eval/replay_prefix_audit_test.exs test/exphil/eval/float_input_build_test.exs test/exphil/data/processed_input_test.exs > eval_runs/0915_float_input/v2_tests.log 2>&1
bash eval_runs/0915_float_input/smoke10.sh

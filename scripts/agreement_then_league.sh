#!/usr/bin/env bash
# 10-21: (1) agreement eval of the candidate with the semi-Markov hold rule +
# press calibration (~18 min); (2) the full league round robin, 12 bots,
# 16 envs x 3600 f x 2 sessions per pair (~4 h at ~550 game-frames/s).
# No training. One beam at a time (NO-MIX LAW).
set -euo pipefail
cd ~/git/exphil
export EDIFICE_LOCAL_NX=1
pgrep -f '[b]eam.smp' >/dev/null && { echo "a beam is alive"; exit 1; }
mx() { devenv shell -- mix run --no-compile "$@"; }
echo "== agreement eval s906_ep2 ($(date +%H:%M))"
mx scripts/agreement_eval.exs --policy checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin --games 24 --stride 2 --out eval_runs/agreement/s906_ep2.json --rows eval_runs/agreement/s906_ep2_rows.jsonl 2>&1 | tee logs/agreement_s906.log | grep -E "RESULT|^\*\*|rror" | cut -c1-500
echo "== league ($(date +%H:%M))"
mx scripts/league.exs --envs 16 --frames 3600 --out eval_runs/league/1021_full 2>&1 | tee logs/league_full.log | grep -E "RESULT|^\*\*|rror|unsupported|ready" | cut -c1-400
echo "== done ($(date +%H:%M))"

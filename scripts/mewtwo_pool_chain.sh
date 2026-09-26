#!/usr/bin/env bash
# Run inside devenv. Serialized GPU stages; no concurrent EXLA clients.
set -euo pipefail
cd /home/blewf/git/exphil
RUN=eval_runs/0925_mewtwo_pool
PRIOR=checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin
OLD=eval_runs/0924_mewtwo_ppo/v1
CONTROL=eval_runs/0924_mewtwo_ppo/prior_head.bin
export EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.80 EXPHIL_EXLA_PRECISION=highest
mkdir -p "$RUN"
phase() {
  python3 - "$RUN" "$1" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); phase=sys.argv[2]
state=dict(phase=phase,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           unit='exphil-mewtwo-pool-v2',log='logs/exphil-mewtwo-pool-v2.log',
           stop_file=str(root/'v2/STOP'))
(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
pathlib.Path('docs/planning/MEWTWO_POOL_LIVE_STATUS.md').write_text(
    '# Mewtwo opponent-pool experiment\n\n'+
    f'Updated: {state["updated_at"]}. Phase: **{phase}**.\n\n'+
    f'Unit: `{state["unit"]}`. Log: `{state["log"]}`.\n\n'+
    'One serialized GPU job: training → evaluation/export. No other training is launched by this chain.\n\n'+
    'Pool: frozen prior + v1 heads70/150/300; v1 head100 excluded for evaluation. '
    'KL0.01 unchanged,200 iterations maximum. This tests within-family opponent diversity, '
    'not broad character/generalist strength. No promotion.\n\n'+
    f'Graceful training stop: create `{state["stop_file"]}`; evaluation then proceeds on the saved head. '
    'Stop the systemd unit to cancel the entire chain.\n\n'+
    'Design and completed review: [September25 resume](PPO_RESUME_2026-09-25.md).\n')
PY
}
trap 'phase failed' ERR
phase training
mix run scripts/ppo_r3.exs --policy "$PRIOR" --character mewtwo \
  --critic eval_runs/0924_mewtwo_ppo/critic_v1_refit/critic_best.bin \
  --opponent-heads "$OLD/head_iter70.bin,$OLD/head_iter150.bin,$OLD/head_iter300.bin" \
  --envs 64 --frames 600 --iters 200 --epochs 2 --minibatch 4096 \
  --lr 3.0e-5 --gamma 0.995 --lambda 0.95 --kl-coef 0.01 --kl-stop 0.5 \
  --save-every 10 --seed 925 --out "$RUN/v2"
HEAD=$(python3 - "$RUN/v2" <<'PY'
import pathlib,re,sys
heads=list(pathlib.Path(sys.argv[1]).glob('head_iter*.bin'))
assert heads, 'Training produced no heads'
print(max(heads,key=lambda p:int(re.search(r'iter(\d+)',p.name)[1])))
PY
)
phase evaluation
# No selection on these games: evaluate the final/guard-stopped saved head.
# Prior-control comparisons use exactly the same opponent within each panel.
for panel in prior heldout100; do
  opponent=()
  if [ "$panel" = heldout100 ]; then opponent=(--opponent-head "$OLD/head_iter100.bin"); fi
  for arm in candidate control; do
    selected=$HEAD
    if [ "$arm" = control ]; then selected=$CONTROL; fi
    mix run scripts/ppo_r3_eval.exs --policy "$PRIOR" --head "$selected" \
      "${opponent[@]}" --games 100 --envs 50 --cap 28800 --fingerprint-games 16 \
      --seed 9251 --out "$RUN/eval/${panel}_${arm}"
  done
  python3 scripts/degeneracy_check.py --baseline "$RUN/eval/${panel}_control" \
    --candidate "$RUN/eval/${panel}_candidate" --out "$RUN/eval/${panel}_drift.json"
done
mix run scripts/ppo_export_policy.exs --policy "$PRIOR" --head "$HEAD" \
  --out checkpoints/mewtwo_ppo_pool_v2_policy.bin
phase complete

#!/usr/bin/env bash
# Bounded matched active-opponent screen. Run inside devenv after checking GPU ownership.
set -euo pipefail
out=${1:?output directory required}
shift
if [ "$#" -eq 0 ]; then set -- 1003 1004 1005 1006; fi
mkdir -p "$out"
opponent=checkpoints/coh_prev_q_ck4/model_best_policy.bin
export EDIFICE_FUSED_CUSTOM_CALL=0
for arch in min_gru mamba mamba2 gru; do
  character=fox
  envs=8
  case "$arch" in
    min_gru) policy=$opponent ;;
    mamba) policy=checkpoints/coh_fullscale_mamba_v1/model_best_policy.bin ;;
    mamba2) policy=checkpoints/mewtwo_combo_newera_mamba2_screen_best_policy.bin; character=mewtwo; envs=4 ;;
    gru) policy=checkpoints/fox_v3_20260917_050901_resume3/model_policy.bin ;;
  esac
  for seed in "$@"; do
    modes=(windowed carried resync30 resync60)
    if [ "$arch" = gru ]; then modes=(carried); fi
    for mode in "${modes[@]}"; do
      label="${arch}_${mode}_s${seed}"
      if [ -s "$out/$label.json" ]; then continue; fi
      flags=()
      if [ "$mode" != windowed ]; then flags+=(--stateful-step); fi
      if [ "$mode" = resync30 ]; then flags+=(--stateful-resync 30); fi
      if [ "$mode" = resync60 ]; then flags+=(--stateful-resync 60); fi
      elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
        scripts/sim_closed_loop.exs --policy "$policy" --label "$label" \
        --opponent "$opponent" --character "$character" --opponent-character fox \
        --envs "$envs" --frames 1800 --seed "$seed" --stage final_destination \
        "${flags[@]}" --out "$out/$label.json" 2>&1 | tee "$out/$label.log"
    done
  done
done

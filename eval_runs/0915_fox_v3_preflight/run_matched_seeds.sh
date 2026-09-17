#!/usr/bin/env bash
# Matched-seed two-epoch fits for the Fox V3 preflight, run sequentially
# under highest EXLA arithmetic. Each seed dir gets train.log + launch.json
# (env, start/end, wall seconds, exit status).
set -u
cd /home/blewf/git/exphil
root=eval_runs/0915_fox_v3_preflight
prefix="${1:-matched}"
for seed in 905 906; do
  dir="$root/${prefix}_seed_$seed"
  mapfile -t args < <(jq -r '.[]' "$dir/train_args.json")
  start=$(date +%s)
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.70 \
    EXPHIL_EXLA_PRECISION=highest mix "${args[@]}" >>"$dir/train.log" 2>&1
  status=$?
  end=$(date +%s)
  jq -n --arg seed "$seed" --arg start "$(date -d @$start -Is)" --arg end "$(date -d @$end -Is)" \
    --argjson wall $((end-start)) --argjson status $status \
    '{seed:$seed, env:{EXLA_TARGET:"cuda",EXPHIL_GPU_MEMORY_FRACTION:"0.70",EXPHIL_EXLA_PRECISION:"highest"}, start:$start, end:$end, wall_seconds:$wall, exit_status:$status}' \
    > "$dir/launch.json"
done

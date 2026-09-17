#!/usr/bin/env bash
# Launch gate on the tag-map recipe: ONE matched seed + parity + held-out.
set -u
cd /home/blewf/git/exphil
root=eval_runs/0915_fox_v3_preflight
dir="$root/tagmap_seed_905"
mapfile -t args < <(jq -r '.[]' "$dir/train_args.json")
start=$(date +%s)
devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.70 EXPHIL_EXLA_PRECISION=highest mix "${args[@]}" >>"$dir/train.log" 2>&1
status=$?; end=$(date +%s)
jq -n --arg start "$(date -d @$start -Is)" --arg end "$(date -d @$end -Is)" --argjson wall $((end-start)) --argjson status $status \
  '{seed:"905", env:{EXLA_TARGET:"cuda",EXPHIL_GPU_MEMORY_FRACTION:"0.70",EXPHIL_EXLA_PRECISION:"highest"}, start:$start, end:$end, wall_seconds:$wall, exit_status:$status}' > "$dir/launch.json"
echo "train exit $status"
for s in parity heldout; do
  t0=$(date +%s)
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.30 EXPHIL_EXLA_PRECISION=highest mix run $root/$s.exs "$dir/model_policy.bin" >"$dir/$s.log" 2>&1
  echo "$s exit $? ($(( $(date +%s) - t0 ))s)"
done

#!/usr/bin/env bash
# Offline V3 preflight gates after the matched-seed fits: trained parity and
# complete held-out scoring per seed (highest arithmetic, fresh process each),
# then the full-corpus inventory (CPU). Logs land next to each artifact.
set -u
cd /home/blewf/git/exphil
root=eval_runs/0915_fox_v3_preflight
prefix="${1:-matched}"
for seed in 905 906; do
  dir="$root/${prefix}_seed_$seed"
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.30 EXPHIL_EXLA_PRECISION=highest \
    mix run $root/parity.exs "$dir/model_policy.bin" >"$dir/parity.log" 2>&1
  echo "parity seed $seed exit $?"
  devenv shell -- env EXLA_TARGET=cuda EXPHIL_GPU_MEMORY_FRACTION=0.30 EXPHIL_EXLA_PRECISION=highest \
    mix run $root/heldout.exs "$dir/model_policy.bin" >"$dir/heldout.log" 2>&1
  echo "heldout seed $seed exit $?"
done
if [ "${2:-}" = "audit" ]; then
  devenv shell -- env EXPHIL_GPU=0 mix run $root/full_corpus_audit.exs >"$root/full_corpus_audit.log" 2>&1
  echo "full corpus audit exit $?"
fi

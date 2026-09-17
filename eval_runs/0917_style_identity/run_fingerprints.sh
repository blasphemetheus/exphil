#!/usr/bin/env bash
# STYLE_IDENTITY step 1 over the two corpora (subject Fox). Sequential.
set -u
cd /home/blewf/git/exphil
out=eval_runs/0917_style_identity
devenv shell -- env EXPHIL_GPU=0 mix run scripts/style_fingerprint.exs \
  "replays/erickfm_ranked/v2_filtered/*.slp" --subject-character Fox \
  --out $out/erickfm_fox.jsonl --quiet >$out/erickfm_fox.log 2>&1
echo "erickfm exit $?"
devenv shell -- env EXPHIL_GPU=0 mix run scripts/style_fingerprint.exs \
  "replays_root/yeti/raw/**/*.slp" --subject-character Fox \
  --out $out/yeti_fox.jsonl --quiet >$out/yeti_fox.log 2>&1
echo "yeti exit $?"

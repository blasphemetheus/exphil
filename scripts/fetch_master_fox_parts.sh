#!/usr/bin/env bash
# Data push (2026-10-31): we hold ONE of SIX master-master Fox parts (a1 = 7,911 games). This pulls the rest from
# the HF ranked dataset into /data/exphil/replays/erickfm_ranked/FOX/, extracts each (≈4.6x), deletes the tarball,
# and links every new game into replays/erickfm_master_full/ (a1 games included). Needs ~160 GB free on /data
# for all five parts (a2 alone ≈ 23 GB); run after the disk decision. Resumable (curl -C -); skips extracted parts.
#   scripts/fetch_master_fox_parts.sh [parts...]      default: a2 a3 a4 a5 a6
set -uo pipefail
base=https://huggingface.co/datasets/erickfm/melee-ranked-replays/resolve/main/FOX
dir=/data/exphil/replays/erickfm_ranked/FOX; ex=$dir/extracted; corpus=/home/blewf/git/exphil/replays/erickfm_master_full
mkdir -p "$ex" "$corpus"
for p in "${@:-a2 a3 a4 a5 a6}"; do
  tgz=$dir/FOX_master-master_$p.tar.gz; mark=$ex/.done_master-master_$p
  [ -f "$mark" ] && { echo "$p already extracted"; continue; }
  free_gb=$(df --output=avail -BG /data | tail -1 | tr -dc 0-9); [ "$free_gb" -ge 45 ] || { echo "only ${free_gb}G free on /data; need ~45G per part. stop."; exit 1; }
  echo "== $p download ($(date +%H:%M))"; curl -L -C - --retry 5 -o "$tgz" "$base/FOX_master-master_$p.tar.gz" || exit 1
  echo "== $p extract"; tar -xzf "$tgz" -C "$ex" --strip-components=0 || exit 1
  rm -f "$tgz"; touch "$mark"; echo "== $p done: $(ls "$ex" | grep -c master-master) master-master files extracted total"
done
find "$ex" -maxdepth 3 -name 'master-master-*.slp' -print0 | while IFS= read -r -d '' f; do ln -sf "$f" "$corpus/$(basename "$f")"; done
echo "corpus $corpus: $(ls "$corpus" | wc -l) games"

#!/usr/bin/env bash
# r17a resume launcher (overnight 2026-07-24). Must run inside `devenv shell`
# — the launcher needs make on PATH (GOTCHAS #60).
set -u
cd /home/blewf/git/exphil

BC=$(cat corpus/curated/r17a_bc.txt)
EXTRA="/home/blewf/Slippi/Game_20260719T213810.slp,/home/blewf/Slippi/Game_20260719T213957.slp,probes/newera8/r14/plain/p1/Game_20260718T073713.slp,probes/newera8/r14/plain/p2/Game_20260718T073726.slp,probes/newera8/r14/plain/p3/Game_20260718T073738.slp,probes/newera8/r14/debounce/p1/Game_20260718T074551.slp,probes/newera8/r14/debounce/p2/Game_20260718T074603.slp,probes/newera8/r14/debounce/p3/Game_20260718T074615.slp,probes/newera8/r16/r13/plain/p1/Game_20260723T022523.slp,probes/newera8/r16/r13/plain/p2/Game_20260723T022535.slp,probes/newera8/r16/r13/plain/p3/Game_20260723T022547.slp"

export NEWERA8_TAG=r17a
export NEWERA8_BACKBONE=mamba_2
export NEWERA8_MAX_EPOCHS=100
export NEWERA8_ROUNDS=1
export NEWERA8_DROPOUT=0.6
export PROBE_STATEFUL=0
export NEWERA8_CONVERSION_WEIGHT=3.0
export NEWERA8_OPENER_WEIGHT=3.0
export NEWERA8_PROBE_REG=0.1
export NEWERA8_PROBE_REG_EVERY=5
export NEWERA8_PROBE_EVAL=10
export NEWERA8_BC_REPLAYS="$BC"
export NEWERA8_LEARN_STYLES=1
export NEWERA8_PREFLIGHT=1
export NEWERA8_EXTRA_ROLLOUTS="$EXTRA"

exec scripts/overnight_newera8.sh

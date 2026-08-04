#!/usr/bin/env bash
# Task A (#30): reproduce the hold-B absorber OFFLINE by shifting STAGE.
#
# Live 2026-08-03/04: the bot multishines on FD and "holds shine" on
# Dreamland. Every drill fixture is FD-only, so DL is off-distribution —
# the same door into the absorber that human pressure opens. If a stand
# dummy on DL reproduces the collapse, the fight-state failure becomes a
# 60-second deterministic test instead of "schedule a friend".
#
# Design: ONE checkpoint, ONE dummy, ONE delay — stage is the only variable.
#   control:   FD  (expect ~424/min, chain ~423 for g4_d2mix)
#   treatment: DL  (expect collapse if the stage hypothesis holds)
#   extra:     BF, YS — is it "not-FD" or specifically DL? A sensitivity
#              curve tells us whether multi-stage training needs 2 stages
#              or all of them.
# Prereg:
#   P1 DL chains collapse (<50) while FD holds (>300) => stage door
#      CONFIRMED; offline repro achieved; task E (multi-stage) justified.
#   P2 DL ~= FD => the live DL failure was pressure or ruleset, not stage;
#      the offline repro must come from a moving opponent instead.
#   P3 ALL stages collapse incl. FD => something about the stage FLAG path
#      differs from netplay stage selection; investigate plumbing first.
# Scored with analyze_shine_source (CHAINS — not qtrace press counts; see
# HUMAN_PLAY_FINDINGS_2026-08-04 F5).
set -uo pipefail
cd "$(dirname "$0")/.."
export ISO="${ISO:-$HOME/isos/melee.iso}"
export DOLPHIN_DIR="$HOME/.config/Slippi Launcher/netplay-beta-nixos"
export XLA_TARGET_EVAL=cuda12
export EXPHIL_GPU_MEMORY_FRACTION=0.25
export EXPHIL_SKIP_NIF_COMPILE=1

CKPT="${CKPT:-checkpoints/ms_g4_d2mix.bin}"

run_stage () { # stage
  local stage=$1
  echo "=== STAGE-ABSORBER $stage $(date +%H:%M:%S)"
  EXLA_TARGET=host bash scripts/eval_live_protocol.sh "$CKPT" \
    "eval_runs/0804_stage_$stage" --runs 3 --dummy stand --runner sync \
    -- --frame-delay 3 --stage "$stage" \
       --headless --emulation-speed 0 --blocking-input --slippi-port 51442
}

run_stage final_destination
run_stage dreamland
run_stage battlefield
run_stage yoshis_story
echo "=== STAGE-ABSORBER done $(date +%H:%M:%S)"

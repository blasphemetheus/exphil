#!/usr/bin/env bash
# Run any `mix run …` (or other shell command) as a systemd user unit from the repo root.
#
#   scripts/launch_unit.sh UNIT 'mix run scripts/foo.exs --bar 1'
#
# Why: `systemd-run --user` starts the unit in $HOME, not the caller's cwd, so a
# relative path in the command makes the unit die instantly with no log (bit us
# four times on 2026-09-22). The command runs inside `devenv shell` from
# /home/blewf/git/exphil; stdout+stderr go to logs/<UNIT>.log.
set -euo pipefail
unit=$1; shift
cmd=$*
repo=/home/blewf/git/exphil
systemd-run --user --unit "$unit" --collect bash -lc "cd $repo && devenv shell -- $cmd > logs/$unit.log 2>&1"
sleep 8
systemctl --user is-active "$unit" || { echo "unit died; log tail:"; tail -20 "$repo/logs/$unit.log"; exit 1; }
tail -c 400 "$repo/logs/$unit.log"

#!/usr/bin/env bash
# 0 (busy) when a BEAM VM that can rebuild or is using the shared exphil/nx build is alive: a VM whose
# working directory is an exphil checkout or the nx fork AND whose parent is a mix / iex / elixir
# invocation (compiles, training units, scripts). Ignored: VMs other repos own, and the VM the
# platfighter game spawns in the exphil dir (the policy server; it never compiles — 2026-10-28, it
# held the width arm back for six hours together with an unrelated project's nix shell).
# Usage: scripts/beam_busy.sh && echo busy
for p in $(pgrep -f '[b]eam.smp'); do
  cwd=$(readlink "/proc/$p/cwd" 2>/dev/null) || continue
  case "$cwd" in /home/blewf/git/exphil*|/home/blewf/git/nx*) ;; *) continue;; esac
  parent=$(ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" 2>/dev/null)
  case "$parent" in *mix\ *|*iex*|*elixir\ *|*devenv*) exit 0;; esac
done
exit 1

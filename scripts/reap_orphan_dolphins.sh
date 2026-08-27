#!/usr/bin/env bash
# Reap orphaned Dolphin emulators + stale libmelee temp homes.
#
# The STRUCTURAL fix for orphans is libmelee_ex's spawn shim
# (Melee.Dolphin, 2026-08-13): Dolphin now dies with its beam on ANY
# death mode, including kill -9 / SIGBUS. This script is belt-and-braces
# for what the shim can't cover: pre-shim leftovers, dolphins launched
# by other paths (manual runs, the legacy python bridge), and the
# /tmp/libmelee_* home dirs they leave behind (4 orphans burned ~5
# cores for 2 days before being noticed, 2026-08-11..13).
#
# Orphan := a dolphin-emu process using a /tmp/libmelee_* user dir whose
# parent is init/systemd (PPID 1) — its launching beam is gone. Live
# sessions (parented by a beam or the spawn shim) are never touched.
#
#   bash scripts/reap_orphan_dolphins.sh          # report + kill
#   bash scripts/reap_orphan_dolphins.sh --dry    # report only
set -euo pipefail
DRY=${1:-}

reaped=0
while read -r pid ppid cmd; do
  case "$cmd" in
    *dolphin-emu*-u\ /tmp/libmelee_*)
      if [ "$ppid" -eq 1 ]; then
        echo "orphan: pid $pid ($cmd)" | cut -c1-120
        if [ "$DRY" != "--dry" ]; then
          kill "$pid" 2>/dev/null || true
          reaped=$((reaped + 1))
        fi
      fi
      ;;
  esac
done < <(ps -eo pid=,ppid=,args= | sed 's/^ *//')

# Stale temp homes: no live process references the dir and it's >1 day old
stale=0
for dir in /tmp/libmelee_*; do
  [ -d "$dir" ] || continue
  if ! pgrep -f "$(basename "$dir")" > /dev/null 2>&1; then
    if [ -n "$(find "$dir" -maxdepth 0 -mtime +1 2>/dev/null)" ]; then
      echo "stale home: $dir"
      if [ "$DRY" != "--dry" ]; then
        rm -rf "$dir"
        stale=$((stale + 1))
      fi
    fi
  fi
done

echo "reaped $reaped orphan(s), removed $stale stale home(s)${DRY:+ (dry run)}"

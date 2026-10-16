#!/usr/bin/env bash
# Serve the exphil review pages with the melee-sim-light viewer (2026-10-16).
# Builds a static root of symlinks under ~/.cache/exphil/viewer_root and runs
# python's http.server on it as the user unit exphil-viewer-8012:
#
#   /tools/      melee-sim-light's built viewer (slippi-viewer dist + zips, msltrace1.js)
#   /coach/      priv/viewer/coach   (coach page + the pane every page embeds)
#   /ga/         priv/viewer/ga
#   /death-review/  priv/viewer/deaths
#   /labels/     priv/viewer/labels  (label / DAgger / teacher review, one run at a time)
#   /runs/ /garuns/ /labelruns/ /death-cases/  -> eval_runs (any export dir under it)
#
#   scripts/viewer_serve.sh            # (re)start; prints the URLs
#   scripts/viewer_serve.sh stop
set -euo pipefail
repo=/home/blewf/git/exphil
msl=${MSL_VIEWER_BUILD:-/home/blewf/git/melee-sim-light/build/viewer/tools}
root=${EXPHIL_VIEWER_ROOT:-$HOME/.cache/exphil/viewer_root}
port=${EXPHIL_VIEWER_PORT:-8012}
unit=exphil-viewer-$port
if [ "${1:-}" = stop ]; then systemctl --user stop "$unit" 2>/dev/null || true; echo "stopped $unit"; exit 0; fi
[ -f "$msl/viewer/msltrace1.js" ] || { echo "no built viewer at $msl (make viewer-build in melee-sim-light, or set MSL_VIEWER_BUILD)"; exit 1; }
mkdir -p "$root"
ln -sfn "$msl" "$root/tools"
ln -sfn "$repo/priv/viewer/coach" "$root/coach"
ln -sfn "$repo/priv/viewer/ga" "$root/ga"
ln -sfn "$repo/priv/viewer/deaths" "$root/death-review"
ln -sfn "$repo/priv/viewer/labels" "$root/labels"
for d in runs garuns labelruns death-cases; do ln -sfn "$repo/eval_runs" "$root/$d"; done
systemctl --user stop "$unit" 2>/dev/null || true
systemd-run --user --unit "$unit" --collect --working-directory="$root" python3 -m http.server "$port" --bind 127.0.0.1 >/dev/null
sleep 1
systemctl --user is-active "$unit" >/dev/null || { echo "unit $unit failed"; exit 1; }
echo "serving $root on http://127.0.0.1:$port/"
echo "  labels:  http://127.0.0.1:$port/labels/?run=/labelruns/<export dir under eval_runs>"
echo "  coach:   http://127.0.0.1:$port/coach/?run=/runs/<dir>     ga: /ga/?run=/garuns/<dir>"

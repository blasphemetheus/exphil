#!/usr/bin/env bash
# GOTCHA #114: on this rig Xwayland :0 keeps listening on /tmp/.X11-unix/X0_
# while the X0 path that clients use gets unlinked (three times on 09-10/11,
# each time while Bradley was away from the machine; deleter unidentified).
# Every X11 client — both Slippi Dolphin builds — then fails with
# "could not connect to display :0", which the bridge reports only as a
# console-connect timeout. Repair the path without restarting anything.
d=/tmp/.X11-unix
if [ ! -e "$d/X0" ] && [ -S "$d/X0_" ]; then
  ln -s X0_ "$d/X0" && echo "[x11_socket_fix] restored $d/X0 -> X0_" >&2
fi

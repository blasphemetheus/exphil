"""Launch the captured BEAM command directly under Nsight Systems (2026-09-30).

Sibling of mamba_sanitizer_direct.py: injection tools break `mix`/erlexec, so
the BEAM is exec'd with BINDIR/ROOTDIR from crash/beam_argv.json. Default
script = the race reproducer with the pre-fix workspace path restored
(EDIFICE_SSB_WORKSPACE=pool), which crashes within minutes: the report then
shows which stream each backward kernel and each cudaFreeAsync landed on and
what the other threads enqueued between them.

  devenv shell -- env EDIFICE_FUSED_CUSTOM_CALL=1 EXLA_TARGET=cuda \
    EXPHIL_EXLA_PRECISION=highest EXPHIL_GPU_MEMORY_FRACTION=0.45 \
    EDIFICE_SSB_WORKSPACE=pool \
    python3 scripts/mamba_nsys_direct.py NSYS_PATH [OUT_REPORT] [SCRIPT ARGS...]

nsys is not in devenv; get it with
  nix shell nixpkgs#cudaPackages.nsight_systems -c which nsys
and pass that path. Run only with no other beam alive. On the Xid crash nsys
still finalizes the .nsys-rep (it flushes on process exit); if it does not,
rerun with --duration to stop before the crash and inspect the steady state.
"""
import json
import os
from pathlib import Path
import sys

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

nsys = sys.argv[1]
report = sys.argv[2] if len(sys.argv) > 2 else "eval_runs/0925_fox_mamba/crash/nsys_pool_crash"
args = json.loads(Path("eval_runs/0925_fox_mamba/crash/beam_argv.json").read_text())
os.environ.update(BINDIR=args[args.index("-bindir") + 1],
                  ROOTDIR=args[args.index("-root") + 1], EMU="beam", PROGNAME="erl")
os.environ.setdefault("EDIFICE_SSB_WORKSPACE", "pool")
idx = args.index("scripts/mamba_crash_stream.exs")
args[idx:] = sys.argv[3:] or ["scripts/mamba_race_probe.exs",
    "eval_runs/0925_fox_mamba/crash/step600",
    "eval_runs/0925_fox_mamba/crash/race_nsys_pool", "real", "6000"]
Path(report).parent.mkdir(parents=True, exist_ok=True)
os.execv(nsys, [nsys, "profile",
                "--trace=cuda,nvtx,osrt",
                "--cuda-memory-usage=true",
                "--cuda-um-cpu-page-faults=false",
                "--sample=none", "--cpuctxsw=none",
                "--force-overwrite=true",
                "-o", report, *args])

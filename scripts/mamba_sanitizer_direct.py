"""Launch the captured BEAM command directly, bypassing shell/erlexec tracing.

Diagnostic only. Requires crash/beam_argv.json captured from the same dev shell.
"""
import json
import os
from pathlib import Path
import sys

sanitizer = sys.argv[1]
args = json.loads(Path("eval_runs/0925_fox_mamba/crash/beam_argv.json").read_text())
os.environ.update(BINDIR=args[args.index("-bindir") + 1],
                  ROOTDIR=args[args.index("-root") + 1], EMU="beam", PROGNAME="erl")
idx = args.index("scripts/mamba_crash_stream.exs")
args[idx:] = sys.argv[2:] or ["scripts/mamba_crash_probe.exs",
    "eval_runs/0925_fox_mamba/crash/step600",
    "eval_runs/0925_fox_mamba/crash/direct_sanitizer", "fixed", "2"]
os.execv(sanitizer, [sanitizer, "--target-processes", "application-only",
                    "--tool", "memcheck", "--track-stream-ordered-races", "all",
                    "--report-api-errors", "no", "--error-exitcode", "99", *args])

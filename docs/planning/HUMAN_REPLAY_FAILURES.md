# Human replay failures: signed-zero arithmetic compatibility

2026-09-15. Follow-up to `FLOAT_INPUT_INJECTION_REVIEW.md`.

## Finding

The four failures were four prefixes of one human Fox-vs-Fox recording,
`eval_runs/local_multishine_20260913_224806/2026-09-Mainline/Game_20260913T224817.slp`,
with handoffs 1155, 4234, 4452, and 1319. They were not four independent games.

The complete diagnostic found exactly one differing field: port 1's vertical
knockback velocity, `speed_y_attack`, on frames **783–800**. Its original f32
bits are `80000000` (-0.0); the old JIT records `00000000` (+0.0). Every other
exposed state field matches through frame 4451. Both ports' recorded inputs,
physical fields, raw sticks, and RNG match throughout each prefix.
The shorter human prefixes (handoffs 302 and 744) never reach the discrepancy.

`eval_runs/0915_float_input/human_audit.exs` reproduces the full-field diagnostic;
`human_audit.json` stores the results, including float bits so JSON numeric
equality cannot hide the sign. The production audit remains bit-exact.

## Root cause and fix

PowerPC `fnmsub` evaluates `-(a*c - b)`. Ishiiruka's JIT evaluated `b - a*c`,
including using x86 `VFNMADD` on FMA-capable hosts. With exact cancellation
under the normal rounding mode, the former yields negative zero and the
latter positive zero. Melee's airborne vertical knockback decay uses
`fnmsubs` at `0x8006B9E4`, then stores it at fighter offset `0x90`. The Slippi
post-frame recorder copies this word directly.

Changing only the CPU engine to the interpreter makes the 1,194-frame human
prefix exact. Correcting the JIT subtraction/negation makes all six human
prefixes exact; forcing the non-FMA JIT path also fixes the 1,194-frame case.
No input injection, state restoration, or audit tolerance was changed.

However, **an unconditional arithmetic correction breaks compatibility with
older CPU recordings**. Checking the longest mined prefix of each of the
13 sources rejected seven of the twelve CPU games, with zero-sign differences
in the opposite direction. This supersedes the initial three-game CPU smoke
check, which passed. The failed unconditional experiment is retained as
`nmsub_longest`; it is not a release configuration.

The final implementation therefore adds **`AccurateNmsub`**, default false,
to Dolphin's `[Core]` configuration. It preserves the historical JIT behavior
unless explicitly enabled. libmelee writes the selected value on every
session, including false, preventing stale settings from leaking between runs.

## Usage

The new installation is
`~/.local/share/slippi/exi-ai-float-nmsub-option-v2/dolphin-emu-headless`.
Its SHA-256 is
`b694802a4210a5b0b5b6307713fc05e7596d57e6c7b4b44ae09aa7dc70bf644c`.
The protocol, assembly hook, and Gecko hashes are unchanged.
The default experimental `exi-ai-float` symlink now points to this installation;
all older installations and the production `exi-ai-flush` are retained.

- For these older CPU sources, retain the default arithmetic.
- For this Mainline human source, add `--accurate-nmsub` to the float-input
  suite invocation.
- Mixed-source manifests may specify `"accurate_nmsub": true` or `false` per
  entry, overriding the CLI default. This is an explicit source compatibility
  declaration; do not infer it from whether the players are human or CPU.
- The suite rejects the accurate option before launch if the installation
  manifest does not advertise support. Each run records its selected mode.

`nmsub_matched_manifest.json` declares the appropriate mode for the longest
prefix of each source. These intervals cover all 77 originally mined prefixes.
`nmsub_profiles.exs` reproduces those declarations and the full 77-entry
`nmsub_compatible_mined.json` for future experiments.
`nmsub_matched.sh` runs that matrix; `human_option.sh` separately runs all six
human prefixes using the public CLI flag. These are prefix regression checks,
not a repeat of the policy-training experiment. Their short response windows
do not measure the original chain-length gate.

The final source matrix passed **13/13 exact prefixes**: all twelve CPU games
with legacy arithmetic and the human game with accurate arithmetic. The CLI
check also passed **6/6 exact human prefixes**, resolving all four failures.
The dependency commits are Dolphin `42f7c9d94` and libmelee `9d0f4af`.
The final Elixir checks passed: 11 ExPhil tests and 75 libmelee tests plus one doctest.
The earlier full libmelee suite also passed before the arithmetic option was
added. Results are summarized in `human_checks_summary.json`.

## Upstream recommendation

Current Dolphin already handles signed-zero semantics explicitly in
[its JIT implementation](https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/Core/PowerPC/Jit64/Jit_FloatingPoint.cpp#L806).
The [Ishiiruka fork still contains the old calculation](https://github.com/vladfi1/slippi-Ishiiruka/blob/slippi/Source/Core/Core/PowerPC/Jit64/Jit_FloatingPoint.cpp#L290).
Propose a compatibility-aware backport there, with the interpreter/JIT
comparison and an explanation of why changing the default invalidates old
recordings. This is not a new bug to report against modern Dolphin.

Propose the direct processed-input protocol and ASM hook together to the AI
forks, in separate coordinated PRs. Keep the arithmetic fix independent.
The receiver depends on this fork's earlier direct-channel work, so an upstream
PR must include or depend on those prerequisites; do not blindly cherry-pick
the entire local branch or the ExPhil experiment archive.
libmelee_ex and ExPhil are already published in their primary repositories.
No external PR has been opened.

The trained candidate and original coverage results remain unchanged. Using
the newly recovered human examples in training would be a separate experiment.

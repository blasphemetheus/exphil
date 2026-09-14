# Recorded input verification

The scenario verifier compares the game's recorded **readback**, not literal
requested analog values. It remains fail-closed for missing/reordered frames,
incorrect timing, button mismatches, and unexpected analog values.

## Standard quantized pipe contract

- Stick requests map to integer axes using half-even rounding of
  `(value - 0.5) * 160`.
- Melee clamps the two-dimensional vector to radius 80, truncating resulting
  integer components. Each component with absolute value below 23 reads neutral.
- Remaining components normalize as `0.5 + component / 160`.
- Analog shoulder goes to L, quantized to 140 steps. Digital L/R clicks each
  force that side's trigger readback to 1 independently. Never sum the sides.
- Comparison tolerance is 0.00001 (floating-point readback), not the old 0.02.

Source investigation: local Melee `gm/gmmain.c` and
`sysdolphin/baselib/controller.c`; Dolphin `ControllerInterface/Pipes/Pipes.cpp`
and `ControllerEmu.h`. Live recordings independently pin the complete policy
17-by-17 stick grid, added deadzone boundary points, shoulder buckets, and
digital L/R combinations. Both main and C sticks are checked.

## Transport correction and historical profiles

The old libmelee_ex pipe shoulder path sent the raw Slippi byte fraction through
a bipolar `Axis +` binding. Requests 0/0.25/0.5/0.75 produced zero; request 1
produced approximately 0.178571. This was a transport bug, not normal deadzone
behavior. `Melee.Controller.fix_pipe_analog_trigger/1` now inverts that binding;
`fix_analog_trigger/1` remains unchanged for raw Slippi pad serialization.

- `pipe_v2` is the default corrected transport contract.
- `pipe_v1` is an explicit historical profile for old pipe recordings. It is
  never auto-selected by whichever profile best matches observed data.
- Reports identify the profile and verifier version, and include component-level
  mismatch examples. Old scoreboards are not overwritten by re-verification.
- This does not certify unquantized inputs, other pad mappings, direct transports,
  arbitrary game builds, or unrelated online timing configurations.

Native dependencies need not rebuild, but the corrected **Elixir Controller
module must load** along with the verifier. The suite rejects a stale Controller
without the new pipe conversion API rather than claiming the new contract.

## Reproduce

With the updated modules compiled or explicitly loaded:

```bash
mix run scripts/probe_input_readback.exs --out NEW_PROBE_DIRECTORY
mix run scripts/validate_input_readback.exs \
  --input NEW_PROBE_DIRECTORY/readback.json --profile pipe_v2 --out NEW_REPORT.json
mix run scripts/reverify_scenario_inputs.exs \
  --scores OLD_SCOREBOARD.json --profile pipe_v1 --out NEW_TIMING_REPORT.json
```

Do not overlap the live probe with another Dolphin job. Probe paths and report
files must be new. Artifacts: `eval_runs/0913_analog_legacy/` and
`eval_runs/0913_analog_fixed/`, 549/549 sent and issued comparisons passing each
under their declared profiles. Curated independent readbacks and replay hashes:
`test/fixtures/analog_readback.json`.

## Matched cold-history evaluation

Use `scripts/scenario_suite.exs --prefix-history cold` for the boundary-fixed
tiny-overfit cold-start training contract. Policy prefix observations are skipped,
and the real agent's buffers, cached action, and controller queue reset at handoff.
Physical prefix replay and pending delayed-input delivery remain unchanged.

`applied` and `committed` retain warmed-history behavior; defaults are unchanged.
Scores must identify which convention was used. Cold success is not evidence of
continuous recovery with a populated recurrent history.

Live smoke: `eval_runs/0913_cold_history.json` uses the saved untrained checkpoint
at neutral-start 2228 and recovery 4. Both have complete 120-frame responses,
zero drift/errors, and valid input verification. Their behavior fails, as expected
for an untrained model; this validates mechanics, not a trained bot.

The existing ep57 warmed-history positive control also still passes: chain 14,
120 response frames, zero drift/errors, and valid pipe_v2 verification in
`eval_runs/0913_readback_positive_control.json`.

Validation: 18 Exphil regression tests pass; sibling controller/raw-pad tests
pass 9 doctests, 2 properties, and 18 tests. Native dependencies were not rebuilt.

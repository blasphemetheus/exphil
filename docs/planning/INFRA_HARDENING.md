# Infrastructure hardening plan (2026-09-01, post-GOTCHA #107)

Bradley's directive: actively hunt for bugs in the #107 class, and harden —
turn copy-pasted script logic into unit-tested library functions.

## 1. The class, precisely

GOTCHA #107's shape: a CONVENTION (subject = port 1, opponent = port 2) is
hardcoded in downstream consumers (embedding `own_port=1`, AWBC
`standard_rewards(port \\ 1)`), while upstream grows features that break the
convention (per-file port resolution) — and nothing asserts it. Silent
keyword defaults (`opponent_port: 2`) let call sites half-migrate. Smoke
tests verified plumbing (proportions, loss descending), not semantics
(frame content) — and loss DESCENDS fine on wrong-slot data.

## 2. AUDIT RESULT — the class has a SECOND LIVING INSTANCE (E1c)

Read-only sweep 09-01 evening (~34 scripts resolve non-port-1 subjects):

**Sub-class A — raw-frame scorecards: CLEAN.** defense/commitment/
neutral-range scorecards, situation_hist, exchange/punish tools read
`game_state.players[port]` with the resolved port passed explicitly and
never embed. Their numbers stand.

**Sub-class B — embedding-path instruments: SWAPPED PERSPECTIVE on
non-port-1 files (~44% of fox-resolved expert files).** Anything that
resolves a fox port and then goes through `Activations.capture_replay` /
`embed_frames` / `CriticFeatures.extract_replay`: frames keep ACTUAL port
keys, `Data.precompute_frame_embeddings` → `embed_states_fast(states, 1)`
puts the real opponent in the OWN slot and the imitated fox in the
OPPONENT slot. (Unlike the training-path #107, the opponent is present —
ports were passed — but the perspective is inverted.)

Affected instruments and standing results (all DILUTED ~44%, not void —
p1 files + dittos are correct; several validated against live data):
- `scripts/lib/critic_features.exs` → critic extracts, D2 critic, ladder
  runs (`0831_critic_ar`, `0831_critic_refit`)
- `scripts/train_ar_head.exs` → the 8a heads, v1.2 refit, v1.3 refit
  (chain stage 3 — left running deliberately: v1.2 and v1.3 refits share
  the contamination, preserving the pre-registered v1.1→v1.3 comparison)
- `scripts/coincidence_probe.exs` → L_cond/R_state numbers
- `scripts/dynamics_spike.exs` → the G3b dynamics model + saved
  `dynamics_fox_v11AR.bin`; `dynamics_action_sensitivity.exs`
- `scripts/vrollout_eval.exs` (plan-c campaign)
- Leg S AR runs to the extent they embedded fox-resolved expert files
CLEAN by construction: position probe + sweep (livelook replays, port 1),
F3c scorecard (sub-class A), anything on port-1-only corpora or livelook.

**Consequence policy:** re-baseline, don't re-litigate. After the fix,
rerun the cheap standing probes (coincidence, action-sensitivity) and the
refit (v1.3b) on clean captures; retrain dynamics; flag the critic-ladder
absolute numbers as diluted in their RESULTS files. Directional verdicts
that validated against live play (probe↔live lift) likely survive.

## 3. Fix design (QUEUED — no lib edits while `v13-portfix` runs, L7)

One choke point covers most of sub-class B:
1. `Activations.capture_replay/3` and callers of `to_training_frames`
   inside lib: pass `remap_ports: true` (the #107 fix's remap).
2. `CriticFeatures.extract_replay`: same (it parses independently).
3. Scripts that parse directly (coincidence_probe, dynamics_spike,
   vrollout_eval, my scripts this week): migrate onto the new library
   functions (below) instead of patching 30 call sites.

## 4. Library-ification (the copy-paste census)

Each block below is copy-pasted 4–30× across scripts/ — every copy is a
place the next convention change silently misses:

| new function | replaces | copies |
|---|---|---|
| `ExPhil.Eval.Corpus.resolve_subject(path, char_id)` | the metadata→single-port-of-character→skip-dittos block | ~10 |
| `ExPhil.Eval.Corpus.load_frames(path, port, opts)` | parse → to_training_frames(remap!) → reject frame<0 | ~30 |
| `ExPhil.Eval.Corpus.pick_files(glob, char_id, n)` | wildcard→resolve→take | ~8 |
| `ExPhil.Eval.Corpus.embeds(frames, policy_config)` | embed_frames → embedded_frames rank-normalize (the dynamics-spike crash was exactly this misuse) | ~5 |
| `ExPhil.Eval.Corpus.rtg(frames, port, gamma, horizon)` | standard_rewards→return_to_go | ~4 |

All with unit tests against a SYNTHETIC PORT-2 FIXTURE — promote the
`synthetic_replay` builder from `character_port_test.exs` into
`test/support/replay_fixtures.ex` (port-2 subject, distinct characters,
known positions) so every test can assert "subject in slot 1, opponent
present, distance > 0" through any path.

## 5. Invariant enforcement (make the class impossible, not just fixed)

1. **Stamp the convention**: `to_training_frames` sets
   `game_state.own_port = 1` after remap (field exists, nil offline
   today). `embed_states_fast/embed` RAISES if `own_port` is set and ≠
   the own_port argument. A swapped perspective becomes a loud crash at
   the first embedded frame instead of a silent 44% corruption.
2. **Kill the silent default**: `to_training_frames` warns (once per
   process) when `player_port != 1` and `remap_ports` is not passed —
   half-migrated call sites self-report.
3. **`scripts/corpus_doctor.exs`** (semantic smoke, GOTCHA #107 rule 2 as
   a command): given a glob + char, sample N files and print players-key
   histogram, own-slot character histogram, distance==0 rate, zero-self
   embed-block rate. RULE: run before any training launch that touched
   the loader, and paste its output in the launch log.
4. **Property test**: embedding a port-2 file with remap == embedding the
   port-swapped copy of the same data as port-1 (bitwise on the player
   blocks).

## 6. Testing philosophy changes (the cheap rules)

- "Loss descending" is NEVER corpus validation (GOTCHA #107 rule 2).
- A smoke for a data change INSPECTS one non-default file's frames.
- New convention-crossing features get an adversarial fixture test
  (the port-2 fixture) before first training use.
- Script → library rule of thumb: the third copy-paste of a block is the
  signal to lift it into lib with a test (this week alone: 4 copies of
  embeds_and_frames, 3 of resolve, 2 of window-stacking).

## 7. Order of work (after v13-portfix completes)

1. Empirically confirm E1c on one port-2 file through capture_replay
   (5 min — the mirror of the 09-01 port_check).
2. Lib fix (capture_replay remap + own_port stamp + embed assert) +
   fixture module + tests.
3. `ExPhil.Eval.Corpus` + migrate the 5 instruments that feed standing
   numbers (critic_features, train_ar_head, coincidence_probe,
   dynamics_spike, vrollout_eval).
4. corpus_doctor.exs.
5. Re-baseline: v1.3b refit on clean captures → coincidence probe →
   retrain dynamics model → mark diluted RESULTS files.

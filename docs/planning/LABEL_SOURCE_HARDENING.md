# Label-source contract hardening

Implemented directly in the checkout while Claude owns the G26 guard work.
Only `Training.Labels` and `Training.Data.shift_actions` were changed in the
runtime path. No Mix build or edit to the trainer/guard/queued readout was made.

## Defect and resulting behavior

Both the list source resolver and the expert-shift guard previously looked at
the first frame. A recorded frame followed by expert frames could therefore
send expert labels through the recorded-future shift. The new resolver checks
every frame and rejects mixed sources, mixed expert modules, and malformed tags.
This also applies at delay zero: identity operations cannot carry malformed
source lists past a validation boundary.

Recorded shifts now check every intermediate frame counter. For example,
`[0, 4, 2]` previously passed a delay-two endpoint check despite crossing a
discontinuity; it now yields no shifted example.

## Compatibility and explicit strict modes

- Legacy untagged lists still mean recorded data. A partially untagged expert
  list is a source conflict. If *all* provenance is stripped, legacy mode cannot
  infer that the list used to contain expert labels.
- `Labels.source(frames, require_tagged: true)` and
  `Labels.at_delay(frames, delay, require_tagged: true, ...)` reject missing
  provenance, including at delay zero. Use this at ingestion boundaries for
  exports whose schema guarantees tags. Existing legacy loaders are not silently
  switched to this mode.
- `require_projection: true` requires a loaded expert module to export
  `label_ahead/4`. Otherwise the default remains its current commitment for
  compatibility with existing non-projecting experts. This checks callback
  availability, not correctness of the returned future.
- `off_loop: :drop` remains the way to omit states the expert does not identify
  as on-loop. Unknown off-loop modes, invalid delays, and unexpected expert
  result shapes now fail explicitly instead of silently producing dropped data.
- `Labels.tag/2` is an intentional relabeling operation; callers remain
  responsible for the truth of a newly assigned source. Serialization tests
  verify that ordinary term round trips preserve existing tags.

## Validation

Regression tests cover recorded-first and expert-first mixtures, delays zero
and positive, partial and complete tag loss, malformed tags, multiple experts,
serialization, reset/gap combinations, valid recorded successors, unsupported
projection, and invalid callback results. Existing real-fixture expert/label
tests and the general data suite are run alongside them.

Tests compile the edited source in a separate Elixir process against independent
copies of dependency beams/native libraries, with EXLA explicitly configured
for CPU. This does not recompile or replace the native libraries in the active
checkout's build. It is not a clean dependency bootstrap.

Result: 22 label/expert/contract tests passed. The expanded run had 85 tests
with one failure: `DataTest`'s process-dictionary cache test looks up
`:frames_array_cache`, while the implementation uses a size-keyed tuple.
The unchanged baseline reproduces the same failure (63 tests, one failure).
The other 62 data tests pass with the changes. This unrelated test was not edited.

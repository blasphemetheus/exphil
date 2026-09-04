# Style identity: re-identification + perceived-player clustering

Direction set by Bradley 2026-09-04. Status: PLANNED (blocked on GPU-free
for the metadata probe; design ready). Standing example player:
`[INFP]` = Michael Schlag (Bradley's friend) — use for all
player-targeted examples (`--style-tag INFP`, "does it play like
Michael when conditioned").

## Problem

The ranked corpus is anonymized ("Master Player", hashed filenames) —
name conditioning currently collapses ~8k anonymous games into ONE fake
identity (id 0), reproducing the averaging-into-blandness problem inside
the anonymous bucket. Meanwhile the partner corpus has real filename
tags (72 in the v1.5 registry; thousands of tagged games in
FALCO/ZELDA_SHEIK).

## Plan (in order)

0. **Cheap probe FIRST** (needs mix; run when no beam is live): check
   anonymized files' metadata for surviving connect codes / stable
   per-player residue. If anonymization only scrubbed display names,
   no ML is needed. `Peppi.metadata` over a sample of
   `master-master-*` files, diff all fields across files.
1. **Fingerprints — BUILT 09-04 (tests queued behind the v15 beam)**:
   `ExPhil.Interp.StyleFingerprint` (rides `ExPhil.Options.events/2`) +
   `scripts/style_fingerprint.exs` (one JSONL row per game with the
   FilenameTags label) + `test/exphil/interp/style_fingerprint_test.exs`.
   ~45 features: option rates/min (dashdance, wavedash, rolls, ...),
   forced-choice mixes (tech, ledge, throw direction), aerial mix +
   c-stick attribution, and controller micro-mechanics — X-vs-Y jump
   button (near-biometric), per-button press rates, light-shield frac,
   c-stick reliance, 3x3 stick occupancy, input-rhythm mean/CV.
   `vector/1` + `distance/2` are the matcher primitives. Candidate v2
   features (need animation analysis, deferred): L-cancel timing,
   wavedash airdodge-angle, DI-in-hitstun tendencies, OOS latency,
   SH/FH ratio (jumpsquat-length dependent).
2. **Calibrate on tagged games** (ground truth for free): split each
   tagged player's games; same-player vs different-player distance
   distributions -> retrieval accuracy + a confidence scale for match
   claims. Optional upgrade: metric-learned embedding (contrastive on
   tagged pairs) — usually beats hand features.
3. **Perceived players**: cluster anonymous-game fingerprints
   (HDBSCAN/k-means, k via silhouette on the tagged-side clusters) ->
   pseudo-tags `~cluster07`. Emit a `game_path -> pseudo_tag` map
   (JSON). LOADER ALREADY SUPPORTS THIS: `maybe_filename_tags`
   (streaming.ex) is a per-file tag override seam; the registry holds
   synthetic tags fine; cache key already includes the registry.
4. **Re-identification (the funny part)**: anonymous games vs tagged
   centroids -> ranked hypotheses ("this master-master game sits
   inside [INFP]'s cluster"). Scope: WITHIN-corpus, pseudonymous tags
   only — style analysis, not unmasking.

## Ditto handling (Bradley 09-04: "deal with dittos intelligently")

MEASURED 09-04: the "filename `A + B` = port 1 + 2" convention holds
only **69.2%** (386/558 non-ditto MARTH-fox files,
`--validate-order`) — positional ditto tags would mislabel ~31% of
frames, worse than no label. Therefore:
- Training + registry: dittos stay ANONYMOUS (subject_tag/2 semantics).
- `FilenameTags.subject_tag/3` (positional) exists but is
  matcher-gated — never feed it to training labels.
- **The intelligent path**: a ditto file names its own TWO candidate
  tags; fingerprint rows for ditto ports carry `candidates: [A, B]`
  and the calibrated matcher assigns each side 2-way (vastly easier
  than open-set). Assignments flow back via the `--player-tag-map`
  override — same seam as perceived-player clusters.

## Wiring notes

- Pseudo-tag map consumption: extend the loader override to consult an
  optional `--player-tag-map <json>` (path -> tag) BEFORE the filename
  fallback. ~20 lines in streaming.ex + a flag.
- Registry: pseudo-tags count toward the 112 cap — budget (e.g. top-40
  clusters + real tags). Anonymous games outside any cluster stay id 0.
- Eval hook: conditioning sanity check = generate with
  `--style-tag INFP` vs `--style-id 0` and diff the instrument panel
  (does the conditioned arm move toward Michael's fingerprint?).

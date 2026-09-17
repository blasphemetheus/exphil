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

## Tags are EVIDENCE, not ground truth (Bradley 09-04)

Two real-world failure modes: people enter the WRONG nametag
(borrowed setup profile), and DIFFERENT people share a tag (4-char
collisions). So the identity model is:

- **Entity** (latent player) is the unit of identity; the registry
  ultimately keys on entity ids. **Tags are aliases** attached to
  entities, many-to-many both ways.
- Three noisy observations per game: tag, fingerprint, session context
  (timestamp adjacency on a setup — consecutive local games are strong
  same-player evidence; fingerprint rows carry `started_at` for this).
- **Reconciliation report** (post-clustering, thresholds from the step-2
  calibration):
  - Per TAG: cluster purity. A tag spanning >1 well-separated style
    cluster (within-character) = COLLISION suspect -> split into
    entities `YETI/1`, `YETI/2`.
  - Per GAME: tag-vs-cluster agreement. A tagged game far from its
    tag's entity cluster = MISLABEL suspect -> flag; reassign only
    above the calibrated confidence bar.
  - Suspects surface to Bradley for adjudication (he knows the scene —
    the human oracle names/merges/splits entities).
- SubjectResolver's `:identity` rung resolves the ROLE by tag and is
  marked `provenance: :identity`; the fingerprint audit is what
  upgrades/downgrades trust in that claim — provenance exists exactly
  so mislabeled-tag games can be found and re-resolved later.
- **Same tag, different characters (same player)**: fingerprint
  features split into character-DEPENDENT (option rates, aerial mixes
  — cluster WITHIN character) and character-INVARIANT (controller
  micro: jump button, light-shield, stick occupancy, rhythm — the
  hands travel with the person). Entities LINK across characters via
  shared tag + `StyleFingerprint.invariant_distance/2`; a tag whose
  cross-character link fails the invariant check is a collision
  suspect instead ("YETI-on-Fox" vs "YETI-on-Marth" as different
  people).

## Wiring notes

- Pseudo-tag map consumption: extend the loader override to consult an
  optional `--player-tag-map <json>` (path -> tag) BEFORE the filename
  fallback. ~20 lines in streaming.ex + a flag.
- Registry: pseudo-tags count toward the 112 cap — budget (e.g. top-40
  clusters + real tags). Anonymous games outside any cluster stay id 0.
- Eval hook: conditioning sanity check = generate with
  `--style-tag INFP` vs `--style-id 0` and diff the instrument panel
  (does the conditioned arm move toward Michael's fingerprint?).

## 2026-09-17 status update (V3 preflight session)

**The premise above was partly wrong: the name channel was DEAD, not
merely collapsed.** GOTCHA #117: in-file cartridge tags are full-width
(`ＦＯＸ`); `placeholder?` was false, the filename tag was never
substituted, the ASCII registry matched nothing — 0/61 tagged files in
the V3 subset trained with a non-zero id. Fixed at `Peppi.get_player_tag`
(normalize). Every pre-09-17 style/context readout on erickfm data
(w180 context lever, g25a opp-context, v1.1 "conditioning real-but-
diffuse") was made with `name_id == 0` everywhere and needs re-reading.

Two more registry defects fixed the same night: the registry was built
from FILENAME tags while frames carry the IN-FILE tag when one exists
(16 files with `????`-mangled filenames would mismatch) — now both use
`Streaming.subject_tag/3`; and `from_tags` kept the first 111 tags in
file order — now ordered by game count (test: "registry ids follow game
count"). The trainer prints `player registry: K of N tags kept (a/b
tagged games; c anonymous)`.

Measured on the full V3 corpus (28,452 Fox games, scratch `corpus_tags.exs`):

| | |
| --- | --- |
| Games with a resolvable subject tag | 3,605 (12.7 %; 12.3 % of frames) |
| Anonymous | 7,853 hashed `master-master-*` ("Master Player") + ~17k untagged partner files |
| In-file vs filename tag, both present | 2,890 agree, 16 disagree (all filename-mangled non-ASCII) |
| Distinct train tags | 368 for 111 slots; top-111 covers 2,917/3,605 tagged games; 62 singletons |
| Top tags | 314 (281), SKWA (160), CUMB (130), C2 (129), NELL (107), E4F4 (98), LI (91), EASY (89), FOX (82), PNUT (79) ... INFP (54) |
| Last-16 validation files | **0 tagged** — the full-run held-out monitor cannot test conditioning; anonymous == registry there is EXPECTED |

### Name-channel verification still owed

1. **Liveness as a standing gate**: `heldout.exs`-style paired scoring
   (anonymous vs registry) on a *tagged* held-out sample — assert the
   scores differ on tagged files and are identical on untagged ones. The
   V3 last-16 split has no tagged file, so the trained V3 artifact needs
   a separate tagged eval set (e.g. 16 games of the top-10 tags held out
   by hash from `full_corpus.json`).
2. **Live conditioning parity**: the Agent embeds `name_id` at inference
   (`--style-tag`); pin that a frame embedded live with id k equals the
   training-side embedding with id k (extend `name_conditioning_test`).
3. **Effect size**: after V3, the STYLE_IDENTITY eval hook — generate
   with `--style-tag INFP` vs id 0 and diff the instrument panel toward
   Michael's fingerprint. Only meaningful now that ids were actually
   trained.
4. Registry budget: 111 slots vs 368 tags. Options: raise
   `num_player_names` (embedding width change — a recipe change, own
   checks), or accept top-111 (81 % of tagged games). V3 launches with
   top-111.

### Where the de-anonymization plan stands

Step 0 (metadata probe of `master-master-*` files) — NOT RUN; note the
inventory shows their in-file name is the literal placeholder, so
residue would have to be in other fields. Step 1 (fingerprints) — built
09-04, tests present. Steps 2-4 (calibration on tagged games, clustering
into perceived players, re-identification) and the `--player-tag-map`
loader seam — NOT STARTED. The 3,605 tagged games are the free ground
truth for step 2; with 368 tags and 12 % coverage, clustering the other
88 % into pseudo-players is where the identity signal would mostly come
from — that is the ask Bradley raised again 09-17.

## Pre-V3 program (Bradley, 2026-09-17 late): finish de-anonymization first

V3 waits on this. Evidence root `eval_runs/0917_style_identity/`.

**Yeti corpus is an identity source** (measured 09-17, 1,500-file sample of
the 20,781 loose `.slp`; 362 archives still unextracted): 31 % of player
slots carry an in-game tag, 192 distinct tags, and it is the SAME scene as
the erickfm partner tags (C2, INFP, 314, CUMB, NELL, TITP, OASI, ALLY...).
Extrapolated: ~14k tagged slots, ~3k tagged Fox games, INFP's Mewtwo.
Cross-corpus same-tag pairs are the strongest calibration test we can get
(different setups, different days, same hands).

| Step | What | Gate / test | Status |
| --- | --- | --- | --- |
| S0 residue probe | raw `.slp` header + metadata tail of `master-master-*` | any surviving name/code/startAt? | **DONE 09-17: NEGATIVE** — display names both "Master Player", no connect code, `metadata{}` empty. Identity for the 88 % must come from fingerprints. |
| S1 fingerprints | `scripts/style_fingerprint.exs` (in-game tag first, normalized) over every Fox game in `v2_filtered` and the Yeti loose files | one JSONL row per game with tag/nil | RUNNING 09-17 03:00 (`run_fingerprints.sh`, unit `exphil-fingerprints`; ~65 files/s) |
| S2 calibration | same-player vs different-player distance distributions on tagged games, WITHIN erickfm, WITHIN Yeti, and CROSS-corpus for shared tags; leave-one-out nearest-centroid retrieval top-1/top-5; threshold at a chosen false-match rate | `scripts/style_calibrate.exs` + synthetic unit test; report `calibration.json` | next |
| S3 perceived players | cluster anonymous-game fingerprints (k-means in Nx; k by silhouette against the tagged-side clusters; min cluster size); emit `player_tag_map.json` path -> `~cNN` | cluster report: sizes, silhouette, purity of tagged games assigned to the same clustering | after S2 |
| S4 reconciliation | anonymous clusters vs tagged centroids (ranked hypotheses, pseudonymous), per-tag purity (collision suspects), per-game tag-vs-cluster (mislabel suspects) | `reconciliation.md` for Bradley to adjudicate | after S3 |
| S5 wiring | `--player-tag-map <json>` consulted by `Streaming.subject_tag/3` BEFORE the filename fallback; registry budget decision (top-K by games, or raise `num_player_names`); trainer prints tag-map coverage | pipeline tests: map overrides filename, held-out map entries cannot enter the registry, frequency ordering with pseudo-tags | after S3 |
| S6 name-channel verification | (a) paired anonymous-vs-registry held-out scoring on a TAGGED sample — differ on tagged, identical on untagged; (b) live embedding parity for `--style-tag` k vs training id k; (c) after training: `--style-tag INFP` vs id 0 instrument-panel diff | tests + `heldout_tagged.json` | before V3 GO |

Then V3 relaunch decision: same recipe + `--player-tag-map` + the registry
budget chosen in S5. The 09-17 mechanical GO stays valid for the
mechanics; the data recipe changes, so re-run the cheap gates (matched
seeds, held-out liveness) on the new registry before the 50 h run.

### S1 + S2 results (2026-09-17 03:27, `eval_runs/0917_style_identity/`)

S1 (after the NIF non-finite fix + 32 workers, ~6 min for both corpora,
0 failed): erickfm Fox 28,452 games / 3,605 tagged / 368 tags; **Yeti Fox
10,567 games / 3,650 tagged / 222 tags** (Yeti has more tagged Fox games
than erickfm). Rows carry costume, player_type, cpu_level, team,
started_at, random_seed, opponent character/costume, stage.

S2 (`calibration.json`, `--min-games 5`, hand features, unweighted
centroids, no z-scoring):

| Set | tags | games | same p50 | diff p50 | top-1 | top-5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| erickfm, full metric | 152 | 3,127 | 2.26 | 4.85 | **0.547** | 0.819 |
| erickfm, invariant | 152 | 3,127 | 0.97 | 4.20 | 0.442 | 0.768 |
| Yeti, full | 116 | 3,399 | 2.37 | 4.54 | 0.347 | 0.730 |
| Yeti, invariant | 116 | 3,399 | 1.01 | 3.79 | 0.263 | 0.635 |
| cross erickfm→Yeti gallery, full | 44 shared | 1,543 q | | | 0.365 | 0.761 |
| cross Yeti→erickfm gallery, full | 44 shared | 2,714 q | | | 0.438 | 0.711 |

Chance top-1 is 1/152, 1/116, 1/44. Thresholds (erickfm full): at 1 %
false-match rate d ≤ 2.00 accepts 30 % of same-player pairs; at 5 % FMR
d ≤ 2.59 accepts 71 %. Yeti at 1 % FMR accepts only 15 %.

**Closed-set estimate on Yeti** (from the leave-one-out rank
distribution, median rank 2 of 116): top-1 with 2 candidates 0.94, with
3 → **0.90**, 5 → 0.85, 10 → 0.76. This is the regime Bradley's costume
priors create (YETI_SCENE_PRIORS.md), so the current hand features are
already usable there; the open-set erickfm case (hashed games, no
prior) is where 55 % top-1 is not enough on its own.

Reading: real signal, survives across corpora and years (same hands,
different setups: 37-44 % top-1 among 44), but the invariant subset is
weaker than the full metric everywhere — controller micro-mechanics
alone are not yet a biometric with these features. Cheap upgrades in
order: per-feature z-scoring on the tagged pool (distances are currently
dominated by the largest-variance rates), a metric-learned projection on
same/different pairs (contrastive on 3.1k+3.4k tagged games), then the
v2 animation-timing features (L-cancel timing, wavedash angle, DI).

Next: S3 needs (a) the closed-set Yeti matcher with the costume prior —
requires Bradley's costume-index → colour table for Fox (NOT guessed;
confirm on a known game), and (b) open-set clustering for the erickfm
hashed games with the z-scored metric; S4 report; S5 wiring.

### S2b: z-scoring + learned metric (v1 features, 03:55)

`scripts/style_metric_eval.exs`, `ExPhil.Interp.StyleMetric` (linear NCA
projection, k=24, trained on one corpus's players, evaluated on the
other's — never-seen players), `metric_eval.json` (kept in
`v1_features/` after the v2 pass).

| queries | raw | z-scored | NCA trained on the other corpus |
| --- | ---: | ---: | ---: |
| erickfm within (152-way) | 0.585 | 0.670 | 0.688 |
| erickfm -> Yeti cross (44 shared tags) | 0.483 | 0.547 | 0.572 |
| Yeti within (116-way) | 0.441 | 0.483 | 0.515 |
| Yeti -> erickfm cross | 0.480 | 0.534 | 0.591 |

Held-out 20 % of erickfm players (31-way gallery): z-score 0.915, NCA
0.910 — a closed-set-of-30 number, not comparable to the 152-way rows.
Z-scoring is a free +8 points; NCA another +2..6, and its top weights are
X/Y press rates, `jump_x_ratio`, R/L press rates, c-stick aerial fraction,
press interval — the hands dominate the learned metric. Hence v2.

Fixed on the way: NCA loss needed a stable log-softmax (row-max
subtraction) — the Yeti fit produced NaN without it.

### v2 timing features (`ExPhil.Interp.StyleTiming`, 11 keys, 6 invariant)

No per-character frame-data tables; every feature is a motor habit
readable from the animation + controller stream:

| Feature | Definition |
| --- | --- |
| `lcancel_attempt_frac`, `lcancel_press_offset_mean/cv` | L/R/Z press edge (digital or analog >= 0.30) in the 7 frames before an aerial landing (70-74); offset = frames before touchdown |
| `wavedash_angle_mean/cv`, `wavedash_jumpsquat_frame_mean` | KNEE_BEND -> AIRDODGE (<= 3 f) -> LANDING_SPECIAL (<= 4 f); stick angle below horizontal on the airdodge frame; which jumpsquat frame the airdodge came out on |
| `di_active_frac`, `di_perp_mean` | on entering a DAMAGE_* state (75-91): stick past the dead zone; \|sin\| of stick vs knockback vector (1 = perpendicular DI) |
| `oos_latency_mean`, `oos_jump_frac` | frames from leaving SHIELD_STUN to the first non-shield action; share that are jumps |
| `short_hop_frac` | takeoff `speed_y_self` < 0.75 x the game's max takeoff speed |

Invariant set additions: L-cancel offset mean/cv, wavedash angle mean/cv,
wavedash jumpsquat frame, OOS latency. Tests:
`test/exphil/interp/style_timing_test.exs` (one synthetic sequence per
detector). Fingerprint rows now carry `costume_name`
(`ExPhil.Data.Costumes`, from the decomp costume tables).

### v2 result (04:22) and tooling

| metric | v1 | v2 (timing features) |
| --- | ---: | ---: |
| erickfm z-scored within (152-way) | 0.670 | 0.673 |
| erickfm -> Yeti cross, z-scored (44 shared) | 0.547 | 0.571 |
| Yeti z-scored within (116-way) | 0.483 | 0.499 |
| Yeti -> erickfm cross, z-scored | 0.534 | 0.553 |
| NCA cross e->Y / Y->e | 0.591 / 0.572 | 0.610 / 0.586 |

+2..2.5 points cross-corpus; the linear-metric ceiling on hand features is
~0.68 open-set. v2 stays. Found on the way (and why the firing-rate audit
now exists): the first v2 pass fired on 0 % of games for wavedash/DI —
controller rows are `%{main_stick: %{x,y}, l_shoulder}` not the flat test
shape, real wavedashes go KNEE_BEND -> JUMPING (1 f) -> AIRDODGE, and
2020-era replays carry no velocities (short hop / DI now use positions).
`style_fingerprint.exs` writes `<out>_audit.json` and warns on any feature
below 5 % firing. `StyleCalibration.retrieval/2` is vectorized (Nx,
leave-one-out via centroid sums): metric eval 13 min -> 87 s.

### S3a closed-set matcher — DONE 04:28 (`ExPhil.Interp.StyleMatcher`, `scripts/style_identify.exs`)

Posterior ∝ P(e) · P(costume|e) · LR(d); LR from 1-D Gaussians fitted to
the gallery's own leave-one-out game-to-centroid distances (same: mean
1.99 sd 0.76; different: mean 4.16 sd 1.02, NCA space); explicit UNKNOWN
hypothesis (LR 1, prior = one average entity). Gallery: 232 entities
(tags with >= 5 games across erickfm + Yeti, 6,676 games).

Leave-one-out per corpus (accuracy on assigned / coverage):

| | top-1 | τ0.7 | τ0.8 | τ0.9 |
| --- | ---: | --- | --- | --- |
| erickfm, with costume | 0.705 | 0.89 / 49 % | **0.91 / 37 %** | 0.96 / 19 % |
| erickfm, no costume | 0.648 | 0.92 / 30 % | 0.96 / 17 % | 0.97 / 9 % |
| Yeti, with costume | 0.641 | 0.83 / 43 % | **0.83 / 33 %** | 0.88 / 17 % |
| Yeti, no costume | 0.623 | 0.84 / 28 % | 0.87 / 16 % | 0.83 / 9 % |

Costume evidence: +6 top-1 points, coverage at fixed accuracy roughly
doubled — Bradley's "just a data point" is worth a lot as a likelihood.
Flat prior helps erickfm top-1 (0.741) but collapses coverage; the count
prior stays on. Assignment of untagged games at τ0.8
(`player_tag_map.json`, 3,259 entries): Yeti 1,651 / 9,093 (18 %; with
the 3,650 tagged ≈ 50 % of Yeti Fox games now identified), erickfm 1,652
/ 25,831 (6.4 % — hashed ranked games are mostly not this scene; S3b
clustering owns the rest). Collision suspect for S4: entity "FOX" (a
generic tag; 494 assignments). Frequent assignees LI, USSR, KHAG, TJOM,
314 look like real players — Bradley to confirm handles.

### S3b clustering — DONE 04:32 (`ExPhil.Interp.StyleClusters`, `scripts/style_cluster.exs`)

K-means (k-means++ seeding, Nx) on the 24,098 erickfm games still
anonymous after S3a, in NCA space. Tagged games (6,676, not used to fit)
projected into the clusters as the purity probe:

| k | tagged purity | clusters >= 30 | largest | wall |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.670 | 19 | 1,984 | 11 s |
| 40 | 0.656 | 39 | 1,155 | 10 s |
| 80 | 0.629 | 77 | 747 | 18 s |

Chosen k=40 (registry budget: ~111 slots − real tags); games within the
cluster's median radius get `~cNN` (11,649 games, 39 pseudo-tags).
`player_tag_map.json` = 14,908 entries (3,259 matched + 11,649
clustered), `cluster.json` the sweep. Whole S3 pipeline runs in ~2 min.

Registry budget for S5: combined pool has 232 real tags >= 5 games plus 39
pseudo-tags; 111 slots keeps the most-played (pseudo clusters are ~300
games each and will take many slots). Decision needed: raise
`num_player_names` (embedding width change) or accept top-111.

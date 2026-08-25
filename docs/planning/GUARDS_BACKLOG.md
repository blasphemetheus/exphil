# Guards backlog — silent-failure classes without tripwires

2026-08-24 (Bradley: "are there other things we should have guards
for but don't"), written the night the delay-id guard made its first
catch (against our own bad metadata). Ordered by (history of burns x
cheapness). Each entry names the incidents that justify it.

## 1. Train-vs-live embedding FINGERPRINT canary — [BUILT 08-24: ExPhil.Embeddings.Canary; save stores the batched-path fingerprint, Agent load re-embeds via the live path and refuses on divergence; pre-canary checkpoints skip]

**The class**: any silent divergence between what training embedded
and what the live agent embeds. Burns on record: the base-block
reversal ("scrambled features invisible to val_loss"), the af
convention (GOTCHA #81), the 288-vs-336 queue-layout mismatch, and
TODAY's stage id-space bug (every live game embedded stage as
"other" — for months, masked only by the policy's stage-blindness).

**The guard**: at checkpoint save, embed ONE canned synthetic
gamestate (fixed players/stage/controller covering every feature
family) and store the vector hash + a few marker values in metadata.
At Agent load, embed the same canned state through the LIVE path and
compare. Any mismatch = hard error naming the first divergent dim
range. One canary kills the whole class — id spaces, ordering,
scaling, gating — including bugs we haven't written yet.

## 2. In-game frame-loop starvation alarm — [BUILT 08-24: windowed-fps check in StatsMonitor, 2 ticks <45fps = loud alarm; EXPHIL_STARVATION_FATAL=1 aborts]

**The class**: the frame loop degrades and the bot plays garbage while
everything "works". THREE incidents today alone (stage-merge watcher
snapshot, volatile-address watch churn, exec-cache mid-game load) —
each diagnosed by a human watching fps decay or a statue on screen.

**The guard**: AsyncRunner alarm when in-game fps < 45 sustained for
>3s: loud "FRAME LOOP STARVED (Nfps)" log + optionally fatal for eval
sessions (a starved eval is garbage data — see EXPOSURE_BIAS 0c). The
in-game sibling of MENU STUCK.

## 3. Gate-sweep infrastructure-failure abort — [BUILT 08-24: 5 consecutive failures = exit 7 "INFRASTRUCTURE?"]

**The class**: eval infrastructure fails and the harness happily
concludes "the policy scored nothing". Tonight: 60x "GATE FAILED" →
"ARGMAX: at -1/min" → phase 1 marched on to replicate 2's doomed
sweep. (Opposite failure 08-21: one grep no-match killed a whole
sweep — the fix overcorrected to never-abort.)

**The guard**: N consecutive gate failures (N=5) = ABORT the sweep
with a nonzero exit and "INFRASTRUCTURE?" in the message. Policy
failures are scores of 0; process failures are not scores.

## 4. RAM-merge sanity ranges

**The class**: memory addresses silently drift (Slippi update, boot
mode) and garbage flows into gamestates as truth. Seen: netplay FoD
reads (4e-34 floats) — currently handled by gating netplay off, but a
LOCAL drift after a Dolphin update would flow straight in.

**The guard**: merge_stage rejects out-of-band values (heights outside
[-10, 45], digit not in the known byte set) — skip + warn-once
"stage RAM out of band, merge disabled this session".

## 5. GPU-beam lockfile (the no-mix / second-client law, enforced)

**The class**: a mix invocation beside a live GPU beam SIGBUSes it.
Enforced today purely by discipline + pgrep idioms, both of which
failed at least once this session (self-matching wrappers; one
mix-test ran beside a live try-4 beam on luck).

**The guard**: sessions that create a CUDA EXLA client write
`~/.cache/exphil/gpu.pid`; app boot (and a `mix exphil.guard` you can
put in aliases) checks for a LIVE holder and refuses GPU work with
the holder's cmdline in the message. Escape hatch env for deliberate
sharing (policy-server sessions are CPU-only and exempt).

## 6. Checkpoint save-time lint — [BUILT 08-25 (width lint): export_policy resolves embed_size against the PARAMS' actual leading dims (embed_config candidate preferred, config scalar fallback) and RAISES if neither matches any param tensor; warns when the canary length disagrees with the exported width (agent prefers canary → non-default embed opts won't deploy). Same commit killed the root cause: Pipeline's two hand-rolled Embeddings.config whitelists now pass full resolved opts, and Trainer.new falls back to pipeline.embed_config width for the streaming path (which has no upfront embedded tensor). Remaining sub-lints unbuilt: with_delay_id⇒train_delays consistency, fingerprint-present assert]

**The class**: wrong metadata baked at save, discovered at deploy
(train_delays [0]; the 0825 pilot's config said 296 while the params
were 288). **The guard**: after building the config map, assert
consistency before writing: with_delay_id => train_delays non-empty
and != [0] when a multi-delay pool was built; queue_depth/embed sizes
agree with the dataset's embed_config; fingerprint (#1) present.
Refuse to save a checkpoint that lies about itself.

## 7. Convergence-exit trust (already recommended, still unbuilt)

From 0819_g15r2: the drill's converged check trusts a single-epoch
loss; a >100x one-epoch drop is divergence-suspect, not convergence.
Guard: require target-loss for 2 consecutive epochs; flag the drop
class and export best-epoch instead. (The collapse guard covers the
NaN path; this covers the "exported the cliff" path.)

## Non-guards (deliberately)

- Chain-metric misuse: handled by labeling (analyzer prints
  "COMMANDED, not chains") + the memory law; a hard guard would block
  legitimate press-analysis.
- Frozen-stadium intent: it's a preference, not an error; the flag
  exists and PS work now knows to pass it.
- Deploy-knob drift beyond delay-id: DEPLOY_KNOBS.md + the guard (#1
  delay-id) cover the burn history; more would be config-freezing.

## 8. Matchmaking-timeout error handling (added 08-24, live incident)

**The class**: Slippi Direct shows "Error: Matchmaking timed out,
please try again (press Z to clear)" when the peer is slow to start
searching; the bot sits on the error screen forever (a human cleared
it this time). **The guard**: hunt the error screen's RAM signature
(one park-and-scan session at a forced timeout — search with nobody
on the other side and wait), then MenuHelper: on the error state,
press Z + re-enter the search flow. Blind periodic Z is NOT
acceptable (Z cancels an ACTIVE search — the depth-word arc). Until
the signature exists, the mitigation is operational: relaunch.

## 9. Unknown-flag rejection in standalone-parser scripts (added 08-25)

**The class**: scripts with their own OptionParser strict lists
(dagger_drill and kin) silently IGNORE unrecognized flags —
`--stage-internals` on the 0825 overnight arm was dropped without a
word and the arm trained the plain recipe (a wasted-premise run; only
the checkpoint metadata exposed it). **The guard**: every standalone
parser checks OptionParser's invalid/unknown list and ERRORS with the
offending flags. One helper, applied to each script's parse site.

# PRE-REGISTRATION — transfer decider (D7)

**Status: DRAFT — three decisions marked `[DECISION]` are Bradley's and
must be settled before the first game. Written 2026-08-28, before any
scored game is played.**

This decider was launched by hand on 2026-08-24 with no run script and
no prereg — the only one of its family without a written decision rule
(0812, 0820, 0824_crown all have one). That gap is what this file
closes. Nothing here may be edited after the first scored game.

## The question

Does the "generalist → sharpen" recipe **transfer off the character it
was tuned on**, or is the champion a Fox-vs-popo specialist?

This is the only active evidence for the project's founding goal
(low-tier characters). Until it resolves, that goal has none.

## The arms (blinded)

| color | file | md5 |
|---|---|---|
| green | `arm_green.bin` | `5efc3619317b2a4546e801d9c3b8b97b` |
| orange | `arm_orange.bin` | `f1ac87fca4a8171935a1f608f3a6632f` |

They are `ms_g19_ep4.bin` (the reigning champion) and `ms_g19r3_ep4.bin`
(the transfer candidate) in an order sealed in `key.txt`, unread.

**Archive location: `/home/blewf/decider2_archive/`** (arms, `key.txt`,
`order.txt`). The originals lived in another session's `/tmp` scratchpad
on a 99%-full root filesystem; `key.txt` is 29 bytes and irreplaceable.

Sealed order (`order.txt`): `green orange orange green green orange`.

### Blinding hazard — read before scoring

The arms are **size-leaky**: `arm_green.bin` is 3,793,856 B and
`arm_orange.bin` is 3,793,948 B, and those sizes match the two source
checkpoints exactly. **A single `ls -l` on the arm directory defeats the
blind.** Whoever scores must not list that directory. Bradley's
impressions are unaffected — he sees colors on screen.

They were deliberately NOT re-blinded by padding: it is unknown whether
the loader tolerates trailing bytes, and corrupting an irreplaceable
artifact to fix a hygiene problem is a bad trade. The real protection is
this prereg — a decision rule fixed in advance cannot be fitted after the
fact.

## The distinguishing hypothesis

From `eval_runs/0824_joint_sweep/RESULTS.md`: the champion is
**popo-only** (chain-1 against Mewtwo and Yoshi), while `ms_g19r3_ep4`
chains broadly (344 / 274 / 363). If that offline result is real, the
arms should separate **only against a non-Fox opponent** — which is why
the opponent character is the load-bearing variable here, not a detail.

## Protocol

- Netplay, connect code `DBTD#411`, bot plays Fox.
- **Stage pinned FD** (`--require-stage final_destination`) — the decider
  stage-pin law; an unpinned decider measures stage luck.
- Deploy knobs, copied not reconstructed (`docs/guides/DEPLOY_KNOBS.md`):
  `--frame-delay 4 --delay-id-override 3 --deterministic`.
  Delay id 4 is UNTRAINED; the override is mandatory.
- 6 scored games, 3 per arm, in the sealed order. Consecutive same-color
  games may share one launch (4 launches total).
- **GPU must be idle.** A live beam starves the game loop and flatlines
  both arms — the failure the 0820 script's hard guard exists to prevent.
  Check `systemctl --user list-units --state=active | grep -E "bracket|train"`
  and `pgrep -c beam.smp` first.

### The 3 already-banked games are DISCARDED

`2026-08-Mainline/` holds 3 games from the 08-24 attempt. All three are
protocol-broken: all under **green** (the sealed order wanted green,
orange, orange…), the opponent played **Fox** not Mewtwo, and one game
lasted 43 seconds. They are kept as artifacts and excluded from scoring.

## `[DECISION 1]` — what does the human play?

The informative condition is a human playing **Mewtwo**. The confound:
if Bradley does not play Mewtwo well, "the bot chains against Mewtwo" is
indistinguishable from "the human is bad at Mewtwo". Nothing in the
original setup addressed this.

- [ ] **A — 6 Mewtwo games** (recommended if Bradley's Mewtwo is
      passable). Maximum power on the actual hypothesis; the confound is
      stated as a caveat rather than removed.
- [ ] **B — 3 Mewtwo + 3 Fox** (12 games total, 6/arm). Fox games are
      the within-decider control: if the arms separate on Mewtwo but not
      Fox, that IS the transfer result and the human-skill confound is
      largely defused. Costs a second session.
- [ ] **C — 6 Fox games.** Cheapest, and answers a *different*, less
      interesting question (the champion's home turf, where the crown
      decider already ran).

**Recommendation: B if there is appetite for two sessions, else A.** B is
the only option that separates "transfer" from "Bradley's Mewtwo".

## `[DECISION 2]` — the decision rule

Pre-registered, primary metric **canonical chains scored from `.slp`
only** (`scripts/analyze_shine_source.exs`, both ports — bot port flips
with connect order). Never press counts.

Proposed rule:

1. **Candidate ≥ champion on chains against the non-Fox opponent, and
   champion visibly popo-only (chain ≤ 2 where candidate chains) →
   TRANSFER CONFIRMED.** The candidate becomes the transfer-line
   production checkpoint. The crown for Fox-vs-Fox does **not** move on
   this evidence alone.
2. **Candidate collapses against a human while champion chains →
   transfer refuted**, champion retains, and the offline broad-chain
   numbers are recorded as not predicting live transfer (itself a
   valuable finding about the offline metric).
3. **Neither separates by 2× → NO VERDICT.** Champion retains on
   incumbency.

- [ ] approve  - [ ] edit

## `[DECISION 3]` — does Bradley's blind impression gate the result?

The standing g6 lesson says CPU rankings invert against humans and that
nothing is crowned without a human read. Here the human *is* the
opponent, so his per-game blind impressions (by color, recorded before
unblinding) are available.

- [ ] **A — impressions are RECORDED but chains decide** (recommended).
      Keeps the primary metric objective; impressions become evidence
      about the metric.
- [ ] **B — impressions can veto** a chain-based verdict.

**Recommendation: A**, with impressions and an arm guess written down
per game *before* `key.txt` is opened.

## Declared in advance

- **n=3/arm cannot resolve a difference under 2×**, and this project's
  baselines are not reproducible across days. **"No verdict" is a likely
  and legitimate outcome** and must be reported as such, not narrated
  into a winner.
- The standing law that **single-run transfer claims are void** (the 10×
  lottery) applies: a win here licenses "worth a real campaign", not
  "transfer works".
- **Unverified assumption, recorded:** the `--stage-internals` embedding
  and the 08-24 live stage-id-space fix both landed *after* these two
  checkpoints trained. Both arms are affected equally so the A/B should
  remain fair, but this was not verified.

## Finish sequence

1. Settle the three `[DECISION]`s above; delete this line.
2. Confirm GPU idle.
3. Play the games in sealed order, recording blind impressions per game.
4. Score every replay with `analyze_shine_source.exs` on the bot port
   (verify port by connect order). **Do not `ls -l` the arm directory.**
5. Write impressions + arm guesses down.
6. Unseal `key.txt`, write `RESULTS.md`.
7. Update DEPLOY_KNOBS / handoff / memory only if the verdict moves
   something.

# 2026-09-10 — multishine under the causal recipe, gated under SAMPLING

Bradley's direction: "best version we can of the multishine bot, local
play, no delay", with v3's recipe. Prereg: `run_g20.sh` (arms a/b),
`../0910_g21_sharp/run_g21.sh` (arms a/b). All numbers: stand-fox FD,
headless sync runner, 60 s games, ShineChain v3; T=1.0 unless marked.

## 1. The g19 record was an argmax artifact

| policy | decode | rung | shines/min | max chain |
|---|---|---|---|---|
| ms_g19_ep4 (the card) | argmax | d3 id3 | 437.4 | 438 |
| ms_g19_ep4 | T=1.0 (x2) | d3 id3 | 44.9 / 42.9 | 3 / 2 |
| ms_g19_ep40 | T=1.0 | d3 id3 | 101.9 | 9 |
| ms_g19_ep58 | T=1.0 | d3 id3 | 126.8 | 12 |

A 438-chain is ~4,000 frame-exact inputs; only a near-deterministic
distribution survives that under sampling. Per-state probe
(`scripts/probe_ms_state_confidence.exs`): g19_ep4 sits at 0.77-0.94 on
the loop decisions, ep58 at 0.90-1.00. Bradley (12:xx): "relying on
deterministic was a silly thing to do" -> `gate_sweep.sh` now samples at
T=1.0 by default (`GATE_TEMP=0` restores argmax). **The peak-at-ep4 /
decay-after lore (0820) was the argmax metric**: under sampling the LATE
epochs are better, monotone with loss (0.21 -> 0.010).

## 2. g20 arms (g19 recipe + causal labels + clean loss; rungs {0,1,2,3}, 24 ep)

| arm | head | SS | d1 sweep (24 ep) | argmax ep | notes |
|---|---|---|---|---|---|
| g20a | autoregressive | no | 79-100/min, chain 1-4, FLAT | ep11 99.9 | rung-insensitive (d1/d2/d3 all ~80-95), temperature-insensitive (T=1/0.5/0.25 ~83; argmax WORSE 56 c1) |
| g20b | independent | 0.5/10 | 78-103/min, chain 1-3, FLAT | ep13 102.9 | offline fixture agreement 0.888 @offset 2 (g19: 0.887 @5; g20a 0.777 @4) |

Trace (`trace_multishine.exs`, g20a ep11 vs g19 ep4 at T=1): JC 39 vs 29,
shines-from-jumpsquat 19 vs 9, empty hops 35 vs 26 — both jump-cancel
fine; the cycle breaks AFTER the jump.

Per-state confidence (issued-input semantics; "fixture B@off" = the
fixture's issued B rate at that offset; the policies track offset 2):

| state | g20a p(B) | g20b p(B) | g19_ep4 p(B) | g19_ep58 p(B) | fixture B@2 |
|---|---|---|---|---|---|
| ground reflector af1 | 0.37 | 0.91 | 0.87 | 1.00 | 1.0 |
| ground reflector af2 (release) | 0.38 | 0.02 | 0.82 | 0.90 | 0.0 |
| jumpsquat af0/1/2 | 0.97 | 1.00/0.93/0.96 | 0.83-0.87 | 0.94-0.99 | 1.0 |
| aerial reflector af1/2/3 | 0.33-0.37 | 0.99 | 0.89/0.94/0.77 | 0.96/0.98/0.91 | 1.0 |
| reflector-open af0 | 0.22 | 1.00 | 0.13 | 0.06 | 1.0 |

- **g20a (AR head)** genuinely fails: p(hold B) ~0.33 through the aerial
  reflector -> releases mid-air -> the 18-frame wind-down. Argmax makes
  it worse (release whenever p<0.5): 56/min c1. AR head is OUT for the
  ms line.
- **g20b (independent)** is decisive and correct (0.91-1.00) and STILL
  chains 1-3: a 9-frame cycle at ~0.95/frame chains ~3. Chain length
  under sampling ~ 1/(1-p_cycle); a real multishiner needs ~0.999/frame,
  i.e. a loss floor near 1e-3, not 1e-2.

## 3. Where the loss floor comes from (label conflicts)

`scripts/probe_ms_label_dilution.exs` over the 27-replay rollout pool
(expert-relabeled, issued semantics): aerial reflector af1-3 -> B=1.0
(7,380 frames); reflector-open af0 -> B=1.0 (2,414); reflector-open
af1..12+ -> B=0.0 (**32,000 frames** — the bot FLOATS in rollouts). Two
conflicts follow:
1. reflector-open af0 (hold) vs af1+ (release) differ by ONE frame; the
   1/60-scaled action_frame scalar cannot separate them -> a blend
   (g20a 0.22, g19 0.06-0.13 there). The bucketized action-frame
   embedding (`--action-frame-buckets`, wired into dagger_drill today)
   is the direct fix — arm g21b.
2. Multi-rung shifted labels: at base+2..5 the label for an aerial-shine
   state reaches into the float (release) for the higher rungs and stays
   hold for the lower -> the same state is labeled both ways across ids,
   and the id channel is weak -> g20a's 0.33. Single rung = arm g21a.

## 3b. CORRECTION (13:05): every "d1" gate above ran delay-id 1, not 0

The sync runner (`scripts/play_dolphin.exs`) still copied `--frame-delay`
into `delay_id` (the async script was fixed 09-09; this one was not).
So g20a/g20b "d1" = live d1 at id 1 (label offset 3); "d2" = id 2
(offset 4); "d3" = id 3 (offset 5). g21a (trained id 0 only) was REFUSED
by the untrained-id guard on its first sweep — which is how this
surfaced. Fixed (the runner passes nil; the Agent derives id N-1 for
causal checkpoints). The per-state probe used `--delay-id 0` explicitly
and is unaffected. g20b ep13 is being re-gated at the true id 0.

## 4. g21 (single rung 0, 60 ep, gated late under sampling)

- g21a (g20b recipe, --multi-delay "0"): loss 0.17 (ep20) -> 0.08 (ep40)
  -> 0.017 (ep56) -> 0.029 (ep60): NOT below g19's 0.010 (a quarter of
  the pool; SS + AWBC make the loss noisy). Sweep at the TRUE id 0, d1,
  T=1.0: ep8..60 all **chain 1** (86-114/min; argmax ep48 113.9). g20b
  ep13 at true id 0: 89.9/min chain 1 (vs chain 3 at id 1). VERDICT: the
  single-rung quarter-pool arm converged worse and does not chain; the
  local rung is not obviously id 0 either (both ids chain <=3, not
  decisive). g21b NOT run.
- **g22** (`../0910_g22_converge/run_g22.sh`, 13:10): g20b recipe (4
  rungs, full pool) to 60 epochs — the direct causal successor of the
  g19 late epochs; gate late at d3 id 3 against g19_ep58 (127/min c12)
  and at d1. Arm b adds --action-frame-buckets 24.

## 4b. g22a (g20b recipe, full pool, 60 ep) — converged, sharp, chains 4-5

Loss 0.039 (ep26) -> 0.0077 (ep56) -> **0.0036 (ep59)** -> 0.063 (ep60,
a bounce: per-epoch snapshots matter). Gates (T=1.0, stand FD):

| rung | id | epochs | shines/min | chain |
|---|---|---|---|---|
| d3 | 3 (WRONG: g19's "id3" is OLD numbering == new id 2) | 30..60 | 78-91 | 1-2 |
| d1 | derived 0 | 48/55/57/59 | 90-96 | 1 |
| d1 | 1 | 48/55/57/59 | 78-84 | 1-2 |
| **d3** | **2 (= g19_ep58's rung)** | 55 / 59 | 104.9 / 105.9 | **5 / 4** |
| d2 | 1 | 59 | 108.9 | 4 |

g19_ep58 at the same rung: 126.8/min chain 12. Per-state (ep59, id 2):
0.98-1.00 everywhere EXCEPT the two boundary frames — last aerial-
reflector frame 0.93 and the ground-reflector release frame 0.94
(reflector-open af0 0.05 is CORRECT at this offset: 4 frames on is the
jumpsquat release). 0.93 x 0.94 x the rest ~ 0.8 per cycle -> chain ~5.
Both soft spots are "which frame am I on" decisions -> arm g22b
(`--action-frame-buckets 24`, launched 15:15) is the direct test.
Rung law for the DRILL line, empirically: live d3 <-> new id 2 (offset
4), live d2 <-> id 1 (offset 3): i.e. **live N <-> id N-1**, which is
what the Agent derives for causal checkpoints. The d1/id0 chain-1
results say the tight rung has a further problem (sync runner's
effective delay = async+1, HANDOFF 07-28 — d1 sync may be below the
harness floor); the local no-delay target should be re-read as "the
tightest rung that chains", to be found by sweeping d1/d2 on g22b.

## 4c. g22b (+ --action-frame-buckets 24): loss 5e-5, STILL chains 2-4

Loss 0.0051 (ep55) -> **7e-5 (ep57)** -> 0.029 -> 0.0050 -> **5e-5
(ep60)**: two orders below g22a. Gates (T=1.0): d3 id2 ep48-60 =
101-113/min chain 2-4; d2 id1 97-108 c2-3; d1 id0 106-107 c2-3.
Per-state (ep60, id2): 0.99-1.00 everywhere except the SAME two boundary
frames: last aerial-reflector frame 0.95 (was 0.93), ground-reflector
release 0.96 (was 0.94). A policy that fits its pool to 5e-5 and sits at
0.95 on a fixture state = the POOL's labels at that state are a 95/5 mix.

**Found the conflict:** `scripts/snippet_mine.exs` builds its expert
table from `fox_multishine_closed.slp` (its default) while the drill's
table comes from `fox_multishine_closed_d1.slp` — the "d1" fixture
triggers one frame earlier (07-28). 12.5k snippet frames (2.5% of the
pool, concentrated on exactly the boundary decisions) carried labels one
frame off. This is true of TODAY'S arms AND of g19 (the 0804 snippets
were mined the same way). So the residual 5% on the boundary frames is
not resolution and not convergence — it is two teachers one frame apart.

**g23** (`../0910_g23_consistent/run_g23.sh`, launched 17:39): g22b
recipe with (a) snippets re-mined with the d1 fixture
(`eval_runs/0910_snippets_human_causal_d1`), (b) NO snippets — the
clean control. Read: boundary frames -> 0.99+ and chains in the tens.

## 4d. g23a — THE FIRST SAMPLED MULTISHINER (19:50)

g22b recipe (independent head, SS 0.5/10, clean loss, causal labels,
4 rungs, `--action-frame-buckets 24`, 60 ep) with the snippets re-mined
against the drill's own d1 fixture. Loss 0.0061 (ep40), 0.0042 (ep57),
0.0067 (ep60). Sweep at d3 id2, T=1.0: ep48 121.8 c6, ep53 121.8 c6,
ep55 100.9 c5, **ep57 436.4 c436**, ep59 106.9 c6, ep60 22.0 c2.
**Confirmation ep57 d3 id2 x3 (T=1.0): 437.4 c438 / 110.8 c4 / 426.4
c423.** Two of three games are game-long chains under SAMPLING; the
third broke at chain 4 and did not re-enter. Per-state (ep57, id2):
1.00 on eight of nine loop decisions, 0.98 on the last aerial-reflector
frame and the ground-reflector release (were 0.93/0.94 on g22a). ep57 is
a sharp snapshot (ep59/60 do not chain) — per-epoch snapshots + a
sampled gate sweep remain mandatory. Other rungs: d2 id1 c2, d1 id0 c2
— the tight rungs still do not chain (harness floor question, §4b).

What it took, in order: gate under sampling (not argmax); independent
head (AR releases B mid-air); train to convergence (60 ep, not 24);
bucketized action frame (loss 5e-3 -> 5e-5); and remove the one-frame
teacher conflict in the snippets (0.95 -> 0.98 on the boundary frames).
Control g23b (no snippets) launched 19:55.

**Deploy for a local look (Bradley):** `checkpoints/ms_g23a_ep57.bin`,
`--frame-delay 3 --delay-id-override 2 --temperature 1.0
--buttons-temperature 1.0 --stateful-step` (the id is the NEW numbering;
the Agent would derive 2 at d3 on its own). NO crown from stand numbers
(g6 rule).

## 4e. g23b (no snippets, control): best chain 59 — consistent snippets HELP

Loss 0.057 (ep40), 0.010 (ep55), 3e-5 (ep57), 0.0036 (ep59), 0.069
(ep60). Sweep d3 id2 T=1.0: ep48 114.8 c7, ep53 129.8 c12, ep55 112.8
c5, ep56 55.9 c3, ep57 26.0 c3, ep58 98.9 c5, **ep59 140.8 c59**, ep60
111.8 c20. Per-state (ep57): last aerial-reflector frame 0.96 (g23a
0.98). Verdict: without the human snippets the recipe reaches chains in
the tens; with them re-mined against the drill's own fixture it reaches
game-long chains on 2 of 3 games. The snippets contribute, and only
once their teacher agrees with the drill's. Both arms show epoch-to-
epoch volatility at the end (ep56/57 collapse on b, ep59/60 on a):
per-epoch snapshots + a sampled gate sweep are mandatory for selection.

**Candidate: `checkpoints/ms_g23a_ep57.bin`** (d3, id 2, T=1.0).

## 4f. Bradley's local look (09-11 00:41) + the metric correction

Bradley, async windowed d3 id2 T=1.0: "didn't seem that impressive".
Session replay (2.7 min): 103 shines, 95 self-initiated, 8 hit-induced,
**35.7/min, max chain 2**. Same checkpoint, same rung, headless:

| opponent | runner | d | shines/min | max chain |
|---|---|---|---|---|
| stand dummy | sync | 3 | 437 / 111 / 426 | 438 / 4 / 423 |
| stand dummy | async | 2 | 246.7 | 189 |
| stand dummy | async | 3 | 296.5 | 277 |
| stand dummy | async | 4 | 189.1 | 134 |
| **level-1 CPU** | async | 3 | **54.3 / 61.7** | **2 / 3** |
| Bradley | async | 3 | 35.7 | 2 |

Delay is NOT the cause (async d3 is the aligned rung; sync d3 == async
d2, HANDOFF 07-28). The cause is STATE COVERAGE: the policy multishines
in fixture-like states and collapses in any other (opponent moving,
approaching, getting hit). Bradley's restatement of the goal: it must
multishine "no matter where Fox is, what Fox is doing, no matter what
the opponent is doing" — a methodology test. Corrections:
1. **Gate metric:** shines/min + re-entry over a full game vs a MOVING
   opponent (`--dummy cpu`, tech_random), not max chain vs stand. The
   stand number stays as the technique floor. Every "record" before
   today was the floor.
2. **Data:** DAgger rounds on the failing states — human sessions and
   CPU-dummy rollouts relabeled by the expert (the drill's own loop).
   First pool: Bradley's 0911 session + the CPU games from this readout.

## 4g. g24a — first DAgger round on the failing states (09-11 04:30)

g23a recipe + 9 rollouts (Bradley's session, 8 CPU games of ep57;
expert corrected 63-75% of their frames vs ~59% on the old dummy
rollouts). 698k frames, 60 ep, loss 0.0064 (ep59). **CPU gate** (async
d3 id2, T=1.0, 2x90s, shines/min · max chain), same night, same gate:

| snapshot | run 1 | run 2 |
|---|---|---|
| g23a ep57 (reference) | 49.3 · 3 | 56.3 · 2 |
| g24a ep48 | 56.3 · 3 | 51.8 · 2 |
| g24a ep53 | 62.1 · 3 | 56.9 · 8 |
| **g24a ep55** | **71.1 · 13** | **68.5 · 8** |
| g24a ep57 | 64.6 · 7 | 62.7 · 2 |
| g24a ep59 | 57.6 · 4 | 65.3 · 5 |
| g24a ep60 | 53.1 · 2 | 57.6 · 3 |

Stand floor, ep55 (async d3 id2): 111.0/min c27 and 92.3/min c3 — vs
g23a ep57's 296.5 c277. **Verdict: one round of DAgger on the failing
states raised the moving-opponent number (+30% rate, chains 2-3 -> 8-13)
and paid for it with the isolated technique (c277 -> c27).** A mixed
pool with 9 new rollouts is a small step; the loop is designed to be
iterated (rollouts from ep55 -> relabel -> retrain), and the two metrics
must be gated together — the stand floor is the guard that the
technique survives, the CPU number is the goal. Both are still far from
"multishines most of the time no matter what" (a clean cycle is ~400/min).

## 6. Instruments (09-12) — pool label auditor + closed-loop correction validation

**Instrument 1, `scripts/audit_ms_pool_labels.exs`** (pre-train gate in
run_g24.sh): per loop-state key, B/X label rates per SOURCE (fixture,
relabeled rollouts, snippet file, openers), CONFLICT when sources with
n>=20 differ by >0.05. Validation: flags the 0804 snippet file at exactly
{361,2} (X: the jump-cancel press) and {24,2} (B: the release) with
"smallest source: snippets"; clears the re-mined d1 file.

**Closed-loop correction validation** (Astra's note): `scenario_suite.exs
--driver teacher|policy|neutral --character fox` on break moments mined
by `scripts/mine_ms_breaks.exs` (36 from the 0911 CPU rollouts +
Bradley's session), new `:multishine_reentry` scorer (re-entry frame,
ShineChain v3 max_chain in the 120f window, ended_by, empty hops;
pass = re-entry <= 60f AND max_chain >= 5). First 12 moments:

| driver | re-entry (frames) | v3 max chain | note |
|---|---|---|---|
| neutral | never | 0-1 | control |
| teacher, first run | 7-51 | **1** | every path = 365x3 -> **366x18** (the aerial-shine-one-frame-late float) -> 368 -> 363x14: a 43f loop, not the 9f cycle |
| teacher, af live->parsed | **4-12** (one 40, started mid-float) | **10-14**, ended only by the window | the corrections DO restore the tight cycle |
| policy ep55, derived id 0 | 8-108 / never | 1 | rung in this harness unverified; id sweep running |

The first teacher run failed for the GOTCHA #81 reason (table keyed in
PARSED action_frame numbering, bridge reports LIVE numbering): executed
one frame late, every re-entry floated. That is precisely the class the
instrument exists to catch, and it is a HARNESS convention, not a rule
defect — the drill's relabel labels parsed frames and is unaffected.
Verdict so far: the expert's corrections are executable and recover the
loop from these break states within ~10 frames.

**Policy driver (07:15-07:31):** ms_g24a_ep55 at delay-id 0/1/2 (22 runs
each): max_chain 1-2 everywhere, re-entry scattered/never; with
`--live-af` (same conversion the teacher needed) it is WORSE (score
0.18). **CONTROL (`scenarios/ms_midchain_control.json`):** handoffs
INSIDE ms_g23a_ep57's own 438-chain stand game (frames 900/1500/2100):
teacher chains (0.965); ep57 — which chained 438 in that very replay on
the sync runner — does NOT (ids 0/1/2: 0.29/0.11/0.19). So the harness
cannot yet evaluate the POLICY: `run_one` resets the Agent at handoff,
and a queue-as-input GRU policy then starts a 9-frame cycle from an
EMPTY 60-frame window and an EMPTY own-input queue. The teacher is
memoryless and is unaffected. **Needed:** an Agent observe-only step
during the prefix (embed each observed frame into the window and push
the RECORDED p1 input into the controller queue as if the agent had
issued it), so the policy is warm at handoff. Until then the closed-loop
validator is a TEACHER instrument (which it has already paid for) and the
policy's break-state behavior must be read from live rollouts (the CPU
gate) instead.

## 5. Harness (GOTCHA #114)

Every gate before 11:27 ran the NETPLAY AppImage headless (global
`DOLPHIN_DIR`); it needs a display and died when X went stale. Sweeps
now FORCE the exi-ai headless build, kill TERM-immune survivors per
gate, and run as `systemd-run --user` units (no display needed; retires
the tool-shell 10-min/memory-kill chunking).

## Open

- Whether the loss floor is reachable by data cleanliness (fixture vs
  expert-table one-frame disagreement; float-heavy rollouts:
  `--rollout-cap-per-state` exists) or needs a sharpening term.
- Local look (Bradley) at the best sampled snapshot at `--frame-delay 1`.
- The Agent derives id N-1 at live N for causal checkpoints; for the
  drill line g20b tracks offset 2 at id 0 and physically d1 needs offset
  2 — consistent. g19 (legacy) at d3 uses id 3 (offset 4).

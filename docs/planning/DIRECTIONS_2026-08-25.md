# Directions — 2026-08-25 zoom-out

Written while fox_gen_v1 (7,911-game generalist Fox BC) trains overnight.
Companion to GOALS.md (what the bot should do) and HANDOFF_2026-08-25
(live resume state). This doc: where we are, what's banked, what's open,
and a menu of directions to pick from.

---

## Where we are, in three sentences

The **specialist line peaked**: ms_g19_ep4 is crowned, multishines at
record levels through real netplay latency, and its recipe science is
finished — the stand gate is saturated and further specialist gains are
lottery-shaped. The **infrastructure matured past the bot**: menus are
closed-loop, JIT is solved, RAM truth is wired, guards catch our own
metadata lies, and (as of tonight) the training pipeline actually scales
to ~90M frames. The **generalist line just opened**: the first real
full-corpus BC run is in flight, and the strategic question has shifted
from "can we execute a technique under delay" to "can a bot learn to
*play Melee* from master-level data, then be sharpened."

## What's banked (don't rebuild, don't relitigate)

- **Track B (Fox execution) RESOLVED** — multishine under delay, ladder
  to d4, all-time netplay chain 62. The delay campaign recipe
  (multi-delay {2,3} + SS-on-queue + queue-as-input) is the proven
  template for "skill under latency."
- **Deployment stack** — policy server (0ms checkout), stateful probes
  (1.5s), closed-loop online/local menus incl. Direct code entry,
  stage-pinned deciders, DEPLOY_KNOBS.md discipline.
- **Observability** — RAM merge + memory-watch program, ShineChain,
  situation labels (47), coach knowledge model v0, rewind viewer,
  CycleSim, early-reject probes. Players-as-probes as a method.
- **Guards program** — untrained delay-id, embed canary, params-width
  lint (#6, tonight), starvation alarm, gate-sweep abort, unknown-flag
  abort. The theme: metadata must not lie, silence must not pass.
- **Recipe science** — argmax epoch is a lottery → gate-sweep is
  mandatory; loss anti-correlates with behavior; mewtwo-transfer is the
  hardest and most predictive opponent test; every retrain carries all
  validated mixes.
- **Scale pipeline (tonight)** — streaming lazy-layout path: ~90M
  frames trainable on this machine, embedding cached after epoch 1.

## What's genuinely open

1. **The decision problem (Track A)** — the bot doesn't *go in*,
   convert, or chain on purpose. GOALS.md rung 1 (supervise the
   decision, not the buttons) is UNSTARTED. This is the intellectual
   heart of the project.
2. **The fight-state gap** — vs humans and pressure, behavior degrades;
   needs the human corpus and pressure labels.
3. **Generalist quality** — pilot at 300 games = idles/crouches
   (mode-averaged BC). Whether 26x data fixes the baseline is exactly
   what fox_gen_v1 answers tomorrow.
4. **RL** — never seriously attempted (deliberately). BC-then-RL is the
   agreed road; nothing built beyond the PPO trainer skeleton.
5. **The Direct exhibition milestone** — take ≥1 stock off a real human
   in consented Direct. Still ungated-through.
6. **Low-tier long game** — Mewtwo/Ganon/Link/Zelda/ICs/G&W bots. One
   Mewtwo arm exists; nothing at scale.

---

## Directions menu

Ordered roughly near-term → far-term, not by priority. Each entry: what
it is, what question it answers, rough cost, and risk/dependency notes.

### D1. Generalist gate + iterate (the run already in flight)
Gate-sweep fox_gen_v1's 10 epoch checkpoints (argmax = lottery law),
live look at the best 1-2, compare against the pilot's idle/crouch
baseline. Then apply the curation playbook that worked for the
specialist (edge/conversion snippet mixes, dose discipline) to the
generalist corpus.
**Answers:** does scale fix BC mode-averaging? Is the generalist line
alive?
**Cost:** hours (run finishes overnight; sweep is scripted).
**Notes:** this is the committed next step; everything else forks on
its result.

### D2. BC-then-RL: offline RL first (F5), then self-play
Take the best generalist checkpoint as the base policy. Start with
offline RL (advantage-weighted regression / IQL-style on the replay
corpus — the F5 thread) before touching self-play PPO. The corpus
already has outcomes; AWBC is already validated in the champion line.
**Answers:** can sharpening beat imitation's ceiling — the "surpass
Phillip-Mewtwo" decision problem, on Fox first where eval is easy.
**Cost:** days-to-weeks; new training loop code but reuses everything
else.
**Notes:** the philosophy verdict already chose this road. Needs a
value model, which Track C also wants — shared investment.

### D3. Yeti corpus ingest
Collect the yeti tournament-series replays, filter Fox 1v1s (parse
metadata: character pair + game mode), and fold into the generalist
corpus — or hold out as a distinct arm. Yeti's distinguishing property:
**consistent players across years**, which the erickfm ranked corpus
lacks.
**Answers:** does more/different master data help; sets up
player-conditioned training.
**Cost:** small (pipeline is proven; the filter is one metadata pass).
**Notes:** cheap to do the *inventory* now (count Fox 1v1s, players,
patch versions) even before training on it. Slippi version floor
matters (pre-3.18 files lack some fields — the erickfm-ranked lesson).

### D4. OGSwaglord bot (player-style imitation)
Train a bot that plays like a *specific person*: fine-tune the
generalist on that player's games, or use the existing player-registry
/ style-conditioning machinery (learn_player_styles) with their tag as
the conditioning token. Yeti's consistent-player property is the
enabler.
**Answers:** can we capture an individual's style, not just
"master-level Fox" — also the emotional/demo payoff direction.
**Cost:** small once D1+D3 exist; the machinery is built but barely
exercised.
**Notes:** Bradley's stated order: current-data lessons first, then
this. A blind "which one is OGSwaglord" decider is the natural eval —
we know how to run those.

### D5. Track A rung 1: the decision-supervising expert (Mewtwo)
Build the expert that supervises *when to commit*, not which buttons:
reaction-conditioned decisions, tech-episode frames upweighted (P4's
offset curve is the acceptance test). Then the A1/A2/A3 gates
(approaches/min ≥1, conversion ≥25%, chained aerials ≥2 — A3 metric
still needs building).
**Answers:** the core thesis question — can drills teach decisions?
**Cost:** the big one; expert design work, not just training runs.
**Notes:** this is GOALS.md's stated next work and has been idle since
08-03. The generalist line is partly a bet that master data contains
the decisions the drills couldn't encode — D1's result should inform
how urgent D5 remains.

### D6. Fight-state / pressure program
Human replay corpus (always --replay-dir; sessions exist), pressure
labels on top of the 47 situation labels, and the bot-Improoover loop
(bookmark → scenario manifest → practice → mine successes → mix). First
manifest = the 6 decider games (already chosen, never executed).
**Answers:** why the bot degrades under pressure; closes the 22→4
chain-gap class from the behavior side.
**Cost:** medium; mostly labeling + session time with Bradley.
**Notes:** two known flaws (shield-loops, camping) are unmeasurable vs
CPU — this direction *requires* human sessions.

### D7. Finish the transfer decider (paused mid-setup)
ms_g19r3_ep4 (broad-transfer candidate) vs ep4 champion; arms are
already blinded in scratchpad decider2, key unread; mewtwo games are
the informative ones.
**Answers:** whether the deployed champion should be the transfer
generalist instead of the popo specialist.
**Cost:** one session with Bradley. Cheapest open loop on the board.

### D8. Netplay passive threads (let them ride)
Resync-arm tail sampling in organic netplay; FoD/PS address probes
(~10s log lines on netplay stage games) → flip online stage merge;
#8 matchmaking-timeout RAM signature hunt.
**Answers:** closes the remaining netplay-truth gaps.
**Cost:** near-zero — piggybacks on any netplay session; just read the
logs afterward.

### D9. Infra debts (do opportunistically, not as a project)
Guard #4 (RAM sanity ranges), #5 (GPU-beam lockfile — tonight's
two-beam near-miss argues for it), #7 (convergence trust), #6 sub-lints
(train_delays consistency); streaming batch-count estimate (~5x over);
streaming val rigor (val split source unaudited); mmap-corpus path for
repeat 90M-frame runs.
**Answers:** nothing new — prevents relearning old lessons.
**Cost:** each is an hour-scale task; good filler during training runs.

### D10. Direct exhibition milestone
Take ≥1 stock off a real human over consented Direct with the champion
(or the generalist, if D1 surprises). The infrastructure excuse is
gone: menus, code entry, policy server, stage pinning all work.
**Answers:** the public proof-of-life gate in GOALS.md.
**Cost:** scheduling + a session; zero new code.

### D11. Coach (Track C) increment
The knowledge model v0 + viewer + situation stats exist. A thin next
increment: per-situation "what's good" queries exposed during replay
review (Bradley + brother-in-law's product idea), fed by the same
labels the bot program needs anyway.
**Answers:** long-term product direction; keeps the shared substrate
(labels, value model) honest.
**Cost:** incremental; do when D2's value model lands (shared prereq).

### D12. Low-tier at scale (the actual long-term goal)
Multi-character generalist or per-character lines for Mewtwo, Ganon,
Link, Zelda, ICs, G&W from the full erickfm/HF dataset (the second
ingestion path: slippi-frame-extractor parquet). Corpus sparsity per
matchup is the known wall (also Track C's wall).
**Answers:** the founding goal of the project.
**Cost:** large; gated on D1/D2 proving the generalist→sharpen recipe
on Fox first.

---

## A suggested reading of the board

**This week:** D1 (gate + live look, already committed) → D7 (cheapest
open loop) → D3-inventory (count what yeti actually contains — one
metadata pass, no training). D8/D9 ride along for free.

**The fork after D1:**
- Generalist shows *any* structured play → double down: D2 (offline RL
  on top) + D3/D4 (yeti + style conditioning). Fox stays the fast-eval
  substrate.
- Generalist still mode-averages at 26x data → the "data contains the
  decisions" bet weakens → D5 (decision-supervising expert) returns to
  the front, and D6's curated human corpus becomes the data story.

**Standing throughout:** D10 (the exhibition) should be scheduled the
moment any policy takes stocks off Bradley in casual Direct play — it's
a session, not a project.

The quiet meta-lesson of the last month: every time we invested in
*instruments before iterations* (canaries, deciders, probes, profilers)
the next science got 10x cheaper. Whichever direction wins, build its
measurement first — A3's chain-length metric for D5, a style-match
decider for D4, an offline-RL eval gate for D2.

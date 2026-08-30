# EVAL_DIRECTIONS batch 2 — D1b, E2, A2, C1, C4, B3 (ran 2026-08-29 23:32 → 23:51)

Chain `scripts/eval_directions_chain.sh`. Expert = first 800 erickfm ranked Fox
games (port 1); E2 on all 7,911. Bot sets as in A1: ep10 vs CPU, buttons-0.6 vs
CPU, mode-of-16 vs CPU, ep10 T=1.0 vs Bradley (08-28), ep10 T=0.5 vs Bradley
(08-28), B1/B2/B3 vs Bradley (08-29). Full tables in each `eval_runs/0829_*/RESULTS.md`.

## D1b — noise floor for TV distance (`0829_noise_floor/tv_floor.md`)

Same checkpoint + decode, separate batches, mean TV to expert:
F05 (T=0.5/0.5): 0.52 · 0.55 · 0.53 — spread **0.03**.
F10 (T=0.5/raw): 0.63 · 0.59 · 0.62 · 0.62 — spread **0.04**.
Per-situation cells (n≥100) move ≤0.05; small cells (n<60) move up to 0.2.
**Floor: 0.05 on the mean, 0.1 on a single well-populated situation.** The
B1/B2/B3 (0.46–0.47) vs ep10 (0.53–0.56) gap is 0.06–0.10 → at the floor,
consistent with the human read but not decisive on its own. mode-16 (0.72)
and F10-vs-F05 (0.10, the buttons-temperature effect) are resolved.

## E2 — rare-event coverage, full corpus (`0829_rare_events/RESULTS.md`)

| behaviour the bot lacks | expert count (7,911 games) | per game | ≈ window labels / epoch (stride 5) |
|---|---|---|---|
| dash initiation | 787,335 | 99.6 | 157,000 |
| dash-dance bursts | 84,010 | 10.6 | 16,800 |
| throw from a held grab | 29,711 | 3.8 | 5,900 |
| up-B start offstage | 33,798 | 4.3 | 6,800 |
| side-B start offstage | 16,084 | 2.0 | 3,200 |
| airdodge offstage (the bot's substitute) | 4,949 | 0.6 | 1,000 |

Mean grab hold 8.2 frames per grab entry (the expert throws fast).
**Not a data-quantity problem.** Every missing behaviour has thousands of
training labels per epoch; the corpus teaches up-B offstage 7× more often
than airdodge, and the bot does the opposite. The loss is downstream: the
states the bot reaches (dash probe), and/or what the recipe does with these
labels (window/stride alignment, sampling weights, the 17-bucket stick).

## A2 — edgeguard / recovery scorecards (`0829_edge_scorecard/RESULTS.md`)

Edgeguarding (opponent off, subject on): expert kills **17.3%** of episodes,
first option dash 42% / double-jump 17%. Bot vs Bradley: kills **2–6%**, first
option grab 12–20%, special 6–24%, *nothing* 12–40%; dash 0–6%. Vs the CPU
the bot kills 17% (ep10) — the dummy falls off on its own.

Recovering (subject off): expert gets back **79%**, dies 12.5%; routes
double-jump 24%, ledge 29%, up-B 16%, side-B 10%, airdodge 5%. Bot vs Bradley:
back **20–53%**, dies **47–67%**; routes side-B 28–39%, **airdodge 27–41%**,
double-jump 0–7%, up-B 5–9%, ledge ≤2%. Death rate given route: side-B
48–87% (expert 15%), airdodge 54–75% (expert 14%), none 73–86% (expert 42%).
The bot picks the worst routes and executes them worse. mode-16: 82% "no
route", 91% died.

## C1 — neutral exchanges (`0829_neutral_exchange/RESULTS.md`)

| set | exchanges/min | subject win rate (decided) | wins by whiff punish | losses by being whiff-punished |
|---|---|---|---|---|
| expert (ditto, ~50% by construction) | 32.9 | 50.7% | 25.1% | 24.4% |
| ep10 vs CPU | 55.1 | 63.4% | 4.4% | 19.1% |
| ep10 T=1.0 vs Bradley | 38.9 | 43.2% | 13.7% | 31.8% |
| ep10 T=0.5 vs Bradley | 43.6 | 38.6% | 15.6% | 25.0% |
| **B1 vs Bradley** | 42.7 | **20.1%** | 7.9% | **47.0%** |
| B2 vs Bradley | 37.0 | 38.4% | 18.7% | 26.2% |
| B3 vs Bradley | 37.0 | 38.3% | 13.1% | 23.8% |
| mode-16 vs CPU | 66.6 | 4.5% | 0 | 6.3% (67% got first-hit; 23% stall) |

Against Bradley the bot wins ~38–43% of exchanges (ep10, B2, B3) — closer
than the death counts suggest — and **B1 wins only 20%, losing 47% of its
exchanges by getting whiff-punished.** That is the opposite of the
loops/coach read (B1 "cleanest") and of Bradley's "probably better":
B1 commits to attacks that get punished; B2's "option spam" trades better.
Expert wins 25% of exchanges by whiff punish; the bot 4–19%. The dense
number the program wanted; it disagrees with the sparse ones, and n=304
exchanges for B1 makes it the better-powered read.

## C4 — death classifier (`0829_death_classifier/RESULTS.md`)

| | expert | ep10 cpu | ep10 vs Bradley (T=0.5) | B1 | B2 | B3 | mode-16 |
|---|---|---|---|---|---|---|---|
| deaths/game | 3.18 | 2.25 | 4.00 | 4.00 | 4.00 | 3.20 | 4.00 |
| **unforced (walk-off + fall)** | **27.9%** | 50% | 32% | 29% | **50%** | 25% | **81%** (75% walk-off) |
| edgeguarded | 39% | 11% | 18% | 29% | 23% | 6% | 6% |
| neutral kill | 13% | 33% | 32% | 32% | 23% | 56% | 13% |
| mean % at death (hit deaths) | 109 | 52 | 93 | 52 | 67 | 71 | 63 |

Even the expert loses 28% of stocks unforced (this detector counts late
recovery deaths with no hit in 150 f). The bot's distinctive number is
**% at death: 52–71 vs 109** — it dies at half the expert's percent, i.e.
it gets killed by single hits in neutral (neutral_kill 23–56% vs 13%) and
never gets to high percent. B2's 50% unforced matches Bradley's "goes
sideways off the stage sometimes". mode-16: 75% walk-offs, confirmed.

## B3 — policy entropy by situation (`0829_entropy/RESULTS.md`)

Buttons entropy at deploy (T=0.5) is 2.5–3.1 bits in most situations and
**4.1 bits in pummel_throw_decision** — the policy is *least* sure what to
press exactly where the expert is *most* decisive (expert option-entropy
0.46 bits there: throw, 86%). Main-stick entropy at deploy is 0.5–0.8 bits
everywhere (the T=0.5 stick is near-deterministic: the decode has already
removed most stick variety); c-stick ≈ 0. The over-confident/loop candidates
are neutral, approach, tech_chase (stick H 0.49–0.55 at deploy while the
expert's option entropy is 2.6–3.3); the dithering candidate is the grab:
high button entropy, expert decisive.

## What batch 2 changes

1. **E2 closes the data-quantity branch**: the recipe sees every missing
   behaviour thousands of times per epoch. Combined with the dash probe
   (head is calibrated at expert states) the gap is *state distribution* —
   the bot's own trajectories leave the states the corpus teaches from.
   That is the training question to work on: how to make the model's
   closed-loop states look like the expert's (DAgger-style relabelling of
   bot states, or training on the bot's own replays with expert labels).
2. **C1 is the "harder to hit" instrument** and it disagrees with the
   sparse metrics on B1 vs B2. Before trusting either: run C1 on the D1
   batches for its floor, and ask Bradley for a blind B1/B2 pair.
3. **A2 + C4 name the death mechanism**: wrong recovery route (side-B /
   airdodge, no double-jump, no ledge), executed at half the expert's
   success rate, from a state the expert would double-jump out of.
4. **B3 flips the pummel story**: it is not an over-confident loop; the
   policy is *uncertain* holding a grab (4.1 bits) and the sampled decode
   lands on non-throw buttons. Uncertainty where the data is decisive is a
   training-signal question (are throw frames under-weighted / lost to the
   stride?), which E2 says have 5,900 labels per epoch.

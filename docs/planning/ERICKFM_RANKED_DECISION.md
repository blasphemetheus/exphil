# Decision doc — erickfm/melee-ranked-replays (accept/reject)

Written 2026-08-19 (planning queue). Dataset facts verified against the
HF card this day.

## What it is

`huggingface.co/datasets/erickfm/melee-ranked-replays` — raw `.slp`,
**~850k unique replays at platinum+ rank** (1.19M listed entries;
each replay appears in BOTH players' character buckets → ~90%
duplication unless mirror — dedupe by filename via the
`metadata_a{N}.json` files). Six rank pairs (plat-plat → master-master),
labels per replay in the metadata JSONs. Organized as per-character
tarballs `CHAR/CHAR_{rank_pair}_a{1-6}.tar.gz`; 25 fighter buckets
(SHEIK/ZELDA and POPO/NANA merged). 1.43 TB total; the Fox bucket is
the ~180 GB slice quoted in HANDOFF_2026-08-12. MIT, anonymized.

## What it is for (and NOT for)

The 08-12 framing stands: **knowledge-model upgrade, not event
source.**

Value, in order:
1. **Rank-conditioned knowledge model** — situation_stats/Options
   (1.5M events today, rank-blind) rebuilt with a rank axis: "what do
   master-rank players choose in this situation" is a direct coach
   upgrade and a curation prior (mine snippets from top pairs only).
2. **Fight-state corpus at volume** — the standing gap is human
   pressure data; 850k ranked games is orders of magnitude more than
   the current human corpus, WITH a skill floor (plat+) the current
   corpus lacks.
3. **Rank-curated snippet mining** — the round-2 lesson says precision
   beats volume, but precision MINING needs a big haystack; failure→
   outcome links mined from master-master Fox games are the best
   available imitation targets for flaw counters.

NOT for: stage-event-dependent work. **Pre-3.18 replays carry no stage
events** (peppi.ex GameFrame: FoD side-platform heights and PS
transformation events nil on older replays) → per-stage-ledge (#25),
PS overlay modeling, and any FoD/PS geometry-dependent Situations
labels are absent/degraded on 2 of 6 competitive stages. Consumers
already nil-guard (Inspect: "event-less replays stay absent") — the
cost is coverage, not correctness. Own-recorded corpus remains the
source for stage-event work.

## Costs / risks

- **Storage/transfer**: 1.43 TB full / ~180 GB Fox. Goes to B2 per
  REPLAY_STORAGE (never only-local at this size). Download time is the
  real cost; tarball granularity lets us take rank pairs one at a time.
- **Parse compute**: Peppi handles old versions — the 08-19 commit
  extended the differential to pre-2.2.0 (9,087/9,092, zero
  divergences), so format risk is low; 850k games is still days of
  parsing → staged, cached, and only for slices we use.
- **Anonymized** → no per-player continuity (can't model an individual
  opponent's habits). Acceptable: the knowledge model is population-
  level anyway.
- **Netplay-era ranked** → all games are under netplay conditions;
  fine for a knowledge model, and arguably BETTER matched to the
  deploy regime (delay campaign) than local replays.

> **Stage 1 STARTED 2026-08-20 23:11** — pilot tarball
> `FOX/FOX_master-master_a1.tar.gz` (6.72 GB) downloading to
> `replays/erickfm_ranked/FOX/`. CAVEAT found at download time: the
> README documents `metadata/metadata_a{N}.json` but NO metadata files
> exist in the repo (API: "metadata does not exist on main") — rank
> labels come only from tarball names (sufficient for the pilot;
> re-check or contact erickfm before Stage 2 if per-replay char/rank
> detail matters).
>
> **Gate (a) PASSED 2026-08-21**: 300-replay sample (pilot_sample/),
> Peppi parse 300/300, peppi<->libmelee_ex differential 300/300, zero
> issues; 1.3k-21.6k frames/replay (median ~10.2k). The corpus is
> pipeline-clean. Gate (b) (rank-conditioned distributional check vs
> the existing corpus) still owed before Stage 2.

## DECISION: ACCEPT, staged — commit only to Stage 1 now

- **Stage 1 (pilot, ~cheap):** metadata JSONs + ONE Fox tarball
  (master-master slice). Gates: (a) Peppi parse divergence ~0 on the
  sample; (b) rank-split situation_stats on the pilot vs the existing
  corpus shows a MEASURABLE distributional difference (if plat+ play
  is indistinguishable from the current corpus in the option model,
  the rank axis buys nothing and we stop at "extra volume, someday").
- **Stage 2 (on pilot pass):** full Fox bucket (~180 GB) → B2 →
  knowledge model v3 with the rank axis; coach reads from it.
- **Stage 3 (conditional, needs its own prereg):** rank-curated
  snippet mining feeding fight-state mixes — enters the drill/mix
  pipeline under the standard dose discipline, after the retune
  program settles the base.
- **Reject/stop conditions:** parse divergences above the noise floor;
  Stage-1 gate (b) fails; or storage/egress costs exceed what B2
  budget tolerates (check before Stage 2).

Non-goal at every stage: training directly on the raw 850k (the
convC/curation program already proved blanket volume doesn't buy
chain skill — mixes stay curated and dosed).

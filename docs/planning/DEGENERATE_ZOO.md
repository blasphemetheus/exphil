# The Degenerate Strategy Zoo

Opened 2026-09-25 at Bradley's request, after the Mewtwo PPO iteration-300 head
turned out to have found one: *"that's just like an example of a degenerate
strategy. So that's something we can add to our uh zoo of degenerate strategies
… the solution is probably the zoo."*

**What this is for.** RL against a fixed opponent finds strategies that win the
sim and lose to a person. Each one we catch goes in here with (a) the human
description, (b) a **numeric signature in `StyleFingerprint` features**, and (c)
what it cost us to find. The point is that the second time a strategy appears we
detect it automatically instead of discovering it by playing a human.

**The load-bearing discovery (2026-09-25): degeneracy detection needs NO human
corpus.** The style gate was blocked for Mewtwo because no Mewtwo human
fingerprint corpus exists (coordination ASK 12). But a degenerate strategy shows
up as **drift from the PRIOR's own fingerprint**, measured through the same
harness — and the prior is always available. Humans are needed to say "is this
human-like"; they are *not* needed to say "this policy stopped playing Melee".

---

## Entry 1 — "Ledge roll → grab → back throw" (Mewtwo, PPO iter 300)

**Found:** 2026-09-25, by Bradley playing the exported policy in Dolphin.
**Artifact:** `checkpoints/mewtwo_ppo_v1_iter300_policy.bin`, head
`eval_runs/0924_mewtwo_ppo/v1/head_iter300.bin`.
**Human description (Bradley):** *"definitely was spot dodging and rolling to the
edge and then trying to grab and then like back throw … I think it only really
works against like a static opponent like what it had to face."*

### Numeric signature

Mean over the 8 fingerprint games of each selection arm, vs the untouched prior
through the same evaluator (`eval_runs/0924_mewtwo_ppo/eval/*/fingerprint.jsonl`):

| feature | prior (control) | iter70 | iter150 | **iter300** | iter300 vs prior |
| --- | --- | --- | --- | --- | --- |
| `roll_backward_per_min` | 1.81 | 2.33 | 6.86 | **23.82** | **13.1×** |
| `roll_forward_per_min` | 2.91 | 3.37 | 9.19 | **14.24** | **4.9×** |
| `grab_per_min` | 2.01 | 2.33 | 5.05 | **7.51** | **3.7×** |
| `throw_back_mix` | 0.062 | 0.000 | 0.125 | **0.250** | **4.0×** |
| `ledge_roll_mix` | 0.000 | 0.125 | 0.250 | 0.250 | 0 → 0.25 |
| `ledge_getup_mix` | 0.000 | 0.000 | 0.125 | 0.125 | 0 → 0.125 |
| `aerial_per_min` | 9.77 | 6.47 | 2.98 | **0.78** | **−92 %** |
| `spotdodge_per_min` | 0.388 | 0.129 | 0.259 | 0.259 | −33 % |

**The detector, stated as a rule:** rolls up >4×, grabs up >3×, `throw_back_mix`
up >3×, ledge mixes appearing from zero, **and `aerial_per_min` collapsing** —
that last one is the clincher, because a policy that has stopped throwing
aerials has stopped playing Melee.

**One correction to the human report, kept because it matters for the detector:**
**spotdodge is NOT elevated** (0.26 vs the prior's 0.39, i.e. *down* a third).
The impression of spotdodging was wrong; the behaviour is rolling. Do not put
`spotdodge_per_min` in this signature.

### It is a trajectory, not a cliff

`iter150` shows **the same signature at roughly half magnitude** (rolls 6.9/3.4×,
grabs 2.5×, aerials already −70 %). Bradley judged iter150 "definitely better"
live, and it is — but it is **on the same path, caught earlier**, not clean. So:

- Selecting an earlier checkpoint mitigates this; it does not solve it.
- The fingerprint moves monotonically with training, so it can be watched
  **during** a run and used as an early-stop signal, not only post hoc.

### Why the sim could not see it

Both `iter150` and `iter300` went **60W/0L** against the frozen prior, and the
200-game test returned 99.5 % with a clean 50.0 % control. The win-rate gate was
saturated and blind to this. Bradley's own explanation is the correct one: the
strategy *only works against a static opponent like what it had to face.*

### Cost of finding it this way

One 6,133 s training run, a 25-minute three-arm evaluation, and a human playing
two sessions. The fingerprint numbers above were already sitting in the eval
output the whole time — nobody had compared them. **That is the cheap win: the
comparison is read-only, takes seconds, and needs no GPU.**

---

## How to check a new arm (read-only, no GPU, no human)

`scripts/ppo_style_half.py` does this for Fox against the human corpus. For a
character with no human corpus, compare arms to the **prior control** instead —
the one-off used for Entry 1:

```python
# eval_runs/<run>/eval/{control,sel_*}/fingerprint.jsonl ; rows carry
# 'fingerprint' (sim) or 'features' (replays); mean each feature per arm and
# ratio against control. Watch: roll_*_per_min, grab_per_min, throw_back_mix,
# ledge_*_mix, aerial_per_min.
```

**Implemented September 25:** `scripts/degeneracy_check.py` compares same-harness
fingerprints to a prior control, writes hashed JSON evidence, and now runs at
the end of `mewtwo_eval_chain.sh`. It flags iter300's compound signature and
iter150's reduced-aerial rate; iter70 triggers neither. Tests cover five cases,
including missing data, character mismatch and zero baselines. Command:

```bash
python3 scripts/degeneracy_check.py \
  --baseline eval_runs/0924_mewtwo_ppo/eval/control \
  --candidate eval_runs/0924_mewtwo_ppo/eval/sel_head_iter{70,150,300} \
  --out eval_runs/0924_mewtwo_ppo/eval/degeneracy_report.json
```

**Interpretation correction:** the thresholds are retrospective heuristics fitted
to this observed case, not a validated classifier. Fewer aerials alone cannot
prove degeneracy or that a policy has "stopped playing Melee". The compound
signature and human playtest support this particular finding. The program uses
combined forward+backward roll rate, flags aerial reductions over50% for review,
requires at least8 finite samples/feature/arm, and keeps zero denominators null.
Event mixtures are sparse/noisy. Missing evidence is not a pass; no known
signature is not a promotion. Capture/opponent/stage settings must be matched.

**Still pending:** an in-training behavior tripwire with matched evaluation
rollouts. Do not threshold short training trajectories against these longer
eval captures, or turn this new heuristic into a blanket aerial quota.

---

## Open hypothesis, not a finding

**Why was Fox's PPO better behaved than Mewtwo's?** Bradley, 2026-09-25: *"the
results on the fox were a lot better, and that might just be because we had more
data, better data, um higher level play."*

Supporting facts, as facts: the Fox prior is V3.1-ep3 off a large high-level
corpus and is a genuinely strong bot (netplay chain records, a crowned lineage);
the Mewtwo prior is 418 deduped games / 5.24 M frames, validation loss 3.1043,
and self-destructs about every 7.6 s in its own ditto. Fox's PPO arm reached
91 % (not saturated) with only `aerial_per_min` drifting out of the human range;
Mewtwo's saturated at ~100 % and went degenerate.

**Untested alternative explanations**, which is why this is a hypothesis:
Mewtwo used a looser anchor (`--kl-coef 0.01` vs Fox's 0.05) and ran 300
iterations vs 200 — either could account for more drift without any appeal to
data quality. **A clean test exists: rerun Mewtwo PPO at `--kl-coef 0.05` for
200 iterations and compare the fingerprint drift.** Until that runs, do not
state the data-quality explanation as a finding.

---

## Entry 1 check applied to the opponent-pool candidate (2026-09-25 23:40)

`eval_runs/0925_mewtwo_pool/v2/head_iter200` (opponent pool: prior + v1 heads
70/150/300, v1 head100 held out; KL 0.01, 200 iterations). The checker was run
with **matched** baselines — each candidate arm against the prior control that
faced the *same* opponent — rather than one pooled baseline:

```bash
python3 scripts/degeneracy_check.py \
  --baseline eval_runs/0925_mewtwo_pool/eval/prior_control \
  --candidate eval_runs/0925_mewtwo_pool/eval/prior_candidate \
  --out eval_runs/0925_mewtwo_pool/eval/degeneracy_report_vs_prior.json
python3 scripts/degeneracy_check.py \
  --baseline eval_runs/0925_mewtwo_pool/eval/heldout100_control \
  --candidate eval_runs/0925_mewtwo_pool/eval/heldout100_candidate \
  --out eval_runs/0925_mewtwo_pool/eval/degeneracy_report_vs_heldout100.json
```

Both: `no_known_signature_detected`, zero alerts, 16 games per arm.

| feature (candidate / control) | vs prior | vs held-out head100 |
| --- | --- | --- |
| roll_backward_per_min | ×0.50 | ×0.59 |
| roll_forward_per_min | ×0.44 | ×0.61 |
| grab_per_min | ×0.41 | ×0.26 |
| throw_back_mix | 0.06 → 0.00 | 0.13 → 0.00 |
| ledge_roll_mix | 0.06 → 0.00 | ×1.0 |
| aerial_per_min | **×0.58** | ×0.75 |

Reading: the Entry 1 signature is rolls/grabs/back-throws UP and aerials DOWN
together. Here rolls, grabs and back-throws all went DOWN, so the compound
signature is absent, and the ledge-roll behaviour never appeared. The one
feature still drifting in the Entry 1 direction is aerials, ×0.58 against the
prior — just above the ×0.5 review line — and only ×0.75 against the stronger
opponent. That is the same reduced-aerial tendency v1 iter150 carried, milder.
It is worth watching across future arms, not an alert.

What this does and does not say: no Entry 1 signature, which is the only
signature the zoo has. A pool-trained policy could have found a *different*
exploit the zoo does not know yet; the 66 % against a held-out opponent that
crushes the prior 95-5 makes that less likely than for v1, but the playtest is
the only test that can add an Entry 2. **Pool candidate playtest still owed.**

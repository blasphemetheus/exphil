# 8a §6 verdict — ARhead vs INDhead (recipe-fixed, bracket 2)

**2026-08-30 20:10.** 8 × 120 s vs CPU per arm, T=0.5/buttons 0.5, delay 0,
headless. Knob + head-banner assertions passed both arms. AR ran the fused
sampler (12.5 inf/frame, staleness normal). Bracket 1
(`../0830_arhead_score/`) is discarded (plain-CE fit artifact).

## Health (no-collapse gate): PASS

| | AR | IND | ep10 ref |
|---|---|---|---|
| duration | 107.0s [42.9–124.5], 6/8 cap | 111.5s [70.8–124.8], 6/8 cap | 124.5s, 8/8 cap |
| frozen-input frac | 0.00 | 0.00 | – |
| loops/min | 0.18 | 0.34 | – |

## PRIMARY 1 — A2 recovery (subject offstage; expert n=11,187; AR n=48, IND n=47)

| | expert | **AR** | IND | ep10_cpu |
|---|---:|---:|---:|---:|
| first route: **up-B** | 16.3 | **8.3** | 0.0 | 14.3 |
| first route: double_jump | 24.8 | 4.2 | 2.1 | 7.1 |
| first route: **airdodge** | 4.8 | **20.8** | 53.2 | 28.6 |
| first route: none | 14.6 | 43.8 | 31.9 | 25.0 |
| outcome: died % | 12.7 | **35.4** | 51.1 | 60.7 |
| outcome: back % | 79.1 | **60.4** | 44.7 | 32.1 |

Rule check: up-B + double-jump share AR 12.5% vs IND 2.1% (**6×**, ≥2× ✓);
airdodge AR 20.8 vs IND 53.2 (**≤½** ✓); recovery deaths AR 35.4 < IND 51.1
✓ (per-route died% cells are n<10 — read the aggregate).

## PRIMARY 2 — live joint coincidence (from the arms' own replays)

| | expert (audit) | **AR** | IND |
|---|---:|---:|---:|
| P(stick up \| B) / P(stick up) | 3.2× | **2.20×** | 1.05× |
| offstage P(up \| B) / P(up) | 2.9× | **2.65×** | 1.97× |

The conditioning wire works live: the AR head presses B-with-up at 2.2–2.65×
its marginal (expert 2.9–3.2×); the independent control sits at ~1× — the
exact signature the joint-head audit said no trunk could fix.

## Secondary

- **B2 TV**: AR 0.62 vs IND 0.61 (diff 0.01 < 0.05 floor — "not worse" ✓).
  Both above ep10's 0.53: a 4-epoch frozen-trunk head refit is still behind
  the jointly-trained original overall.
- **C4**: unforced deaths AR 65% vs IND 61% (expert 27) — dominated by
  unforced falls, ≈ equal across arms; walk-offs AR 0%, IND 4.3%.
- taunts/min AR 1.63, IND 0.46 (expert baseline ~2.2–2.5 — both below).

## VERDICT (pre-registered rule, plan §6): **SIGNAL**

AR beats IND on the exact criteria the head was built for, with no collapse
and TV within the floor. Per the rule: AR becomes the default-head
CANDIDATE and **the next gate is Bradley's live look** (g6: the look gates
the recipe, not the metric).

Caveats for the look: absolute play is below ep10 (TV 0.62 vs 0.53; 6/8 vs
8/8 cap) — the frozen-trunk refit trades overall polish for the joint
structure. If the look confirms the up-B/recovery improvement, the plan's
items 8/9 (3-epoch unfreeze, AR vs IND control) are the path to getting
both.

Live command (local, deploy knobs per DEPLOY_KNOBS.md):

    devenv shell -- mix run scripts/play_dolphin_async.exs \
      --policy checkpoints/fox_gen_v1.1_ARhead_policy.bin \
      --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" --iso "$HOME/isos/melee.iso" \
      --character fox --stage final_destination --frame-delay 0 \
      --temperature 0.5 --buttons-temperature 0.5 --port 1 --opponent-port 2 \
      --replay-dir eval_runs/0830_livelook_ARhead --verbose

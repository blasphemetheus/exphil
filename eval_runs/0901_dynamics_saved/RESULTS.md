# G3b dynamics spike — RESULTS

f: (embed_t ⊕ action_13) -> delta(embed), 512x512 MLP,
52 train / 8 held-out replays
(545921 pairs), normalized space, 2 epochs.

- **held-out 1-step R^2: 0.995** (gate > 0.9)
- **k-step open-loop cosine (ground-truth actions, 100 rollouts):**

| k | cos sim |
|---|---:|
| 1 | 0.978 |
| 2 | 0.951 |
| 3 | 0.907 |
| 4 | 0.895 |
| 5 | 0.879 |
| 6 | 0.870 |
| 7 | 0.855 |
| 8 | 0.832 |
| 9 | 0.828 |
| 10 | 0.824 |

**Gate (declared pre-run): PASS — build V-rollouts on this**

Caveats: single-perspective embeds (opponent modeled only through the
embedding's opponent block); cosine in normalized space; no collision/
blast-zone hard constraints — a V-rollout consumer must treat rollouts
as SHORT-horizon (k <= 10) texture, not simulation truth
(GOTCHA: this is the learned rung of the fidelity ladder; headless
Dolphin savestates are the mechanics-true audit — INTERP_GEN_V1 G3b).

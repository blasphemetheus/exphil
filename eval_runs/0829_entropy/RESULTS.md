# Entropy by situation (B3)

Policy `fox_gen_v1_20260825_210355_ep10.bin`, 20 expert files, port 1, 80 frames per label.
Policy entropies in bits from the head logits at the expert's states (T=1 = the learned
distribution; @deploy = what sampling actually draws from). Buttons = sum of 8 Bernoulli
entropies (max 8); main/c = mean of x,y categorical entropies over 17 buckets (max 4.09);
shoulder max 2. Expert option-entropy = entropy of the expert's next-option histogram in
that situation (max ≈ 4.4 over ~21 options) — a different quantity, shown as the
"how diverse is correct play here" reference, comparable across rows not across columns.

| situation | frames | expert option-entropy (bits) | policy H buttons T=1 | H main T=1 | H c T=1 | H shoulder T=1 | H buttons @0.5 | H main @0.5 | H c @0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 80 | 3.23 (n=4988) | 5.36 | 1.81 | 1.01 | 0.92 | 2.58 | 0.49 | 0.04 |
| approach | 80 | 3.29 (n=2021) | 5.27 | 1.97 | 1.03 | 0.90 | 2.46 | 0.61 | 0.05 |
| retreat | 80 | 2.89 (n=1231) | 5.53 | 2.06 | 1.01 | 0.99 | 2.83 | 0.67 | 0.04 |
| advantage | 80 | 3.08 (n=1284) | 5.42 | 1.97 | 1.03 | 0.81 | 2.57 | 0.55 | 0.05 |
| disadvantage | 80 | 2.95 (n=319) | 5.93 | 2.31 | 1.06 | 1.06 | 3.27 | 0.73 | 0.03 |
| conversion_open | 80 | 3.56 (n=3157) | 5.66 | 2.18 | 1.08 | 0.98 | 3.00 | 0.79 | 0.06 |
| combo_active | 80 | 3.16 (n=1331) | 5.42 | 1.91 | 1.03 | 0.82 | 2.67 | 0.54 | 0.05 |
| tech_chase | 80 | 2.57 (n=268) | 5.32 | 1.94 | 1.01 | 0.90 | 2.47 | 0.55 | 0.03 |
| edgeguard | 80 | 2.88 (n=851) | 5.31 | 1.87 | 1.00 | 0.91 | 2.51 | 0.59 | 0.03 |
| pummel_throw_decision | 80 | 0.46 (n=102) | 6.29 | 1.89 | 0.98 | 0.97 | 4.12 | 0.44 | 0.02 |
| shield_pressure_theirs | 80 | 1.93 (n=97) | 6.00 | 2.27 | 0.98 | 1.40 | 3.62 | 0.68 | 0.02 |
| being_edgeguarded | 80 | 1.66 (n=192) | 5.72 | 1.94 | 0.95 | 0.91 | 3.04 | 0.55 | 0.02 |
| recovery_low | 80 | 2.01 (n=229) | 5.75 | 1.97 | 0.93 | 0.88 | 3.08 | 0.66 | 0.01 |
| recovery_high | 80 | 1.97 (n=137) | 5.50 | 2.02 | 0.96 | 0.85 | 2.78 | 0.61 | 0.02 |
| cornered | 80 | 2.94 (n=621) | 5.66 | 2.21 | 1.02 | 1.08 | 3.03 | 0.81 | 0.03 |
| offstage | 80 | 2.13 (n=366) | 5.63 | 1.87 | 0.95 | 0.80 | 2.96 | 0.57 | 0.02 |
| respawn_invincible | 80 | 2.22 (n=251) | 5.63 | 2.09 | 1.02 | 0.98 | 2.80 | 0.63 | 0.02 |
| percent_lead | 80 | 3.17 (n=1068) | 5.33 | 1.87 | 1.05 | 0.90 | 2.54 | 0.53 | 0.05 |
| percent_deficit | 80 | 3.59 (n=398) | 5.74 | 2.20 | 1.01 | 1.02 | 3.10 | 0.82 | 0.03 |

Read: rows where the policy's main/buttons entropy is LOW while the expert's option
entropy is HIGH are over-confident states (loop candidates). Rows where the policy is
high-entropy while the expert is decisive are dithering candidates.

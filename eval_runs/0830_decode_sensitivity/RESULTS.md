# D3 — decode-vs-model sensitivity

Sweep: eval_runs/0828_loop_rescore/buttons_sweep/report.json · knob values [0.5, 0.6, 0.7, 1.0] (per group entry, in order).
Spearman rho between knob and per-arm mean; |rho| ≥ 0.8 with range ratio
> 1.5 = the decode steers this metric — do not read it as model evidence.

| metric | rho(knob) | per-arm means | verdict |
|---|---:|---|---|
| dpad_per_min | 1.00 | 112.27 · 170.54 · 247.85 · 430.16 | **decode-driven** |
| longest_action_run | 1.00 | 215.20 · 275.80 · 371.00 · 373.80 | **decode-driven** |
| longest_input_run | -1.00 | 16.00 · 8.40 · 5.60 · 4.00 | **decode-driven** |
| loops_per_min | 1.00 | 1.57 · 1.96 · 1.96 · 2.35 | decode-leaning (small range) |
| action_long_frac | 0.80 | 0.26 · 0.24 · 0.30 · 0.34 | decode-leaning (small range) |
| input_long_frac | -0.80 | 0.00 · 0.00 · 0.00 · 0.00 | decode-leaning (small range) |
| max_loop_repeats | 0.80 | 10.80 · 10.40 · 12.00 · 13.40 | decode-leaning (small range) |
| taunts_per_min | 0.00 | 2.25 · 1.17 · 2.94 · 2.15 | noisy (big range, no order) |

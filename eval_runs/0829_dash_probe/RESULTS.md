# Dash probe — decode artifact or learned gap?

Policy `fox_gen_v1_20260825_210355_ep10.bin`, 20 expert files, port 1, 16 samples/frame,
deploy main-stick T=0.5. Full tilt = stick ≤0.1 or ≥0.9 (buckets 0–1, 15–16 of 17).

| at frames where the expert… | n | expert stick full-tilt | policy mass on full tilt, T=0.5 | T=1.0 | argmax is full tilt | sampled flick rate (deploy T=0.5, n=16) | policy mass mid-zone T=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| …initiates a DASH next frame | 400 | 69.3% | **74.5%** | 62.4% | 76.8% | **74.3%** | 10.2% |
| …stays STANDING (control) | 400 | 0.0% | 0.7% | 3.7% | 0.5% | 0.7% | 1.8% |


**Verdict:** LEARNED/STATE: the head produces full tilt at dash-initiation states (74.5% mass, 74.3% sampled flicks vs expert 69.3%). The missing dash is upstream of the decode — the bot does not get itself into these states (WAIT 0.4% of its frames). Training / behaviour question, not a decode one.

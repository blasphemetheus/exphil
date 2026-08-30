# Mode-of-N mechanism — per-frame button marginals by situation

Policy `fox_gen_v1_20260825_210355_ep10.bin`, 20 expert files, port 1; buttons at T=0.5, sticks T=0.5,
16 joint draws for the majority vote. Marginals are the policy's per-frame press probabilities at
the expert's states. A majority vote over joint draws can only press a button when its marginal is
> 0.5 on that frame; "mode presses NOTHING" is the empirical vote outcome.

| situation | n | P(A) | P(B) | P(X) | P(Y) | P(Z) | P(L) | P(R) | P(dup) | frames with any button > 0.5 | mode has jump | mode has B | mode presses NOTHING | expert pressing jump / B / nothing (this frame) | modal stick (x,y bucket of 17; 8=center) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| neutral | 150 | 9.2 | 7.9 | 9.9 | 13.3 | 4.4 | 12.2 | 11.7 | 2.5 | 22.0 | 10.0 | 6.0 | **72.7** | 11.3 / 3.3 / 69.3 | (15,8) in 24.0% |
| cornered | 150 | 13.4 | 7.1 | 9.4 | 11.2 | 6.3 | 23.3 | 20.8 | 2.5 | 31.3 | 6.7 | 1.3 | **62.7** | 7.3 / 1.3 / 56.0 | (0,8) in 29.3% |
| edge_danger | 150 | 2.5 | 3.1 | 2.2 | 2.8 | 2.5 | 2.0 | 2.2 | 3.2 | 0.0 | 0.0 | 0.0 | **100.0** | 0.7 / 0.0 / 99.3 | (8,8) in 34.7% |
| offstage | 150 | 7.3 | 20.9 | 11.4 | 13.6 | 4.1 | 8.9 | 7.3 | 2.8 | 20.7 | 6.7 | 13.3 | **77.3** | 7.3 / 16.0 / 72.7 | (0,8) in 23.3% |
| recovery_low | 150 | 8.4 | 22.9 | 12.7 | 15.6 | 4.4 | 8.7 | 9.2 | 2.7 | 22.7 | 7.3 | 16.7 | **70.7** | 12.0 / 16.7 / 66.0 | (8,8) in 29.3% |
| recovery_high | 150 | 8.1 | 17.4 | 10.7 | 12.7 | 3.3 | 7.6 | 5.9 | 2.4 | 10.0 | 7.3 | 10.7 | **78.7** | 7.3 / 10.0 / 78.0 | (0,8) in 37.3% |
| being_edgeguarded | 150 | 7.6 | 19.1 | 12.8 | 15.5 | 4.1 | 9.3 | 7.9 | 2.7 | 18.0 | 7.3 | 17.3 | **70.7** | 12.0 / 12.0 / 71.3 | (0,8) in 28.7% |

Read: if every button marginal sits well under 0.5 in offstage / recovery states while the expert
presses B or jump on 20–40% of those frames, the mode is structurally unable to recover — the
press is a 1–2-frame event whose per-frame probability never crosses one half. Sampling fires it
within a few frames; the vote never does. The same in neutral explains the walk-off: modal stick
held, no jump.

## Read (2026-08-30 01:15)

The sparse-press hypothesis is HALF right. Per frame, every button marginal is
≤23%, and a majority vote can only press a button on the 10–31% of frames
where some marginal exceeds 0.5 — so the vote presses *nothing* on 63–79%
of frames. But the expert also presses nothing on 56–78% of frames, and the
vote's per-frame B rate offstage (13–17%) matches the expert's (12–17%).
Per-frame press *rates* are not what the vote destroys.

What it destroys is the STICK and the TIME STRUCTURE:

1. **Modal stick is a hard hold in one direction.** In cornered / offstage /
   recovery_high / being_edgeguarded the single most common vote output is
   main stick (0, 8) — full LEFT, y centre — on 23–37% of frames, or dead
   centre (8, 8). Never up. So when the vote does press B, the stick is
   sideways or neutral: side-B (illusion) or a laser, never up-B. And
   holding full-left with no jump from the P1 side is the walk-off.
2. **Holds replace edges.** The vote returns the same modal input frame after
   frame; a jump or B needs a press EDGE (released then pressed). Sampling
   produces edges by construction; the mode produces plateaus.
3. **edge_danger is the clean case**: 100% "nothing" (expert 99.3% too) with
   the modal stick at centre — the expert is teetering and about to act on
   the next frame; the mode has no next frame that differs from this one.

So the interp answer to "why does mode-of-N run off stage": the mode of a
per-frame policy is a *held direction with no button edges*. Direction is
the densest signal (present every frame), presses are sparse, and variety
in the stick — which is what carries the up-B and the jump — is exactly what
the vote averages away (B3: stick entropy at deploy is already only 0.5–0.8
bits; the vote takes it to ~0). Left specifically: the modal direction is
full-left in the edge situations sampled from a port-1-normalized corpus;
port 2 would flip it (untested).

Generalizes: any mode-seeking decode of a per-frame policy keeps the dense
channel (stick hold) and loses the sparse one (press edges + stick
excursions). Argmax buttons (08-28) and mode-of-16 (08-29) are the same
failure.

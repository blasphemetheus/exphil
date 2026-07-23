# Stateful mamba-2 step — design (the 60fps / rollback path)

Draft 2026-07-22 (design only — written with no GPU; the kernel is
untested and must be validated, see "Verification"). Decision #3's NEXT
BUILD. Lives in edifice (the arch lib); exphil wires the agent path.

## Why

Live inference today re-runs the WHOLE window every frame: to act on
frame t the agent embeds frames [t-W+1 .. t] and runs the full mamba_2
forward, because a cold recurrent state is off-distribution (GOTCHA #65 —
step paths must replicate windowed warmup). That is O(W) work per frame
(W=60) and needs a warmup every time. Mamba's whole selling point is a
recurrent form that is **O(1) per step**; we don't use it live.

Two things need the recurrent form:
1. **60 fps headroom / lower latency** — O(1) vs O(W) per frame, no
   per-frame warmup recompute.
2. **Rollback netcode** — the netcode saves state at frame N, predicts,
   and on a misprediction restores frame N's state and re-simulates. That
   requires a small, cheaply serializable per-frame state to save/restore
   — exactly the SSM hidden state, and exactly what the windowed forward
   does NOT expose.

mamba_2 has no `step/2` today (retnet, liquid, gated_ssm do — use them as
the interface template). This is net-new for the mamba family.

## The recurrence

SSD training uses the chunked scan (mamba_ssd.ex: intra-chunk matmul +
inter-chunk scan). The mathematically-equivalent recurrent step, per
token, per (head, state) is:

    h_t = Ā_t ⊙ h_{t-1} + B_t x_t          # state update  [H, N, P]
    y_t = C_t · h_t + D ⊙ x_t              # readout       [H, P]

where Ā_t = exp(dt_t · A) is the per-step discretization, x_t the (post
in-proj, post-conv) input, and B_t/C_t/dt_t the selective params for
frame t. This is O(H·N·P) — constant in sequence length. Producing the
SAME y_t as the windowed scan at frame t (given equal warmup) is the
correctness bar.

## Carry state (what step/2 threads and rollback saves)

Two pieces, both plain tensors (no structs — must serialize trivially):

1. **ssm_state** `h`: shape `[batch, heads, d_state, d_head]` (the
   `h_t` above). Zeros = cold.
2. **conv_state**: mamba's causal depthwise conv1d over the input needs
   the last `conv_size-1` inputs. A ring buffer `[batch, conv_size-1,
   d_inner]`. Zeros = cold. (Easy to forget — the block is conv → SSM;
   the conv is stateful too.)

Carry = `{ssm_state, conv_state}` per layer → a list across layers. Keep
it a flat tensor tuple so rollback save/restore is a copy, not a walk.

## Interface (scaffold — signatures, not kernel)

    # edifice: Edifice.Mamba2 (or the exphil mamba_ssd block)
    @doc "Initial carry (cold = zeros) for a batch."
    def init_state(config, batch \\ 1)

    @doc "One frame. carry in, {y_t, carry'} out. O(1) in seq len."
    defn step(params, x_t, carry)
      # x_t: [batch, d_model]   carry: per-layer {ssm_state, conv_state}
      # 1. in-proj + gate split
      # 2. conv_state update (shift-in x_t) -> conv out
      # 3. selective params (dt, B, C) from conv out
      # 4. Ā = exp(dt*A); h' = Ā*h + B*x; y = C·h' + D*x
      # 5. out-proj(gate * y) -> y_t ; return {y_t, {h', conv'}}

Warm start on connect: run the existing windowed forward over a real
prefix, but capture the FINAL h and conv states (not just the output) and
seed the carry — avoids the cold-state off-distribution hit (#65) while
still going O(1) thereafter. This means the windowed path needs a
"return final state" mode; add it as an opt so training is unchanged.

## Verification (the whole point — untestable until GPU frees)

1. **Parity test**: feed one random sequence [1, T, d_model] through
   (a) the windowed SSD forward and (b) T calls to `step` seeded from a
   warmup prefix; assert `y_t` equal within bf16 tolerance for t past
   warmup. This is the gate — a stateful step that doesn't match the
   trained forward silently degrades play.
2. **Rollback test**: run to frame N, snapshot carry, run M more, restore,
   re-run M — assert identical outputs (deterministic state save/restore).
3. **Drift check**: over a full ~28k-frame game in bf16, compare step vs
   windowed periodically — bf16 state accumulation may drift; if it does,
   keep `h` in f32 (it's small) while compute stays bf16.
4. **Latency**: confirm per-step << windowed (target the 8.9 ms windowed
   figure dropping to sub-ms).

## Risks / open questions

- **bf16 state drift** over long games (mitigation: f32 state above).
- **dt discretization cost per step** — `exp(dt*A)` every frame; cheap
  but not free; measure.
- **Parity is subtle**: the SSD chunked matmul and the naive recurrence
  can disagree at the edges (initial-state handling, D skip term); the
  parity test must past-warmup only.
- **Scope creep**: this is the mamba-2 step ONLY. The multi-arch ladder
  (measure the others, don't assume — decision #3) is downstream; don't
  generalize the interface prematurely, but DO match retnet/liquid's
  existing `step` shape so the ladder can call them uniformly.

## Next action

Implement `init_state` + `step` + windowed "return final state" mode in
edifice behind the parity test, once training frees mix. Start from
mamba_ssd.ex's `ssd_scan` (the recurrent inter-chunk path already walks
states — the per-step logic is the inner body of that scan lifted out).

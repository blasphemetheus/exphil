# PPO status: NOT ready — the r17 fallback branch needs real work

Investigated 2026-07-23 (attempted the "PPO smoke" that has sat on the
board as *repaired, UNRUN*). **"Repaired" was optimistic.** The script had
never executed even once — it fails on its very first line of work — and
behind the surface breakages sits an architecture mismatch that a smoke
test can't paper over.

**Why this matters:** `R17_ACCEPTANCE_2026-07-23.md` makes PPO
approach-shaping the fallback if r17a doesn't move gate-10. That fallback
is NOT a quick pivot. Budget real implementation time, or plan a
different escalation.

## What I fixed (committed — these are genuine, it now gets much further)

1. `@flag_groups` module attribute at .exs top level → `cannot invoke @/1
   outside module`. Script died instantly, every invocation.
2. Same attribute referenced again in the help call.
3. `import PPOScript` — compile-time dep on a same-file top-level module;
   Elixir rejects it. Now fully-qualified calls.
4. `action_to_controller/1` built a `ControllerState` with a `buttons:`
   map and flat `main_x`/`c_x` floats. The real struct has
   `main_stick: %{x,y}` / `c_stick: %{x,y}` and flat `button_*` fields →
   KeyError. Now delegates to `Networks.to_controller_state/2` (what the
   live agent uses).
5. `PPO.load_pretrained/2` did a raw `binary_to_term`. Every policy since
   2026-07-16 is the Edifice manifest format (flag byte + meta size) →
   "invalid or unsafe external representation". Now uses
   `Checkpoint.load_policy/2`, which handles both formats.
6. `Embeddings.Game.embed(game_state, embed_config)` — real signature is
   `embed(gs, prev_action, own_port, opts)`, so the config was being fed
   into the prev-action slot. Now `embed(gs, nil, 1, config: ...)`.

## What is NOT fixed (design-level — stopped here deliberately)

7. **Architecture mismatch (the blocker).** `PPO.new/1` builds a fresh
   `ActorCritic.build_combined` **MLP** (`hidden_sizes: [512, 512]`) and
   then `merge_params` copies the pretrained policy's params into it. But
   our imitation policies are **temporal Mamba-2 trunks** (hidden 256).
   An MLP is not a Mamba trunk — it dies with
   `dot/zip ... 512 does not equal 256`. Fine-tuning r16 with PPO
   therefore *cannot work as written*: it needs an actor-critic built on
   the SAME temporal backbone (Mamba trunk + value head), not a
   parallel MLP. `merge_params`' "handles slightly different
   architectures" comment badly understates this.
8. **Head-shape drift.** `sample_action/1` expects a per-head map
   (`policy_logits.buttons`, `.main_x`, …); the ActorCritic's `policy`
   output is a single tensor. The whole rollout path (sample → log-prob →
   `stack_actions`) is written against a head-map API that no longer
   matches.

Even with random init (no `--pretrained`), the loop dies at (8) — so the
rollout→advantage→update machinery has never been exercised end-to-end
either. Nothing about PPO is validated.

## What a real repair looks like

1. Build the actor-critic from the shared policy trunk: reuse
   `Networks.Policy.build_temporal_trunk` (as `ProbeRegularizer` does) and
   attach a value head, so pretrained params load by name and the
   backbone matches. Drop `ActorCritic.build_combined`'s parallel MLP for
   the fine-tuning path.
2. Reconcile the head API: make `sample_action` / `compute_log_probs` /
   `stack_actions` consume the same 6-head shape the imitation policy and
   `Networks.Policy.sample` already use.
3. Only then a mock smoke, then a real short Dolphin run.

## Bearing on r17

- r17a (imitation-shaping) is unaffected and stays the next run.
- If r17a FAILS the primary gate, do **not** assume PPO is a week-1
  option. Either fund items 1–3 above, or consider cheaper escalations
  first (heavier opener weight, more curated go-in data, seeded self-play
  from `selfplay_rollouts.exs` for interactive neutral).

# Platform X-silence attribution by state-level patching (W1 follow-up,
# the fast-tracked causal-attribution tier: patching on contrastive pairs).
#
# Finding to explain (2026-08-04): on YS platforms the X (jump-cancel)
# head is hard-off (fire 0.0000, max logit < 0 over ~5400 frames) while
# the down+B shine motif keeps running — the absorber mechanism.
#
# Within the YS replays, platform-vs-ground context differs ONLY in own
# position + window history (same stage flags, x ~= -42 in both, dummy
# static at +42): the leading suspect is the OWN-Y channel. Both
# directions, on matched JC-phase states (reflector af>=3 — the states
# where X must fire):
#
#   RESTORE (platform windows, X silent): patch y -> ground; does X wake?
#   KILL    (ground windows, X firing):   patch y -> 23.45;  does X die?
#
# each at two strengths: current-frame-only vs whole-window (does the
# context live in the last frame or accumulate through the GRU?).
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_platform_patch.exs \
#     [--policy checkpoints/ms_g4_d2mix.bin] [--delay-id 3] \
#     [--out eval_runs/interp/platform_patch.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, delay_id: :integer, out: :string]
  )

policy_path = opts[:policy] || "checkpoints/ms_g4_d2mix.bin"
delay_id = opts[:delay_id] || 3
out_path = opts[:out]

# Ground reflector family (multishine JC states; margin-cartography
# conventions) — X is expert-correct at af>=3.
reflector = [360, 361, 362]
plat_y = 23.45

Output.banner("Platform patch probe (own-y attribution)")
Output.config([{"Policy", policy_path}, {"Delay id", delay_id}])

loaded = Activations.load_heads(policy_path)
window = loaded.window

load = fn path ->
  {:ok, replay} = Peppi.parse(path)

  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
end

patch_y = fn frames, y ->
  Enum.map(frames, fn f ->
    p = f.game_state.players[1]
    %{f | game_state: %{f.game_state | players: Map.put(f.game_state.players, 1, %{p | y: y})}}
  end)
end

embed = fn frames ->
  ds = Activations.embed_frames(frames, loaded.config, delay_id: delay_id)
  Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
end

# JC-phase sample indices (t >= window so full windows exist)
sample = fn frames, plat? ->
  frames
  |> Enum.with_index()
  |> Enum.filter(fn {f, t} ->
    p = f.game_state.players[1]

    t >= window and p.action in reflector and p.action_frame >= 3 and
      if plat?, do: p.y > 15, else: p.y < 15
  end)
  |> Enum.map(fn {_, t} -> t end)
end

# Windows from a base tensor, optionally splicing the last row from an
# alternate tensor (current-frame-only patches).
logits_at = fn ts, base_emb, last_emb ->
  ts
  |> Enum.chunk_every(256)
  |> Enum.flat_map(fn chunk ->
    wins =
      Enum.map(chunk, fn t ->
        hist = Nx.slice_along_axis(base_emb, t - window + 1, window - 1, axis: 0)
        last = Nx.slice_along_axis(last_emb, t, 1, axis: 0)
        Nx.concatenate([hist, last], axis: 0)
      end)

    out = loaded.predict_fn.(loaded.params, Nx.stack(wins))
    buttons = elem(out, 0)
    Enum.zip(Nx.to_flat_list(buttons[[.., 1]]), Nx.to_flat_list(buttons[[.., 2]]))
  end)
end

stat = fn pairs, label ->
  n = length(pairs)
  xs = Enum.map(pairs, &elem(&1, 1))
  bs = Enum.map(pairs, &elem(&1, 0))
  fire = Enum.count(xs, &(&1 > 0.0)) / max(n, 1)

  Output.puts(
    "  #{String.pad_trailing(label, 22)} n=#{n} X_mean=#{Float.round(Enum.sum(xs) / max(n, 1), 3)} " <>
      "X_fire=#{Float.round(fire, 4)} X_max=#{Float.round(Enum.max(xs, fn -> 0.0 end), 2)} " <>
      "B_mean=#{Float.round(Enum.sum(bs) / max(n, 1), 3)}"
  )

  %{label: label, n: n, x_fire: fire, x_mean: Enum.sum(xs) / max(n, 1)}
end

results =
  for {name, path, plat?, patch_to} <- [
        {"RESTORE r2 (plat->ground)", "eval_runs/0804_stage_yoshis_story/r2.slp", true, 0.0},
        {"RESTORE r3 (plat->ground)", "eval_runs/0804_stage_yoshis_story/r3.slp", true, 0.0},
        {"KILL r1 (ground->plat)", "eval_runs/0804_stage_yoshis_story/r1.slp", false, plat_y}
      ] do
    frames = load.(path)
    ts = sample.(frames, plat?)
    Output.puts("#{name}: #{length(ts)} JC-phase windows")

    base_emb = embed.(frames)
    patched_emb = embed.(patch_y.(frames, patch_to))

    [
      stat.(logits_at.(ts, base_emb, base_emb), "baseline"),
      stat.(logits_at.(ts, base_emb, patched_emb), "y-current"),
      stat.(logits_at.(ts, patched_emb, patched_emb), "y-window")
    ]
    |> then(&%{experiment: name, variants: &1})
  end

if out_path do
  File.mkdir_p!(Path.dirname(out_path))
  File.write!(out_path, Jason.encode!(results))
  Output.success("Wrote #{out_path}")
end

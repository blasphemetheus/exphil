# Input attribution for the basin B-decision (INIT_FORENSICS option 2).
#
# Hypothesis-driven saliency: for deep-basin windows, differentiate the B
# LOGIT (not the argmax logit — Attribution.saliency's objective) w.r.t.
# the input window and aggregate |grad x input| into the empirically
# discovered dimension groups. The contrast: do escapers' B-decisions load
# on the PREV-ACTION dims (release-when-prev-B — the edge rule) while
# failed seeds load elsewhere (or nowhere)?
#
# Windows probed: the live_held deep-basin variant (prev = B held) — the
# exact state where release must fire — and live_absorbed (prev = no B)
# for the press decision.
#
# Usage:
#   XLA_TARGET=cpu mix run scripts/probe_b_attribution.exs \
#     [--policies "checkpoints/ms_crouch_*.bin"]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Bridge.ControllerState
alias ExPhil.Data.{Peppi, RecoverySynth}
alias ExPhil.Interp.{Activations, Attribution}
alias ExPhil.Training.{Data, Output}

{opts, _, _} = OptionParser.parse(System.argv(), strict: [policies: :string])
policy_glob = opts[:policies] || "checkpoints/ms_crouch_*.bin"
policies = Path.wildcard(policy_glob) |> Enum.sort()

Output.banner("B-logit attribution (deep basin)")

# ---------------------------------------------------------------------------
# Deep-basin windows, live_held and live_absorbed prev regimes (same
# construction as probe_crouch_boundary.exs)
# ---------------------------------------------------------------------------
fixture = "test/fixtures/replays/fox_multishine_closed.slp"
{:ok, replay} = Peppi.parse(fixture)

frames =
  replay
  |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
  |> Enum.reject(&(&1.game_state.frame < 0))
  |> Enum.reject(fn %{controller: c} ->
    c.main_stick.x < 0.25 and c.main_stick.y > 0.4 and not c.button_b and not c.button_x
  end)

block = RecoverySynth.build_crouch(frames, port: 1, max_af: 40, lead_in: 16, ratio: 0.001)
n = length(block)
tail_start = n - 42
base = hd(block).game_state.frame

renumbered =
  block
  |> Enum.with_index()
  |> Enum.map(fn {f, i} -> %{f | game_state: %{f.game_state | frame: base + i}} end)

down = %ControllerState{
  main_stick: %{x: 0.5, y: 0.0},
  c_stick: %{x: 0.5, y: 0.5},
  l_shoulder: 0.0,
  r_shoulder: 0.0,
  button_a: false,
  button_b: false,
  button_x: false,
  button_y: false,
  button_z: false,
  button_l: false,
  button_r: false
}

variant = fn held? ->
  renumbered
  |> Enum.with_index()
  |> Enum.map(fn {f, j} ->
    if j >= tail_start, do: %{f | controller: %{down | button_b: held?}}, else: f
  end)
end

embed_windows = fn fr ->
  ds =
    fr
    |> Data.from_frames()
    |> Data.precompute_frame_embeddings(use_prev_action: true, show_progress: false)

  emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
  {total, _} = Nx.shape(emb)

  # Deep-basin windows only: last frame at SquatWait af >= 10 (position
  # tail_start + 2 + 9 onward).
  deep_from = tail_start + 11

  windows = for j <- deep_from..(total - 1), do: Nx.slice_along_axis(emb, j - 15, 16, axis: 0)
  Nx.stack(windows)
end

windows = %{held: embed_windows.(variant.(true)), no_b: embed_windows.(variant.(false))}

# Dimension groups. discover_dims' :prev_action probe controller never
# toggles button B, so it misses the dim that matters here — replace it with
# the full contiguous 13-run from prev_action_dim_range, and add :prev_b,
# the single dim that flips when ONLY prev-B toggles (located empirically).
[prev_offset, 13] = Attribution.prev_action_dim_range()

prev_b_dims =
  (fn ->
     gs = hd(renumbered).game_state
     e0 = ExPhil.Embeddings.Game.embed(gs, down, 1) |> Nx.backend_transfer(Nx.BinaryBackend)

     e1 =
       ExPhil.Embeddings.Game.embed(gs, %{down | button_b: true}, 1)
       |> Nx.backend_transfer(Nx.BinaryBackend)

     Nx.not_equal(e0, e1)
     |> Nx.to_flat_list()
     |> Enum.with_index()
     |> Enum.filter(fn {d, _} -> d == 1 end)
     |> Enum.map(&elem(&1, 1))
   end).()

groups =
  Attribution.discover_dims(nil, use_prev_action: true)
  |> Map.put(:prev_action, Enum.to_list(prev_offset..(prev_offset + 12)))
  |> Map.put(:prev_b, prev_b_dims)

Output.puts("Dim groups: #{inspect(Map.new(groups, fn {k, v} -> {k, length(v)} end))} prev_b=#{inspect(prev_b_dims)}")

# ---------------------------------------------------------------------------
# |grad x input| of the B logit, per seed
# ---------------------------------------------------------------------------
b_saliency = fn predict_fn, params, states ->
  grads =
    Nx.Defn.jit_apply(
      fn p, s ->
        Nx.Defn.grad(s, fn s2 ->
          out = predict_fn.(p, s2)
          buttons = elem(out, 0)
          Nx.sum(buttons[[.., 1]])
        end)
      end,
      [params, states],
      compiler: EXLA
    )

  grads |> Nx.multiply(states) |> Nx.abs() |> Nx.sum(axes: [1])
end

live = %{
  "ms_crouch_a" => "escape/universal",
  "ms_crouch_b" => "escape",
  "ms_crouch_c" => "escape/universal",
  "ms_crouch_d" => "escape",
  "ms_crouch_e" => "FAIL/silent",
  "ms_crouch_f" => "FAIL/silent",
  "ms_crouch_g" => "FAIL/hold-B",
  "ms_crouch_h" => "FAIL/hold-B",
  "ms_crouch_i" => "escape/universal",
  "ms_crouch_j" => "FAIL/hold-B",
  "ms_crouch_k" => "escape",
  "ms_crouch_l" => "FAIL/oscillator"
}

IO.puts(
  String.pad_trailing("seed", 13) <>
    String.pad_trailing("variant", 8) <>
    String.pad_trailing("prev_all", 10) <>
    String.pad_trailing("prev_b", 9) <>
    String.pad_trailing("own_action", 12) <> "outcome"
)

for path <- policies do
  seed = Path.basename(path, ".bin")
  loaded = Activations.load_heads(path)

  for {vname, states} <- windows do
    sal = b_saliency.(loaded.predict_fn, loaded.params, states)
    shares = Attribution.group_shares(sal, groups)

    mean_share = fn g ->
      shares[g] |> Nx.mean() |> Nx.to_number() |> Float.round(3)
    end

    IO.puts(
      String.pad_trailing(seed, 13) <>
        String.pad_trailing("#{vname}", 8) <>
        String.pad_trailing("#{mean_share.(:prev_action)}", 10) <>
        String.pad_trailing("#{mean_share.(:prev_b)}", 9) <>
        String.pad_trailing("#{mean_share.(:own_action)}", 12) <> "#{live[seed]}"
    )
  end
end

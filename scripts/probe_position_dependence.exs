# Position-dependence probe (0831 live-look follow-up) — Bradley's empirical
# question: "does it actually value where the opponent is?" The v1.2-ARrefit
# live look: no positional play, same behavior at either ledge/center, spams
# shield-grab/multi-jab regardless of opponent position.
#
# Two separable claims, measured mechanistically on the SAME states:
#
#   PERCEPTION — does the output distribution move AT ALL when the
#     opponent's position is counterfactually changed? (W2 machinery,
#     upgraded from B/X logits to all heads.)
#   DIFFERENTIATION — does it move DIRECTIONALLY: place the opponent just
#     left vs just right of the bot and read the main-stick expectation.
#     approach_delta = E[main_x | opp right] - E[main_x | opp left];
#     ~0 = position-blind steering, >0 = steers toward the opponent.
#
# States come from the LIVE-LOOK replays by default (the states where the
# complaint lives — the bot's own games vs Bradley), bot on --player-port.
# Every variant re-embeds the same frames with a perturbed opponent and
# runs the same trunk+heads; deltas are within-policy so AR teacher-forcing
# uses a fixed neutral prefix throughout (prefix confound cancels).
#
# Variants: baseline · mirror (opp x -> -x) · far (+120, W2's consultation
# knob) · close_left / close_right (opp at self x -/+ 40, self y).
#
#   mix run scripts/probe_position_dependence.exs \
#     --set ARrefit=checkpoints/fox_gen_v1.2_ARrefit_policy.bin \
#     --set INDrefit=checkpoints/fox_gen_v1.2_INDrefit_policy.bin \
#     --replays 'eval_runs/0831_livelook_v12ar/2026-08-Mainline/*.slp' \
#     --player-port 1 --out eval_runs/0831_position_probe/RESULTS.md
#
# Options: --limit-files (6) · --stride (4) · --max-states (2500) ·
#   --player-port (1) · --opponent-port (2) · --offset (40.0)
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Networks.Policy.Heads
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [set: :keep, replays: :string, limit_files: :integer, stride: :integer,
             max_states: :integer, player_port: :integer, opponent_port: :integer,
             offset: :float, situation: :string, out: :string]
  )

sets =
  Keyword.get_values(opts, :set)
  |> Enum.map(fn s -> [n, p] = String.split(s, "=", parts: 2); {n, p} end)

if sets == [], do: raise("--set NAME=POLICY_PATH required")
glob = opts[:replays] || raise("--replays required")
limit_files = opts[:limit_files] || 6
stride = opts[:stride] || 4
max_states = opts[:max_states] || 2500
p_port = opts[:player_port] || 1
o_port = opts[:opponent_port] || 2
offset = opts[:offset] || 40.0
# --situation neutral: restrict decision frames to states carrying this
# Situations label (e.g. :neutral excludes hitstun/advantage frames, where
# stick-away is legitimate DI, from the approach_delta read).
situation = if s = opts[:situation], do: String.to_existing_atom(s)

files = glob |> Path.wildcard() |> Enum.sort() |> Enum.take(limit_files)
if files == [], do: raise("no replays match #{glob}")

Output.banner("Position-dependence probe")

Output.config(
  Enum.map(sets, fn {n, p} -> {n, Path.basename(p)} end) ++
    [{"Replays", "#{length(files)} files"}, {"Bot port", p_port}, {"Stride", stride},
     {"close offset", offset}]
)

base_frames =
  files
  |> Enum.flat_map(fn f ->
    case Peppi.parse(f) do
      {:ok, replay} ->
        replay
        |> Peppi.to_training_frames(player_port: p_port, opponent_port: o_port)
        |> Enum.reject(&(&1.game_state.frame < 0))

      _ -> []
    end
  end)

Output.puts("  #{length(base_frames)} frames loaded")

allowed =
  if situation do
    sits =
      base_frames
      |> Enum.map(& &1.game_state)
      |> ExPhil.Situations.label_states(p_port, as: :set)
      |> List.to_tuple()

    set =
      0..(tuple_size(sits) - 1)
      |> Enum.filter(&MapSet.member?(elem(sits, &1), situation))
      |> MapSet.new()

    Output.puts("  situation #{situation}: #{MapSet.size(set)}/#{length(base_frames)} frames eligible")
    set
  end

perturb = fn frames, fun ->
  Enum.map(frames, fn f ->
    self = f.game_state.players[p_port]
    opp = f.game_state.players[o_port]
    %{f | game_state: %{f.game_state | players: Map.put(f.game_state.players, o_port, fun.(opp, self))}}
  end)
end

variants = [
  {:baseline, base_frames},
  {:mirror, perturb.(base_frames, fn o, _s -> %{o | x: -o.x} end)},
  {:far, perturb.(base_frames, fn o, _s -> %{o | x: o.x + 120.0} end)},
  {:close_left, perturb.(base_frames, fn o, s -> %{o | x: s.x - offset, y: s.y} end)},
  {:close_right, perturb.(base_frames, fn o, s -> %{o | x: s.x + offset, y: s.y} end)}
]

# Bucket-center expectation for a 17-bucket axis head: value(i) = (i-8)/8.
axis_vals = Nx.divide(Nx.subtract(Nx.iota({17}), 8), 8.0)

probe_one = fn {name, policy} ->
  Output.puts("== #{name}")
  trunk = Activations.load_trunk(policy)
  heads = Activations.load_heads_only(policy)
  window = trunk.window
  config = Map.get(trunk, :config, %{})

  feats_for = fn frames ->
    ds = Activations.embed_frames(frames, config)
    emb = Nx.backend_transfer(ds.embedded_frames, Nx.BinaryBackend)
    n = Nx.axis_size(emb, 0)

    rows =
      (window - 1)..(n - 1)//stride
      |> Enum.filter(fn r -> allowed == nil or MapSet.member?(allowed, r) end)
      |> Enum.take(max_states)

    rows
    |> Enum.chunk_every(256)
    |> Enum.map(fn ts ->
      wins = Enum.map(ts, &Nx.slice_along_axis(emb, &1 - window + 1, window, axis: 0))
      trunk.predict_fn.(trunk.params, Nx.stack(wins))
    end)
    |> Nx.concatenate(axis: 0)
  end

  heads_on = fn feats ->
    m = Nx.axis_size(feats, 0)

    inputs =
      case heads.head do
        :independent ->
          %{"trunk" => feats}

        :autoregressive ->
          neutral = Nx.broadcast(8, {m})

          tf =
            Heads.tf_inputs(%{
              buttons: Nx.broadcast(0, {m, 8}),
              main_x: neutral,
              main_y: neutral,
              c_x: neutral,
              c_y: neutral,
              shoulder: Nx.broadcast(0, {m})
            })

          Map.merge(%{"trunk" => feats}, tf)
      end

    {b_l, mx_l, my_l, _cx, _cy, _sh} = heads.predict_fn.(heads.params, inputs)

    %{
      p_buttons: Nx.sigmoid(b_l),
      p_mx: Axon.Activations.softmax(mx_l, axis: -1),
      p_my: Axon.Activations.softmax(my_l, axis: -1)
    }
  end

  outs =
    Map.new(variants, fn {vname, frames} ->
      {vname, heads_on.(feats_for.(frames))}
    end)

  base = outs[:baseline]
  mean = fn t -> Nx.to_number(Nx.mean(t)) end
  # Per-frame total variation distance between categorical rows, averaged.
  tv = fn a, b -> mean.(Nx.multiply(Nx.sum(Nx.abs(Nx.subtract(a, b)), axes: [1]), 0.5)) end
  e_mx = fn o -> mean.(Nx.dot(o.p_mx, axis_vals)) end

  deltas =
    for {vname, o} <- outs, vname != :baseline do
      %{
        variant: vname,
        d_buttons: mean.(Nx.abs(Nx.subtract(o.p_buttons, base.p_buttons))),
        tv_mx: tv.(o.p_mx, base.p_mx),
        tv_my: tv.(o.p_my, base.p_my),
        e_mx: e_mx.(o)
      }
    end

  approach = e_mx.(outs[:close_right]) - e_mx.(outs[:close_left])

  # Grab-button (z, index 4) sensitivity to opponent proximity: does P(z)
  # rise when the opponent is actually next to it vs far away?
  p_z = fn o -> mean.(o.p_buttons[[.., 4]]) end
  z_near = (p_z.(outs[:close_left]) + p_z.(outs[:close_right])) / 2
  z_far = p_z.(outs[:far])

  %{name: name, head: heads.head, n: Nx.axis_size(base.p_mx, 0), deltas: deltas,
    e_mx_base: e_mx.(base), approach: approach, z_near: z_near, z_far: z_far}
end

results = Enum.map(sets, probe_one)

f3 = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 3) end

per_variant =
  Enum.map_join(results, "\n", fn r ->
    rows =
      Enum.map_join(r.deltas, "\n", fn d ->
        "| #{r.name} | #{d.variant} | #{f3.(d.d_buttons)} | #{f3.(d.tv_mx)} | #{f3.(d.tv_my)} | #{f3.(d.e_mx)} |"
      end)

    rows
  end)

summary_rows =
  Enum.map_join(results, "\n", fn r ->
    "| #{r.name} | #{r.head} | #{r.n} | #{f3.(r.e_mx_base)} | **#{f3.(r.approach)}** | " <>
      "#{f3.(r.z_near)} | #{f3.(r.z_far)} |"
  end)

report = """
# Position-dependence probe — RESULTS

States: #{length(base_frames)} frames from #{length(files)} live-look replays
(bot port #{p_port}), stride #{stride}#{if situation, do: ", situation filter :#{situation}", else: ""}. Counterfactual opponent placement,
same trunk+heads per policy; AR heads teacher-forced on a fixed neutral
prefix for every variant (prefix confound cancels in the deltas).

## Perception — output movement vs baseline per variant

| policy | variant | mean dP(buttons) | TV(main_x) | TV(main_y) | E[main_x] |
|---|---|---:|---:|---:|---:|
#{per_variant}

## Differentiation — summary

| policy | head | states | E[main_x] base | approach_delta | P(z) near | P(z) far |
|---|---|---:|---:|---:|---:|---:|
#{summary_rows}

Reading guide:
- Perception rows ~0 across variants = the network does not READ opponent
  position at all (training/curation lever).
- Perception alive but approach_delta ~0 = it sees position but the learned
  policy doesn't STEER by it (selection lever — same class as the F1
  airdodge finding: option-selection, not perception).
- approach_delta > 0 = steers toward the opponent when placement flips.
- P(z) near vs far: does grab probability track actual grab range?
"""

IO.puts("\n" <> report)

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, report)
  Output.success("wrote #{out}")
end

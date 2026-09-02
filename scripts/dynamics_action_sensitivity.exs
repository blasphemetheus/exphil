# Action-sensitivity diagnostic for the G3b dynamics model (09-01).
#
# The G3b gate (1-step R^2, teacher-forced cos@k) can be passed by a model
# that IGNORES its action input entirely — identity dominates frame-to-frame
# dynamics. The v3 selector's null (scores identical across candidates,
# shuffled control == real) predicts exactly that. Test: from real states,
# roll k steps under maximally different held actions and compare the
# divergence between the resulting trajectories against the magnitude of
# the rollout's own drift.
#
#   mix run scripts/dynamics_action_sensitivity.exs \
#     --dynamics checkpoints/dynamics_fox_v11AR.bin \
#     --policy checkpoints/fox_gen_v1.2_ARrefit_policy.bin \
#     --replay replays/erickfm_ranked/FOX/extracted/<any>.slp --k 10
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Interp.Activations
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [dynamics: :string, policy: :string, replay: :string, k: :integer,
             char_id: :integer, out: :string]
  )

dyn_path = opts[:dynamics] || raise "--dynamics required"
policy = opts[:policy] || raise "--policy required"
replay_glob = opts[:replay] || raise "--replay required"
k_steps = opts[:k] || 10
char_id = opts[:char_id] || 2

dyn = dyn_path |> File.read!() |> :erlang.binary_to_term()
d = dyn.config.embed_dim
hidden = dyn.config.hidden

dyn_model =
  Axon.input("x", shape: {nil, d + 13})
  |> Axon.dense(hidden, activation: :relu)
  |> Axon.dense(hidden, activation: :relu)
  |> Axon.dense(d)

{_, dyn_predict} = Axon.build(dyn_model, mode: :inference)
norm = fn e -> Nx.divide(Nx.subtract(e, dyn.mu), dyn.sd) end

trunk = Activations.load_trunk(policy)
config = Map.get(trunk, :config, %{})

path = replay_glob |> Path.wildcard() |> List.first() || raise("no replay matches")

port =
  case Peppi.metadata(path) do
    {:ok, meta} ->
      case Enum.filter(meta.players, &(&1.character == char_id)) do
        [%{port: p}] -> p
        _ -> 1
      end

    _ -> 1
  end

opp = if port == 1, do: 2, else: 1
{:ok, replay} = Peppi.parse(path)

frames =
  replay
  |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
  |> Enum.reject(&(&1.game_state.frame < 0))

ds = Activations.embed_frames(frames, config)

emb =
  case Nx.rank(ds.embedded_frames) do
    2 -> ds.embedded_frames
    3 -> ds.embedded_frames[[.., 0, ..]]
  end

n = Nx.axis_size(emb, 0)
starts = Enum.take_every(60..(n - k_steps - 2)//1, 30) |> Enum.take(100)
e0 = Nx.take(emb, Nx.tensor(starts), axis: 0) |> norm.()
m = Nx.axis_size(e0, 0)

Output.banner("Dynamics action-sensitivity")
Output.puts("  #{m} start states from #{Path.basename(path)}")

# 13-dim continuous controller layout (Controller.embed_continuous_batch):
# 8 buttons {0,1} + main_x, main_y, c_x, c_y in [-1, 1] (0 = NEUTRAL, the
# (raw - 0.5) * 2 convention) + shoulder [0, 1].
#
# CONVENTION BUG FIXED 09-01 (stick-space instance of the GOTCHA #107
# class): the first version passed RAW 0..1 stick values (0.5 = neutral)
# into the embedded-space slots — "hard_left" was actually x=0 = neutral,
# "neutral" was x=0.5 = half-right. The action-sensitive verdict survives
# a fortiori (even those mild/mislabeled vectors diverged 15-33% of
# drift), but per-action numbers from the first run are mislabeled.
mk = fn btns, mx, my ->
  Nx.tensor([btns ++ [mx, my, 0.0, 0.0, 0.0]], type: :f32) |> Nx.broadcast({m, 13})
end

actions = [
  {:neutral, mk.([0, 0, 0, 0, 0, 0, 0, 0], 0.0, 0.0)},
  {:hard_left, mk.([0, 0, 0, 0, 0, 0, 0, 0], -1.0, 0.0)},
  {:hard_right, mk.([0, 0, 0, 0, 0, 0, 0, 0], 1.0, 0.0)},
  {:jump_x, mk.([0, 0, 1, 0, 0, 0, 0, 0], 0.0, 0.0)},
  {:down_b, mk.([0, 1, 0, 0, 0, 0, 0, 0], 0.0, -1.0)}
]

roll = fn a ->
  Enum.reduce(1..k_steps, e0, fn _j, cur ->
    Nx.add(cur, dyn_predict.(dyn.params, %{"x" => Nx.concatenate([cur, a], axis: 1)}))
  end)
end

finals = Enum.map(actions, fn {name, a} -> {name, roll.(a)} end)

dist = fn a, b ->
  Nx.to_number(Nx.mean(Nx.LinAlg.norm(Nx.subtract(a, b), axes: [1])))
end

{_, f_neutral} = hd(finals)
drift = dist.(f_neutral, e0)

Output.puts("\n  drift |f^#{k_steps}(s, neutral) - s|: #{Float.round(drift, 3)} (rollout's own movement scale)")
Output.puts("  divergence |f^#{k_steps}(s, a) - f^#{k_steps}(s, neutral)| per action:")

rows =
  for {name, f} <- tl(finals) do
    dv = dist.(f, f_neutral)
    ratio = dv / max(drift, 1.0e-9)
    Output.puts("    #{String.pad_trailing(to_string(name), 12)} #{Float.round(dv, 4)}  (#{Float.round(ratio * 100, 1)}% of drift)")
    {name, dv, ratio}
  end

max_ratio = rows |> Enum.map(&elem(&1, 2)) |> Enum.max()

verdict =
  if max_ratio < 0.1,
    do: "ACTION-BLIND: max divergence #{Float.round(max_ratio * 100, 1)}% of drift — the model ignores its action input; G3b's gate did not test this",
    else: "action-sensitive: max divergence #{Float.round(max_ratio * 100, 1)}% of drift"

Output.puts("\n  VERDICT: #{verdict}")

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))

  body =
    Enum.map_join(rows, "\n", fn {name, dv, ratio} ->
      "| #{name} | #{Float.round(dv, 4)} | #{Float.round(ratio * 100, 1)}% |"
    end)

  File.write!(out, """
  # Dynamics action-sensitivity — RESULTS

  #{m} start states, k=#{k_steps}, drift under neutral hold: #{Float.round(drift, 3)}.

  | held action | divergence from neutral rollout | % of drift |
  |---|---:|---:|
  #{body}

  **#{verdict}**
  """)

  Output.success("wrote #{out}")
end

# Coverage readout for the expert-distribution labeler: squared distance
# from a state to its NEAREST expert row, on (a) held-out expert games and
# (b) the bot's own states in a DAgger set. Where the bot's distances exceed
# the expert's own, the sampled label is an extrapolation (10-08 22:12:
# `airdodge_with_jump` 0.316 at 5x dose). Prints quantiles and the
# suggested gate = the held-out expert's 95th percentile, and the label
# hazards inside the set above/below that gate.
#   mix run --no-compile scripts/expert_labeler_distance.exs --split data/silent_fall/heldout_fd_fox_split.json \
#     --index data/silent_fall/expert_recovery_index.bin --games 24 --set data/silent_fall/sim_dagger_expert_r2.frames --out X.json
require Logger
Logger.configure(level: :warning)
for app <- [:nx, :jason], do: Application.ensure_all_started(app)
Code.require_file("scripts/lib/expert_recovery_labeler.exs")

alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Data.Peppi
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [split: :string, index: :string, games: :integer, set: :string, out: :string])
index = L.load(opts[:index] || "data/silent_fall/expert_recovery_index.bin")
edge = GA.stage_edge(32)
Output.banner("Expert labeler coverage (#{index.n} rows)")

quantile = fn l, q -> s = Enum.sort(l); Enum.at(s, min(length(s) - 1, trunc(q * length(s)))) end
qs = fn l -> [0.5, 0.9, 0.95, 0.99] |> Enum.map(fn q -> "q#{trunc(q * 100)} #{Float.round(quantile.(l, q) * 1.0, 2)}" end) |> Enum.join("  ") end

# (a) held-out expert states
expert_d2 =
  case opts[:split] do
    nil -> []
    split ->
      files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation")
      games =
        files
        |> Enum.flat_map(fn path ->
          case Peppi.metadata(path) do
            {:ok, %{stage: 32} = meta} ->
              own = Enum.find(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
              opp = own && Enum.find(meta.players, &(&1.port != own.port))
              if own && opp, do: [{path, own.port, opp.port}], else: []
            _ -> []
          end
        end)
        |> Enum.take(opts[:games] || 24)

      Enum.flat_map(games, fn {path, own, opp} ->
        {:ok, replay} = Peppi.parse(path, player_port: own)
        frames = replay |> Peppi.to_training_frames(player_port: own, opponent_port: opp, remap_ports: true) |> Enum.reject(&(&1.game_state.frame < 0))
        states =
          frames
          |> Enum.chunk_every(2, 1, :discard)
          |> Enum.filter(fn [f0, f1] -> f1.game_state.frame == f0.game_state.frame + 1 and L.labelable?(f1.game_state.players[1], edge) end)
          |> Enum.map(fn [f0, f1] -> {f1.game_state.players[1], f0.controller, f1.game_state.players[2], f0.game_state.players[1]} end)
        L.nearest_d2_batch(index, states, edge)
      end)
  end

if expert_d2 != [], do: Output.puts("RESULT nearest-d2 held-out expert states n=#{length(expert_d2)}: #{qs.(expert_d2)}")
gate = if expert_d2 == [], do: nil, else: quantile.(expert_d2, 0.95) * 1.0

# (b) the bot's states in a set (labels already sampled; prev = the set's prev_controller)
set_rows =
  case opts[:set] do
    nil -> []
    path ->
      set = path |> File.read!() |> :erlang.binary_to_term()
      frames = set.frame_lists |> List.flatten() |> Enum.reject(&(&1[:input_only] == true))
      states = Enum.map(frames, fn f -> {f.game_state.players[1], f.prev_controller, f.game_state.players[2], f[:prev_player]} end)
      Enum.zip(frames, L.nearest_d2_batch(index, states, edge))
  end

if set_rows != [] do
  d2 = Enum.map(set_rows, &elem(&1, 1))
  Output.puts("RESULT nearest-d2 bot states in #{opts[:set]} n=#{length(d2)}: #{qs.(d2)}")

  if gate do
    far = Enum.filter(set_rows, fn {_, d} -> d > gate end)
    near = Enum.filter(set_rows, fn {_, d} -> d <= gate end)
    jump? = fn c -> c.button_x or c.button_y end
    shoulder? = fn c -> c.button_l or c.button_r or (c.l_shoulder || 0.0) > 0.3 or (c.r_shoulder || 0.0) > 0.3 end
    up? = fn c -> (c.main_stick[:y] || 0.5) >= 0.75 end
    rate = fn rows, g -> if rows == [], do: "-", else: "#{Float.round(100 * Enum.count(rows, fn {f, _} -> g.(f.controller) and not g.(f.prev_controller) end) / length(rows), 2)} %" end
    Output.puts("RESULT gate d2 <= #{Float.round(gate, 2)} (held-out q95): #{length(near)} near / #{length(far)} far (#{Float.round(100 * length(far) / length(set_rows), 1)} % of the set beyond the expert's own range)")
    Output.puts("RESULT label edges near | far: jump #{rate.(near, jump?)} | #{rate.(far, jump?)}   shoulder/airdodge #{rate.(near, shoulder?)} | #{rate.(far, shoulder?)}   stick-up #{rate.(near, up?)} | #{rate.(far, up?)}")
    # where are the far states? jumps in hand, height
    far_j1 = Enum.count(far, fn {f, _} -> (f.game_state.players[1].jumps_left || 0) > 0 end)
    far_low = Enum.count(far, fn {f, _} -> (f.game_state.players[1].y || 0.0) < -20.0 end)
    Output.puts("RESULT far states: jump in hand #{Float.round(100 * far_j1 / max(length(far), 1), 1)} %  y<-20 #{Float.round(100 * far_low / max(length(far), 1), 1)} %")

    # which features carry the distance: mean squared diff per dim between a
    # far state and its nearest expert row (scaled, weighted space = the
    # distance's own units; the dims sum to the mean d2)
    sample = far |> Enum.shuffle() |> Enum.take(4000) |> Enum.map(&elem(&1, 0))
    qvecs = Enum.map(sample, fn f -> L.features(f.game_state.players[1], f.prev_controller, f.game_state.players[2], edge, f[:prev_player]) |> L.vector() end)
    {idx, _} = qvecs |> Enum.chunk_every(256) |> Enum.map(&L.nearest_d2(index, &1, 1)) |> Enum.reduce({[], []}, fn {i, d}, {is, ds} -> {is ++ Nx.to_list(i), ds ++ Nx.to_list(d)} end)
    xs = index.x |> Nx.to_list() |> List.to_tuple()
    per_dim =
      Enum.zip(qvecs, idx)
      |> Enum.map(fn {q, [j]} -> Enum.zip(q, elem(xs, j)) |> Enum.map(fn {a, b} -> (a - b) * (a - b) end) end)
      |> Enum.zip_with(fn col -> Enum.sum(col) / length(col) end)
    Output.puts("RESULT far-state d2 by feature (mean sq diff to nearest expert row, n=#{length(sample)}): " <>
      (Enum.zip(L.dim_names(), per_dim) |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.map(fn {n, v} -> "#{n} #{Float.round(v, 3)}" end) |> Enum.join("  ")))
  end
end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{gate_q95: gate, expert_n: length(expert_d2), set_n: length(set_rows),
    expert_q: Enum.map([0.5, 0.9, 0.95, 0.99], &(if expert_d2 == [], do: nil, else: quantile.(expert_d2, &1))),
    set_q: Enum.map([0.5, 0.9, 0.95, 0.99], &(if set_rows == [], do: nil, else: quantile.(Enum.map(set_rows, fn {_, d} -> d end), &1)))}, pretty: true))
end

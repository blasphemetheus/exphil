# Self-check of the expert-distribution labeler on HELD-OUT expert games
# (not in the index): relabel the expert's own offstage frames with the k-NN
# labeler (prev = the expert's real previous input) and compare the label
# hazards with the expert's actual inputs on the same frames — jump edge,
# B edge, stick-up onset once the jump is spent, hold share, and the
# agreement rate. A labeler that cannot reproduce the expert on the
# expert's states cannot label the bot's.
#
#   elixir -pa _build/dev/lib/*/ebin scripts/expert_labeler_selfcheck.exs \
#     --split checkpoints/coh_X/split.json --index data/silent_fall/expert_recovery_index.bin --games 24
require Logger
Logger.configure(level: :warning)
for app <- [:nx, :jason], do: Application.ensure_all_started(app)
Code.require_file("scripts/lib/expert_recovery_labeler.exs")

alias ExPhil.Agents.ExpertRecoveryLabeler, as: L
alias ExPhil.Data.Peppi
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, _} = OptionParser.parse(System.argv(), strict: [split: :string, index: :string, games: :integer])
split = opts[:split] || raise("--split required")
files = split |> File.read!() |> Jason.decode!() |> Map.fetch!("validation")
index = L.load(opts[:index] || "data/silent_fall/expert_recovery_index.bin")
edge = GA.stage_edge(32)
:rand.seed(:exsss, {1, 2, 3})
Output.banner("Expert labeler self-check (#{index.n} rows)")

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

Output.puts("#{length(games)} held-out FD Fox games")

rows =
  Enum.flat_map(games, fn {path, own, opp} ->
    {:ok, replay} = Peppi.parse(path, player_port: own)
    frames =
      replay
      |> Peppi.to_training_frames(player_port: own, opponent_port: opp, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))

    pairs =
      frames
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.filter(fn [f0, f1] ->
        f1.game_state.frame == f0.game_state.frame + 1 and L.labelable?(f1.game_state.players[1], edge)
      end)

    states = Enum.map(pairs, fn [f0, f1] -> {f1.game_state.players[1], f0.controller, f1.game_state.players[2]} end)
    labels = L.label_batch(index, states, edge)

    Enum.zip(pairs, labels)
    |> Enum.map(fn {[f0, f1], label} ->
      p = f1.game_state.players[1]
      %{p: p, prev: f0.controller, actual: f1.controller, label: label}
    end)
  end)

n = length(rows)
jump? = fn c -> c.button_x or c.button_y end
up? = fn c -> (c.main_stick[:y] || 0.5) >= 0.75 end
b? = fn c -> c.button_b end
edge? = fn r, f -> f.(r.actual) and not f.(r.prev) end
ledge? = fn r, f -> f.(r.label) and not f.(r.prev) end
pct = fn k, d -> if d == 0, do: "-", else: "#{Float.round(100 * k / d, 2)} %" end

spent = Enum.filter(rows, &((&1.p.jumps_left || 0) == 0 and (&1.p.y || 0.0) < -20.0 and not up?.(&1.prev)))
in_hand = Enum.filter(rows, &((&1.p.jumps_left || 0) > 0 and (&1.p.y || 0.0) < -20.0 and not jump?.(&1.prev)))

Output.puts("RESULT labeler self-check n=#{n}: hold share actual #{pct.(Enum.count(rows, &L.same_input?(&1.actual, &1.prev)), n)} | label #{pct.(Enum.count(rows, &L.same_input?(&1.label, &1.prev)), n)}; " <>
  "agreement (16-bucket) #{pct.(Enum.count(rows, &L.same_input?(&1.label, &1.actual)), n)}")
Output.puts("RESULT labeler self-check jump edge, jump in hand, y<-20 (n=#{length(in_hand)}): actual #{pct.(Enum.count(in_hand, &edge?.(&1, jump?)), length(in_hand))} | label #{pct.(Enum.count(in_hand, &ledge?.(&1, jump?)), length(in_hand))}")
Output.puts("RESULT labeler self-check stick-up onset, jump spent, y<-20 (n=#{length(spent)}): actual #{pct.(Enum.count(spent, &edge?.(&1, up?)), length(spent))} | label #{pct.(Enum.count(spent, &ledge?.(&1, up?)), length(spent))}")
Output.puts("RESULT labeler self-check B edge, jump spent, y<-20: actual #{pct.(Enum.count(spent, &edge?.(&1, b?)), length(spent))} | label #{pct.(Enum.count(spent, &ledge?.(&1, b?)), length(spent))}")

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

{opts, _, _} = OptionParser.parse(System.argv(), strict: [split: :string, index: :string, games: :integer, out: :string])
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
        f1.game_state.frame == f0.game_state.frame + 1 and L.labelable?(f1.game_state.players[1], edge, index.window, f1.game_state.players[2])
      end)

    states = Enum.map(pairs, fn [f0, f1] -> {f1.game_state.players[1], f0.controller, f1.game_state.players[2], f0.game_state.players[1]} end)
    labels = L.label_batch(index, states, edge)

    Enum.zip(pairs, labels)
    |> Enum.map(fn {[f0, f1], label} ->
      p = f1.game_state.players[1]
      %{p: p, opp: f1.game_state.players[2], prev: f0.controller, actual: f1.controller, label: label}
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

rate = fn l, f -> if l == [], do: 0.0, else: Enum.count(l, f) / length(l) end

# v6 (`:onstage` index): the same actual-vs-label hazard comparison in the
# onstage cells the window was added for — shield (release / jump / grab /
# roll / spot dodge edges), lying after a missed tech (any getup input, getup
# attack), in-threat grounded neutral (A / grab / c-stick smash / B / jump /
# shield edges). Each is an EDGE rate: the input is on at t and was off at
# t-1 (actual) — the label's edge is label-on and prev-off.
cells =
  if index.window == :onstage do
    sx = fn c -> ((c.main_stick[:x] || 0.5) - 0.5) * 2 end
    sy = fn c -> ((c.main_stick[:y] || 0.5) - 0.5) * 2 end
    cx = fn c -> ((c.c_stick[:x] || 0.5) - 0.5) * 2 end
    cy = fn c -> ((c.c_stick[:y] || 0.5) - 0.5) * 2 end
    on = %{
      "shield" => &L.shielding?/1,
      "release" => &(not L.shielding?(&1)),
      "jump" => jump?,
      "a" => & &1.button_a,
      "z" => & &1.button_z,
      "grab" => &(&1.button_z or (&1.button_a and L.shielding?(&1))),
      "b" => b?,
      "roll" => &(abs(sx.(&1)) >= 0.6),
      "spot" => &(sy.(&1) <= -0.6),
      "smash" => &(abs(cx.(&1)) >= 0.6 or abs(cy.(&1)) >= 0.6),
      "any_input" => &(abs(sx.(&1)) >= 0.6 or abs(sy.(&1)) >= 0.6 or &1.button_a or &1.button_b or &1.button_x or &1.button_y or &1.button_z)
    }
    cell_rows = %{
      "shield" => Enum.filter(rows, &((&1.p.action || 0) in 178..182)),
      "lying" => Enum.filter(rows, &((&1.p.action || 0) in [184, 192])),
      "threat" => Enum.filter(rows, &(L.onstage_cell(&1.p, &1.opp) == :threat and &1.p.on_ground == true and (&1.p.action || 0) < 178 and (&1.p.hitstun_frames_left || 0) == 0))
    }
    events = %{"shield" => ~w(release jump grab roll spot), "lying" => ~w(any_input a), "threat" => ~w(a z smash b jump shield)}

    Map.new(cell_rows, fn {cell, rs} ->
      ev =
        Map.new(events[cell], fn e ->
          f = on[e]
          {e, %{actual: rate.(rs, &edge?.(&1, f)), label: rate.(rs, &ledge?.(&1, f))}}
        end)
      Output.puts("RESULT labeler self-check #{cell} (n=#{length(rs)}): " <>
        Enum.map_join(events[cell], "  ", fn e -> "#{e} actual #{pct.(Enum.count(rs, &edge?.(&1, on[e])), length(rs))} | label #{pct.(Enum.count(rs, &ledge?.(&1, on[e])), length(rs))}" end))
      {cell, %{n: length(rs), events: ev}}
    end)
  else
    %{}
  end

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{
    n: n, games: length(games), n_spent: length(spent), n_in_hand: length(in_hand),
    hold_actual: rate.(rows, &L.same_input?(&1.actual, &1.prev)), hold_label: rate.(rows, &L.same_input?(&1.label, &1.prev)),
    agreement: rate.(rows, &L.same_input?(&1.label, &1.actual)),
    actual_jump: rate.(in_hand, &edge?.(&1, jump?)), label_jump: rate.(in_hand, &ledge?.(&1, jump?)),
    actual_up: rate.(spent, &edge?.(&1, up?)), label_up: rate.(spent, &ledge?.(&1, up?)),
    actual_b: rate.(spent, &edge?.(&1, b?)), label_b: rate.(spent, &ledge?.(&1, b?)),
    cells: cells
  }, pretty: true))
end

# Pool label-conflict auditor (instrument #1, HANDOFF_2026-09-12 §3).
#
# A policy cannot be sharper than its labels: where the assembled drill pool
# disagrees with itself on a loop state, p(correct) is capped at the majority
# share — g22b fit its pool to 5e-5 and still sat at 0.95 on two boundary
# frames because 12.5k snippet frames carried a teacher one frame off. This
# audits every SOURCE the drill assembles, per state key, at the RAW
# (unshifted, causal) label, and names the disagreeing source.
#
#   mix run scripts/audit_ms_pool_labels.exs \
#     [--fixture test/fixtures/replays/fox_multishine_closed_d1.slp] \
#     [--rollouts "glob,glob"] [--snippets path.frames] [--openers "glob"] \
#     [--port 1] [--min-n 20] [--conflict 0.05]
#
# Output: one row per loop-state key {action, af, grounded}: per-source n and
# B/X rates, pooled B/X rate, and CONFLICT when any two sources with n>=min-n
# differ by more than --conflict on B or X. Exit 1 on any conflict, so it can
# gate a prereg script.

alias ExPhil.Agents.MultishineExpert
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [fixture: :string, rollouts: :string, snippets: :string, openers: :string, port: :integer, min_n: :integer, conflict: :float]
  )

fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
port = opts[:port] || 1
min_n = opts[:min_n] || 20
thresh = opts[:conflict] || 0.05

roll =
  opts[:rollouts] ||
    "eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"

globs = fn s -> s |> String.split(",", trim: true) |> Enum.flat_map(&Path.wildcard/1) end

{:ok, fx} = ExPhil.Data.Peppi.parse(fixture, player_port: port)
fixture_frames = ExPhil.Data.Peppi.to_training_frames(fx, player_port: port, opponent_port: if(port == 1, do: 2, else: 1), remap_ports: true)
expert = MultishineExpert.from_frames(fixture_frames, player_port: 1)

key_of = fn f ->
  p = f.game_state.players[1]
  {trunc(p.action || 0), min(trunc(p.action_frame || 0), 12), p.on_ground == true}
end

label_row = fn f, c -> {key_of.(f), c.button_b == true, c.button_x == true} end

relabel = fn frames ->
  recorded = Map.new(frames, fn f -> {f.game_state.frame, f.controller} end)

  Enum.flat_map(frames, fn f ->
    p = f.game_state.players[1]
    prev = recorded[f.game_state.frame - 1]

    case p && MultishineExpert.label(expert, p, prev) do
      {:ok, c} -> [label_row.(f, c)]
      _ -> []
    end
  end)
end

load_replays = fn paths ->
  Enum.flat_map(paths, fn path ->
    case ExPhil.Data.Peppi.parse(path, player_port: port) do
      {:ok, r} -> relabel.(ExPhil.Data.Peppi.to_training_frames(r, player_port: port, opponent_port: if(port == 1, do: 2, else: 1), remap_ports: true))
      _ -> []
    end
  end)
end

sources =
  [
    {"fixture", Enum.map(fixture_frames, fn f -> label_row.(f, f.controller) end)},
    {"rollouts", load_replays.(globs.(roll))}
  ] ++
    if(opts[:snippets],
      do: [
        {"snippets",
         opts[:snippets]
         |> File.read!()
         |> :erlang.binary_to_term()
         |> Map.get(:frame_lists, [])
         |> List.flatten()
         |> Enum.map(fn f -> label_row.(f, f.controller) end)}
      ],
      else: []
    ) ++
    if(opts[:openers], do: [{"openers", load_replays.(globs.(opts[:openers]))}], else: [])

Output.banner("Multishine pool label audit (raw causal labels, per source)")
for {name, rows} <- sources, do: Output.puts("  #{name}: #{length(rows)} labeled frames")

loop_keys = [{361, 1, true}, {361, 2, true}, {361, 3, true}, {24, 0, true}, {24, 1, true}, {24, 2, true}, {365, 1, false}, {365, 2, false}, {365, 3, false}, {366, 0, false}, {366, 1, false}]

by_source = Map.new(sources, fn {name, rows} -> {name, Enum.group_by(rows, &elem(&1, 0))} end)
names = Enum.map(sources, &elem(&1, 0))

rate = fn rows, idx -> if rows == [], do: nil, else: Enum.count(rows, &elem(&1, idx)) / length(rows) end
fmt = fn v -> if v == nil, do: "  -  ", else: String.pad_leading(:erlang.float_to_binary(v * 1.0, decimals: 2), 5) end

Output.puts("")
Output.puts("| state | " <> Enum.map_join(names, " | ", &"#{&1} n · B · X") <> " | verdict |")
Output.puts("|---|" <> String.duplicate("---|", length(names)) <> "---|")

conflicts =
  Enum.flat_map(loop_keys, fn key ->
    cells =
      Enum.map(names, fn n ->
        rows = Map.get(by_source[n], key, [])
        {n, length(rows), rate.(rows, 1), rate.(rows, 2)}
      end)

    valid = Enum.filter(cells, fn {_, n, _, _} -> n >= min_n end)

    spread = fn idx ->
      vals = Enum.map(valid, &elem(&1, idx))
      if length(vals) >= 2, do: Enum.max(vals) - Enum.min(vals), else: 0.0
    end

    sb = spread.(2)
    sx = spread.(3)
    bad = sb > thresh or sx > thresh

    verdict =
      cond do
        bad ->
          culprit =
            valid
            |> Enum.sort_by(fn {_, n, _, _} -> n end)
            |> List.first()
            |> elem(0)

          "CONFLICT B±#{Float.round(sb, 2)} X±#{Float.round(sx, 2)} (smallest source: #{culprit})"

        length(valid) < 2 ->
          "single source"

        true ->
          "ok"
      end

    Output.puts(
      "| #{inspect(key)} | " <>
        Enum.map_join(cells, " | ", fn {_, n, b, x} -> "#{n} · #{fmt.(b)} · #{fmt.(x)}" end) <>
        " | #{verdict} |"
    )

    if bad, do: [key], else: []
  end)

Output.puts("")

if conflicts == [] do
  Output.success("no label conflicts on loop states (threshold #{thresh}, min n #{min_n})")
else
  Output.error("#{length(conflicts)} loop state(s) with conflicting labels across sources: #{inspect(conflicts)}")
  System.halt(1)
end

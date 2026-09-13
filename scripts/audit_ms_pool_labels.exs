# Pool label-conflict auditor (instrument #1, HANDOFF_2026-09-12 §3), now at
# the TRAINING SHIFT (evening extension, g25a verdict).
#
# A policy cannot be sharper than its labels: where the assembled drill pool
# disagrees with itself on a loop state, p(correct) is capped at the majority
# share — g22b fit its pool to 5e-5 and still sat at 0.95 on two boundary
# frames because 12.5k snippet frames carried a teacher one frame off.
#
# Two things can disagree, and the drill trains at a SHIFT, not at the raw
# label:
#
#   * ACROSS sources (the original check): fixture vs rollouts vs snippets
#     vs openers give different B/X rates for the same state key.
#   * WITHIN a source, at the shift: reaction delay k pairs state[t] with the
#     label of frame t+k, i.e. "what the expert does k frames LATER" — which
#     depends on where the subject IS k frames later. On a coherent loop that
#     is the loop; in a rollout whose loop breaks after this state, it is a
#     recovery input. Same state key, two futures, two labels. Invisible at
#     shift 0 (the expert's label for the state itself is unambiguous), and
#     the leading hypothesis for the g24a -> g25a technique-floor decline
#     (RESULTS §8: 365/3a collapsed to 0.31 on the fixture at shift 4).
#
#   mix run scripts/audit_ms_pool_labels.exs \
#     [--fixture test/fixtures/replays/fox_multishine_closed_d1.slp] \
#     [--rollouts "glob,glob"] [--snippets path.frames] [--openers "glob"] \
#     [--shifts "0,3,4,5"] [--port 1] [--min-n 20] [--conflict 0.05] [--ambiguity 0.10]
#
# Output, per shift: one row per loop-state key with per-source n · B · X, and
# a verdict: CONFLICT (two sources with n>=min-n differ by more than --conflict
# on B or X — names the smallest source), AMBIGUOUS (a source's own majority
# share on B or X is below 1 - --ambiguity: the future-dependent kind — names
# the source), or ok. Exit 1 on any CONFLICT or AMBIGUOUS at any shift, so it
# gates a prereg script. --shifts must be the drill's --multi-delay list.

alias ExPhil.Agents.MultishineExpert
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      fixture: :string,
      rollouts: :string,
      snippets: :string,
      openers: :string,
      shifts: :string,
      port: :integer,
      min_n: :integer,
      conflict: :float,
      ambiguity: :float
    ]
  )

fixture = opts[:fixture] || "test/fixtures/replays/fox_multishine_closed_d1.slp"
port = opts[:port] || 1
min_n = opts[:min_n] || 20
thresh = opts[:conflict] || 0.05
amb_thresh = opts[:ambiguity] || 0.10
shifts = (opts[:shifts] || "0") |> String.split(",", trim: true) |> Enum.map(&String.to_integer(String.trim(&1)))

roll =
  opts[:rollouts] ||
    "eval_runs/dagger_d3_round1_collect/r*.slp,eval_runs/d3_div_b1/r*.slp,eval_runs/d3_div_full2/r*.slp,eval_runs/d3_div_r3/r*.slp,eval_runs/0802_d2pool/r*.slp"

globs = fn s -> s |> String.split(",", trim: true) |> Enum.flat_map(&Path.wildcard/1) end

{:ok, fx} = ExPhil.Data.Peppi.parse(fixture, player_port: port)

fixture_frames =
  ExPhil.Data.Peppi.to_training_frames(fx, player_port: port, opponent_port: if(port == 1, do: 2, else: 1), remap_ports: true)

expert = MultishineExpert.from_frames(fixture_frames, player_port: 1)

key_of = fn f ->
  p = f.game_state.players[1]
  {trunc(p.action || 0), min(trunc(p.action_frame || 0), 12), p.on_ground == true}
end

# Lists are TRAINING FRAME LISTS carrying their label source, and the audit
# derives each shift's labels through ExPhil.Training.Labels.at_delay/3 —
# the same function the drill trains through (INVARIANTS item 14) — so what
# is audited is what is trained: recorded lists shift along the recording,
# expert lists ask the expert's label_ahead/4.
recorded_labels = fn frames -> ExPhil.Training.Labels.tag(frames, :recorded) end

relabel = fn frames ->
  recorded = Map.new(frames, fn f -> {f.game_state.frame, f.controller} end)

  frames
  |> Enum.flat_map(fn f ->
    p = f.game_state.players[1]
    prev = recorded[f.game_state.frame - 1]

    case p && MultishineExpert.label(expert, p, prev) do
      {:ok, c} -> [f |> Map.put(:controller, c) |> Map.put(:prev_controller, prev)]
      _ -> []
    end
  end)
  |> ExPhil.Training.Labels.tag({:expert, MultishineExpert})
end

load_replays = fn paths ->
  Enum.flat_map(paths, fn path ->
    case ExPhil.Data.Peppi.parse(path, player_port: port) do
      {:ok, r} ->
        [relabel.(ExPhil.Data.Peppi.to_training_frames(r, player_port: port, opponent_port: if(port == 1, do: 2, else: 1), remap_ports: true))]

      _ ->
        []
    end
  end)
end

# sources :: [{name, [list]}]
sources =
  [
    {"fixture", [recorded_labels.(fixture_frames)]},
    {"rollouts", load_replays.(globs.(roll))}
  ] ++
    if(opts[:snippets],
      do: [
        {"snippets",
         opts[:snippets]
         |> File.read!()
         |> :erlang.binary_to_term()
         |> Map.get(:frame_lists, [])
         # mined snippets are EXPERT-relabeled (snippet_mine runs the expert per
         # frame) — tag them as the drill does, so their delayed labels come
         # from label_ahead, not from shifting the recording.
         |> Enum.map(&ExPhil.Training.Labels.tag(&1, {:expert, MultishineExpert}))}
      ],
      else: []
    ) ++
    if(opts[:openers], do: [{"openers", load_replays.(globs.(opts[:openers]))}], else: [])

# Rows at a shift: the state key of each frame with its reaction-delay-s
# label as the drill would train it. rows :: [{key, b, x}]
rows_at_shift = fn lists, s ->
  Enum.flat_map(lists, fn list ->
    list
    |> ExPhil.Training.Labels.at_delay(s, expert: expert, player_port: 1)
    |> Enum.map(fn f -> {key_of.(f), f.controller.button_b == true, f.controller.button_x == true} end)
  end)
end

Output.banner("Multishine pool label audit — per source, at the training shift(s) #{inspect(shifts)}")

for {name, lists} <- sources,
    do: Output.puts("  #{name}: #{length(lists)} list(s), #{lists |> Enum.map(&length/1) |> Enum.sum()} frames")

loop_keys = [
  {361, 1, true},
  {361, 2, true},
  {361, 3, true},
  {24, 0, true},
  {24, 1, true},
  {24, 2, true},
  {365, 1, false},
  {365, 2, false},
  {365, 3, false},
  {366, 0, false},
  {366, 1, false}
]

names = Enum.map(sources, &elem(&1, 0))
rate = fn rows, idx -> if rows == [], do: nil, else: Enum.count(rows, &elem(&1, idx)) / length(rows) end
fmt = fn v -> if v == nil, do: "  -  ", else: String.pad_leading(:erlang.float_to_binary(v * 1.0, decimals: 2), 5) end
majority = fn r -> max(r, 1.0 - r) end

problems =
  Enum.flat_map(shifts, fn s ->
    by_source = Map.new(sources, fn {name, lists} -> {name, rows_at_shift.(lists, s) |> Enum.group_by(&elem(&1, 0))} end)

    Output.puts("")
    Output.puts("### shift #{s} (state[t] -> label of frame t+#{s})")
    Output.puts("| state | " <> Enum.map_join(names, " | ", &"#{&1} n · B · X") <> " | verdict |")
    Output.puts("|---|" <> String.duplicate("---|", length(names)) <> "---|")

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
      conflict? = sb > thresh or sx > thresh

      # Within-source ambiguity: a source whose own B or X majority share
      # falls below 1 - amb_thresh carries two labels for this state.
      ambiguous =
        valid
        |> Enum.filter(fn {_, _, b, x} -> majority.(b) < 1.0 - amb_thresh or majority.(x) < 1.0 - amb_thresh end)
        |> Enum.map(fn {n, _, b, x} -> "#{n} (B #{fmt.(b) |> String.trim()}, X #{fmt.(x) |> String.trim()})" end)

      # Both signals print when both fire: a cross-source spread AND which
      # sources are internally split (the future-dependent kind).
      verdict =
        cond do
          conflict? or ambiguous != [] ->
            parts =
              if conflict? do
                culprit = valid |> Enum.sort_by(fn {_, n, _, _} -> n end) |> List.first() |> elem(0)
                ["CONFLICT B±#{Float.round(sb, 2)} X±#{Float.round(sx, 2)} (smallest source: #{culprit})"]
              else
                []
              end

            parts = if ambiguous != [], do: parts ++ ["AMBIGUOUS in " <> Enum.join(ambiguous, "; ")], else: parts
            Enum.join(parts, "; ")

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

      Enum.concat(
        if(conflict?, do: [{s, key, :conflict}], else: []),
        if(ambiguous != [], do: [{s, key, :ambiguous}], else: [])
      )
    end)
  end)

Output.puts("")

if problems == [] do
  Output.success("no label conflicts or ambiguities on loop states at shifts #{inspect(shifts)} (conflict #{thresh}, ambiguity #{amb_thresh}, min n #{min_n})")
else
  conflicts = Enum.filter(problems, &(elem(&1, 2) == :conflict))
  ambiguous = Enum.filter(problems, &(elem(&1, 2) == :ambiguous))

  Output.error(
    "#{length(conflicts)} cross-source conflict(s), #{length(ambiguous)} within-source ambiguity(ies) on loop states: " <>
      Enum.map_join(problems, ", ", fn {s, k, kind} -> "shift #{s} #{inspect(k)} #{kind}" end)
  )

  System.halt(1)
end

# #38 scenario-seeded farming — manifest generator.
#
#   mix run --no-compile scripts/gen_scenario_manifest.exs \
#     --replays "corpus/archive/mewtwo/*.slp" \
#     --types idle_deadlock,opponent_behind,tech_chase,edgeguard \
#     --max 200 --out scenarios/seed_manifest.json
#
# Scans source replays for the situations the bot never reaches on its own
# (idle_deadlock = neutral standoff / approach decision — the r15
# initiation gap; opponent_behind = spacing blindness; tech_chase/edgeguard
# = conversion breadth), and writes a manifest scenario_suite.exs consumes.
# Feed the resulting seeded rollouts to r16's --rollouts.
#
# HARD FILTER: the suite's input-prefix replay is deterministic ONLY for
# Mewtwo(port1) vs Fox(port2) on Final Destination (no items/hazards). Any
# other stage/matchup/port-order drifts and gets excluded at handoff, so we
# reject them up front. Corrupt/empty archive replays are skipped.
#
# NO-MIX-safe: pure parse + scan, no GPU, no Dolphin.

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ScenarioScan
alias ExPhil.Training.Output

# Internal IDs (libmelee enum, per CLAUDE.md): Mewtwo 16, Fox 1, FD 32
mewtwo = 16
fox = 1
fd = 32

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, types: :string, max: :integer, per_replay: :integer, out: :string]
  )

replays =
  (opts[:replays] || "")
  |> String.split(",", trim: true)
  |> Enum.flat_map(&Path.wildcard(Path.expand(&1)))

types =
  case opts[:types] do
    nil -> [:idle_deadlock, :opponent_behind, :tech_chase, :edgeguard]
    s -> s |> String.split(",", trim: true) |> Enum.map(&String.to_existing_atom/1)
  end

max_total = opts[:max] || 200
per_replay = opts[:per_replay] || 12
out = opts[:out] || "scenarios/seed_manifest.json"

if replays == [], do: raise("--replays required")

Output.banner("Scenario seed manifest (#38)")
Output.config([
  {"Source replays", length(replays)},
  {"Types", types},
  {"Max entries", max_total},
  {"Cap/replay", per_replay},
  {"Out", out}
])

# FD, Mewtwo(p1) vs Fox(p2)? Returns :ok | {:skip, reason}.
compat = fn path ->
  case Peppi.parse(path) do
    {:ok, replay} ->
      frames =
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      case frames do
        [] ->
          {:skip, :empty}

        [f | _] ->
          gs = f.game_state
          p1 = gs.players[1] && trunc(gs.players[1].character || -1)
          p2 = gs.players[2] && trunc(gs.players[2].character || -1)
          stage = trunc(gs.stage || -1)

          cond do
            stage != fd -> {:skip, {:stage, stage}}
            p1 == mewtwo and p2 == fox -> :ok
            true -> {:skip, {:matchup, p1, p2}}
          end
      end

    {:error, r} ->
      {:skip, {:parse, r}}
  end
end

{entries, stats} =
  Enum.reduce(replays, {[], %{ok: 0, skip: 0, reasons: %{}}}, fn path, {acc, st} ->
    case compat.(path) do
      :ok ->
        {:ok, %{frames: frames}} = ScenarioScan.load(path)
        cands = ScenarioScan.scan(frames, types: types) |> Enum.take(per_replay)

        rows =
          Enum.map(cands, fn c ->
            %{"slp" => path, "frame" => c.frame, "type" => to_string(c.type), "note" => c.note}
          end)

        {rows ++ acc, %{st | ok: st.ok + 1}}

      {:skip, reason} ->
        tag = if is_tuple(reason), do: elem(reason, 0), else: reason
        {acc, %{st | skip: st.skip + 1, reasons: Map.update(st.reasons, tag, 1, &(&1 + 1))}}
    end
  end)

# Balance across types up to max_total (round-robin so no single type
# dominates), then cap.
by_type = Enum.group_by(entries, & &1["type"])

balanced =
  by_type
  |> Map.values()
  |> Enum.map(&Enum.shuffle/1)
  |> then(fn lists ->
    # interleave the per-type lists
    max_len = lists |> Enum.map(&length/1) |> Enum.max(fn -> 0 end)

    for i <- 0..max_len, list <- lists, entry = Enum.at(list, i), entry != nil, do: entry
  end)
  |> Enum.take(max_total)

type_counts = balanced |> Enum.frequencies_by(& &1["type"])

Output.puts("")
Output.puts("compatible replays: #{stats.ok}/#{length(replays)} (skipped #{stats.skip}: #{inspect(stats.reasons)})")
Output.puts("candidates by type: #{inspect(type_counts)}")
Output.puts("manifest entries:   #{length(balanced)}")

File.mkdir_p!(Path.dirname(out))
File.write!(out, Jason.encode!(%{"entries" => balanced}, pretty: true))
Output.success("manifest -> #{out}")

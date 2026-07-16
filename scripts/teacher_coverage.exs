# Teacher-quality audit (task: expert-table blindness + coverage metrics).
# Builds the combo expert EXACTLY as dagger_drill does (all four fair
# fixtures), relabels pool replays, and reports:
#   1. label-source routing (chase / edge / fine / coarse / recovery branches)
#   2. jump-label rate (X or Y pressed) per source, grounded vs airborne
#   3. the headline: what fraction of ALL labels press jump, and how much of
#      that comes from the :jump_restart default vs learned table cells
#
#   mix run scripts/teacher_coverage.exs <replay.slp> [more.slp ...]

alias ExPhil.Agents.{MewtwoComboExpert, MewtwoFairExpert, MewtwoTechChaseExpert}
alias ExPhil.Interp.ReplayStats

fixtures =
  [
    "test/fixtures/replays/mewtwo_fair_chains.slp",
    "test/fixtures/replays/mewtwo_shfair_only.slp",
    "test/fixtures/replays/mewtwo_approach_fair.slp",
    "test/fixtures/replays/mewtwo_turnaround_fair.slp"
  ]

frames =
  Enum.flat_map(fixtures, fn path ->
    {:ok, replay} = ExPhil.Data.Peppi.parse(path)
    ExPhil.Data.Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)
  end)

fair = MewtwoFairExpert.from_frames(frames)
chase = MewtwoTechChaseExpert.new()
IO.puts("expert built from #{length(fixtures)} fixtures (#{length(frames)} frames)")

jump? = fn c -> c.button_x or c.button_y end

tally =
  for path <- System.argv(), reduce: %{} do
    acc ->
      data = ReplayStats.load(path)

      frames_p = Enum.zip([data.p1.players, data.p1.controllers, data.p2.players])

      {acc, _prev} =
        for {p1, ctrl, p2} <- frames_p, p1 != nil, reduce: {acc, nil} do
          {acc, prev} ->
            result =
              case MewtwoTechChaseExpert.label(chase, p1, prev, p2) do
                {:ok, c} -> {:ok, c, :chase}
                :skip -> MewtwoFairExpert.label_traced(fair, p1, prev, p2)
              end

            acc =
              case result do
                :skip ->
                  Map.update(acc, {:skip, nil, nil}, 1, &(&1 + 1))

                {:ok, c, source} ->
                  key = {source, p1.on_ground, jump?.(c)}
                  Map.update(acc, key, 1, &(&1 + 1))
              end

            {acc, ctrl}
        end

      acc
  end

total = tally |> Map.values() |> Enum.sum()
labeled = for {{s, _, _}, n} <- tally, s != :skip, reduce: 0, do: (acc -> acc + n)

IO.puts("\ntotal frames: #{total}  labeled: #{labeled}")
IO.puts(String.pad_trailing("\nsource", 18) <> " grounded?  labels   jump-labels   jump%")

tally
|> Enum.filter(fn {{s, _, _}, _} -> s != :skip end)
|> Enum.group_by(fn {{s, g, _}, _} -> {s, g} end)
|> Enum.map(fn {{s, g}, entries} ->
  n = entries |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  j = entries |> Enum.filter(fn {{_, _, jp}, _} -> jp end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  {s, g, n, j}
end)
|> Enum.sort_by(fn {_, _, n, _} -> -n end)
|> Enum.each(fn {s, g, n, j} ->
  IO.puts(
    String.pad_trailing("#{s}", 18) <>
      String.pad_trailing("#{g}", 10) <>
      String.pad_trailing("#{n}", 9) <>
      String.pad_trailing("#{j}", 14) <>
      "#{Float.round(j * 100 / max(n, 1), 1)}%"
  )
end)

all_jump = for {{s, _, jp}, n} <- tally, s != :skip, jp, reduce: 0, do: (acc -> acc + n)
jr_jump = for {{s, _, jp}, n} <- tally, s == :jump_restart, jp, reduce: 0, do: (acc -> acc + n)

IO.puts("\nHEADLINE: #{Float.round(all_jump * 100 / max(labeled, 1), 1)}% of all labels press jump;")
IO.puts("#{Float.round(jr_jump * 100 / max(all_jump, 1), 1)}% of jump labels come from the :jump_restart default")

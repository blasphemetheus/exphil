# On-stage double-jump spend hazard, expert (FD replays) vs a DAgger set of
# the bot's own states (2026-10-10). recovery_jump_spend.js found every
# carried-off Illusion trip starts with the double jump spent 24-30 frames
# earlier, 45-67 units INSIDE the edge, mid first jump, stick toward the edge
# -- outside the wide labeler's window (|x| > edge - 15), so no round has
# labelled that frame. This gives the rate per airborne-over-stage frame with
# the double jump in hand, by distance inside the edge, for the expert and for
# the set.
#
#   mix run --no-compile scripts/expert_onstage_double_jump.exs [--split SPLIT.json] [--stage 32]
#     [--max-games 150] [--character fox] [--set data/silent_fall/sim_dagger_expert_r7w.frames]
alias ExPhil.Data.Peppi
alias ExPhil.Sim.GA
alias ExPhil.Training.Output

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [split: :string, stage: :integer, max_games: :integer, character: :string, set: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

split = (opts[:split] || "checkpoints/coh_base/split.json") |> File.read!() |> Jason.decode!()
stage = opts[:stage] || 32
char = opts[:character] || "fox"
edge = GA.stage_edge(stage)
files = split["validation"] ++ split["train"]
bands = ["-15..0", "-30..-15", "-50..-30", "-80..-50", "<-80"]
band = fn d -> cond do d > -15 -> "-15..0"; d > -30 -> "-30..-15"; d > -50 -> "-50..-30"; d > -80 -> "-80..-50"; true -> "<-80" end end

# frames = [%{p: player, c: controller}] in order; counts exposure frames and spends per band
tally = fn frames ->
  frames
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.reduce(%{}, fn [a, b], acc ->
    pa = a.p
    if pa.on_ground != true and abs(pa.x || 0.0) < edge and (pa.jumps_left || 0) == 1 and (pa.hitstun_frames_left || 0) == 0 and (pa.action || 0) > 13 do
      k = band.(abs(pa.x) - edge)
      spent = (b.p.jumps_left || 0) == 0 and b.p.on_ground != true
      s = if pa.x >= 0, do: 1, else: -1
      toward = b.c != nil and (b.c.main_stick.x - 0.5) * 2 * s >= 0.6
      acc
      |> Map.update({k, :n}, 1, &(&1 + 1))
      |> Map.update({k, :spent}, if(spent, do: 1, else: 0), &(&1 + if(spent, do: 1, else: 0)))
      |> Map.update({k, :spent_toward}, if(spent and toward, do: 1, else: 0), &(&1 + if(spent and toward, do: 1, else: 0)))
    else
      acc
    end
  end)
end

merge = fn maps -> Enum.reduce(maps, %{}, fn m, acc -> Map.merge(acc, m, fn _, x, y -> x + y end) end) end

print = fn label, t ->
  Output.puts("RESULT #{label} on-stage double-jump spend per airborne-with-jump frame, by distance inside the edge:")
  for k <- bands do
    n = t[{k, :n}] || 0
    sp = t[{k, :spent}] || 0
    tw = t[{k, :spent_toward}] || 0
    pct = fn a, b -> if b == 0, do: "-", else: :erlang.float_to_binary(100 * a / b, decimals: 2) <> " %" end
    Output.puts("  #{String.pad_trailing(k, 9)} frames #{String.pad_leading(to_string(n), 7)}  spends #{String.pad_leading(to_string(sp), 5)}  hazard #{pct.(sp, n)}  stick toward edge >=0.6 at the spend #{pct.(tw, sp)}")
  end
  n = Enum.sum(for k <- bands, do: t[{k, :n}] || 0)
  sp = Enum.sum(for k <- bands, do: t[{k, :spent}] || 0)
  Output.puts("  all       frames #{n}  spends #{sp}  hazard #{if n == 0, do: "-", else: :erlang.float_to_binary(100 * sp / n, decimals: 2)} %")
end

Output.banner("On-stage double-jump spend: #{char} on stage #{stage}")

chosen =
  files
  |> Task.async_stream(fn p -> with {:ok, m} <- Peppi.metadata(p), do: {p, m} end, max_concurrency: 16, timeout: 60_000)
  |> Enum.flat_map(fn
    {:ok, {p, %{stage: ^stage, players: [_, _] = ps}}} ->
      case Enum.filter(ps, &(String.downcase(&1.character_name || "") == char)) do
        [own] -> [{p, own}]
        _ -> []
      end

    _ ->
      []
  end)
  |> Enum.take(opts[:max_games] || 150)

Output.puts("#{length(chosen)} games (of #{length(files)} scanned)")

expert =
  chosen
  |> Task.async_stream(
    fn {path, own} ->
      {:ok, replay} = Peppi.parse(path, player_port: own.port)

      replay
      |> Peppi.to_training_frames(player_port: own.port)
      |> Enum.reject(&(&1.game_state.frame < 0))
      |> Enum.map(&%{p: &1.game_state.players[own.port], c: &1.controller})
      |> tally.()
    end,
    max_concurrency: 8,
    timeout: 300_000
  )
  |> Enum.map(fn {:ok, t} -> t end)
  |> merge.()

print.("expert (#{length(chosen)} FD games)", expert)

if set = opts[:set] do
  m = File.read!(set) |> :erlang.binary_to_term()

  bot =
    m.frame_lists
    |> Enum.map(fn l -> l |> Enum.map(&%{p: &1.game_state.players[1], c: &1[:actual] || &1.controller}) |> tally.() end)
    |> merge.()

  print.("bot (#{Path.basename(set)}, #{length(m.frame_lists)} lists)", bot)
end

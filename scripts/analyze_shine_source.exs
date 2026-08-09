# Self-initiated shines vs ones the opponent knocked you into.
#
#   mix run scripts/analyze_shine_source.exs <replay.slp> [more.slp ...]
#     [--port N] [--lookback N]
#
# Raw shine count is a CONTAMINATED metric for comparing policies. Observed
# 2026-07-26 while watching a run: the CPU walks up and jabs the bot out of a
# held shine, and the bot then shines again. Those re-shines are the
# opponent's doing, not learned recovery — so a policy can score better purely
# because it got hit more often.
#
# This is the same shape as an earlier trap in this investigation: a policy
# that "shined 66 times" under stochastic sampling was being rescued by random
# RELEASES, not by knowing what to do. Twice now, apparent competence turned
# out to be an outside perturbation. Split the two before believing a number.
#
# A shine onset counts as SELF-initiated when the player had no hitstun in the
# preceding --lookback frames (default 30).
#
# Also reports max chain length from ExPhil.Eval.ShineChain, the qualitative
# marker: an isolated shine is max chain 1, a real multishine chains. Prefer
# these over the off-manifold agreement metric, whose run-to-run spread
# (44-77% at one config) swamps most effects — see
# docs/planning/EXPOSURE_BIAS.md item 0.

alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(), strict: [port: :integer, lookback: :integer])

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/analyze_shine_source.exs <replay.slp> ...")
  System.halt(2)
end

lookback = opts[:lookback] || 30

# Which port is the BOT on? In Direct netplay connect order decides it,
# and the wrong-port default has now produced phantom-zero scores TWICE
# (07-30 "smear", 08-09 decider briefly read as 0 shines all session).
# Autodetect per replay from metadata, loudest-signal first:
#   1. netplay name containing "exphil" (the bot's connect-code tag)
#   2. the unique Fox, when exactly one port plays Fox (external id 2)
#   3. fall back to port 1
# --port N still overrides everything.
detect_port = fn path ->
  case ExPhil.Data.Peppi.metadata(path) do
    {:ok, meta} ->
      by_name =
        Enum.find(meta.players, fn p ->
          name = String.downcase("#{p.netplay_name || ""} #{p.tag || ""}")
          String.contains?(name, "exphil") or String.contains?(name, "exph")
        end)

      foxes = Enum.filter(meta.players, &(&1.character == 2))

      cond do
        by_name != nil -> {by_name.port, "netplay tag"}
        length(foxes) == 1 -> {hd(foxes).port, "unique Fox"}
        true -> {1, "default"}
      end

    _ ->
      {1, "default"}
  end
end

Output.puts("replay                    frames  shines  self  hit-ind  self/min  maxchain")

Enum.each(paths, fn path ->
  name = Path.basename(path, ".slp")

  {port, port_src} =
    case opts[:port] do
      nil -> detect_port.(path)
      p -> {p, "--port"}
    end

  if port_src != "--port", do: Output.puts("  [#{name}] scoring port #{port} (#{port_src})")

  case ExPhil.Data.Peppi.parse(path) do
    {:ok, replay} ->
      rows =
        Enum.flat_map(replay.frames, fn f ->
          case f.players[port] do
            nil -> []
            p -> [{trunc(p.action), trunc(p.hitstun_frames_left || 0)}]
          end
        end)

      n = max(length(rows), 1)

      onsets =
        rows
        |> Enum.with_index()
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.filter(fn [{{a, _}, _}, {{b, _}, _}] ->
          ShineChain.family(a) != :ground_reflect and ShineChain.family(b) == :ground_reflect
        end)
        |> Enum.map(fn [_, {_, i}] -> i end)

      {self_shines, hit_shines} =
        Enum.split_with(onsets, fn i ->
          rows
          |> Enum.slice(max(i - lookback, 0), min(lookback, i))
          |> Enum.all?(fn {_, hitstun} -> hitstun == 0 end)
        end)

      chains = rows |> Enum.map(&elem(&1, 0)) |> ShineChain.chains()
      minutes = n / 3600

      Output.puts(
        String.pad_trailing(name, 25) <>
          String.pad_leading("#{n}", 7) <>
          String.pad_leading("#{length(onsets)}", 8) <>
          String.pad_leading("#{length(self_shines)}", 6) <>
          String.pad_leading("#{length(hit_shines)}", 9) <>
          String.pad_leading("#{Float.round(length(self_shines) / max(minutes, 0.01), 1)}", 10) <>
          String.pad_leading("#{Enum.max(chains, fn -> 0 end)}", 10)
      )

    {:error, reason} ->
      # Slippi only finalizes on game end, so a killed run leaves a file peppi
      # refuses outright. Record with --seconds N so it SDs to a clean end.
      Output.error("#{name}: unparseable — #{inspect(reason)} (truncated? use --seconds)")
  end
end)

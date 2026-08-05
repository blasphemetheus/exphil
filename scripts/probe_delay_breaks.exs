# Delay-break forensics (task #20 arc, 2026-08-05): WHERE does the
# multishine cycle break under netplay delay?
#
# Material: the matched three-latency corpus — same policy (g10b), same
# human opponent, same day: LOCAL ~2f (chain 22), netplay d3 (chain 3),
# netplay d4 (chain 4). The 22->4 delta IS the residual human gap; this
# script names the phase where it lives, using ShineChain's built-in
# break taxonomy:
#   :empty_hop    — left the ground from jumpsquat, never shined (the
#                   canonical dropped link: aerial-B misfire)
#   :air_shine    — aerial shine happened but the air gap ran long (the
#                   sloppy full-jump loop)
#   :aerial_jump  — airborne, no shine, not from jumpsquat
#   :other_action — left the loop (hit, shield, wait, ...)
#
# Usage:
#   mix run scripts/probe_delay_breaks.exs \
#     [--replays "glob1,glob2"] [--port 1] [--out eval_runs/interp/delay_breaks.json]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Eval.ShineChain
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, port: :integer, out: :string]
  )

globs =
  (opts[:replays] ||
     "eval_runs/0805_human_g10b/**/*.slp,eval_runs/0805_direct_g10b/**/*.slp,eval_runs/0805_direct_g10b_d4/**/*.slp")
  |> String.split(",", trim: true)

out_path = opts[:out] || "eval_runs/interp/delay_breaks.json"

Output.banner("Delay-break forensics (three-latency corpus)")

results =
  for glob <- globs,
      path <- Path.wildcard(glob) do
    regime =
      cond do
        String.contains?(path, "direct_g10b_d4") -> :netplay_d4
        String.contains?(path, "direct_g10b") -> :netplay_d3
        true -> :local
      end

    case Peppi.parse(path) do
      {:ok, replay} ->
        actions =
          replay
          |> Peppi.to_training_frames(player_port: opts[:port] || 1, opponent_port: 2)
          |> Enum.reject(&(&1.game_state.frame < 0))
          |> Enum.map(& &1.game_state.players[1].action)

        # Per-cycle airborne-stretch lengths: from jumpsquat exit to the next
        # grounded reflector, counting only stretches containing an aerial
        # shine (real cycle links). This measures the STRETCH directly,
        # independent of the analyzer's max_air_gap break threshold.
        segments =
          actions
          |> Enum.map(&ShineChain.family/1)
          |> Enum.chunk_by(& &1)
          |> Enum.map(&{hd(&1), length(&1)})

        air_stretches =
          segments
          |> Enum.reduce({[], nil}, fn {fam, n}, {acc, cur} ->
            case {fam, cur} do
              {:jumpsquat, _} -> {acc, {0, false}}
              {:air_reflect, {len, _}} -> {acc, {len + n, true}}
              {:aerial_jump, {len, shined}} -> {acc, {len + n, shined}}
              {:ground_reflect, {len, true}} -> {[len | acc], nil}
              {:ground_reflect, _} -> {acc, nil}
              # non-cycle family while airborne-tracking: landing lag etc.
              # keeps accumulating up to a sanity cap, else abandon
              {_, {len, shined}} when len + n <= 30 -> {acc, {len + n, shined}}
              _ -> {acc, nil}
            end
          end)
          |> elem(0)

        %{
          replay: Path.basename(path),
          regime: regime,
          chains: ShineChain.chains_detailed(actions),
          air_stretches: air_stretches
        }

      _ ->
        Output.warning("unparseable: #{Path.basename(path)}")
        nil
    end
  end
  |> Enum.reject(&is_nil/1)

for regime <- [:local, :netplay_d3, :netplay_d4] do
  chains = results |> Enum.filter(&(&1.regime == regime)) |> Enum.flat_map(& &1.chains)
  n = length(chains)

  if n > 0 do
    real = Enum.filter(chains, &(&1.length >= 2))
    ends_all = chains |> Enum.frequencies_by(& &1.ended_by) |> Enum.sort_by(fn {_, c} -> -c end)
    ends_real = real |> Enum.frequencies_by(& &1.ended_by) |> Enum.sort_by(fn {_, c} -> -c end)
    lens = chains |> Enum.map(& &1.length) |> Enum.sort(:desc) |> Enum.take(8)

    Output.puts("")
    Output.puts("#{regime}: #{n} chain attempts, #{length(real)} real chains (len>=2), top lengths #{inspect(lens)}")
    Output.puts("  all ends:        #{inspect(ends_all)}")
    Output.puts("  real-chain ends: #{inspect(ends_real)}")

    stretches =
      results |> Enum.filter(&(&1.regime == regime)) |> Enum.flat_map(& &1.air_stretches)

    if stretches != [] do
      s = Enum.sort(stretches)
      m = length(s)
      pct = fn p -> Enum.at(s, min(trunc(p * m), m - 1)) end
      buckets = Enum.frequencies_by(s, fn l -> cond do
        l <= 5 -> "<=5"; l <= 8 -> "6-8"; l <= 12 -> "9-12"; true -> ">12" end
      end)

      Output.puts(
        "  air-stretches (shined cycles): n=#{m} p50=#{pct.(0.5)} p90=#{pct.(0.9)} " <>
          "max=#{List.last(s)} | #{inspect(Enum.sort(buckets))}"
      )
    end
  end
end

File.mkdir_p!(Path.dirname(out_path))

File.write!(
  out_path,
  Jason.encode!(Enum.map(results, fn r -> %{r | chains: Enum.map(r.chains, &Map.new/1)} end))
)

Output.success("Wrote #{out_path}")

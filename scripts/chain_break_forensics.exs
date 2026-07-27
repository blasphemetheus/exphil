# Why do multishine chains END? — per-break classification.
#
#   mix run scripts/chain_break_forensics.exs <replay.slp> [more.slp ...]
#     [--port N] [--gap N] [--window N]
#
# Max chain 3-7 (vs the teacher's 791) is ambiguous between three stories:
#   opponent  - the CPU jabbed/lasered the bot out of the loop (hitstun in
#               the post-break window): interference, not incompetence
#   pressure  - no hit landed, but the opponent was inside jab range at the
#               break (approach changes the state distribution)
#   unforced  - nobody near: the policy dropped the cycle on its own — the
#               exposure-bias signature, and the only category training
#               interventions can be judged on
# Also prints where the bot LANDS after each unforced break (the top action
# states in the following second) — grounded reflector at high action_frame
# is GOTCHAS #81's absorbing trap, visible here directly.
#
# Chain definition here is ONSET-GAP based (consecutive shine onsets <= --gap
# frames apart, default 15 ≈ the 9-frame cycle + slack), deliberately simpler
# than ExPhil.Eval.ShineChain's family walk — it keeps frame indices, which
# the classification needs. Counts can differ slightly from
# analyze_shine_source.exs; compare trends, not absolutes.

alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [port: :integer, gap: :integer, window: :integer]
  )

if paths == [] do
  IO.puts(:stderr, "usage: mix run scripts/chain_break_forensics.exs <replay.slp> ...")
  System.halt(2)
end

port = opts[:port] || 1
gap = opts[:gap] || 15
window = opts[:window] || 30

reflectors = [360, 361, 362, 363, 364, 365, 366, 367]

Enum.each(paths, fn path ->
  name = Path.basename(path, ".slp")

  case ExPhil.Data.Peppi.parse(path) do
    {:ok, replay} ->
      opp_port = if port == 1, do: 2, else: 1

      rows =
        replay.frames
        |> Enum.map(fn f ->
          p = f.players[port]
          o = f.players[opp_port]

          p &&
            %{
              action: trunc(p.action),
              af: trunc(p.action_frame || 0),
              hitstun: trunc(p.hitstun_frames_left || 0),
              dx: abs((p.x || 0.0) - ((o && o.x) || 999.0))
            }
        end)
        |> Enum.reject(&is_nil/1)

      onsets =
        rows
        |> Enum.with_index()
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.flat_map(fn [{a, _}, {b, i}] ->
          if b.action in reflectors and a.action not in reflectors, do: [i], else: []
        end)

      # Group onsets into chains: consecutive onsets <= gap frames apart.
      chains =
        Enum.reduce(onsets, [], fn i, acc ->
          case acc do
            [[last | _] = cur | rest] when i - last <= gap -> [[i | cur] | rest]
            _ -> [[i] | acc]
          end
        end)
        |> Enum.map(&Enum.reverse/1)
        |> Enum.reverse()

      breaks =
        chains
        |> Enum.map(fn chain ->
          break_at = List.last(chain)
          post = Enum.slice(rows, break_at, window)

          cause =
            cond do
              Enum.any?(post, &(&1.hitstun > 0)) -> :opponent
              Enum.any?(post, &(&1.dx < 15.0)) -> :pressure
              true -> :unforced
            end

          {length(chain), break_at, cause, post}
        end)

      by_cause = Enum.frequencies_by(breaks, fn {_, _, c, _} -> c end)
      chain_lens = Enum.map(breaks, fn {len, _, _, _} -> len end)

      Output.puts("#{name}: #{length(chains)} chains, max #{Enum.max(chain_lens, fn -> 0 end)}")

      Output.puts(
        "  breaks: #{by_cause[:opponent] || 0} opponent-hit, " <>
          "#{by_cause[:pressure] || 0} under-pressure, #{by_cause[:unforced] || 0} UNFORCED"
      )

      unforced_post =
        breaks
        |> Enum.filter(fn {_, _, c, _} -> c == :unforced end)
        |> Enum.flat_map(fn {_, _, _, post} -> post end)

      if unforced_post != [] do
        top =
          unforced_post
          |> Enum.frequencies_by(& &1.action)
          |> Enum.sort_by(fn {_, n} -> -n end)
          |> Enum.take(4)
          |> Enum.map_join(", ", fn {a, n} ->
            afs = unforced_post |> Enum.filter(&(&1.action == a)) |> Enum.map(& &1.af)
            "#{a} (#{n}f, af#{Enum.min(afs)}-#{Enum.max(afs)})"
          end)

        Output.puts("  after unforced breaks: #{top}")

        trap =
          unforced_post
          |> Enum.filter(&(&1.action == 361 and &1.af > 2))
          |> length()

        if trap > 0 do
          Output.puts("  ⚠ #{trap} frames in the GOTCHAS #81 trap (361 at af>2) post-break")
        end
      end

    {:error, reason} ->
      Output.error("#{name}: parse failed: #{inspect(reason)}")
  end
end)

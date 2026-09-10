# Failed-exit scan (2026-09-08; DEFINITION FIXED 09-09 — see below) — do the bot's "leave standing" inputs
# actually leave standing?
#
# Finding chain (V2_PREP passivity verdict): v2 enters WAIT at the expert
# rate and its per-frame non-neutral-input rate during WAIT matches
# experts, yet its WAIT episodes last 6.6x longer (32.8 vs 5.0 frames).
# Prediction: a large fraction of the bot's inputs during WAIT fail to
# END the state — sub-deadzone stick, no-op presses — the jab-smearing
# mechanism in a third costume (right rate, wrong magnitude/kind).
#
# For every WAIT (action 14) frame with a non-neutral input, classify the
# input and check whether the action state changes within --horizon
# frames. Report per group: failed-exit fraction overall and per kind.
#
# DEFINITION FIX (2026-09-09, GOTCHA #113): Slippi records the controller
# on the frame whose state it PRODUCED, so the input that ENDS a WAIT
# frame is recorded on the NEXT frame. The 09-08 numbers classified the
# same-frame controller, which by construction never ends the state
# (expert "full-stick" attempts there were Y-axis crouch/jump inputs).
# The scan now classifies the SUCCESSOR frame's controller: "attempt" =
# the input issued from this standing frame; "failed" = the state is
# still WAIT --horizon frames later.
#
#   mix run scripts/failed_exit_scan.exs --port 1 'eval_runs/0907_v2_live/**/*.slp'
#   mix run scripts/failed_exit_scan.exs --char-id 2 --limit-files 300 \
#     'replays/erickfm_ranked/FOX/extracted/*.slp'
# Options: --port N | --char-id N [2] · --limit-files N [400] · --horizon N [3]
#          --max-opp-idle F [1.0] (AFK gate) · --concurrency N [8] · --out FILE.md
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [port: :integer, char_id: :integer, limit_files: :integer, horizon: :integer,
             max_opp_idle: :float, concurrency: :integer, out: :string]
  )

files =
  paths
  |> Enum.flat_map(fn p -> if String.contains?(p, "*"), do: Path.wildcard(p), else: [p] end)
  |> Enum.filter(&String.ends_with?(&1, ".slp"))
  |> Enum.uniq()
  |> Enum.sort()
  |> Enum.take(opts[:limit_files] || 400)

if files == [], do: raise("no .slp matched")

horizon = opts[:horizon] || 3
max_opp_idle = opts[:max_opp_idle] || 1.0

Output.banner("Failed-exit scan")
Output.config([{"Files", length(files)}, {"Subject", opts[:port] || "char #{opts[:char_id] || 2}"}, {"Horizon", "#{horizon} frames"}])

defmodule FailedExit do
  @moduledoc false
  @wait 14
  @kinds [:stick_dead, :stick_mid, :stick_full, :button, :both]

  def kinds, do: @kinds

  def resolve_port(path, opts) do
    if opts[:port] do
      {:ok, opts[:port]}
    else
      char = opts[:char_id] || 2

      case Peppi.metadata(path) do
        {:ok, meta} ->
          case Enum.filter(meta.players, &(&1.character == char)) do
            [%{port: p}] -> {:ok, p}
            _ -> :skip
          end

        _ ->
          :skip
      end
    end
  end

  def scan(path, opts, horizon) do
    with {:ok, port} <- resolve_port(path, opts),
         {:ok, %{frames: frames}} when length(frames) > 1800 <- Peppi.parse(path) do
      arr = :array.from_list(frames)
      n = length(frames)

      subj = fn i ->
        f = :array.get(i, arr)
        Map.get(f.players || %{}, port)
      end

      opp_idle =
        frames
        |> Enum.count(fn f ->
          case Enum.find(f.players || %{}, fn {p, _} -> p != port end) do
            {_, o} -> trunc(o.action || 0) == @wait
            _ -> false
          end
        end)

      init = %{
        wait_frames: 0,
        neutral_wait_frames: 0,
        attempts: Map.new(@kinds, &{&1, 0}),
        failed: Map.new(@kinds, &{&1, 0}),
        opp_idle_frac: opp_idle / n
      }

      acc =
        Enum.reduce(0..(n - horizon - 1), init, fn i, acc ->
          case subj.(i) do
            %{action: a} when is_number(a) ->
              if trunc(a) == @wait do
                acc = %{acc | wait_frames: acc.wait_frames + 1}

                # the input issued FROM this frame (INVARIANTS.md item 9:
                # one definition, successor-aligned, never frames[i].controller)
                case ExPhil.Interp.Labels.kind(ExPhil.Interp.Labels.issued_input(arr, i, port)) do
                  :none ->
                    %{acc | neutral_wait_frames: acc.neutral_wait_frames + 1}

                  kind ->
                    exited? =
                      Enum.any?(1..horizon, fn d ->
                        case subj.(i + d) do
                          %{action: a2} when is_number(a2) -> trunc(a2) != @wait
                          _ -> false
                        end
                      end)

                    acc = update_in(acc.attempts[kind], &(&1 + 1))
                    if exited?, do: acc, else: update_in(acc.failed[kind], &(&1 + 1))
                end
              else
                acc
              end

            _ ->
              acc
          end
        end)

      {:ok, acc}
    else
      _ -> :skip
    end
  end

  def summarize(rows) do
    n = length(rows)
    sum = fn f -> rows |> Enum.map(f) |> Enum.sum() end
    wait = sum.(& &1.wait_frames)
    neutral = sum.(& &1.neutral_wait_frames)

    per_kind =
      Enum.map(@kinds, fn k ->
        a = sum.(& &1.attempts[k])
        f = sum.(& &1.failed[k])
        {k, a, f, if(a > 0, do: Float.round(f / a, 3), else: nil)}
      end)

    total_a = per_kind |> Enum.map(&elem(&1, 1)) |> Enum.sum()
    total_f = per_kind |> Enum.map(&elem(&1, 2)) |> Enum.sum()

    %{
      games: n,
      wait_frames: wait,
      neutral_frac: if(wait > 0, do: Float.round(neutral / wait, 3), else: nil),
      attempts: total_a,
      failed_frac: if(total_a > 0, do: Float.round(total_f / total_a, 3), else: nil),
      per_kind: per_kind
    }
  end
end

conc = opts[:concurrency] || 8

{afk, groups} =
  files
  |> Task.async_stream(fn p ->
    group =
      p |> Path.dirname() |> Path.split()
      |> Enum.reject(&Regex.match?(~r/^20\d\d-\d\d(-Mainline)?$/, &1))
      |> List.last()

    {group, FailedExit.scan(p, opts, horizon)}
  end, max_concurrency: conc, timeout: 120_000, on_timeout: :kill_task)
  |> Enum.reduce({%{}, %{}}, fn
    {:ok, {group, {:ok, row}}}, {afk, groups} ->
      if row.opp_idle_frac > max_opp_idle,
        do: {Map.update(afk, group, 1, &(&1 + 1)), groups},
        else: {afk, Map.update(groups, group, [row], &[row | &1])}

    _, acc ->
      acc
  end)

Enum.each(afk, fn {g, n} -> Output.warning("#{g}: #{n} game(s) EXCLUDED (opponent idle > #{max_opp_idle})") end)

kind_cols = FailedExit.kinds() |> Enum.map(&"#{&1} (n / failed)") |> Enum.join(" | ")
header = "| group | games | WAIT frames | neutral-input frac | attempts | **failed-exit frac** | #{kind_cols} |"
sep = "|---|---:|---:|---:|---:|---:|" <> String.duplicate("---:|", length(FailedExit.kinds()))

lines =
  groups
  |> Enum.sort()
  |> Enum.map(fn {g, rows} ->
    s = FailedExit.summarize(rows)
    kinds = s.per_kind |> Enum.map(fn {_k, a, f, frac} -> "#{a} / #{frac || "-"}" end) |> Enum.join(" | ")
    "| #{g} | #{s.games} | #{s.wait_frames} | #{s.neutral_frac} | #{s.attempts} | **#{s.failed_frac}** | #{kinds} |"
  end)

report = Enum.join([header, sep | lines], "\n")
IO.puts("\n" <> report <> "\n")
IO.puts("Kinds: stick_dead = 0.15-0.30 deflection (inside Melee's ~0.29 deadzone), stick_mid = 0.30-0.70, stick_full = >0.70, button = any of A/B/X/Y/Z/L/R/d-up with stick neutral, both = button + stick. failed = action still WAIT #{horizon} frames later.")

if opts[:out] do
  File.mkdir_p!(Path.dirname(opts[:out]))
  File.write!(opts[:out], "# Failed-exit scan — #{DateTime.utc_now() |> DateTime.to_iso8601()}\n\n" <> report <> "\n")
  Output.success("Wrote #{opts[:out]}")
end

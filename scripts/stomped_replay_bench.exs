#!/usr/bin/env elixir
# Time the stomped.dev policy bot on a real arena game, offline.
#
# A stored .slp is cut back into the messages the arena sent
# (ExPhil.Stomped.Replay) and fed through the same ExPhil.Stomped.Match the
# socket uses, with the policy bot in one seat. Each frame's handling time is
# the bot's whole compute share of the arena's 11 ms budget (decode + policy +
# pad); the network round trip comes on top of it.
#
# The replay does not react to our pads, so this measures TIME, not play: the
# policy sees another bot's game from that bot's seat.
#
# Usage (from the repo root, inside `devenv shell`):
#
#   mix run scripts/stomped_replay_bench.exs \
#     --policy eval_runs/0923_ppo/eval_candidate/candidate_policy.bin \
#     --replay test/fixtures/replays/stomped_fd_rR2hER7t.slp --port 1
#
# Flags:
#   --policy PATH      exported policy (required)
#   --replay PATH      .slp to replay (required)
#   --port N           the seat to play, 0 or 1 (default 1)
#   --max-frames N     stop after N frames (default: the whole game)
#   --windowed         windowed inference instead of stateful step
#   --deterministic    argmax instead of sampling
#   --temperature T    sampling temperature (default 1.0)

alias ExPhil.Stomped.{Match, Replay}
alias ExPhil.Stomped.Bots.Policy
alias ExPhil.Training.Output

{opts, rest, invalid} =
  OptionParser.parse(System.argv(),
    strict: [
      policy: :string,
      replay: :string,
      port: :integer,
      max_frames: :integer,
      windowed: :boolean,
      deterministic: :boolean,
      temperature: :float
    ]
  )

if invalid != [] or rest != [] or opts[:policy] == nil or opts[:replay] == nil do
  Output.error(
    "Usage: --policy PATH --replay PATH [--port 0|1] (bad: #{inspect(invalid ++ rest)})"
  )

  System.halt(1)
end

port = Keyword.get(opts, :port, 1)

Output.banner("ExPhil — stomped.dev policy timing")

Output.config([
  {"Policy", opts[:policy]},
  {"Replay", opts[:replay]},
  {"Seat (port)", port},
  {"Inference", if(opts[:windowed], do: "windowed", else: "stateful step")},
  {"EXLA client", inspect(Application.get_env(:exla, :default_client))}
])

Output.puts("⏳ JIT compiling model (warm-up)... this may take 2-5 minutes")

{:ok, policy} =
  Policy.start(
    policy: opts[:policy],
    stateful_step: not Keyword.get(opts, :windowed, false),
    deterministic: Keyword.get(opts, :deterministic, false),
    temperature: Keyword.get(opts, :temperature, 1.0)
  )

Output.success("Agent warm in #{policy.warmup_ms} ms")

{:ok, [header | messages]} = Replay.read_messages(opts[:replay])
messages = if n = opts[:max_frames], do: Enum.take(messages, n), else: messages

step = fn name, extra ->
  Jason.encode!(
    Map.merge(
      %{
        "step" => name,
        "matchId" => "bench",
        "port" => port,
        "inputDelayFrames" => 1,
        "stage" => 0
      },
      extra
    )
  )
end

match = Match.new(bot: {Policy, seat: policy.seat}, hold_ahead: 1)
{_, match} = Match.handle_text(step.("character", %{}), match)
{_, match} = Match.handle_text(step.("ready", %{}), match)
{_, match} = Match.handle_text(step.("game", %{"frame" => -124}), match)
{_, match} = Match.handle_binary(header, match)

total = length(messages)

{times, pads, match} =
  messages
  |> Enum.with_index(1)
  |> Enum.reduce({[], 0, match}, fn {message, index}, {times, pads, match} ->
    if rem(index, 500) == 0 do
      IO.write(:stderr, "\r  frame #{index}/#{total}\e[K")
    end

    started = System.monotonic_time(:microsecond)
    {replies, match} = Match.handle_binary(message, match)
    elapsed = System.monotonic_time(:microsecond) - started
    {[elapsed | times], pads + length(replies), match}
  end)

IO.write(:stderr, "\r\e[K")

sorted = Enum.sort(times)
count = length(sorted)
at = fn q -> Enum.at(sorted, min(count - 1, floor(q * count))) / 1000 end
ms = fn v -> :erlang.float_to_binary(v * 1.0, decimals: 2) end
budget_ms = 11.0
over = Enum.count(sorted, &(&1 / 1000 > budget_ms))
# The first frames after a cold trunk are slower; report the steady state too.
steady = times |> Enum.reverse() |> Enum.drop(120) |> Enum.sort()
steady_max = if steady == [], do: 0.0, else: List.last(steady) / 1000

stats = Policy.stats(match.bot_state)

Output.config([
  {"Frames", count},
  {"Pads sent", pads},
  {"Mean", "#{ms.(Enum.sum(sorted) / count / 1000)} ms"},
  {"p50 / p95 / p99", "#{ms.(at.(0.50))} / #{ms.(at.(0.95))} / #{ms.(at.(0.99))} ms"},
  {"Max (all / after 120 frames)", "#{ms.(List.last(sorted) / 1000)} / #{ms.(steady_max)} ms"},
  {"Frames over #{budget_ms} ms", "#{over} (#{ms.(100 * over / count)}%)"},
  {"Undecodable / no-decision frames", stats.errors}
])

if stats.errors > 0 do
  Output.error("The bot failed on #{stats.errors} frames; the timing above is not a clean run.")
  System.halt(1)
end

Output.success("Done")

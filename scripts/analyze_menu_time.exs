# Menu-time profiler: where does a session's wall time go between
# launch and gameplay? Parses a MeleePort session log (netplay or
# local), builds a timestamped phase timeline, compares against the
# expected budget, and flags known-unnecessary waits (e.g. blind-CSS
# retry windows that ran after RAM already confirmed the pick).
#
#   mix run scripts/analyze_menu_time.exs eval_runs/<run>/g1b.log
#
# Works best on --verbose logs (scene words + press points at info);
# degrades to the Frame/menu_state timeline otherwise.
alias ExPhil.Training.Output

path =
  case System.argv() do
    [p | _] -> p
    [] -> raise "usage: mix run scripts/analyze_menu_time.exs <session.log>"
  end

lines = File.stream!(path)

# --- timestamp helpers ------------------------------------------------
# Logger lines: "00:21:36.233 [info] ..."; Output lines: "[00:21:36] ...".
parse_ts = fn line ->
  case Regex.run(~r/^(\d\d):(\d\d):(\d\d)(?:\.(\d+))?\s/, line) ||
         Regex.run(~r/^\[(\d\d):(\d\d):(\d\d)\]/, line) do
    [_, h, m, s | rest] ->
      ms =
        case rest do
          [frac] -> String.to_integer(String.pad_trailing(frac, 3, "0"))
          _ -> 0
        end

      String.to_integer(h) * 3_600_000 + String.to_integer(m) * 60_000 +
        String.to_integer(s) * 1_000 + ms

    nil ->
      nil
  end
end

classify = fn line ->
  cond do
    line =~ ~r/Step 3\/5: Initializing Dolphin/ -> :launch
    line =~ ~r/Dolphin initialized and connected/ -> :connected
    line =~ ~r/RAM scene word -> 0x\w*04 / -> :scene_game
    line =~ ~r/RAM scene word -> 0x08080100/ -> :scene_css
    line =~ ~r/RAM scene word/ -> :scene_other
    line =~ ~r/blind CSS press point/ && line =~ ~r/SKIPPING/ -> :press_skip
    line =~ ~r/blind CSS press point/ -> :press_point
    line =~ ~r/retrying A press/ -> :retry
    line =~ ~r/handing back to helper/ -> :handback
    line =~ ~r/#10 delay probe/ -> :in_game_tick
    line =~ ~r/menu hold/ -> :hold_suppressed
    line =~ ~r/MENU STUCK/ -> :stuck
    true -> nil
  end
end

events =
  lines
  |> Enum.flat_map(fn line ->
    with kind when kind != nil <- classify.(line),
         ts when ts != nil <- parse_ts.(line) do
      [{ts, kind, String.trim(line)}]
    else
      _ -> []
    end
  end)

if events == [] do
  Output.error("no recognizable events — is this a --verbose MeleePort log?")
  System.halt(1)
end

# Midnight wrap: normalize so time is monotone.
{events, _} =
  Enum.map_reduce(events, nil, fn {ts, k, l}, prev ->
    ts = if prev && ts < prev - 12 * 3_600_000, do: ts + 24 * 3_600_000, else: ts
    {{ts, k, l}, ts}
  end)

t0 = elem(hd(events), 0)
fmt = fn ms -> :io_lib.format("~6.1f", [ms / 1000]) |> IO.iodata_to_binary() end

Output.banner("Menu-time profile: #{Path.basename(path)}")

# --- phase fold -------------------------------------------------------
# Budgets in ms; nil = report-only (human/JIT dependent).
budgets = %{
  "launch -> console connected" => 3_000,
  "connected -> online CSS" => 3_000,
  "CSS -> press point (steer + JIT)" => nil,
  "press -> handback (pick + START pulses)" => 7_000,
  "handback -> in-game (code entry + opponent)" => nil,
  "postgame CSS -> next in-game" => nil
}

defmodule Prof do
  def first(events, kind), do: Enum.find(events, &(elem(&1, 1) == kind))
  def after_t(events, t, kind), do: Enum.find(events, &(elem(&1, 0) > t and elem(&1, 1) == kind))
end

launch = Prof.first(events, :launch)
connected = Prof.first(events, :connected)
first_css = Prof.first(events, :scene_css)
first_press = Prof.first(events, :press_point) || Prof.first(events, :press_skip)
first_handback = Prof.first(events, :handback)
first_game = Prof.first(events, :in_game_tick) || Prof.first(events, :scene_game)

rows =
  [
    {"launch -> console connected", launch, connected},
    {"connected -> online CSS", connected, first_css},
    {"CSS -> press point (steer + JIT)", first_css, first_press},
    {"press -> handback (pick + START pulses)", first_press, first_handback},
    {"handback -> in-game (code entry + opponent)", first_handback, first_game}
  ]
  |> Enum.map(fn {name, a, b} ->
    case {a, b} do
      {{ta, _, _}, {tb, _, _}} -> {name, tb - ta}
      _ -> {name, nil}
    end
  end)

Output.puts("Phase durations (first game cycle):")

for {name, dur} <- rows do
  budget = budgets[name]

  flag =
    cond do
      dur == nil -> "  (not observed)"
      budget == nil -> ""
      dur > budget -> "  <-- OVER budget #{fmt.(budget)}s"
      true -> ""
    end

  Output.puts("  #{String.pad_trailing(name, 46)} #{if dur, do: fmt.(dur) <> "s", else: "  -"}#{flag}")
end

# --- waste detection: retry windows after a RAM-confirmed pick --------
skips = Enum.filter(events, &(elem(&1, 1) == :press_skip))
retries = Enum.filter(events, &(elem(&1, 1) == :retry))

wasted =
  for {tr, _, _} <- retries,
      Enum.any?(skips, fn {ts, _, _} -> abs(ts - tr) < 1_000 end) do
    tr
  end

if wasted != [] and first_handback != nil do
  {th, _, _} = first_handback
  saved = th - hd(wasted)

  Output.warning(
    "#{length(wasted)} retry window(s) ran AFTER RAM confirmed the pick — " <>
      "~#{fmt.(saved)}s of pure wait (fixed by the selection-confirmed early handback)"
  )
end

# --- per-game cycles --------------------------------------------------
game_starts = Enum.filter(events, &(elem(&1, 1) == :scene_game))
css_returns = Enum.filter(events, &(elem(&1, 1) == :scene_css))

if length(game_starts) > 1 do
  Output.puts("")
  Output.puts("Rematch cycles (game end -> next game start):")

  for {{tg, _, _}, i} <- Enum.with_index(tl(game_starts), 2) do
    prev_css = css_returns |> Enum.filter(fn {t, _, _} -> t < tg end) |> List.last()

    case prev_css do
      {tc, _, _} -> Output.puts("  game #{i}: CSS -> in-game #{fmt.(tg - tc)}s")
      _ -> :ok
    end
  end
end

for {t, k, l} <- events, k in [:stuck, :hold_suppressed] do
  Output.puts("")
  Output.puts("  [#{fmt.(t - t0)}s] #{if k == :stuck, do: "STUCK", else: "hold suppressed"}: #{String.slice(l, 0, 100)}")
end

Output.puts("")
Output.puts("Total observed: #{fmt.(elem(List.last(events), 0) - t0)}s from first event")

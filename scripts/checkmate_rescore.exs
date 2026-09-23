# Re-score a checkmate sweep (sweep.json) with the current model, without
# re-simulating: the sim verdicts stay, the model runs again on each saved
# state. Facing is read back from the replay.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/checkmate_rescore.exs eval_runs/0923_checkmate/sweep.json 'checkpoints/*/live_bradley_*/*/*.slp'
alias ExPhil.Melee.Checkmate
[sweep | globs] = System.argv()
rows = Jason.decode!(File.read!(sweep))
paths = globs |> Enum.flat_map(&Path.wildcard/1) |> Map.new(&{Path.basename(&1), &1})

facings =
  rows
  |> Enum.group_by(& &1["replay"])
  |> Enum.flat_map(fn {replay, rs} ->
    {:ok, r} = ExPhil.Data.Peppi.parse(paths[replay])
    by = Map.new(r.frames, &{&1.frame_number, &1})
    for x <- rs, do: {{replay, x["port"], x["frame"]}, by[x["frame"]].players[x["port"]].facing}
  end)
  |> Map.new()

scored =
  rows
  |> Task.async_stream(fn x ->
    st = x["state"] |> Map.new(fn {k, v} -> {String.to_atom(k), v} end)
    f = facings[{x["replay"], x["port"], x["frame"]}]
    st = Map.put(st, :facing, if(f && f < 0, do: -1.0, else: 1.0))
    Map.put(x, "model_now", Checkmate.checkmate?(st))
  end, max_concurrency: System.schedulers_online(), timeout: :infinity)
  |> Enum.map(fn {:ok, x} -> x end)

agree = fn key -> Enum.count(scored, &(&1[key] == &1["sim_checkmate"])) end
IO.puts("#{length(scored)} positions: old model agrees on #{agree.("model")}, current model on #{agree.("model_now")}")
IO.puts("current confusion {model, sim}: #{inspect(Enum.frequencies_by(scored, &{&1["model_now"], &1["sim_checkmate"]}))}")
IO.puts("by kind {kind, agrees}: #{inspect(Enum.frequencies_by(scored, &{&1["kind"], &1["model_now"] == &1["sim_checkmate"]}))}")
IO.puts("\ndisagreements:")
for x <- scored, x["model_now"] != x["sim_checkmate"] do
  s = x["state"]
  IO.puts("  #{x["replay"]} p#{x["port"]} f#{x["frame"]} #{x["kind"]}: model #{if x["model_now"], do: "checkmate", else: "recoverable"}, sim #{x["made"]}/#{x["tried"]}  stage #{s["stage"]} x #{Float.round(s["x"], 1)} y #{Float.round(s["y"], 1)} jumps #{s["jumps_left"]} kb (#{Float.round(s["kb_vx"] * 1.0, 2)}, #{Float.round(s["kb_vy"] * 1.0, 2)})")
end
File.write!(Path.rootname(sweep) <> "_rescored.json", Jason.encode!(scored, pretty: true))

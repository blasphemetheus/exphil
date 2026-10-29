# Per-game skill score for the Fox player (2026-10-29): the master-only line plateaued
# (data ceiling; width did not help), and only the 7,853 "master-master-*" files carry a tier —
# the other 56k are anonymous. This scores EVERY Fox game from play itself (ExPhil.Eval.PlayStats
# rates + the fidelity scorecard's derived L-cancel / short-hop measures) so the master-master
# games can calibrate a "master-like" selector over the untiered ones.
#
#   mix run --no-compile scripts/corpus_skill_score.exs --replays replays/erickfm_ranked/v2_filtered \
#     --out eval_runs/1029_skill/scores.jsonl [--workers 8] [--limit N]
#
# One JSON row per (file, Fox port): tier (master | untiered), ditto?, opponent character, stage,
# minutes, win?, PlayStats rates, l_cancel_rate, short_hop_share, apm (input changes / min).
# Rollback duplicate frames are dropped (GOTCHA #144). CPU only (Peppi NIF, ~30 ms per game).

alias ExPhil.Data.Peppi
alias ExPhil.Eval.PlayStats
alias ExPhil.Sim.GA
alias ExPhil.Training.Output
alias ExPhil.Bridge.ControllerState

{opts, _, bad} = OptionParser.parse(System.argv(), strict: [replays: :string, out: :string, workers: :integer, limit: :integer])
if bad != [], do: raise("bad options: #{inspect(bad)}")
dir = opts[:replays] || "replays/erickfm_ranked/v2_filtered"
out = opts[:out] || "eval_runs/1029_skill/scores.jsonl"
workers = opts[:workers] || 8
File.mkdir_p!(Path.dirname(out))

files = Path.wildcard(Path.join(dir, "*.slp")) |> Enum.sort()
files = if opts[:limit], do: Enum.take(files, opts[:limit]), else: files
Output.banner("Corpus skill score: #{length(files)} files, #{workers} workers")

ctrl = fn c ->
  %ControllerState{
    main_stick: %{x: c.main_stick_x, y: c.main_stick_y}, c_stick: %{x: c.c_stick_x, y: c.c_stick_y},
    l_shoulder: c.l_trigger, r_shoulder: c.r_trigger,
    button_a: c.button_a, button_b: c.button_b, button_x: c.button_x, button_y: c.button_y,
    button_z: c.button_z, button_l: c.button_l, button_r: c.button_r, button_d_up: c.button_d_up
  }
end

neutral_ctrl = %ControllerState{main_stick: %{x: 0.5, y: 0.5}, c_stick: %{x: 0.5, y: 0.5}, l_shoulder: 0.0, r_shoulder: 0.0,
  button_a: false, button_b: false, button_x: false, button_y: false, button_z: false, button_l: false, button_r: false, button_d_up: false}

derived = fn s ->
  # histogram keys are STRINGS (PlayStats); normalized shares, so hit+miss is a share not a count
  lag = s.hists["landing_lag"] || %{}
  p = fn ks -> Enum.sum(Enum.map(ks, &Map.get(lag, "#{&1}", 0.0))) end
  hit = p.([7, 9, 10, 11]); miss = p.([15, 18, 20, 22])
  peaks = s.hists["jump_peak"] || %{}
  ki = fn k -> case Integer.parse(to_string(k)) do {v, _} -> v; _ -> -1 end end
  low = peaks |> Enum.filter(fn {k, _} -> ki.(k) in 5..19 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  real = peaks |> Enum.filter(fn {k, _} -> ki.(k) >= 5 end) |> Enum.map(&elem(&1, 1)) |> Enum.sum()
  rr = fn a, b -> if b == 0 or b == 0.0, do: nil, else: Float.round(a / b, 3) end
  %{l_cancel_rate: rr.(hit, hit + miss), l_cancel_n: hit + miss, short_hop_share: rr.(low, real)}
end

score_file = fn path ->
  try do
    {:ok, meta} = Peppi.metadata(path)
    foxes = Enum.filter(meta.players, &(String.downcase(&1.character_name || "") == "fox"))
    if foxes == [] do
      []
    else
      {:ok, replay} = Peppi.parse(path)
      raw = replay.frames |> Enum.filter(&(&1.frame_number >= 0)) |> Enum.reduce(%{}, fn f, acc -> Map.put(acc, f.frame_number, f) end) |> Map.values() |> Enum.sort_by(& &1.frame_number)
      stage = meta.stage
      edge = GA.stage_edge(stage)
      tier = if String.starts_with?(Path.basename(path), "master-master-"), do: "master", else: "untiered"
      for fox <- foxes do
        opp = Enum.find(meta.players, &(&1.port != fox.port))
        rows =
          raw
          |> Enum.flat_map(fn f ->
            p = Map.get(f.players, fox.port); o = Map.get(f.players, opp.port)
            if p && o, do: [%{own: p, opp: o, controller: (if p.controller, do: ctrl.(p.controller), else: neutral_ctrl)}], else: []
          end)
        if length(rows) < 1800 do
          []
        else
          s = PlayStats.from_game(rows, edge) |> PlayStats.summarize()
          last = List.last(rows)
          # APM-ish: frames where the controller differs from the previous frame, per minute
          minutes = length(rows) / 3600
          # actions per minute = button presses + dashes + ground jumps + wavedashes (raw controller diffs count analog noise)
          r = s.rates
          actions = Enum.sum(for {k, v} <- r, String.starts_with?(k, "press_") and is_number(v), do: v) + (r["dashes_per_min"] || 0) + (r["ground_jumps_per_min"] || 0) + (r["wavedashes_per_min"] || 0)
          [Map.merge(%{file: Path.basename(path), port: fox.port, tier: tier, ditto: length(foxes) > 1, opponent: opp.character_name, stage: stage,
             minutes: Float.round(minutes, 2), win: (last.own.stock || 0) > (last.opp.stock || 0), final_stocks: [last.own.stock, last.opp.stock],
             apm: Float.round(actions * 1.0, 1), netplay_name: fox.netplay_name},
            Map.merge(Map.new(s.rates, fn {k, v} -> {String.to_atom(k), v} end), derived.(s)))]
        end
      end
    end
  rescue
    e -> [%{file: Path.basename(path), error: Exception.message(e) |> String.slice(0, 120)}]
  end
end

t0 = System.monotonic_time(:millisecond)
{:ok, io} = File.open(out, [:write])
n =
  files
  |> Task.async_stream(score_file, max_concurrency: workers, timeout: 120_000, ordered: false, on_timeout: :kill_task)
  |> Stream.with_index(1)
  |> Enum.reduce(0, fn {res, i}, acc ->
    rows = case res do {:ok, rows} -> rows; _ -> [] end
    Enum.each(rows, &IO.write(io, Jason.encode!(&1) <> "\n"))
    if rem(i, 2000) == 0, do: Output.puts("  #{i}/#{length(files)} files, #{acc + length(rows)} rows (#{div(System.monotonic_time(:millisecond) - t0, 1000)} s)")
    acc + length(rows)
  end)
File.close(io)
Output.puts("RESULT skill scores: #{n} Fox-game rows from #{length(files)} files in #{div(System.monotonic_time(:millisecond) - t0, 1000)} s -> #{out}")

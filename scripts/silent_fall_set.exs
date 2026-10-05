# Silent-fall correction set (2026-10-05, lever 2 for the silent fall —
# INPUT_COHERENCE_2026-10-01.md "10-05 12:55"): MANUFACTURE the offstage
# states the bot drives itself into and the expert never produces, and label
# them with the expert's own decision.
#
# For each expert decision frame t (offstage, actionable, double jump in
# hand, the expert's input at t is active after a neutral frame at t-1):
#   1. seed the sim bit-exactly at t-1 (ExPhil.Sim.Seed; skipped on divergence);
#   2. inject k frames of NEUTRAL on the subject's lane while the opponent,
#      RNG and everything else follow the real game (replay-exact rows with
#      one lane overwritten) — the silent fall, stopped early if the subject
#      is hit, lands, grabs the ledge, goes helpless or dies;
#   3. oracle check: from the silent state, play the expert's recorded inputs
#      from t (time-shifted) against a neutral opponent; keep the LARGEST k
#      from which the expert's decision still returns (ledge or ground, stock
#      kept), down to k_min;
#   4. export one frame list per seed: `prefix` real frames (t-prefix..t-1)
#      then the k injected states; injected frame j carries
#      controller = expert's decision at t when j >= k_min (else neutral) and
#      prev_controller = neutral (what was actually pressed), in the
#      scripts/export_drill_frames.exs format for `--mix-frames`.
#
#   mix run scripts/silent_fall_set.exs [--split checkpoints/coh_base/split.json] [--max-games 150]
#     [--k 30] [--k-min 12] [--prefix 90] [--oracle-frames 90] [--y-min -60] [--y-max 40]
#     [--out data/silent_fall/fd_fox.frames] [--report eval_runs/1005_silent_fall/set_report.json]
alias ExPhil.Data.Peppi
alias ExPhil.Sim.{Drill, Env, GA, Seed}
alias ExPhil.Training.{Output, SilentFallWeighting}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [split: :string, max_games: :integer, k: :integer, k_min: :integer, prefix: :integer, oracle_frames: :integer,
             y_min: :float, y_max: :float, out: :string, report: :string, stage: :integer, action_delay: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

split = (opts[:split] || "checkpoints/coh_base/split.json") |> File.read!() |> Jason.decode!()
stage = opts[:stage] || 32
k_max = opts[:k] || 30
k_min = opts[:k_min] || 12
prefix = opts[:prefix] || 90
oracle_frames = opts[:oracle_frames] || 90
y_min = opts[:y_min] || -60.0
y_max = opts[:y_max] || 40.0
out = opts[:out] || "data/silent_fall/fd_fox.frames"
report_path = opts[:report] || "eval_runs/1005_silent_fall/set_report.json"
action_delay = opts[:action_delay] || 0
edge = GA.stage_edge(stage)
neutral = Drill.neutral()
# one 16-byte MslReplayInputPlayer lane with nothing pressed (flags byte 3 = lanes valid)
neutral_lane = <<0::unsigned-16-little, 0::signed-8, 0::signed-8, 0::signed-8, 0::signed-8, 0::8, 0::8,
                 0::signed-8, 0::signed-8, 0::signed-8, 0::signed-8, 3::8, 0::size(3 * 8)>>

Output.banner("Silent-fall correction set")
Output.config([{"stage", stage}, {"k", "#{k_min}..#{k_max}"}, {"prefix", prefix}, {"oracle", "#{oracle_frames} f vs neutral opp"},
  {"y range", "#{y_min}..#{y_max}"}, {"out", out}])

# training files only — the validation split feeds the probes
files = split["train"]

chosen =
  files
  |> Task.async_stream(fn p -> with {:ok, m} <- Peppi.metadata(p), do: {p, m} end, max_concurrency: 16, timeout: 60_000)
  |> Enum.flat_map(fn
    {:ok, {p, %{stage: ^stage, players: [_, _] = ps} = m}} ->
      case Enum.filter(ps, &(String.downcase(&1.character_name || "") == "fox")) do
        [own] -> [{p, m, own, Enum.find(ps, &(&1.port != own.port))}]
        _ -> []
      end

    _ ->
      []
  end)
  |> Enum.take(opts[:max_games] || 150)

Output.puts("#{length(chosen)} games (of #{length(files)} scanned)")

offstage? = fn p -> p.on_ground != true and (abs(p.x || 0.0) > edge or (p.y || 0.0) < -12.0) end

replace_lane = fn row, idx, lane ->
  <<head::binary-size(16 + 16 * idx), _::binary-size(16), rest::binary>> = row
  head <> lane <> rest
end
lane_of = fn row, idx -> binary_part(row, 16 + 16 * idx, 16) end

# the subject's slot in the sim is its position in port order (+1 for the players map)
remap = fn gs, own_slot ->
  opp_slot = 3 - own_slot
  %{gs | players: %{1 => gs.players[own_slot], 2 => gs.players[opp_slot]}} |> Map.put(:own_port, 1)
end

dead? = fn p -> (p.action || 0) <= 12 end
returned? = fn p -> (p.action || 0) in [252, 253] or (p.on_ground == true and abs(p.x || 0.0) <= edge + 2.0 and not dead?.(p)) end

# a silent-fall frame is still a silent fall: airborne, untouched, not on the ledge / helpless / dead
still_falling? = fn p, p0 ->
  p.on_ground != true and (p.stock || 0) == (p0.stock || 0) and (p.percent || 0.0) <= (p0.percent || 0.0) + 0.01 and
    SilentFallWeighting.falling?(%{game_state: %{players: %{1 => p}}})
end

results =
  Enum.map(chosen, fn {path, meta, own, opp} ->
    {:ok, replay} = Peppi.parse(path, player_port: own.port)

    tf =
      replay
      |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port, remap_ports: true)
      |> Enum.reject(&(&1.game_state.frame < 0))

    tft = List.to_tuple(tf)
    by_num = Map.new(Enum.with_index(tf), fn {f, i} -> {f.game_state.frame, i} end)

    # decision frames: offstage, actionable, jump in hand, in the height band,
    # expert ACTS at t after a neutral t-1
    cands =
      for i <- prefix..(tuple_size(tft) - 1),
          f = elem(tft, i),
          p = f.game_state.players[1],
          offstage?.(p) and SilentFallWeighting.falling?(f) and (p.jumps_left || 0) >= 1 and
            (p.y || 0.0) >= y_min and (p.y || 0.0) <= y_max and
            not SilentFallWeighting.neutral?(f.controller) and SilentFallWeighting.neutral?(elem(tft, i - 1).controller),
          # the prefix must be one continuous replay segment
          elem(tft, i - prefix).game_state.frame == f.game_state.frame - prefix,
          do: f.game_state.frame

    if cands == [] do
      Output.puts("  #{Path.basename(path)}: no decision frames")
      %{file: path, candidates: 0, seeded: 0, injected: 0, kept: 0, silence_kills: 0, lists: []}
    else
      {:ok, raw} = Peppi.parse(path)
      port_ids = meta.players |> Enum.map(& &1.port) |> Enum.sort()
      own_idx = Enum.find_index(port_ids, &(&1 == own.port))
      opp_idx = 1 - own_idx
      own_slot = own_idx + 1
      gf = Map.new(raw.frames, &{&1.frame_number, &1})
      row_at = fn num -> Seed.replay_row(gf[num], port_ids) end

      {:ok, seeded} = Seed.from_replay(path, frame: Enum.max(cands) - 1, frames: Enum.map(cands, &(&1 - 1)))
      sim = seeded.sim
      good_saves = seeded.saves |> Enum.reject(& &1.diverged?) |> Map.new(&{&1.frame + 1, &1})

      lists =
        cands
        |> Enum.filter(&Map.has_key?(good_saves, &1))
        |> Enum.flat_map(fn t ->
          save = good_saves[t]
          p0 = save.state.players[own_slot]
          decision = elem(tft, by_num[t]).controller

          # 1-2. the silent fall: k frames of neutral on the subject's lane, everything else as recorded
          {:ok, _} = Env.restore(sim, 0, save.blob)

          {injected, _} =
            Enum.reduce_while(1..k_max, {[], nil}, fn j, {acc, _} ->
              num = t + j - 1

              case gf[num] && Env.step_replay(sim, [replace_lane.(row_at.(num), own_idx, neutral_lane)]) do
                {:ok, [gs], _} ->
                  if still_falling?.(gs.players[own_slot], p0) do
                    {:ok, blob} = Env.save(sim, 0)
                    {:cont, {[{j, gs, blob} | acc], nil}}
                  else
                    {:halt, {acc, :stopped}}
                  end

                _ ->
                  {:halt, {acc, :stopped}}
              end
            end)

          injected = Enum.reverse(injected)

          # 3. oracle: from the silent state after k frames, the expert's inputs from t vs a neutral opponent
          oracle = fn {k, _gs, blob} ->
            {:ok, _} = Env.restore(sim, 0, blob)

            Enum.reduce_while(0..(oracle_frames - 1), :fell, fn i, _ ->
              src = gf[t + i]
              cur = gf[t + k + i] || src

              if src == nil do
                {:halt, :fell}
              else
                row = row_at.(cur.frame_number) |> replace_lane.(own_idx, lane_of.(row_at.(src.frame_number), own_idx)) |> replace_lane.(opp_idx, neutral_lane)

                case Env.step_replay(sim, [row]) do
                  {:ok, [gs], _} ->
                    p = gs.players[own_slot]
                    cond do
                      (p.stock || 0) < (p0.stock || 0) or dead?.(p) -> {:halt, :died}
                      returned?.(p) -> {:halt, :returned}
                      true -> {:cont, :fell}
                    end

                  _ ->
                    {:halt, :died}
                end
              end
            end)
          end

          # does silence alone kill from the deepest state? (the label matters)
          silence_kills =
            case List.last(injected) do
              nil -> nil
              {k, _gs, blob} ->
                {:ok, _} = Env.restore(sim, 0, blob)
                Enum.reduce_while(0..(oracle_frames - 1), false, fn i, _ ->
                  case gf[t + k + i] && Env.step_replay(sim, [row_at.(t + k + i) |> replace_lane.(own_idx, neutral_lane)]) do
                    {:ok, [gs], _} ->
                      p = gs.players[own_slot]
                      cond do
                        (p.stock || 0) < (p0.stock || 0) or dead?.(p) -> {:halt, true}
                        returned?.(p) -> {:halt, false}
                        true -> {:cont, false}
                      end
                    _ -> {:halt, false}
                  end
                end)
            end

          kept =
            injected
            |> Enum.filter(fn {k, _, _} -> k >= k_min end)
            |> Enum.reverse()
            |> Enum.find(fn entry -> oracle.(entry) == :returned end)

          case kept do
            nil ->
              [%{t: t, injected: length(injected), kept: 0, silence_kills: silence_kills, list: nil}]

            {k, _, _} ->
              real = for i <- (by_num[t] - prefix)..(by_num[t] - 1), do: elem(tft, i)
              tag = elem(tft, by_num[t])[:player_tag]

              made =
                injected
                |> Enum.take(k)
                |> Enum.map(fn {j, gs, _} ->
                  %{game_state: remap.(gs, own_slot), controller: if(j >= k_min, do: decision, else: neutral),
                    prev_controller: neutral, player_tag: tag}
                end)

              [%{t: t, injected: length(injected), kept: k, silence_kills: silence_kills, y: Float.round((p0.y || 0.0) * 1.0, 1), list: real ++ made}]
          end
        end)

      Env.stop(sim)
      kept = Enum.filter(lists, &(&1.list != nil))

      Output.puts("  #{Path.basename(path)}: #{length(cands)} decision frames, #{map_size(good_saves)} seeded, " <>
        "#{Enum.count(lists, &(&1.injected >= k_min))} fell silent >= #{k_min} f, #{length(kept)} kept (k med #{if kept == [], do: "-", else: Enum.sort_by(kept, & &1.kept) |> Enum.at(div(length(kept), 2)) |> Map.get(:kept)}), " <>
        "silence kills #{Enum.count(lists, &(&1.silence_kills == true))}/#{Enum.count(lists, &(&1.silence_kills != nil))}" <>
        if(seeded.divergence, do: "  (diverged at #{inspect(elem(seeded.divergence, 0))})", else: ""))

      %{file: path, candidates: length(cands), seeded: map_size(good_saves), injected: Enum.count(lists, &(&1.injected >= k_min)),
        kept: length(kept), silence_kills: Enum.count(lists, &(&1.silence_kills == true)),
        divergence: seeded.divergence && inspect(seeded.divergence), lists: Enum.map(kept, & &1.list),
        ks: Enum.map(kept, & &1.kept), ys: Enum.map(kept, & &1.y)}
    end
  end)

frame_lists = Enum.flat_map(results, & &1.lists)
act_frames = frame_lists |> Enum.flat_map(& &1) |> Enum.count(&(is_map_key(&1, :prev_controller) and not SilentFallWeighting.neutral?(&1.controller)))
total = frame_lists |> Enum.map(&length/1) |> Enum.sum()

File.mkdir_p!(Path.dirname(out))
File.write!(out, :erlang.term_to_binary(%{
  expert: "silent_fall_expert_decision",
  exported_at: DateTime.utc_now() |> DateTime.to_iso8601(),
  action_delay: action_delay,
  label_convention: ExPhil.Data.LabelConvention.current(),
  frame_lists: frame_lists
}, [:compressed]))

sum = fn k -> results |> Enum.map(&Map.get(&1, k, 0)) |> Enum.sum() end
File.mkdir_p!(Path.dirname(report_path))
File.write!(report_path, Jason.encode!(%{
  "games" => length(results), "candidates" => sum.(:candidates), "seeded" => sum.(:seeded), "fell_silent" => sum.(:injected),
  "kept" => sum.(:kept), "silence_kills" => sum.(:silence_kills), "frames" => total, "act_frames" => act_frames,
  "k_hist" => results |> Enum.flat_map(&Map.get(&1, :ks, [])) |> Enum.frequencies(),
  "per_game" => Enum.map(results, &Map.drop(&1, [:lists, :ks, :ys]))
}, pretty: true))

Output.puts("RESULT silent-fall set: #{sum.(:candidates)} decision frames -> #{sum.(:seeded)} seeded -> #{sum.(:injected)} fell silent >= #{k_min} f -> " <>
  "#{sum.(:kept)} kept (expert's decision still returns); silence alone kills #{sum.(:silence_kills)}; " <>
  "#{total} frames (#{act_frames} labelled act) -> #{out}")

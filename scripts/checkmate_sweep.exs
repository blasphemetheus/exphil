# Calibrate ExPhil.Melee.Checkmate against the sim over real replays.
# For every Fox port and every offstage episode: evaluate the model at each
# decision frame (actionable, not mid-special / air dodge / helpless);
# record the first checkmate flip and the decision frame before it; also
# every stock loss's first decision frame after the last hit. Then seed each
# replay once with all candidate frames and TAS every candidate
# (ExPhil.Sim.RecoveryProbe). Disagreements are where the model is wrong.
#
#   elixir -pa '_build/dev/lib/*/ebin' scripts/checkmate_sweep.exs --out eval_runs/0923_checkmate 'checkpoints/*/live_bradley_*/*/*.slp'
alias ExPhil.Data.Peppi
alias ExPhil.Melee.Checkmate
alias ExPhil.Sim.{Env, RecoveryProbe, Seed}

{opts, globs, _} = OptionParser.parse(System.argv(), strict: [out: :string, limit: :integer])
out = opts[:out] || "eval_runs/0923_checkmate"
File.mkdir_p!(out)
paths = globs |> Enum.flat_map(&Path.wildcard/1) |> Enum.sort() |> then(&if(opts[:limit], do: Enum.take(&1, opts[:limit]), else: &1))
# Frozen Stadium only in the sim; Dolphin's Stadium is unfrozen, so skip stage 3.
stages = [2, 8, 28, 31, 32]
busy = MapSet.new([35, 36, 37, 236] ++ Enum.to_list(341..372) ++ Enum.to_list(252..263))
batch = 128

# hitstun_frames_left is a reused slot: after hitstun it holds an int read as a denormal float (2.9e-44)
decision? = fn p -> not p.on_ground and (p.hitstun_frames_left || 0) < 0.5 and (p.hitlag_left || 0) < 0.5 and p.action > 12 and not MapSet.member?(busy, p.action) end
model_state = fn p, stage ->
  %{stage: stage, x: p.x, y: p.y, vx: p.speed_air_x_self || 0.0, vy: p.speed_y_self || 0.0, kb_vx: p.speed_x_attack || 0.0, kb_vy: p.speed_y_attack || 0.0, jumps_left: p.jumps_left || 0}
end

rows =
  Enum.flat_map(paths, fn path ->
    with {:ok, meta} <- Peppi.metadata(path),
         true <- meta.stage in stages || {:skip, "stage #{meta.stage}"},
         {:ok, replay} <- Peppi.parse(path) do
      edge = ExPhil.Situations.geometry(meta.stage).edge
      frames = Enum.sort_by(replay.frames, & &1.frame_number)
      offstage? = fn p -> abs(p.x) > edge or p.y < -5.0 end
      fox_ports = for p <- meta.players, p.character_name == "Fox", do: p.port

      cands =
        for port <- fox_ports, reduce: [] do
          acc ->
            # offstage episodes: first flip + the decision frame before it
            {acc, _} =
              Enum.reduce(frames, {acc, %{prev: nil, done: false}}, fn fr, {acc, ep} ->
                p = fr.players[port]
                cond do
                  p == nil -> {acc, ep}
                  p.on_ground or not offstage?.(p) -> {acc, %{prev: nil, done: false}}
                  ep.done or not decision?.(p) -> {acc, ep}
                  Checkmate.checkmate?(model_state.(p, meta.stage)) ->
                    acc = [%{port: port, frame: fr.frame_number, kind: :flip, model: true, state: model_state.(p, meta.stage)} | acc]
                    acc = if ep.prev, do: [%{ep.prev | kind: :before_flip} | acc], else: acc
                    {acc, %{ep | done: true}}
                  true -> {acc, %{ep | prev: %{port: port, frame: fr.frame_number, kind: :before_flip, model: false, state: model_state.(p, meta.stage)}}}
                end
              end)

            # stock losses: first decision frame after the last hit of that stock
            deaths =
              frames
              |> Enum.chunk_every(2, 1, :discard)
              |> Enum.filter(fn [a, b] -> a.players[port] && b.players[port] && b.players[port].stock < a.players[port].stock end)
              |> Enum.map(fn [_, b] -> b.frame_number end)

            Enum.reduce(deaths, acc, fn d, acc ->
              stock_frames = Enum.filter(frames, &(&1.frame_number < d and &1.players[port] && &1.players[port].stock == Enum.find(frames, fn f -> f.frame_number == d - 1 end).players[port].stock))
              last_hit =
                stock_frames |> Enum.chunk_every(2, 1, :discard) |> Enum.filter(fn [a, b] -> b.players[port].percent > a.players[port].percent end) |> List.last()
              from = if last_hit, do: hd(tl(last_hit)).frame_number, else: nil
              first = from && Enum.find(stock_frames, &(&1.frame_number > from and decision?.(&1.players[port]) and offstage?.(&1.players[port])))
              if first do
                st = model_state.(first.players[port], meta.stage)
                [%{port: port, frame: first.frame_number, kind: :death_first_act, model: Checkmate.checkmate?(st), state: st, death: d} | acc]
              else
                acc
              end
            end)
        end
        |> Enum.uniq_by(&{&1.port, &1.frame})

      IO.puts("#{Path.basename(path)} stage #{meta.stage}: #{length(cands)} candidates")

      if cands == [] do
        []
      else
        {:ok, seed} = Seed.from_replay(path, frame: Enum.max(Enum.map(cands, & &1.frame)), frames: Enum.map(cands, & &1.frame), warm: 0)
        Env.stop(seed.sim)
        saves = Map.new(seed.saves, &{&1.frame, &1})
        {:ok, sim} = Env.start(:nif, stage: seed.stage, players: seed.players, batch_size: batch, seed: 7, ucf_cardinals: 1)

        rows =
          for c <- cands, save = saves[c.frame], save != nil, not save.diverged? do
            {:ok, id} = Env.upload(sim, save.blob)
            results = RecoveryProbe.run(sim, {:id, id}, c.port, save.state.players[c.port], batch: batch)
            made = Enum.filter(results, fn {_, o} -> RecoveryProbe.made_it?(o) end)
            row = Map.merge(c, %{replay: Path.basename(path), made: length(made), tried: length(results), best: made |> List.first() |> then(&(&1 && elem(&1, 0))), sim_checkmate: made == []})
            IO.puts("  p#{c.port} f#{c.frame} #{c.kind}: model #{if c.model, do: "checkmate", else: "recoverable"}, sim #{length(made)}/#{length(results)}#{if c.model != row.sim_checkmate, do: "  <-- DISAGREE", else: ""}")
            row
          end

        Env.stop(sim)
        rows
      end
    else
      {:skip, why} -> IO.puts("#{Path.basename(path)}: skipped (#{why})"); []
      other -> IO.puts("#{Path.basename(path)}: #{inspect(other)}"); []
    end
  end)

agree = Enum.count(rows, &(&1.model == &1.sim_checkmate))
IO.puts("\n#{length(rows)} probed positions, model agrees with the sim on #{agree}")
IO.puts("confusion {model_checkmate, sim_checkmate}: #{inspect(Enum.frequencies_by(rows, &{&1.model, &1.sim_checkmate}))}")
IO.puts("by kind: #{inspect(Enum.frequencies_by(rows, &{&1.kind, &1.model == &1.sim_checkmate}))}")
File.write!(Path.join(out, "sweep.json"), Jason.encode!(Enum.map(rows, &Map.update!(&1, :kind, fn k -> to_string(k) end)), pretty: true))
IO.puts("wrote #{Path.join(out, "sweep.json")}")

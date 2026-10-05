# Recovery-means scorecard (2026-10-05): does the policy reach for the
# expert's recovery tool from the same spot? Scores EVERY offstage episode
# (not only deaths — see sd_review.exs for the death list) by situation
# bucket (height x distance x jumps) against the expert table from
# scripts/expert_recovery_means.exs.
#
#   mix run scripts/recovery_means.exs --policy P.bin [--label L] [--opponent self|idle]
#     [--envs 32] [--frames 3600] [--seeds 1001,1002,1003] [--reference FILE.json] [--out FILE.json]
#   mix run scripts/recovery_means.exs --label L [--bot-port 1] GAME.slp ...     # live / recorded games
#
# Headline: mismatch_rate (share of episodes whose first means the expert
# picks < 10 % of the time from that bucket) read against the expert's
# split-half floor; named defects side_b_low / airdodge_with_jump /
# nothing_died; return rate by height band.
alias ExPhil.Agents.Agent
alias ExPhil.Data.Peppi
alias ExPhil.Eval.RecoveryMeans
alias ExPhil.Sim.{Env, Drill, GA}
alias ExPhil.Training.{Checkpoint, Output}

{opts, files, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, label: :string, reference: :string, opponent: :string, envs: :integer,
             frames: :integer, seeds: :string, stateful_step: :boolean, bot_port: :integer, min_bytes: :integer, out: :string])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

ref = (opts[:reference] || "eval_runs/1002_fidelity/expert_recovery_means_fd.json") |> File.read!() |> Jason.decode!()
edge = GA.stage_edge(32)
label = opts[:label] || (opts[:policy] && Path.basename(Path.dirname(opts[:policy]))) || "recovery_means"
Output.banner("Recovery means: #{label}")

episodes =
  if files != [] do
    # recorded games: the bot's port, FD only (the reference is FD)
    bot_port = opts[:bot_port] || 1
    min_bytes = opts[:min_bytes] || 150_000

    files
    |> Enum.filter(&(File.stat!(&1).size >= min_bytes))
    |> Enum.flat_map(fn path ->
      {:ok, meta} = Peppi.metadata(path)
      own = Enum.find(meta.players, &(&1.port == bot_port))
      opp = Enum.find(meta.players, &(&1.port != bot_port))

      if meta.stage != 32 or own == nil or opp == nil do
        Output.warning("skip #{Path.basename(path)}: stage #{meta.stage}, ports #{inspect(Enum.map(meta.players, & &1.port))}")
        []
      else
        {:ok, replay} = Peppi.parse(path, player_port: own.port)
        eps =
          replay
          |> Peppi.to_training_frames(player_port: own.port, opponent_port: opp.port)
          |> Enum.reject(&(&1.game_state.frame < 0))
          |> Enum.map(&%{own: &1.game_state.players[own.port], opp: &1.game_state.players[opp.port], controller: &1.controller})
          |> RecoveryMeans.episodes(edge)
        Output.puts("  #{Path.basename(path)}: #{length(eps)} offstage episodes")
        eps
      end
    end)
  else
    policy = opts[:policy] || raise("--policy or GAME.slp files required")
    n = opts[:envs] || 32
    frames = opts[:frames] || 3600
    seeds = (opts[:seeds] || "1001,1002,1003") |> String.split(",") |> Enum.map(&String.to_integer/1)
    neutral = Drill.neutral()

    start_agent = fn path, extra ->
      {:ok, export} = Checkpoint.load_policy(path)
      contract = ExPhil.Networks.Policy.ExecutionContract.load(export.config)
      {:ok, agent} =
        Agent.start_link([policy_path: path, deterministic: false, temperature: 1.0, af_convention: :parsed,
          frame_delay: 0, reaction_delay: 0, harness: :sync_runner,
          stateful_step: Keyword.get(extra, :stateful_step) || contract.recurrent_state == :carried_zero] ++ Keyword.delete(extra, :stateful_step))
      {:ok, _} = Agent.warmup(agent)
      :ok = Agent.batch_init(agent, n)
      agent
    end

    Output.config([{"Policy", policy}, {"Opponent", opts[:opponent] || "self"}, {"Envs x frames", "#{n} x #{frames}"}, {"Seeds", inspect(seeds)}])
    actor_opts = [stateful_step: opts[:stateful_step] || false]
    actor = start_agent.(policy, actor_opts)
    opponent =
      case opts[:opponent] do
        "idle" -> :idle
        o when o in [nil, "self"] -> start_agent.(policy, actor_opts)
        path -> start_agent.(path, [])
      end
    players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]

    rollout = fn seed ->
      :ok = Agent.batch_reset_rows(actor, Enum.to_list(0..(n - 1)))
      if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, Enum.to_list(0..(n - 1)))
      {:ok, sim} = Env.start(:nif, stage: "final_destination", players: players, batch_size: n, seed: seed)
      {:ok, _, _} = Env.observe(sim)
      {:ok, gs0} = Env.frames(sim)
      starts = for i <- 0..(n - 1), into: %{}, do: ({:ok, blob} = Env.save(sim, i); {i, blob})

      {games, open, _} =
        Enum.reduce(1..frames, {[], List.duplicate([], n), gs0}, fn _t, {games, open, states} ->
          {:ok, c1s} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
          c2s =
            case opponent do
              :idle -> List.duplicate(neutral, n)
              agent ->
                {:ok, cs} = Agent.batch_get_controllers(agent, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
                cs
            end
          {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || neutral, b || neutral] end))
          open = Enum.zip_with([open, states, c1s], fn [h, s, c] -> [%{own: s.players[1], opp: s.players[2], controller: c || neutral} | h] end)
          finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))

          if finished == [] do
            {games, open, nexts}
          else
            for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
            :ok = Agent.batch_reset_rows(actor, finished)
            if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, finished)
            {:ok, reset_states} = Env.frames(sim)
            done = for i <- finished, do: Enum.at(open, i)
            open = open |> Enum.with_index() |> Enum.map(fn {h, i} -> if i in finished, do: [], else: h end)
            {done ++ games, open, reset_states}
          end
        end)

      eps = (open ++ games) |> Enum.reject(&(&1 == [])) |> Enum.flat_map(&RecoveryMeans.episodes(Enum.reverse(&1), edge))
      Output.puts("  seed #{seed}: #{length(eps)} offstage episodes")
      eps
    end

    Enum.flat_map(seeds, rollout)
  end

score = RecoveryMeans.score(episodes, ref["table"])
expert = ref["self_score"]
floor = ref["split_half"]

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(%{"label" => label, "score" => score, "expert" => expert, "split_half" => floor,
    "episodes" => Enum.map(episodes, &Map.new(&1, fn {k, v} -> {Atom.to_string(k), if(is_atom(v) and not is_boolean(v), do: Atom.to_string(v), else: v)} end))}, pretty: true))
end

fmt = fn v -> if v == nil, do: "-", else: inspect(v) end
Output.puts("RESULT #{label} recovery means: #{score["episodes"]} offstage episodes  mismatch #{fmt.(score["mismatch_rate"])} (expert split-half #{fmt.(floor["mismatch_rate"])})  means_js #{fmt.(score["means_js"])} (floor #{fmt.(floor["means_js"])})  return #{fmt.(score["return_rate"])} (expert #{fmt.(expert["return_rate"])})")
Output.puts("RESULT #{label} named defects model|expert: side_b_low #{fmt.(score["side_b_low"])}|#{fmt.(expert["side_b_low"])}  airdodge_with_jump #{fmt.(score["airdodge_with_jump"])}|#{fmt.(expert["airdodge_with_jump"])}  nothing_died #{fmt.(score["nothing_died"])}|#{fmt.(expert["nothing_died"])}")
Output.puts("RESULT #{label} by height n/return/first-means model | expert: " <>
  Enum.map_join(~w(high ledge low deep), "  ", fn h ->
    m = score["by_height"][h] || %{"n" => 0, "return_rate" => nil, "first" => %{}}
    e = expert["by_height"][h] || %{"n" => 0, "return_rate" => nil, "first" => %{}}
    top = fn b -> b["first"] |> Enum.sort_by(&(-elem(&1, 1))) |> Enum.take(2) |> Enum.map_join(",", fn {k, c} -> "#{k} #{Float.round(c / max(b["n"], 1), 2)}" end) end
    "#{h}: #{m["n"]}/#{fmt.(m["return_rate"])}/#{top.(m)} | #{e["n"]}/#{fmt.(e["return_rate"])}/#{top.(e)}"
  end))
Output.puts("RESULT #{label} first means: " <> Enum.map_join(Enum.sort_by(score["first_means"], &(-elem(&1, 1))), "  ", fn {k, c} -> "#{k} #{c}" end))

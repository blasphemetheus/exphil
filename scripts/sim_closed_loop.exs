# Closed-loop sim rollouts: does the policy play, or freeze and fall off? (2026-10-01)
#
# The offline scoreboard (offline_input_coherence.exs) replays EXPERT states,
# so it cannot see failures caused by the policy living with its own inputs
# (fox-mamba-v2-prevact: healthy offline, 4.5 self-destructs/min live). Here
# the policy drives port 1 in the NIF sim for real, batched over N envs.
#
#   mix run scripts/sim_closed_loop.exs --policy P --label L [--opponent idle|POLICY]
#     [--envs 32] [--frames 3600] [--seed 1001] [--stage final_destination]
#     [--ablate-prev-action] [--stateful-step] [--out FILE.json]
#
# Reports (port 1, pooled over envs):
#   sd_per_min        stocks lost with no hit taken in the previous 90 frames
#   deaths_per_min    all stocks lost
#   offstage episodes and the share that ended back on stage (recovery rate)
#   neutral_share     frames with a fully neutral emitted controller
#   repeat_prev       frames whose emitted controller equals the previous one
#   max_frozen_run    longest run of identical emitted controllers (frames)
#   damage dealt/taken per minute (is it playing at all?)
alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Env, Drill, GA}
alias ExPhil.Training.{Checkpoint, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, label: :string, opponent: :string, envs: :integer, frames: :integer,
             seed: :integer, stage: :string, ablate_prev_action: :boolean, out: :string,
             stateful_step: :boolean, stateful_resync: :integer])
if bad != [], do: raise("invalid options: #{inspect(bad)}")

policy = opts[:policy] || raise("--policy required")
label = opts[:label] || Path.basename(Path.dirname(policy))
n = opts[:envs] || 32
frames = opts[:frames] || 3600
stage = opts[:stage] || "final_destination"
stage_id = %{"final_destination" => 32, "battlefield" => 31, "yoshis_story" => 8, "dreamland" => 28, "pokemon_stadium" => 3, "fountain_of_dreams" => 2}[stage] || 32
edge = GA.stage_edge(stage_id)
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

Output.banner("Closed-loop sim rollout: #{label}")
Output.config([{"Policy", policy}, {"Opponent", opts[:opponent] || "idle"}, {"Envs", n}, {"Frames/env", frames}, {"Stage", stage}])
# --stateful-step forces carried-state inference (O(1)/frame) for a policy whose
# training contract is windowed — e.g. Mamba (2026-10-01); --stateful-resync K
# rebuilds the state from the last window every K frames (single path only).
actor = start_agent.(policy, [ablate_prev_action: opts[:ablate_prev_action] || false, stateful_step: opts[:stateful_step] || false] ++
  if(opts[:stateful_resync], do: [stateful_resync: opts[:stateful_resync]], else: []))
opponent = case opts[:opponent] do
  nil -> :idle
  "idle" -> :idle
  path -> start_agent.(path, [])
end

players = [%{character: "fox", costume: 1}, %{character: "fox", costume: 0}]
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: opts[:seed] || 1001)
{:ok, _, _} = Env.observe(sim)
{:ok, gs0} = Env.frames(sim)
starts = for i <- 0..(n - 1), into: %{} do
  {:ok, blob} = Env.save(sim, i)
  {i, blob}
end

neutral? = fn c ->
  not (c.button_a or c.button_b or c.button_x or c.button_y or c.button_z or c.button_l or c.button_r) and
    abs(c.main_stick.x - 0.5) < 0.1 and abs(c.main_stick.y - 0.5) < 0.1
end
key = fn c -> {c.button_a, c.button_b, c.button_x, c.button_y, c.button_z, c.button_l, c.button_r,
               Float.round(c.main_stick.x * 1.0, 2), Float.round(c.main_stick.y * 1.0, 2),
               Float.round(c.c_stick.x * 1.0, 2), Float.round(c.c_stick.y * 1.0, 2)} end
offstage? = fn p -> not p.on_ground and (abs(p.x) > edge or p.y < -5.0) end

# per-env running state
env0 = %{last_hit: -10_000, prev_key: nil, run: 0, max_run: 0, off: false}
acc0 = %{deaths: 0, sds: 0, neutral: 0, repeat: 0, total: 0, off_eps: 0, off_recovered: 0, off_died: 0,
         dmg_dealt: 0.0, dmg_taken: 0.0, kills: 0, games: 0, max_run: 0}

t0 = System.monotonic_time(:millisecond)
{acc, _envs, _states} =
  Enum.reduce(1..frames, {acc0, List.duplicate(env0, n), gs0}, fn t, {acc, envs, states} ->
    {:ok, c1s} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
    c2s =
      case opponent do
        :idle -> List.duplicate(neutral, n)
        agent ->
          {:ok, cs} = Agent.batch_get_controllers(agent, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
          cs
      end
    {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a || neutral, b || neutral] end))

    {acc, envs} =
      [states, nexts, c1s, envs]
      |> Enum.zip()
      |> Enum.reduce({acc, []}, fn {s, nx, c, e}, {acc, out} ->
        p0 = s.players[1]
        p1 = nx.players[1]
        o0 = s.players[2]
        o1 = nx.players[2]
        c = c || neutral
        k = key.(c)
        hit? = p1.percent > p0.percent or (p1.hitstun_frames_left || 0) > 0
        last_hit = if hit?, do: t, else: e.last_hit
        died? = (p1.stock || 0) < (p0.stock || 0)
        sd? = died? and t - e.last_hit > 90
        run = if k == e.prev_key, do: e.run + 1, else: 1
        was_off = e.off
        now_off = offstage?.(p1) and not died?
        acc = %{acc |
          total: acc.total + 1,
          neutral: acc.neutral + if(neutral?.(c), do: 1, else: 0),
          repeat: acc.repeat + if(k == e.prev_key, do: 1, else: 0),
          deaths: acc.deaths + if(died?, do: 1, else: 0),
          sds: acc.sds + if(sd?, do: 1, else: 0),
          kills: acc.kills + if((o1.stock || 0) < (o0.stock || 0), do: 1, else: 0),
          dmg_dealt: acc.dmg_dealt + max(0.0, (o1.percent - o0.percent) * 1.0),
          dmg_taken: acc.dmg_taken + max(0.0, (p1.percent - p0.percent) * 1.0),
          off_eps: acc.off_eps + if(now_off and not was_off, do: 1, else: 0),
          off_recovered: acc.off_recovered + if(was_off and not now_off and not died?, do: 1, else: 0),
          off_died: acc.off_died + if(was_off and died?, do: 1, else: 0),
          max_run: max(acc.max_run, run)}
        {acc, [%{e | last_hit: last_hit, prev_key: k, run: run, max_run: max(e.max_run, run), off: now_off} | out]}
      end)

    envs = Enum.reverse(envs)
    finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))

    {nexts, envs, acc} =
      if finished == [] do
        {nexts, envs, acc}
      else
        for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
        :ok = Agent.batch_reset_rows(actor, finished)
        if opponent != :idle, do: :ok = Agent.batch_reset_rows(opponent, finished)
        {:ok, reset_states} = Env.frames(sim)
        envs = envs |> Enum.with_index() |> Enum.map(fn {e, i} -> if i in finished, do: %{env0 | max_run: e.max_run}, else: e end)
        {reset_states, envs, %{acc | games: acc.games + length(finished)}}
      end

    if rem(t, 600) == 0 do
      Output.puts("  frame #{t}/#{frames}  deaths #{acc.deaths}  SDs #{acc.sds}  kills #{acc.kills}  (#{round((System.monotonic_time(:millisecond) - t0) / t)} ms/step)")
    end

    {acc, envs, nexts}
  end)

minutes = acc.total / 3600
r = fn x, d -> if d == 0, do: nil, else: Float.round(x / d, 3) end
result = %{
  label: label, policy: policy, opponent: opts[:opponent] || "idle", envs: n, frames_per_env: frames,
  env_minutes: Float.round(minutes, 1),
  sd_per_min: r.(acc.sds, minutes), deaths_per_min: r.(acc.deaths, minutes), kills_per_min: r.(acc.kills, minutes),
  damage_dealt_per_min: r.(acc.dmg_dealt, minutes), damage_taken_per_min: r.(acc.dmg_taken, minutes),
  offstage_episodes_per_min: r.(acc.off_eps, minutes),
  recovery_rate: r.(acc.off_recovered, acc.off_recovered + acc.off_died),
  neutral_share: r.(acc.neutral, acc.total), repeat_prev: r.(acc.repeat, acc.total),
  max_frozen_run: acc.max_run, games_finished: acc.games,
  wall_s: Float.round((System.monotonic_time(:millisecond) - t0) / 1000, 1)
}

if out = opts[:out] do
  File.mkdir_p!(Path.dirname(out))
  File.write!(out, Jason.encode!(result, pretty: true))
end

Output.puts("RESULT #{label} vs #{result.opponent}: #{result.env_minutes} env-min  SD/min #{result.sd_per_min}  deaths/min #{result.deaths_per_min}  " <>
  "kills/min #{result.kills_per_min}  dmg dealt/taken per min #{result.damage_dealt_per_min}/#{result.damage_taken_per_min}")
Output.puts("RESULT #{label} offstage eps/min #{result.offstage_episodes_per_min}  recovery rate #{inspect(result.recovery_rate)}  " <>
  "neutral #{result.neutral_share}  repeat_prev #{result.repeat_prev}  max frozen run #{result.max_frozen_run} f  (#{result.wall_s}s wall)")

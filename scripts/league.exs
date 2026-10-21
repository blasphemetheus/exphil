# League (2026-10-21): every notable checkpoint plays every other in the
# batched parity sim; games are scored by stocks; an Elo is fitted to the
# results. Bradley: "they get an ELO through a focused experiment ... this
# one beats this one, but it loses to this one ... rank the imitation
# learning ones based on skill level."
#
#   mix run scripts/league.exs [--roster FILE.json] [--only a,b,c] [--envs 16]
#     [--frames 5400] [--seed 1] [--out eval_runs/league/DIR] [--stage final_destination]
#
# Each pair plays TWO sessions so both bots sit on each port: `envs` parallel
# games of `frames` frames per session (a finished game restarts in its env).
# A game that ends inside the window is scored by who has stocks left; one
# still open at the window's end is scored by stocks remaining, then by
# percent (a draw if both equal). Sessions are independent seeds.
#
# Fighting strength is NOT imitation fidelity: a PPO policy is expected to
# win with a style no human plays (09-23). Read this table beside the
# agreement eval (scripts/agreement_eval.exs), never instead of it.
#
# Roster entries: %{name, policy, character, line}. A policy the batched sim
# path cannot run (no stateful-step trunk, e.g. the old windowed GRUs with
# queue inputs) is reported as unsupported and left out, not faked.
#
# NO-MIX LAW: run only with no other live beam.

alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Env, Drill}
alias ExPhil.Training.{Checkpoint, Output}

{opts, _, bad} =
  OptionParser.parse(System.argv(),
    strict: [roster: :string, only: :string, envs: :integer, frames: :integer, seed: :integer, out: :string, stage: :string, temperature: :float])

if bad != [], do: raise("invalid options: #{inspect(bad)}")

default_roster = [
  %{name: "convC2", policy: "checkpoints/fox_il_v2_convC2_20260811_222129_best_policy.bin", character: "fox", line: "generalist (pre-corpus, 08-11)"},
  %{name: "V3.1_ep3", policy: "checkpoints/fox_v3_1_20260919_022008_ep3/model_best_policy.bin", character: "fox", line: "generalist V3.1 (crowned 09-20, the frozen prior)"},
  %{name: "mix4", policy: "checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin", character: "fox", line: "V3.1 + sim DAgger mix4 (PPO prior)"},
  %{name: "fox_PPO_i200", policy: "eval_runs/0923_ppo/eval_candidate/candidate_policy.bin", character: "fox", line: "RL on the prior (GRU PPO v1_fixed, iter 200)"},
  %{name: "mamba_v1", policy: "checkpoints/fox_mamba_v1_20260925_ep2/model_best_policy.bin", character: "fox", line: "Fox Mamba v1 (09-27)"},
  %{name: "mamba_v2_prevact", policy: "checkpoints/fox_mamba_v2_prevact_20260930/model_best_policy.bin", character: "fox", line: "Mamba + prev-action channel (10-01, froze live)"},
  %{name: "coh_v3_ep3", policy: "checkpoints/coh_fox_mamba_v3_ep3/model_best_policy.bin", character: "fox", line: "coherence recipe, seed 905 ep3"},
  %{name: "s906_ep2", policy: "checkpoints/coh_fox_mamba_v3s906_ep2/model_best_policy.bin", character: "fox", line: "CANDIDATE (coherence recipe, seed 906 ep2)"},
  %{name: "s906_dag9o", policy: "checkpoints/coh_fox_mamba_v3s906_ep3_dag9o/model_best_policy.bin", character: "fox", line: "candidate + expert-labelled DAgger epoch (failed arm)"},
  %{name: "ms_g19_ep4", policy: "checkpoints/ms_g19_ep4.bin", character: "fox", line: "multishine specialist (production 08-24) — control"},
  %{name: "mewtwo_il_v1", policy: "checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin", character: "mewtwo", line: "Mewtwo imitation prior"},
  %{name: "mewtwo_PPO_pool", policy: "checkpoints/mewtwo_ppo_pool_v2_policy.bin", character: "mewtwo", line: "Mewtwo PPO (pool v2)"}
]

roster =
  case opts[:roster] do
    nil -> default_roster
    f -> f |> File.read!() |> Jason.decode!() |> Enum.map(fn m -> Map.new(m, fn {k, v} -> {String.to_atom(k), v} end) end)
  end

roster =
  case opts[:only] do
    nil -> roster
    s -> names = String.split(s, ","); Enum.filter(roster, &(&1.name in names))
  end

n = opts[:envs] || 16
frames = opts[:frames] || 5400
seed0 = opts[:seed] || 1
stage = opts[:stage] || "final_destination"
temperature = opts[:temperature] || 1.0
out = opts[:out] || "eval_runs/league/#{Date.utc_today() |> Date.to_iso8601(:basic)}"
File.mkdir_p!(out)
neutral = Drill.neutral()

Output.banner("League: #{length(roster)} bots, #{n} envs x #{frames} f per session, 2 sessions per pair")

# ---- start every agent once (batched sim contract) ------------------------------
start_agent = fn bot ->
  try do
    unless File.exists?(bot.policy), do: raise("missing #{bot.policy}")
    {:ok, export} = Checkpoint.load_policy(bot.policy)
    contract = ExPhil.Networks.Policy.ExecutionContract.load(export.config)

    {:ok, agent} =
      Agent.start_link(
        policy_path: bot.policy, deterministic: false, temperature: temperature, af_convention: :parsed,
        frame_delay: 0, reaction_delay: 0, harness: :sync_runner,
        stateful_step: contract.recurrent_state == :carried_zero
      )

    {:ok, _} = Agent.warmup(agent)
    :ok = Agent.batch_init(agent, n)
    {:ok, agent}
  rescue
    e -> {:error, Exception.message(e) |> String.slice(0, 300)}
  catch
    :exit, r -> {:error, inspect(r) |> String.slice(0, 300)}
  end
end

{bots, unsupported} =
  Enum.reduce(roster, {[], []}, fn bot, {ok, bad} ->
    case start_agent.(bot) do
      {:ok, agent} ->
        Output.puts("  ready: #{bot.name}")
        {ok ++ [Map.put(bot, :agent, agent)], bad}

      {:error, why} ->
        Output.warning("  unsupported in the batched sim: #{bot.name} — #{why}")
        {ok, bad ++ [Map.put(bot, :why, why)]}
    end
  end)

if length(bots) < 2, do: raise("fewer than two playable bots")

# ---- one session: a (port 1) vs b (port 2) ----------------------------------------
session = fn a, b, seed ->
  rows = Enum.to_list(0..(n - 1))
  :ok = Agent.batch_reset_rows(a.agent, rows)
  :ok = Agent.batch_reset_rows(b.agent, rows)
  players = [%{character: a.character, costume: 1}, %{character: b.character, costume: 0}]
  {:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: n, seed: seed)
  {:ok, _, _} = Env.observe(sim)
  {:ok, gs0} = Env.frames(sim)
  starts = for i <- rows, into: %{}, do: ({:ok, blob} = Env.save(sim, i); {i, blob})
  t0 = System.monotonic_time(:millisecond)

  {games, states, started} =
    Enum.reduce(1..frames, {[], gs0, List.duplicate(0, n)}, fn t, {games, states, started} ->
      {:ok, c1s} = Agent.batch_get_controllers(a.agent, Enum.map(states, &%{&1 | own_port: 1}), player_port: 1)
      {:ok, c2s} = Agent.batch_get_controllers(b.agent, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
      {:ok, nexts, terms} = Env.step(sim, Enum.zip_with(c1s, c2s, fn x, y -> [x || neutral, y || neutral] end))
      finished = terms |> Enum.with_index() |> Enum.filter(fn {term, _} -> (term["done"] || 0) == 1 end) |> Enum.map(&elem(&1, 1))

      if finished == [] do
        {games, nexts, started}
      else
        done =
          Enum.map(finished, fn i ->
            s = Enum.at(nexts, i)
            s1 = s.players[1].stock || 0
            s2 = s.players[2].stock || 0
            %{env: i, frames: t - Enum.at(started, i), finished: true, p1_stocks: s1, p2_stocks: s2, p1_pct: s.players[1].percent, p2_pct: s.players[2].percent}
          end)

        for i <- finished, do: {:ok, _} = Env.restore(sim, i, starts[i])
        :ok = Agent.batch_reset_rows(a.agent, finished)
        :ok = Agent.batch_reset_rows(b.agent, finished)
        {:ok, reset} = Env.frames(sim)
        started = started |> Enum.with_index() |> Enum.map(fn {v, i} -> if i in finished, do: t, else: v end)
        {done ++ games, reset, started}
      end
    end)

  # games still open at the window's end: scored by stocks, then percent
  open =
    states
    |> Enum.with_index()
    |> Enum.map(fn {s, i} ->
      %{env: i, frames: frames - Enum.at(started, i), finished: false, p1_stocks: s.players[1].stock || 0, p2_stocks: s.players[2].stock || 0, p1_pct: s.players[1].percent, p2_pct: s.players[2].percent}
    end)

  Env.stop(sim)
  ms = System.monotonic_time(:millisecond) - t0
  {Enum.reverse(games) ++ open, ms}
end

# outcome for port-1 side: 1 win, 0 loss, 0.5 draw
score_p1 = fn g ->
  cond do
    g.p1_stocks > g.p2_stocks -> 1.0
    g.p1_stocks < g.p2_stocks -> 0.0
    (g.p1_pct || 0) < (g.p2_pct || 0) -> 1.0
    (g.p1_pct || 0) > (g.p2_pct || 0) -> 0.0
    true -> 0.5
  end
end

pairs = for {a, i} <- Enum.with_index(bots), {b, j} <- Enum.with_index(bots), i < j, do: {a, b}
Output.puts("#{length(pairs)} pairs, #{2 * length(pairs)} sessions")

{results, _} =
  Enum.map_reduce(Enum.with_index(pairs), seed0, fn {{a, b}, k}, seed ->
    {g1, ms1} = session.(a, b, seed)
    {g2, ms2} = session.(b, a, seed + 1)
    # normalise to "a's score"
    ga = Enum.map(g1, fn g -> Map.merge(g, %{a_score: score_p1.(g), a_port: 1}) end)
    gb = Enum.map(g2, fn g -> Map.merge(g, %{a_score: 1.0 - score_p1.(g), a_port: 2}) end)
    games = ga ++ gb
    wins = Enum.count(games, &(&1.a_score == 1.0))
    losses = Enum.count(games, &(&1.a_score == 0.0))
    draws = length(games) - wins - losses
    fin = Enum.count(games, & &1.finished)
    fps = round(2 * n * frames / max(ms1 + ms2, 1) * 1000)
    Output.puts("RESULT pair #{k + 1}/#{length(pairs)} #{a.name} vs #{b.name}: #{wins}-#{losses}-#{draws} (#{fin}/#{length(games)} finished)  #{div(ms1 + ms2, 1000)} s  #{fps} game-frames/s")
    {%{a: a.name, b: b.name, wins: wins, losses: losses, draws: draws, games: games}, seed + 2}
  end)

# ---- Elo: Bradley–Terry by iterative logistic fit on every game, mean 1500 ----
names = Enum.map(bots, & &1.name)
game_list = Enum.flat_map(results, fn r -> Enum.map(r.games, &{r.a, r.b, &1.a_score}) end)

rating =
  Enum.reduce(1..400, Map.new(names, &{&1, 0.0}), fn _, r ->
    grad =
      Enum.reduce(game_list, Map.new(names, &{&1, 0.0}), fn {a, b, s}, g ->
        p = 1 / (1 + :math.exp(r[b] - r[a]))
        g |> Map.update!(a, &(&1 + (s - p))) |> Map.update!(b, &(&1 - (s - p)))
      end)

    r = Map.new(r, fn {k, v} -> {k, v + 0.02 * grad[k]} end)
    m = Enum.sum(Map.values(r)) / length(names)
    Map.new(r, fn {k, v} -> {k, v - m} end)
  end)

elo = Map.new(rating, fn {k, v} -> {k, round(1500 + 400 / :math.log(10) * v)} end)

matrix =
  Map.new(names, fn a ->
    {a,
     Map.new(names, fn b ->
       if a == b do
         {b, nil}
       else
         gs = Enum.filter(game_list, fn {x, y, _} -> {x, y} == {a, b} or {x, y} == {b, a} end)
         {b, if(gs == [], do: nil, else: Float.round(Enum.sum(Enum.map(gs, fn {x, _, s} -> if x == a, do: s, else: 1.0 - s end)) / length(gs), 3))}
       end
     end)}
  end)

table = bots |> Enum.map(fn b -> %{name: b.name, line: b.line, character: b.character, elo: elo[b.name],
  games: Enum.count(game_list, fn {x, y, _} -> x == b.name or y == b.name end),
  win_rate: (gs = Enum.filter(game_list, fn {x, y, _} -> x == b.name or y == b.name end);
             if gs == [], do: nil, else: Float.round(Enum.sum(Enum.map(gs, fn {x, _, s} -> if x == b.name, do: s, else: 1.0 - s end)) / length(gs), 3))} end)
  |> Enum.sort_by(&(-&1.elo))

report = %{created: DateTime.utc_now() |> DateTime.to_iso8601(), envs: n, frames: frames, stage: stage, temperature: temperature, seed: seed0,
  roster: Enum.map(bots, &Map.drop(&1, [:agent])), unsupported: unsupported, standings: table, win_matrix: matrix, pairs: results,
  note: "fighting strength in the parity sim (FD, batched path, T=#{temperature}); not imitation fidelity — see the agreement eval"}
File.write!(Path.join(out, "league.json"), Jason.encode!(report, pretty: true))

Output.puts("RESULT standings (Elo, Bradley–Terry over #{length(game_list)} games):")
Enum.each(table, fn r -> Output.puts("  #{String.pad_trailing(r.name, 18)} #{r.elo}  win rate #{r.win_rate}  games #{r.games}  — #{r.line}") end)
Enum.each(unsupported, fn u -> Output.puts("  (unsupported) #{u.name}: #{u.why}") end)
Output.puts("RESULT league -> #{out}/league.json")

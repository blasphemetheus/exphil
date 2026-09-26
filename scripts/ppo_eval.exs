# Batched, port-balanced evaluation of a PPO head against its frozen prior.
# Runs after training; no learning or production-policy promotion.
alias ExPhil.Agents.Agent
alias ExPhil.Sim.{Env, PPO, Critic}
alias ExPhil.Interp.StyleFingerprint
alias ExPhil.Training.Output

{opts, _, invalid} = OptionParser.parse(System.argv(), strict: [policy: :string, head: :string,
  out: :string, games: :integer, envs: :integer, frames: :integer, seed: :integer,
  fingerprint_frames: :integer, character: :string, stage: :string])
if invalid != [], do: raise("Invalid options: #{inspect(invalid)}")
policy = opts[:policy] || raise("--policy required")
out = opts[:out] || raise("--out required")
games = opts[:games] || 200
envs = opts[:envs] || 32
frames = opts[:frames] || 28800
fp_frames = opts[:fingerprint_frames] || 1800
seed = opts[:seed] || 920230
if games < 2 or rem(games, 2) != 0 or envs < 1 or frames < 1, do: raise("Need positive envs/frames and an even games count >=2")
File.mkdir_p!(out)
if File.exists?(Path.join(out, "games.jsonl")), do: raise("Evaluation output already exists: #{out}")
Output.banner("PPO evaluation — frozen opponent, balanced ports")
character = opts[:character] || "fox"
stage = opts[:stage] || "final_destination"
_ = ExPhil.Bridge.SimBatch.character_id(character)
Output.config([{"Policy", policy}, {"Head", opts[:head] || "prior control"}, {"Games", games},
  {"Envs", envs}, {"Frame cap", frames}, {"Seed", seed}, {"Character (ditto)", character},
  {"Stage", stage}, {"Out", out}])
agent_opts = [policy_path: policy, deterministic: false, temperature: 1.0,
  af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]
{:ok, actor} = Agent.start_link(agent_opts)
{:ok, opponent} = Agent.start_link(agent_opts)
for a <- [actor, opponent], do: {:ok, _} = Agent.warmup(a)
if opts[:head] do
  head = opts[:head] |> File.read!() |> :erlang.binary_to_term()
  if head.policy != policy, do: raise("Head provenance differs from --policy")
  :ok = Agent.put_head_params(actor, PPO.to_backend(head.ar))
  {:ok, export} = ExPhil.Training.Checkpoint.load_policy(policy)
  merged = case export.params do
    %Axon.ModelState{} = ms -> %{ms | data: Map.merge(ms.data, head.ar)}
    map -> Map.merge(map, head.ar)
  end
  export_path = Path.join(out, "candidate_policy.bin")
  File.write!(export_path, :erlang.term_to_binary(%{export | params: PPO.to_backend(merged, Nx.BinaryBackend)}))
  {:ok, _} = ExPhil.Training.Checkpoint.load_policy(export_path)
end
players = [%{character: character, costume: 1}, %{character: character, costume: 0}]
all_rows =
  0..(div(div(games, 2) + envs - 1, envs) - 1)
  |> Enum.flat_map(fn pair ->
    n = min(envs, div(games, 2) - pair * envs)
    for port <- [1, 2], reduce: [] do
      accumulated ->
        {:ok, sim} = Env.start(:nif, stage: stage, players: players,
          batch_size: n, seed: seed + pair)
        for a <- [actor, opponent], do: :ok = Agent.batch_init(a, n)
        {:ok, initial} = Env.frames(sim)
        records = for _ <- 1..n, do: %{done: false, reward: 0.0, frames: 0, final: nil, states: [], ca: [], cb: []}
        t0 = System.monotonic_time(:millisecond)
        {records, _} = Enum.reduce_while(1..frames, {records, initial}, fn t, {records, states} ->
          other = 3 - port
          {:ok, ca} = Agent.batch_get_controllers(actor, Enum.map(states, &%{&1 | own_port: port}), player_port: port)
          {:ok, cb} = Agent.batch_get_controllers(opponent, Enum.map(states, &%{&1 | own_port: other}), player_port: other)
          controls = Enum.zip_with(ca, cb, fn a, b -> if port == 1, do: [a, b], else: [b, a] end)
          {:ok, nexts, terminals} = Env.step(sim, controls)
          records = Enum.zip([records, states, nexts, terminals, ca, cb]) |> Enum.map(fn {r, s, next, term, a, b} ->
            if r.done do
              r
            else
              keep = s.frame >= 0 and s.frame < fp_frames
              %{r | done: term["done"] == 1, reward: r.reward + Critic.reward(s, next) * (if port == 1, do: 1, else: -1),
                frames: t, final: next,
                states: if(keep, do: [s | r.states], else: r.states),
                ca: if(keep, do: [a | r.ca], else: r.ca), cb: if(keep, do: [b | r.cb], else: r.cb)}
            end
          end)
          if rem(t, 600) == 0, do: Output.puts("pair #{pair + 1} candidate port #{port}: frame #{t}/#{frames}, finished #{Enum.count(records, & &1.done)}/#{n}")
          if Enum.all?(records, & &1.done), do: {:halt, {records, nexts}}, else: {:cont, {records, nexts}}
        end)
        rows = records |> Enum.with_index() |> Enum.map(fn {r, i} ->
          a = r.final.players[port]
          b = r.final.players[3 - port]
          result = cond do
            not r.done -> "timeout"
            a.stock > 0 and b.stock == 0 -> "win"
            b.stock > 0 and a.stock == 0 -> "loss"
            true -> "draw"
          end
          game = pair * envs * 2 + (port - 1) * n + i + 1
          row = %{game: game, pair: pair, env: i, seed: seed + pair, candidate_port: port,
            result: result, terminal: r.done, frames: r.frames, reward: r.reward,
            candidate_stocks: a.stock, opponent_stocks: b.stock,
            candidate_percent: a.percent, opponent_percent: b.percent}
          File.write!(Path.join(out, "games.jsonl"), Jason.encode!(row) <> "\n", [:append])
          if length(r.states) >= 600 do
            states = Enum.reverse(r.states)
            for {role, p, cs} <- [{"candidate", port, r.ca}, {"prior", 3 - port, r.cb}] do
              fp = StyleFingerprint.fingerprint(states, p, Enum.reverse(cs))
              features = Map.new(fp, fn {k, v} -> {k, if(is_tuple(v), do: Tuple.to_list(v), else: v)} end)
              fr = %{game: game, role: role, port: p, character: String.capitalize(character), stage: ExPhil.Bridge.SimBatch.stage_id(stage),
                path: "#{out}/game#{game}", frames: length(states), features: features}
              File.write!(Path.join(out, "fingerprints.jsonl"), Jason.encode!(fr) <> "\n", [:append])
            end
          end
          row
        end)
        Output.puts("pair #{pair + 1} port #{port}: #{inspect(Enum.frequencies_by(rows, & &1.result))}, #{System.monotonic_time(:millisecond) - t0} ms")
        Env.stop(sim)
        accumulated ++ rows
    end
  end)
counts = Enum.frequencies_by(all_rows, & &1.result)
wins = Map.get(counts, "win", 0)
losses = Map.get(counts, "loss", 0)
summary = %{games: length(all_rows), outcomes: counts, win_rate_all_games: wins / games,
  win_rate_decisive: if(wins + losses > 0, do: wins / (wins + losses), else: nil),
  mean_reward: Enum.sum(Enum.map(all_rows, & &1.reward)) / games,
  by_candidate_port: Map.new([1, 2], fn p -> {p, all_rows |> Enum.filter(&(&1.candidate_port == p)) |> Enum.frequencies_by(& &1.result)} end),
  policy: policy, head: opts[:head], seed: seed, max_frames: frames,
  r3_gate: "NOT ASSESSED: requires human-range identity and defense checks; timeouts are not wins"}
File.write!(Path.join(out, "summary.json"), Jason.encode!(summary, pretty: true))
Output.success("Evaluation complete: #{inspect(counts)}, win rate over all games #{Float.round(100 * wins / games, 1)}%")

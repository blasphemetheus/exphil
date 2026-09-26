# RL_ON_PRIOR R3 gate — does the PPO head actually beat the frozen prior?
#
#   devenv shell -- mix run scripts/ppo_r3_eval.exs \
#     --policy checkpoints/fox_v3_1_step8_mix4/model_best_policy.bin \
#     --head eval_runs/0923_ppo/v1_fixed/head_iter200.bin \
#     --games 200 --envs 100 --out eval_runs/0923_ppo/eval_iter200
#
# The gate (RL_ON_PRIOR.md): win rate > 60 % over >= 200 COMPLETE games against
# the frozen prior, with the style fingerprint still inside the human range.
# Training reward against a frozen opponent is not that number — it is measured
# on the distribution the learner itself induces — so this plays real games to
# stock-out and counts wins.
#
# Two things that would otherwise fake the result:
#   * PORT BIAS. The prior was trained with a port-conditioned embedding, so
#     half the games run the challenger on port 1 and half on port 2.
#   * TIES. A game that hits the frame cap is scored by stocks, then by percent,
#     and anything still level is reported as a draw, never as a win.
#
# Writes summary.json (per-game rows + Wilson CI), fingerprint.jsonl (same shape
# as scripts/sim_prior_play.exs, so sim_r1_compare.exs reads it unchanged).

alias ExPhil.Agents.Agent
alias ExPhil.Interp.StyleFingerprint
alias ExPhil.Sim.{Drill, Env}
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [policy: :string, head: :string, games: :integer, envs: :integer, cap: :integer,
             out: :string, stage: :string, seed: :integer, temperature: :float, fingerprint_games: :integer,
             character: :string, opponent_head: :string]
  )

policy = opts[:policy] || raise("--policy required")
head_path = opts[:head] || raise("--head required (head_iter*.bin from the PPO run)")
games = opts[:games] || 200
envs = opts[:envs] || 100
cap = opts[:cap] || 14_400
stage = opts[:stage] || "final_destination"
seed0 = opts[:seed] || 91
out = opts[:out] || raise("--out required")
fp_games = opts[:fingerprint_games] || 16
File.mkdir_p!(out)

Output.banner("R3 gate — PPO head vs the frozen prior")

head = head_path |> File.read!() |> :erlang.binary_to_term()
ar = head[:ar] || raise("no :ar layers in #{head_path}")

Output.config([
  {"Prior", policy},
  {"Head", "#{Path.basename(head_path)} (iter #{head[:iter]}, #{map_size(ar)} ar_* layers)"},
  {"Games", games}, {"Envs (parallel games)", envs}, {"Frame cap", cap},
  {"Stage", stage}, {"Out", out}
])

# default to the character the head was trained on, so an eval cannot silently
# grade a Mewtwo head in a Fox ditto
character = opts[:character] || head[:character] || "fox"
_ = ExPhil.Bridge.SimBatch.character_id(character)
unless head[:policy] && Path.expand(head.policy) == Path.expand(policy),
  do: raise("Challenger head belongs to a different prior")
if head[:character] && head.character != character, do: raise("Challenger character mismatch")
opponent_head = if opts[:opponent_head] do
  saved = opts[:opponent_head] |> File.read!() |> :erlang.binary_to_term()
  unless saved[:policy] && Path.expand(saved.policy) == Path.expand(policy) && saved[:character] == character,
    do: raise("Opponent head belongs to a different prior or character")
  saved
end
players = [%{character: character, costume: 1}, %{character: character, costume: 0}]
agent_opts = [policy_path: policy, deterministic: false, temperature: opts[:temperature] || 1.0,
              af_convention: :parsed, frame_delay: 0, harness: :sync_runner, reaction_delay: 0, stateful_step: true]

Output.puts("⏳ loading the challenger and the frozen prior…")
{:ok, challenger} = Agent.start_link(agent_opts)
{:ok, prior} = Agent.start_link(agent_opts)
{:ok, _} = Agent.warmup(challenger)
{:ok, _} = Agent.warmup(prior)
:ok = Agent.put_head_params(challenger, ar)
if opponent_head, do: :ok = Agent.put_head_params(prior, opponent_head.ar)
Output.success("challenger = prior trunk + trained head; opponent = #{opts[:opponent_head] || "untouched prior"}")

neutral = Drill.neutral()
{:ok, sim} = Env.start(:nif, stage: stage, players: players, batch_size: envs, seed: seed0)

ensure_batch = fn agent ->
  case Agent.batch_reset_rows(agent, Enum.to_list(0..(envs - 1))) do
    :ok -> :ok
    {:error, :batch_not_initialized} -> :ok = Agent.batch_init(agent, envs)
  end
end

# One batch = `envs` simultaneous games, challenger fixed to `chal_port`.
play_batch = fn batch_idx, chal_port, keep_states? ->
  {:ok, _} = Env.reinit(sim, %{stage: stage, players: players, length: 256, seed: seed0 * 1000 + batch_idx})
  {:ok, _, _} = Env.observe(sim)
  Enum.each([challenger, prior], ensure_batch)
  {:ok, gs0s} = Env.frames(sim)
  opp_port = if chal_port == 1, do: 2, else: 1
  t0 = System.monotonic_time(:millisecond)

  # results.(i) stays nil until env i's game ends; kept.(i) accumulates frames
  # for the fingerprint on the first batch only
  init = {gs0s, Map.new(0..(envs - 1), &{&1, nil}), %{}, 0}

  {final_states, results, kept, frames_done} =
    Enum.reduce_while(1..cap, init, fn t, {states, results, kept, _} ->
      {:ok, c_chal} = Agent.batch_get_controllers(challenger, Enum.map(states, &%{&1 | own_port: chal_port}), player_port: chal_port)
      {:ok, c_opp} = Agent.batch_get_controllers(prior, Enum.map(states, &%{&1 | own_port: opp_port}), player_port: opp_port)

      pairs =
        Enum.zip(c_chal, c_opp)
        |> Enum.map(fn {a, b} ->
          if chal_port == 1, do: [a || neutral, b || neutral], else: [b || neutral, a || neutral]
        end)

      case Env.step(sim, pairs) do
        {:ok, nexts, terms} ->
          kept =
            if keep_states? do
              # only the games we actually fingerprint: keeping all `envs` for
              # 3600 frames is hundreds of thousands of live structs
              slice = {Enum.take(states, fp_games), Enum.take(c_chal, fp_games)}
              Map.update(kept, :frames, [slice], fn acc ->
                if length(acc) < 3600, do: [slice | acc], else: acc
              end)
            else
              kept
            end

          results =
            Enum.zip([0..(envs - 1), states, terms])
            |> Enum.reduce(results, fn {i, prev, term}, acc ->
              cond do
                acc[i] != nil -> acc
                (term["done"] || 0) == 1 -> Map.put(acc, i, {prev, t, :stockout})
                t == cap -> Map.put(acc, i, {prev, t, :cap})
                true -> acc
              end
            end)

          if Enum.all?(0..(envs - 1), &(results[&1] != nil)),
            do: {:halt, {nexts, results, kept, t}},
            else: {:cont, {nexts, results, kept, t}}

        {:error, reason} ->
          raise "sim step failed: #{inspect(reason)}"
      end
    end)

  ms = System.monotonic_time(:millisecond) - t0

  rows =
    Enum.map(0..(envs - 1), fn i ->
      {state, frame, how} = results[i] || {Enum.at(final_states, i), cap, :cap}
      chal = state.players[chal_port]
      opp = state.players[opp_port]

      outcome =
        cond do
          (chal.stock || 0) > (opp.stock || 0) -> :win
          (chal.stock || 0) < (opp.stock || 0) -> :loss
          chal.percent < opp.percent -> :win
          chal.percent > opp.percent -> :loss
          true -> :draw
        end

      %{game: batch_idx * envs + i, chal_port: chal_port, outcome: outcome, ended: how, frames: frame,
        chal_stocks: chal.stock, opp_stocks: opp.stock,
        chal_percent: Float.round(chal.percent * 1.0, 1), opp_percent: Float.round(opp.percent * 1.0, 1)}
    end)

  w = Enum.count(rows, &(&1.outcome == :win))
  l = Enum.count(rows, &(&1.outcome == :loss))
  d = Enum.count(rows, &(&1.outcome == :draw))
  finished = Enum.count(rows, &(&1.ended == :stockout))

  Output.puts(
    "batch #{batch_idx} (challenger port #{chal_port}): #{w}W #{l}L #{d}D of #{envs}; " <>
      "#{finished} ended by stock-out, #{envs - finished} hit the cap; #{frames_done} frames in #{Float.round(ms / 1000, 1)} s"
  )

  {rows, kept}
end

# batches alternate the challenger's port so the win rate is not a port artifact
n_batches = max(1, ceil(games / envs))
t0 = System.monotonic_time(:millisecond)

{all_rows, fp_kept} =
  Enum.reduce(0..(n_batches - 1), {[], nil}, fn b, {acc, fp} ->
    chal_port = if rem(b, 2) == 0, do: 1, else: 2
    {rows, kept} = play_batch.(b, chal_port, b == 0)
    {acc ++ rows, fp || kept[:frames]}
  end)

rows = Enum.take(all_rows, games)
wins = Enum.count(rows, &(&1.outcome == :win))
losses = Enum.count(rows, &(&1.outcome == :loss))
draws = Enum.count(rows, &(&1.outcome == :draw))
n = length(rows)
rate = wins / max(1, n)

# Wilson 95 % interval — a point estimate alone cannot clear a 60 % gate
z = 1.96
denom = 1 + z * z / n
centre = (rate + z * z / (2 * n)) / denom
half = z * :math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
{lo, hi} = {centre - half, centre + half}

by_port =
  Enum.group_by(rows, & &1.chal_port)
  |> Enum.map(fn {p, rs} -> {p, Enum.count(rs, &(&1.outcome == :win)) / max(1, length(rs))} end)
  |> Enum.sort()

# fingerprint the challenger from the first batch's frames
fp_rows =
  case fp_kept do
    nil ->
      Output.warning("no frames kept — skipping the fingerprint")
      []

    frames ->
      {states_seq, ctrl_seq} = frames |> Enum.reverse() |> Enum.unzip()

      Enum.map(0..(min(fp_games, envs) - 1), fn i ->
        st = Enum.map(states_seq, &Enum.at(&1, i)) |> Enum.reject(&(&1.frame < 0))
        ct = Enum.map(ctrl_seq, &Enum.at(&1, i))

        if length(st) >= 600 do
          fp = StyleFingerprint.fingerprint(st, 1, Enum.take(ct, length(st)))
          %{path: "#{out}/game#{i}", tag: "ppo_iter#{head[:iter]}", port: 1, character: String.capitalize(character), ditto: true,
            candidates: [], started_at: DateTime.utc_now() |> DateTime.to_iso8601(), costume: 1, fingerprint: fp}
        end
      end)
      |> Enum.reject(&is_nil/1)
  end

if fp_rows != [] do
  File.write!(Path.join(out, "fingerprint.jsonl"), Enum.map_join(fp_rows, "\n", &Jason.encode!/1) <> "\n")
end

ms = System.monotonic_time(:millisecond) - t0
verdict = if lo > 0.6, do: "R3 WIN-RATE PASSED (95 % CI above 60 %)", else: "R3 WIN-RATE NOT PASSED"

File.write!(Path.join(out, "summary.json"), Jason.encode!(%{
  policy: policy, head: head_path, iter: head[:iter], games: n, envs: envs, cap: cap,
  opponent_head: opts[:opponent_head], character: character, stage: stage, seed: seed0,
  wins: wins, losses: losses, draws: draws, win_rate: rate, wilson95: [lo, hi],
  by_port: Map.new(by_port), stockouts: Enum.count(rows, &(&1.ended == :stockout)),
  verdict: verdict, ms: ms, rows: rows
}, pretty: true))

Output.puts("")
Output.puts("#{n} games: #{wins}W #{losses}L #{draws}D → win rate #{Float.round(rate * 100, 1)} % (95 % CI #{Float.round(lo * 100, 1)}–#{Float.round(hi * 100, 1)} %)")
Output.puts("by challenger port: " <> Enum.map_join(by_port, ", ", fn {p, r} -> "p#{p} #{Float.round(r * 100, 1)} %" end))
Output.puts("ended by stock-out: #{Enum.count(rows, &(&1.ended == :stockout))}/#{n}; fingerprint rows: #{length(fp_rows)}")
Output.puts(verdict)
Output.success("#{Float.round(ms / 1000, 1)} s → #{out}/summary.json")
Env.stop(sim)

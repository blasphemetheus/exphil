defmodule ExPhil.Sim.GA do
  @moduledoc """
  Genetic algorithm for the longest combo from one start state
  (SIM_USES_CHECKLIST "GA longest combo", v0, 2026-09-22).

  * **Genome**: a list of `{macro, hold}` tokens from `ExPhil.Sim.Search.macros/0`
    whose holds sum to `horizon` frames — the same macro-action programs the
    search-as-teacher samples, so mutations stay meaningful (change a token,
    change a hold, insert, delete) and crossover splices two programs at a
    shared frame boundary.
  * **Fitness** (subject = P1 attacker, P2 defender): damage dealt inside the
    best `ExPhil.Eval.AerialChain` chain + 20 per MOVE that lands in it
    (`moves_landed/3`; a multi-hit drill is one move) + 1000 per stock taken,
    minus 500 if P1 dies or leaves the stage, minus the search style penalty
    (`Search.style_counts/1`). v0 weighted moves 100:1 over damage and
    evolved jab pressure against an idle target (run 3, 2026-09-22).
  * **Evaluation**: the whole population is one sim batch — every env restores
    the same start (`Env.upload` once, `Env.restore` per env), then `horizon`
    batched steps. Deterministic given `:seed`.

  `run/3` returns `%{generations: [...], best: %{...}}`; the `:on_generation`
  hook receives each generation's summary (best/mean fitness, elite genome,
  elite rollout states) so a script can write traces for the viewer.
  """

  alias ExPhil.Agents.Agent
  alias ExPhil.Eval.{AerialChain, ScenarioScan}
  alias ExPhil.Sim.{Drill, Env, Search}

  @neutral Drill.neutral()
  @press_macros [:jump, :jump_r, :jump_l, :fair_r, :fair_l, :nair, :uair, :dair, :shine, :lcancel, :grab, :utilt, :dtilt]

  @doc "The macro vocabulary as a map name => controller."
  def vocab, do: Map.new(Search.macros())

  # ---------------------------------------------------------------- genomes

  @doc "A random genome: tokens `{name, hold}` summing to `horizon` frames."
  def random_genome(horizon, max_hold \\ 12) do
    Stream.unfold(0, fn
      t when t >= horizon -> nil
      t ->
        {name, _} = Enum.random(Search.macros())
        hold = min(hold_for(name, max_hold), horizon - t)
        {{name, hold}, t + hold}
    end)
    |> Enum.to_list()
  end

  defp hold_for(name, max_hold), do: if(name in @press_macros, do: :rand.uniform(2), else: :rand.uniform(max_hold))

  @doc "Expand a genome to `horizon` per-frame controllers."
  def to_program(genome, vocab \\ vocab()) do
    Enum.flat_map(genome, fn {name, hold} -> List.duplicate(Map.fetch!(vocab, name), hold) end)
  end

  @doc "Total frames of a genome."
  def frames(genome), do: Enum.sum(Enum.map(genome, &elem(&1, 1)))

  @doc """
  Mutate: each token independently with probability `rate` gets one of
  token-flip / hold-change; plus one insert or delete with probability `rate`.
  The result is re-fitted to exactly `horizon` frames.
  """
  def mutate(genome, horizon, rate \\ 0.15, max_hold \\ 12) do
    names = Enum.map(Search.macros(), &elem(&1, 0))

    g =
      Enum.map(genome, fn {name, hold} ->
        cond do
          :rand.uniform() < rate / 2 -> {Enum.random(names), hold}
          :rand.uniform() < rate / 2 -> {name, max(1, hold + Enum.random([-3, -2, -1, 1, 2, 3]))}
          true -> {name, hold}
        end
      end)

    g =
      cond do
        :rand.uniform() < rate ->
          i = :rand.uniform(length(g) + 1) - 1
          name = Enum.random(names)
          List.insert_at(g, i, {name, hold_for(name, max_hold)})

        :rand.uniform() < rate and length(g) > 1 ->
          List.delete_at(g, :rand.uniform(length(g)) - 1)

        true ->
          g
      end

    fit(g, horizon, max_hold)
  end

  @doc "One-point crossover at a frame boundary shared by both parents (or a random token boundary of `a`)."
  def crossover(a, b, horizon, max_hold \\ 12) do
    cut = Enum.random(1..max(1, horizon - 1))
    fit(take_frames(a, cut) ++ drop_frames(b, cut), horizon, max_hold)
  end

  defp take_frames(g, n) do
    {out, _} =
      Enum.reduce_while(g, {[], 0}, fn {name, hold}, {acc, t} ->
        cond do
          t >= n -> {:halt, {acc, t}}
          t + hold <= n -> {:cont, {[{name, hold} | acc], t + hold}}
          true -> {:halt, {[{name, n - t} | acc], n}}
        end
      end)

    Enum.reverse(out)
  end

  defp drop_frames(g, n) do
    {out, _} =
      Enum.reduce(g, {[], 0}, fn {name, hold}, {acc, t} ->
        cond do
          t + hold <= n -> {acc, t + hold}
          t >= n -> {[{name, hold} | acc], t + hold}
          true -> {[{name, t + hold - n} | acc], t + hold}
        end
      end)

    Enum.reverse(out)
  end

  # pad with neutral / trim so the genome is exactly `horizon` frames, holds >= 1
  defp fit(g, horizon, _max_hold) do
    g = Enum.filter(g, fn {_, h} -> h >= 1 end)
    total = frames(g)

    cond do
      total == horizon -> g
      total > horizon -> take_frames(g, horizon)
      g == [] -> [{:neutral, horizon}]
      true -> g ++ [{:neutral, horizon - total}]
    end
  end

  # ---------------------------------------------------------------- fitness

  @doc "Score one rollout (states from the start, inclusive). Returns `%{fitness, chain, damage, alive?, style}`."
  def score(states, gs0, opts \\ []) do
    style_penalty = Keyword.get(opts, :style_penalty, %{full_hop: 120.0, grab: 60.0, shield_frame: 2.0})
    frames = Enum.map(states, fn s -> %{frame: s.frame, p1: ScenarioScan.player_summary(s.players[1]), p2: ScenarioScan.player_summary(s.players[2])} end)
    openings = AerialChain.openings(frames, window: length(states))
    # the best chain = the one dealing the most DAMAGE; moves that land (a 5-hit drill is one move)
    # are the tiebreaker. v0 weighted moves 100:1 over damage and evolved jab pressure (run 3).
    by_frame = Map.new(states, &{&1.frame, &1})
    chain_damage = fn o ->
      before = Map.get(by_frame, o.frame - 1, hd(states)).players[2].percent
      after_ = Map.get(by_frame, o.chain_end, List.last(states)).players[2].percent
      max(0.0, (after_ - before) * 1.0)
    end

    {chain, hits, chain_dmg} =
      openings
      |> Enum.map(fn o -> {moves_landed(states, o.frame, o.chain_end), o.hits, chain_damage.(o)} end)
      |> Enum.max_by(fn {m, _, d} -> d + 20.0 * m end, fn -> {0, 0, 0.0} end)

    aerials = if openings == [], do: 0, else: Enum.max(Enum.map(openings, & &1.connected_aerials))
    last = List.last(states)
    damage = (last.players[2].percent - gs0.players[2].percent) * 1.0
    stocks_taken = max(0, (gs0.players[2].stock || 0) - (last.players[2].stock || 0))
    p1 = last.players[1]
    alive = p1.stock == gs0.players[1].stock and abs(p1.x) < 90.0
    style = Search.style_counts(states)
    style_cost = style.full_hops * style_penalty.full_hop + style.grabs * style_penalty.grab + style.shield_frames * style_penalty.shield_frame
    move_bonus = Keyword.get(opts, :move_bonus, 20.0)
    kill_bonus = Keyword.get(opts, :kill_bonus, 1000.0)
    fitness = chain_dmg + move_bonus * chain + kill_bonus * stocks_taken - if(alive, do: 0.0, else: 500.0) - style_cost

    %{fitness: fitness, chain: chain, hits: hits, chain_damage: chain_dmg, aerials: aerials, damage: damage, stocks_taken: stocks_taken, alive?: alive, style: style, style_cost: style_cost, openings: length(openings)}
  end

  @doc """
  Distinct P1 attack instances that land on P2 between `from` and `to`
  (frame numbers, inclusive). An instance starts when P1's action changes
  or its action frame counter restarts; it "lands" on any frame in which
  P2's percent rises. Multi-hit moves (drill, nair) count once.
  """
  def moves_landed(states, from, to) do
    states
    |> Enum.filter(&(&1.frame >= from - 1 and &1.frame <= to))
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.reduce({0, MapSet.new()}, fn [a, b], {inst, landed} ->
      pa = a.players[1]
      pb = b.players[1]
      inst = if pb.action != pa.action or (pb.action_frame || 0) < (pa.action_frame || 0), do: inst + 1, else: inst
      landed = if b.players[2].percent > a.players[2].percent, do: MapSet.put(landed, inst), else: landed
      {inst, landed}
    end)
    |> elem(1)
    |> MapSet.size()
  end

  @doc """
  Evaluate a population from one start in one batch. `sim` must have
  `batch_size >= length(genomes)`; `state` is `{:id, id}` from `Env.upload`.
  `defender` is `:idle` or a batched Agent pid. Returns `[%{genome, states, score}]`.
  """
  def evaluate(sim, state, genomes, defender, horizon, opts \\ []) do
    n = length(genomes)
    vocab = vocab()
    for i <- 0..(n - 1), do: {:ok, _} = Env.restore(sim, i, state, frames: false)
    {:ok, _, _} = Env.observe(sim)

    if is_pid(defender) do
      case Agent.batch_reset_rows(defender, Enum.to_list(0..(n - 1))) do
        :ok -> :ok
        {:error, :batch_not_initialized} -> :ok = Agent.batch_init(defender, n)
      end

      for {gs, _c1, _c2} <- Keyword.get(opts, :history, []) do
        :ok = Agent.batch_observe(defender, List.duplicate(%{gs | own_port: 2}, n), player_port: 2)
      end
    end

    {:ok, gs0s} = Env.frames(sim)
    progs = genomes |> Enum.map(&(&1 |> to_program(vocab) |> List.to_tuple()))

    {history, _} =
      Enum.reduce(0..(horizon - 1), {[gs0s], gs0s}, fn t, {acc, states} ->
        c1s = Enum.map(progs, &elem(&1, t))

        c2s =
          if is_pid(defender) do
            {:ok, cs} = Agent.batch_get_controllers(defender, Enum.map(states, &%{&1 | own_port: 2}), player_port: 2)
            Enum.map(cs, &(&1 || @neutral))
          else
            List.duplicate(@neutral, n)
          end

        case Env.step(sim, Enum.zip_with(c1s, c2s, fn a, b -> [a, b] end)) do
          {:ok, nexts, _} -> {[nexts | acc], nexts}
          {:error, reason} -> raise "sim step failed: #{inspect(reason)}"
        end
      end)

    per_env = history |> Enum.reverse() |> Enum.zip() |> Enum.map(&Tuple.to_list/1)

    Enum.zip(genomes, per_env)
    |> Enum.map(fn {g, states} -> %{genome: g, states: states, score: score(states, hd(states), opts)} end)
  end

  # ---------------------------------------------------------------- the loop

  @doc """
  Run the GA. Options: `:population` (must equal the sim batch size),
  `:generations`, `:horizon` (90), `:max_hold` (12), `:elite` (8),
  `:tournament` (4), `:mutation` (0.15), `:crossover` (0.7), `:defender`
  (`:idle`), `:seed`, `:history` (start history for an agent defender),
  `:on_generation` (fn summary -> any), `:initial` (list of genomes to seed
  the population with).
  """
  def run(sim, state, opts) do
    pop = Keyword.fetch!(opts, :population)
    gens = Keyword.fetch!(opts, :generations)
    horizon = Keyword.get(opts, :horizon, 90)
    max_hold = Keyword.get(opts, :max_hold, 12)
    elite_n = Keyword.get(opts, :elite, 8)
    k = Keyword.get(opts, :tournament, 4)
    mut = Keyword.get(opts, :mutation, 0.15)
    cx = Keyword.get(opts, :crossover, 0.7)
    defender = Keyword.get(opts, :defender, :idle)
    on_gen = Keyword.get(opts, :on_generation, fn _ -> :ok end)
    if seed = Keyword.get(opts, :seed), do: :rand.seed(:exsss, {seed, 42, 7})

    initial = Keyword.get(opts, :initial, [])
    genomes = initial ++ for(_ <- 1..max(0, pop - length(initial)), do: random_genome(horizon, max_hold))

    {summaries, _, best} =
      Enum.reduce(1..gens, {[], genomes, nil}, fn gen, {acc, genomes, best} ->
        t0 = System.monotonic_time(:millisecond)
        results = evaluate(sim, state, genomes, defender, horizon, Keyword.take(opts, [:history, :style_penalty, :move_bonus, :kill_bonus]))
        ranked = Enum.sort_by(results, &(-&1.score.fitness))
        elite = hd(ranked)
        best = if best == nil or elite.score.fitness > best.score.fitness, do: Map.put(elite, :generation, gen), else: best
        fits = Enum.map(results, & &1.score.fitness)

        summary = %{
          generation: gen,
          best: elite.score.fitness,
          mean: Enum.sum(fits) / length(fits),
          median: fits |> Enum.sort() |> Enum.at(div(length(fits), 2)),
          best_chain: elite.score.chain,
          best_hits: elite.score.hits,
          best_chain_damage: elite.score.chain_damage,
          best_stocks: elite.score.stocks_taken,
          best_damage: elite.score.damage,
          chains: Enum.frequencies(Enum.map(results, & &1.score.chain)),
          elite: elite,
          best_so_far: best.score.fitness,
          ms: System.monotonic_time(:millisecond) - t0
        }

        on_gen.(summary)

        # next generation: elites verbatim, the rest by tournament + crossover + mutation
        elites = ranked |> Enum.take(elite_n) |> Enum.map(& &1.genome)
        pick = fn -> Enum.min_by(Enum.map(1..k, fn _ -> Enum.random(ranked) end), &(-&1.score.fitness)).genome end

        children =
          for _ <- 1..(pop - elite_n) do
            child = if :rand.uniform() < cx, do: crossover(pick.(), pick.(), horizon, max_hold), else: pick.()
            mutate(child, horizon, mut, max_hold)
          end

        {[Map.drop(summary, [:elite]) |> Map.put(:elite_genome, elite.genome) | acc], elites ++ children, best}
      end)

    %{generations: Enum.reverse(summaries), best: best}
  end
end

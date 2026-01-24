defmodule ExPhil.League.Evolution do
  @moduledoc """
  Self-play evolution for architecture league training.

  Handles the iterative process of:
  1. Running tournament matches
  2. Collecting experiences
  3. Training architectures with PPO
  4. Advancing to the next generation

  ## Evolution Cycle

  ```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          Evolution Iteration                            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │                                                                         │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │  1. Tournament Phase                                              │ │
  │  │     - Round-robin or skill-based matches                         │ │
  │  │     - Collect match experiences                                  │ │
  │  │     - Update Elo ratings                                         │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                              │                                          │
  │                              ▼                                          │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │  2. Training Phase                                                │ │
  │  │     - Pool experiences from all matches                          │ │
  │  │     - Run PPO update for each architecture                       │ │
  │  │     - Track training metrics                                     │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                              │                                          │
  │                              ▼                                          │
  │  ┌───────────────────────────────────────────────────────────────────┐ │
  │  │  3. Evolution Phase                                               │ │
  │  │     - Advance generation counter                                 │ │
  │  │     - Optionally prune weak architectures                        │ │
  │  │     - Save checkpoints                                           │ │
  │  └───────────────────────────────────────────────────────────────────┘ │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
  ```

  ## Usage

      # Run a single evolution iteration
      {:ok, metrics} = Evolution.evolve(league, matches_per_pair: 10, ppo_epochs: 4)

      # Run multiple generations
      {:ok, final_league} = Evolution.run(league,
        generations: 10,
        matches_per_pair: 20,
        checkpoint_dir: "checkpoints/league"
      )

  """

  alias ExPhil.League
  alias ExPhil.League.{ArchitectureEntry, MatchScheduler}
  alias ExPhil.Training.Output

  require Logger

  @default_opts [
    matches_per_pair: 10,
    ppo_epochs: 4,
    ppo_batch_size: 64,
    learning_rate: 3.0e-4,
    clip_epsilon: 0.2,
    value_loss_coef: 0.5,
    entropy_coef: 0.01,
    max_grad_norm: 0.5,
    gamma: 0.99,
    gae_lambda: 0.95,
    verbose: true
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Run one evolution iteration (tournament + training + advance generation).

  ## Parameters

  - `league` - League GenServer pid/name

  ## Options

  - `:matches_per_pair` - Matches between each architecture pair (default: 10)
  - `:ppo_epochs` - PPO training epochs (default: 4)
  - `:ppo_batch_size` - PPO batch size (default: 64)
  - `:learning_rate` - PPO learning rate (default: 3.0e-4)
  - `:verbose` - Print progress (default: true)

  ## Returns

  `{:ok, metrics}` where metrics contains tournament and training stats.

  """
  @spec evolve(GenServer.server(), keyword()) :: {:ok, map()} | {:error, term()}
  def evolve(league, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    verbose = opts[:verbose]

    generation = League.get_generation(league)

    if verbose do
      Output.step(1, 3, "Generation #{generation + 1}: Tournament")
    end

    # 1. Run tournament
    {:ok, tournament_results} = run_tournament_phase(league, opts)

    if verbose do
      Output.puts("  #{length(tournament_results)} matches completed")
      Output.step(2, 3, "Generation #{generation + 1}: Training")
    end

    # 2. Collect experiences and train
    training_metrics = run_training_phase(league, opts)

    if verbose do
      Output.puts("  Trained #{map_size(training_metrics)} architectures")
      Output.step(3, 3, "Generation #{generation + 1}: Evolution")
    end

    # 3. Advance generation
    {:ok, new_generation} = League.advance_generation(league)

    # Compile metrics
    metrics = %{
      generation: new_generation,
      tournament: %{
        matches_played: length(tournament_results),
        results: summarize_tournament(tournament_results)
      },
      training: training_metrics,
      leaderboard: League.get_leaderboard(league, 10)
    }

    if verbose do
      print_generation_summary(metrics)
    end

    {:ok, metrics}
  end

  @doc """
  Run multiple evolution generations.

  ## Parameters

  - `league` - League GenServer pid/name

  ## Options

  - `:generations` - Number of generations to run (default: 10)
  - `:checkpoint_dir` - Save checkpoints here (optional)
  - `:checkpoint_every` - Save checkpoint every N generations (default: 5)
  - `:callback` - Function called after each generation with (generation, metrics)
  - All options from `evolve/2`

  ## Returns

  `{:ok, final_metrics}` with summary of all generations.

  """
  @spec run(GenServer.server(), keyword()) :: {:ok, map()} | {:error, term()}
  def run(league, opts \\ []) do
    generations = Keyword.get(opts, :generations, 10)
    checkpoint_dir = Keyword.get(opts, :checkpoint_dir)
    checkpoint_every = Keyword.get(opts, :checkpoint_every, 5)
    callback = Keyword.get(opts, :callback, fn _, _ -> :ok end)
    verbose = Keyword.get(opts, :verbose, true)

    if verbose do
      Output.banner("Architecture League Evolution")
      Output.config([
        {"Generations", generations},
        {"Architectures", length(League.list_architectures(league))},
        {"Matches per pair", opts[:matches_per_pair] || 10}
      ])
      Output.puts("")
    end

    # Run evolution loop
    all_metrics = Enum.reduce(1..generations, [], fn gen, acc ->
      if verbose do
        Output.puts("=" |> String.duplicate(60))
        Output.puts("Generation #{gen}/#{generations}")
        Output.puts("")
      end

      case evolve(league, opts) do
        {:ok, metrics} ->
          # Callback
          callback.(gen, metrics)

          # Checkpoint
          if checkpoint_dir && rem(gen, checkpoint_every) == 0 do
            gen_dir = Path.join(checkpoint_dir, "gen_#{gen}")
            League.save_all_checkpoints(league, gen_dir)
            if verbose, do: Output.puts("  Checkpoint saved to #{gen_dir}")
          end

          [metrics | acc]

        {:error, reason} ->
          Logger.error("[Evolution] Generation #{gen} failed: #{inspect(reason)}")
          acc
      end
    end)

    # Final checkpoint
    if checkpoint_dir do
      final_dir = Path.join(checkpoint_dir, "final")
      League.save_all_checkpoints(league, final_dir)
      if verbose, do: Output.success("Final checkpoint saved to #{final_dir}")
    end

    # Summary
    final_metrics = %{
      generations_completed: length(all_metrics),
      history: Enum.reverse(all_metrics),
      final_leaderboard: League.get_leaderboard(league),
      final_stats: League.get_stats(league)
    }

    if verbose do
      Output.puts("")
      Output.success("Evolution complete!")
      print_final_summary(final_metrics)
    end

    {:ok, final_metrics}
  end

  @doc """
  Run evolution with pruning of weak architectures.

  Periodically removes architectures that fall below a threshold,
  keeping the league competitive.

  ## Options

  - `:prune_every` - Prune every N generations (default: 5)
  - `:min_elo` - Remove architectures below this Elo (default: 800)
  - `:keep_min` - Always keep at least this many (default: 3)
  - All options from `run/2`

  """
  @spec run_with_pruning(GenServer.server(), keyword()) :: {:ok, map()}
  def run_with_pruning(league, opts \\ []) do
    prune_every = Keyword.get(opts, :prune_every, 5)
    min_elo = Keyword.get(opts, :min_elo, 800)
    keep_min = Keyword.get(opts, :keep_min, 3)

    # Wrap callback to include pruning
    original_callback = Keyword.get(opts, :callback, fn _, _ -> :ok end)

    callback = fn gen, metrics ->
      original_callback.(gen, metrics)

      if rem(gen, prune_every) == 0 do
        prune_weak(league, min_elo, keep_min)
      end
    end

    run(league, Keyword.put(opts, :callback, callback))
  end

  # ============================================================================
  # Tournament Phase
  # ============================================================================

  defp run_tournament_phase(league, opts) do
    matches_per_pair = Keyword.get(opts, :matches_per_pair, 10)

    # Get all architectures
    architectures = League.list_architectures(league)
    arch_ids = Enum.map(architectures, & &1.id)

    if length(arch_ids) < 2 do
      {:ok, []}
    else
      # Generate schedule
      schedule = MatchScheduler.round_robin(arch_ids, matches_per_pair: matches_per_pair)

      # Run matches
      results = Enum.flat_map(schedule, fn {p1, p2} ->
        case League.run_match(league, p1, p2) do
          {:ok, result} -> [result]
          {:error, _} -> []
        end
      end)

      {:ok, results}
    end
  end

  # ============================================================================
  # Training Phase
  # ============================================================================

  defp run_training_phase(league, opts) do
    # Get collected experiences
    experiences = League.get_experiences(league)

    if length(experiences) == 0 do
      Logger.warning("[Evolution] No experiences collected, skipping training")
      %{}
    else
      # Get all architectures
      architectures = League.list_architectures(league)

      # Train each architecture
      metrics = Enum.reduce(architectures, %{}, fn arch, acc ->
        case train_architecture(arch, experiences, opts) do
          {:ok, new_params, train_metrics} ->
            # Update params in league
            League.update_params(league, arch.id, new_params)
            Map.put(acc, arch.id, train_metrics)

          {:error, reason} ->
            Logger.warning("[Evolution] Failed to train #{arch.id}: #{inspect(reason)}")
            acc
        end
      end)

      # Clear experience pool for next generation
      League.clear_experiences(league)

      metrics
    end
  end

  defp train_architecture(arch, experiences, opts) do
    ppo_epochs = Keyword.get(opts, :ppo_epochs, 4)
    batch_size = Keyword.get(opts, :ppo_batch_size, 64)
    learning_rate = Keyword.get(opts, :learning_rate, 3.0e-4)

    if not ArchitectureEntry.has_model?(arch) do
      {:error, :no_model}
    else
      # Prepare rollout data
      rollout = experiences_to_rollout(experiences)

      # Build PPO trainer
      ppo_config = %{
        clip_epsilon: opts[:clip_epsilon] || 0.2,
        value_loss_coef: opts[:value_loss_coef] || 0.5,
        entropy_coef: opts[:entropy_coef] || 0.01,
        max_grad_norm: opts[:max_grad_norm] || 0.5,
        learning_rate: learning_rate,
        batch_size: batch_size,
        num_epochs: ppo_epochs
      }

      # Run PPO update
      case run_ppo_update(arch.model, arch.params, rollout, ppo_config) do
        {:ok, new_params, metrics} ->
          {:ok, new_params, metrics}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp experiences_to_rollout(experiences) do
    # Convert list of experience maps to batched tensors
    %{
      states: experiences |> Enum.map(& &1.state) |> stack_tensors(),
      actions: experiences |> Enum.map(& &1.action) |> stack_actions(),
      rewards: experiences |> Enum.map(& &1.reward) |> Nx.tensor(type: :f32),
      dones: experiences |> Enum.map(&if(&1.done, do: 1.0, else: 0.0)) |> Nx.tensor(type: :f32),
      values: compute_values(experiences),
      log_probs: compute_log_probs(experiences)
    }
  end

  defp stack_tensors(tensors) do
    case tensors do
      [%Nx.Tensor{} | _] -> Nx.stack(tensors)
      _ -> Nx.tensor(tensors, type: :f32)
    end
  end

  defp stack_actions(actions) do
    case hd(actions) do
      %{} = action_map ->
        keys = Map.keys(action_map)
        Map.new(keys, fn key ->
          values = Enum.map(actions, &Map.get(&1, key, 0))
          {key, Nx.tensor(values, type: :f32)}
        end)

      _ ->
        Nx.tensor(actions, type: :f32)
    end
  end

  defp compute_values(experiences) do
    # Placeholder - actual implementation would use value network
    experiences
    |> Enum.map(fn exp -> Map.get(exp, :value, 0.0) end)
    |> Nx.tensor(type: :f32)
  end

  defp compute_log_probs(experiences) do
    # Placeholder - actual implementation would compute from policy
    experiences
    |> Enum.map(fn exp -> Map.get(exp, :log_prob, -1.0) end)
    |> Nx.tensor(type: :f32)
  end

  defp run_ppo_update(model, params, rollout, config) do
    # Compute advantages using GAE
    advantages = compute_gae(
      rollout.rewards,
      rollout.values,
      rollout.dones,
      config[:gamma] || 0.99,
      config[:gae_lambda] || 0.95
    )

    # Compute returns
    returns = Nx.add(advantages, rollout.values)

    # Initialize optimizer - returns {init_fn, update_fn} tuple
    {optimizer_init, optimizer_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    opt_state = optimizer_init.(params)

    # Create minibatches
    num_samples = Nx.axis_size(rollout.states, 0)
    indices = Enum.shuffle(0..(num_samples - 1))

    # Run PPO epochs
    {final_params, _final_opt_state, metrics} =
      Enum.reduce(1..config.num_epochs, {params, opt_state, init_ppo_metrics()}, fn _epoch, {p, opt, m} ->
        # Process minibatches
        batch_indices = Enum.chunk_every(indices, config.batch_size)

        Enum.reduce(batch_indices, {p, opt, m}, fn batch_idx, {p_inner, opt_inner, m_inner} ->
          # Get batch data
          batch = get_batch(rollout, advantages, returns, batch_idx)

          # Compute loss and gradients
          {loss, grads} = compute_ppo_loss_and_grad(model, p_inner, batch, config)

          # Apply updates with gradient clipping
          grads = clip_gradients(grads, config.max_grad_norm)
          {updates, new_opt} = optimizer_update.(grads, opt_inner, p_inner)
          new_params = Polaris.Updates.apply_updates(updates, p_inner)

          # Update metrics
          new_metrics = update_ppo_metrics(m_inner, loss)

          {new_params, new_opt, new_metrics}
        end)
      end)

    {:ok, final_params, finalize_ppo_metrics(metrics)}
  end

  defp compute_gae(rewards, values, dones, gamma, gae_lambda) do
    # Generalized Advantage Estimation
    n = Nx.axis_size(rewards, 0)

    # Add bootstrap value (0 for terminal states)
    values_with_bootstrap = Nx.concatenate([
      values,
      Nx.tensor([0.0], type: :f32)
    ])

    # Compute deltas
    deltas = Nx.subtract(
      Nx.add(rewards, Nx.multiply(gamma, Nx.multiply(Nx.subtract(1.0, dones), Nx.slice(values_with_bootstrap, [1], [n])))),
      Nx.slice(values_with_bootstrap, [0], [n])
    )

    # Compute GAE backwards
    compute_gae_backwards(deltas, dones, gamma * gae_lambda, n - 1, Nx.broadcast(0.0, {n}))
  end

  defp compute_gae_backwards(_deltas, _dones, _discount, -1, advantages), do: advantages

  defp compute_gae_backwards(deltas, dones, discount, t, advantages) do
    delta_t = Nx.slice(deltas, [t], [1]) |> Nx.squeeze()
    done_t = Nx.slice(dones, [t], [1]) |> Nx.squeeze()
    next_adv = if t < Nx.axis_size(advantages, 0) - 1 do
      Nx.slice(advantages, [t + 1], [1]) |> Nx.squeeze()
    else
      Nx.tensor(0.0, type: :f32)
    end

    adv_t = Nx.add(delta_t, Nx.multiply(discount, Nx.multiply(Nx.subtract(1.0, done_t), next_adv)))
    new_advantages = Nx.indexed_put(advantages, Nx.tensor([[t]]), Nx.reshape(adv_t, {1}))

    compute_gae_backwards(deltas, dones, discount, t - 1, new_advantages)
  end

  defp get_batch(rollout, advantages, returns, indices) do
    idx_tensor = Nx.tensor(indices)

    %{
      states: Nx.take(rollout.states, idx_tensor),
      actions: take_actions(rollout.actions, idx_tensor),
      old_log_probs: Nx.take(rollout.log_probs, idx_tensor),
      advantages: Nx.take(advantages, idx_tensor),
      returns: Nx.take(returns, idx_tensor)
    }
  end

  defp take_actions(actions, indices) when is_map(actions) do
    Map.new(actions, fn {k, v} -> {k, Nx.take(v, indices)} end)
  end

  defp take_actions(actions, indices) do
    Nx.take(actions, indices)
  end

  defp compute_ppo_loss_and_grad(model, params, batch, config) do
    Nx.Defn.value_and_grad(params, fn p ->
      # Forward pass
      output = Axon.predict(model, p, %{"state" => batch.states})

      # Compute policy loss (simplified)
      policy_loss = compute_policy_loss(output, batch, config)

      # Compute value loss
      value_loss = if Map.has_key?(output, "value") do
        values = output["value"]
        Nx.mean(Nx.pow(Nx.subtract(values, batch.returns), 2))
      else
        Nx.tensor(0.0, type: :f32)
      end

      # Compute entropy bonus
      entropy = compute_entropy(output)

      # Total loss
      total_loss = Nx.add(
        Nx.add(policy_loss, Nx.multiply(config.value_loss_coef, value_loss)),
        Nx.multiply(-config.entropy_coef, entropy)
      )

      total_loss
    end)
  end

  defp compute_policy_loss(output, batch, config) do
    # Simplified PPO loss - actual implementation would handle all action heads
    case output do
      %{"buttons" => logits} ->
        # Compute new log probs
        new_log_probs = Nx.log(Axon.Activations.softmax(logits) + 1.0e-8)

        # Ratio
        ratio = Nx.exp(Nx.subtract(new_log_probs, batch.old_log_probs))

        # Clipped objective
        clip_ratio = Nx.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
        surr1 = Nx.multiply(ratio, batch.advantages)
        surr2 = Nx.multiply(clip_ratio, batch.advantages)

        Nx.mean(Nx.negate(Nx.min(surr1, surr2)))

      _ ->
        Nx.tensor(0.0, type: :f32)
    end
  end

  defp compute_entropy(output) do
    case output do
      %{"buttons" => logits} ->
        probs = Axon.Activations.softmax(logits)
        Nx.mean(Nx.negate(Nx.sum(Nx.multiply(probs, Nx.log(probs + 1.0e-8)), axes: [-1])))

      _ ->
        Nx.tensor(0.0, type: :f32)
    end
  end

  defp clip_gradients(grads, max_norm) do
    # Compute total norm
    total_norm_sq = grads
    |> Enum.map(fn {_k, g} -> Nx.sum(Nx.pow(g, 2)) end)
    |> Enum.reduce(Nx.tensor(0.0), &Nx.add/2)

    total_norm = Nx.sqrt(total_norm_sq)
    clip_coef = Nx.min(Nx.divide(max_norm, Nx.add(total_norm, 1.0e-6)), 1.0)

    Map.new(grads, fn {k, g} -> {k, Nx.multiply(g, clip_coef)} end)
  end

  defp init_ppo_metrics do
    %{total_loss: 0.0, num_updates: 0}
  end

  defp update_ppo_metrics(metrics, loss) do
    %{
      total_loss: metrics.total_loss + Nx.to_number(loss),
      num_updates: metrics.num_updates + 1
    }
  end

  defp finalize_ppo_metrics(metrics) do
    avg_loss = if metrics.num_updates > 0 do
      metrics.total_loss / metrics.num_updates
    else
      0.0
    end

    %{
      avg_loss: avg_loss,
      num_updates: metrics.num_updates
    }
  end

  # ============================================================================
  # Pruning
  # ============================================================================

  defp prune_weak(league, min_elo, keep_min) do
    architectures = League.list_architectures(league)

    if length(architectures) <= keep_min do
      :ok
    else
      # Sort by Elo
      sorted = Enum.sort_by(architectures, & &1.elo)

      # Find candidates for removal
      to_remove = sorted
      |> Enum.filter(&(&1.elo < min_elo))
      |> Enum.take(length(architectures) - keep_min)

      # Remove weak architectures
      Enum.each(to_remove, fn arch ->
        Logger.info("[Evolution] Pruning #{arch.id} (Elo: #{Float.round(arch.elo, 1)})")
        League.unregister_architecture(league, arch.id)
      end)
    end
  end

  # ============================================================================
  # Reporting
  # ============================================================================

  defp summarize_tournament(results) do
    # Count wins per architecture
    wins_by_arch = Enum.reduce(results, %{}, fn result, acc ->
      winner_id = case result.winner do
        :p1 -> result.p1_id
        :p2 -> result.p2_id
        :draw -> nil
      end

      if winner_id do
        Map.update(acc, winner_id, 1, &(&1 + 1))
      else
        acc
      end
    end)

    %{
      total_matches: length(results),
      wins_by_architecture: wins_by_arch,
      draws: Enum.count(results, &(&1.winner == :draw))
    }
  end

  defp print_generation_summary(metrics) do
    Output.puts("")
    Output.puts("  Leaderboard:")

    Enum.each(metrics.leaderboard, fn entry ->
      win_rate = Float.round(entry.win_rate * 100, 1)
      Output.puts("    #{entry.id}: Elo #{Float.round(entry.elo, 1)} (#{win_rate}% win rate)")
    end)

    Output.puts("")
  end

  defp print_final_summary(metrics) do
    Output.puts("")
    Output.puts("Final Leaderboard:")
    Output.puts("-" |> String.duplicate(50))

    metrics.final_leaderboard
    |> Enum.with_index(1)
    |> Enum.each(fn {entry, rank} ->
      win_rate = Float.round(entry.win_rate * 100, 1)
      Output.puts("  #{rank}. #{entry.id}")
      Output.puts("     Elo: #{Float.round(entry.elo, 1)} | Win Rate: #{win_rate}%")
      Output.puts("     Games: #{entry.games_played} (#{entry.wins}W/#{entry.losses}L/#{entry.draws}D)")
    end)

    Output.puts("")
    Output.puts("Statistics:")
    Output.puts("  Total matches: #{metrics.final_stats.matches_played}")
    Output.puts("  Generations: #{metrics.generations_completed}")
  end
end

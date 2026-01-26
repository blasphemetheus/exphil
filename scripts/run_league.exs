#!/usr/bin/env elixir
# Usage:
#   mix run scripts/run_league.exs \
#     --replays ~/replays/mewtwo \
#     --target-loss 1.0 \
#     --generations 10 \
#     --matches-per-pair 20
#
# With specific architectures:
#   mix run scripts/run_league.exs \
#     --replays ~/replays/mewtwo \
#     --architectures mlp,mamba,lstm \
#     --target-loss 1.5 \
#     --generations 5
#
# Quick smoke test:
#   mix run scripts/run_league.exs \
#     --replays test/fixtures/replays \
#     --architectures mlp,mamba \
#     --target-loss 2.0 \
#     --generations 2 \
#     --matches-per-pair 5

alias ExPhil.Training.Output
alias ExPhil.League
alias ExPhil.League.{Pretraining, Evolution, ArchitectureEntry}
alias ExPhil.Data.ReplayParser

# ============================================================================
# Argument Parsing
# ============================================================================

defmodule LeagueRunner do
  @default_architectures [:mlp, :lstm, :gru, :mamba, :attention, :jamba]

  @default_config %{
    # Pretraining
    target_loss: 1.0,
    max_epochs: 50,
    batch_size: 64,
    learning_rate: 1.0e-4,

    # Competition
    generations: 10,
    matches_per_pair: 20,

    # Training
    ppo_epochs: 4,
    ppo_batch_size: 64,

    # Output
    checkpoint_dir: "checkpoints/league",
    report_path: "league_results.html",
    verbose: true
  }

  def run(args) do
    opts = parse_args(args)
    config = Map.merge(@default_config, Map.new(opts))

    Output.banner("Architecture League System")
    Output.puts("")

    # Step 1: Load and parse replays
    Output.step(1, 5, "Loading replays")
    dataset = load_replays(config.replays_dir)
    Output.puts("  Loaded #{length(dataset)} samples from replays")
    Output.puts("")

    # Step 2: Define architectures
    Output.step(2, 5, "Configuring architectures")
    architectures = build_architecture_specs(config.architectures)
    Output.puts("  Architectures: #{Enum.map(architectures, & &1.id) |> Enum.join(", ")}")
    Output.puts("")

    # Step 3: Imitation pretraining
    Output.step(3, 5, "Imitation pretraining")

    {:ok, trained} =
      Pretraining.train_all(
        architectures,
        dataset,
        target_loss: config.target_loss,
        max_epochs: config.max_epochs,
        batch_size: config.batch_size,
        learning_rate: config.learning_rate,
        checkpoint_dir: Path.join(config.checkpoint_dir, "pretrained"),
        verbose: config.verbose
      )

    Output.puts("")

    # Step 4: Register in league and run evolution
    Output.step(4, 5, "Starting league competition")
    {:ok, _league} = League.start_link(name: LeagueProcess, game_type: :mock)

    # Register all trained architectures
    Enum.each(trained, fn {arch_id, {model, params, final_loss, epochs}} ->
      arch_spec = Enum.find(architectures, &(&1.id == arch_id))

      # Create entry
      {:ok, entry} =
        ArchitectureEntry.new(
          id: arch_id,
          architecture: arch_spec.architecture,
          config: arch_spec.config,
          model: model,
          params: params
        )

      League.register_entry(LeagueProcess, entry)

      Output.puts(
        "  Registered #{arch_id} (loss: #{Float.round(final_loss, 4)}, epochs: #{epochs})"
      )
    end)

    Output.puts("")

    # Run evolution
    {:ok, final_metrics} =
      Evolution.run(LeagueProcess,
        generations: config.generations,
        matches_per_pair: config.matches_per_pair,
        ppo_epochs: config.ppo_epochs,
        ppo_batch_size: config.ppo_batch_size,
        checkpoint_dir: config.checkpoint_dir,
        checkpoint_every: 5,
        verbose: config.verbose
      )

    # Step 5: Generate report
    Output.step(5, 5, "Generating report")
    generate_report(final_metrics, config.report_path)
    Output.success("Report saved to #{config.report_path}")

    Output.puts("")
    Output.success("League complete!")

    # Print final leaderboard
    print_final_results(final_metrics)
  end

  # ============================================================================
  # Argument Parsing
  # ============================================================================

  defp parse_args(args) do
    {opts, _rest, _invalid} =
      OptionParser.parse(args,
        strict: [
          replays: :string,
          architectures: :string,
          target_loss: :float,
          max_epochs: :integer,
          generations: :integer,
          matches_per_pair: :integer,
          ppo_epochs: :integer,
          batch_size: :integer,
          learning_rate: :float,
          checkpoint_dir: :string,
          report_path: :string,
          verbose: :boolean,
          quiet: :boolean,
          help: :boolean
        ],
        aliases: [
          r: :replays,
          a: :architectures,
          g: :generations,
          m: :matches_per_pair,
          o: :checkpoint_dir,
          v: :verbose,
          q: :quiet,
          h: :help
        ]
      )

    if opts[:help] do
      print_help()
      System.halt(0)
    end

    unless opts[:replays] do
      Output.error("Missing required --replays argument")
      Output.puts("")
      print_help()
      System.halt(1)
    end

    # Parse architectures
    architectures =
      case opts[:architectures] do
        nil -> @default_architectures
        str -> str |> String.split(",") |> Enum.map(&String.to_atom/1)
      end

    [
      replays_dir: opts[:replays],
      architectures: architectures,
      target_loss: opts[:target_loss] || 1.0,
      max_epochs: opts[:max_epochs] || 50,
      generations: opts[:generations] || 10,
      matches_per_pair: opts[:matches_per_pair] || 20,
      ppo_epochs: opts[:ppo_epochs] || 4,
      batch_size: opts[:batch_size] || 64,
      learning_rate: opts[:learning_rate] || 1.0e-4,
      checkpoint_dir: opts[:checkpoint_dir] || "checkpoints/league",
      report_path: opts[:report_path] || "league_results.html",
      verbose: not opts[:quiet]
    ]
  end

  defp print_help do
    IO.puts("""
    Architecture League System

    Run a competition between different neural network architectures for Melee AI.

    USAGE:
      mix run scripts/run_league.exs --replays <path> [options]

    REQUIRED:
      -r, --replays <path>          Path to replay directory

    OPTIONS:
      -a, --architectures <list>    Comma-separated list of architectures
                                    (default: mlp,lstm,gru,mamba,attention,jamba)

      PRETRAINING:
      --target-loss <float>         Target validation loss for pretraining (default: 1.0)
      --max-epochs <int>            Max epochs per architecture (default: 50)

      COMPETITION:
      -g, --generations <int>       Number of evolution generations (default: 10)
      -m, --matches-per-pair <int>  Matches between each pair per generation (default: 20)

      TRAINING:
      --ppo-epochs <int>            PPO epochs per generation (default: 4)
      --batch-size <int>            Batch size for training (default: 64)
      --learning-rate <float>       Learning rate (default: 1.0e-4)

      OUTPUT:
      -o, --checkpoint-dir <path>   Checkpoint directory (default: checkpoints/league)
      --report-path <path>          HTML report path (default: league_results.html)

      -v, --verbose                 Verbose output (default)
      -q, --quiet                   Minimal output
      -h, --help                    Show this help

    EXAMPLES:
      # Full league with all architectures
      mix run scripts/run_league.exs \\
        --replays ~/replays/mewtwo \\
        --target-loss 1.0 \\
        --generations 10

      # Quick test with 2 architectures
      mix run scripts/run_league.exs \\
        --replays test/fixtures/replays \\
        --architectures mlp,mamba \\
        --target-loss 2.0 \\
        --generations 2 \\
        --matches-per-pair 5

      # Focus on recurrent architectures
      mix run scripts/run_league.exs \\
        --replays ~/replays/mewtwo \\
        --architectures lstm,gru,mamba \\
        --generations 15
    """)
  end

  # ============================================================================
  # Data Loading
  # ============================================================================

  defp load_replays(replays_dir) do
    unless File.dir?(replays_dir) do
      Output.error("Replay directory not found: #{replays_dir}")
      System.halt(1)
    end

    # Find all .slp files
    replay_files = Path.wildcard(Path.join(replays_dir, "**/*.slp"))

    if length(replay_files) == 0 do
      Output.error("No .slp files found in #{replays_dir}")
      System.halt(1)
    end

    Output.puts("  Found #{length(replay_files)} replay files")

    # Parse replays
    replay_files
    |> Task.async_stream(
      fn file -> parse_replay_file(file) end,
      max_concurrency: System.schedulers_online(),
      timeout: 30_000
    )
    |> Enum.flat_map(fn
      {:ok, frames} -> frames
      {:exit, _} -> []
    end)
  end

  defp parse_replay_file(file) do
    case ReplayParser.parse(file) do
      {:ok, replay} ->
        # Extract training samples (state, action pairs)
        extract_training_samples(replay)

      {:error, _reason} ->
        []
    end
  end

  defp extract_training_samples(replay) do
    # Extract frames with player actions
    replay.frames
    |> Enum.map(fn frame ->
      %{
        state: frame.state,
        action: frame.p1_action
      }
    end)
  rescue
    _ -> []
  end

  # ============================================================================
  # Architecture Configuration
  # ============================================================================

  defp build_architecture_specs(architecture_types) do
    Enum.map(architecture_types, fn arch_type ->
      config = ArchitectureEntry.default_config(arch_type)

      %{
        id: :"#{arch_type}_mewtwo",
        architecture: arch_type,
        config: config
      }
    end)
  end

  # ============================================================================
  # Report Generation
  # ============================================================================

  defp generate_report(metrics, path) do
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Architecture League Results</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4a90d9; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f5f5f5; }
        .rank-1 { background-color: #ffd700 !important; }
        .rank-2 { background-color: #c0c0c0 !important; }
        .rank-3 { background-color: #cd7f32 !important; }
        .stats { display: flex; gap: 30px; margin: 20px 0; }
        .stat-box { background: #f0f0f0; padding: 20px; border-radius: 8px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #333; }
        .stat-label { color: #666; }
        .chart { margin: 20px 0; }
      </style>
    </head>
    <body>
      <h1>Architecture League Results</h1>

      <div class="stats">
        <div class="stat-box">
          <div class="stat-value">#{metrics.generations_completed}</div>
          <div class="stat-label">Generations</div>
        </div>
        <div class="stat-box">
          <div class="stat-value">#{metrics.final_stats.matches_played}</div>
          <div class="stat-label">Total Matches</div>
        </div>
        <div class="stat-box">
          <div class="stat-value">#{length(metrics.final_leaderboard)}</div>
          <div class="stat-label">Architectures</div>
        </div>
      </div>

      <h2>Final Leaderboard</h2>
      <table>
        <tr>
          <th>Rank</th>
          <th>Architecture</th>
          <th>Elo Rating</th>
          <th>Win Rate</th>
          <th>Games</th>
          <th>W/L/D</th>
        </tr>
        #{generate_leaderboard_rows(metrics.final_leaderboard)}
      </table>

      <h2>Generation History</h2>
      <table>
        <tr>
          <th>Generation</th>
          <th>Matches</th>
          <th>Leader</th>
          <th>Leader Elo</th>
        </tr>
        #{generate_history_rows(metrics.history)}
      </table>

      <footer style="margin-top: 40px; color: #999; font-size: 12px;">
        Generated by ExPhil Architecture League System
        <br>#{DateTime.utc_now() |> DateTime.to_string()}
      </footer>
    </body>
    </html>
    """

    File.write!(path, html)
  end

  defp generate_leaderboard_rows(leaderboard) do
    leaderboard
    |> Enum.with_index(1)
    |> Enum.map(fn {entry, rank} ->
      rank_class =
        case rank do
          1 -> "rank-1"
          2 -> "rank-2"
          3 -> "rank-3"
          _ -> ""
        end

      """
      <tr class="#{rank_class}">
        <td>#{rank}</td>
        <td>#{entry.id}</td>
        <td>#{Float.round(entry.elo, 1)}</td>
        <td>#{Float.round(entry.win_rate * 100, 1)}%</td>
        <td>#{entry.games_played}</td>
        <td>#{entry.wins}/#{entry.losses}/#{entry.draws}</td>
      </tr>
      """
    end)
    |> Enum.join("\n")
  end

  defp generate_history_rows(history) do
    history
    |> Enum.map(fn metrics ->
      leader = List.first(metrics.leaderboard) || %{id: "N/A", elo: 0}
      matches = metrics.tournament.matches_played

      """
      <tr>
        <td>#{metrics.generation}</td>
        <td>#{matches}</td>
        <td>#{leader.id}</td>
        <td>#{Float.round(leader.elo, 1)}</td>
      </tr>
      """
    end)
    |> Enum.join("\n")
  end

  defp print_final_results(metrics) do
    Output.puts("")
    Output.puts("=" |> String.duplicate(60))
    Output.puts("FINAL RESULTS")
    Output.puts("=" |> String.duplicate(60))
    Output.puts("")

    metrics.final_leaderboard
    |> Enum.with_index(1)
    |> Enum.each(fn {entry, rank} ->
      medal =
        case rank do
          1 -> "🥇"
          2 -> "🥈"
          3 -> "🥉"
          _ -> "  "
        end

      Output.puts("#{medal} #{rank}. #{entry.id}")

      Output.puts(
        "      Elo: #{Float.round(entry.elo, 1)} | " <>
          "Win Rate: #{Float.round(entry.win_rate * 100, 1)}% | " <>
          "Games: #{entry.games_played}"
      )
    end)

    Output.puts("")
    Output.puts("Total matches played: #{metrics.final_stats.matches_played}")
    Output.puts("Generations completed: #{metrics.generations_completed}")
  end
end

# Run the league
LeagueRunner.run(System.argv())

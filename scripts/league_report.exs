#!/usr/bin/env elixir
# Generate HTML reports from league results
#
# USAGE:
#   # From checkpoint directory
#   mix run scripts/league_report.exs --checkpoint-dir checkpoints/league
#
#   # From saved league state
#   mix run scripts/league_report.exs --league-state checkpoints/league/final/league_state.json
#
#   # Generate with custom output path
#   mix run scripts/league_report.exs --checkpoint-dir checkpoints/league --output my_report.html
#
#   # Generate JSON instead of HTML
#   mix run scripts/league_report.exs --checkpoint-dir checkpoints/league --format json

alias ExPhil.Training.Output
alias ExPhil.League.ArchitectureEntry

defmodule LeagueReporter do
  @default_config %{
    output: "league_report.html",
    format: :html,
    include_charts: true,
    include_history: true,
    theme: :light
  }

  def run(args) do
    opts = parse_args(args)
    config = Map.merge(@default_config, Map.new(opts))

    Output.banner("Architecture League Report Generator")
    Output.puts("")

    # Load league data
    Output.step(1, 2, "Loading league data")
    data = load_league_data(config)
    Output.puts("  Loaded data for #{length(data.architectures)} architectures")
    Output.puts("  Generations: #{data.generations}")
    Output.puts("")

    # Generate report
    Output.step(2, 2, "Generating #{config.format} report")

    case config.format do
      :html ->
        html = generate_html_report(data, config)
        File.write!(config.output, html)
        Output.success("HTML report saved to #{config.output}")

      :json ->
        json = generate_json_report(data)
        File.write!(config.output, json)
        Output.success("JSON report saved to #{config.output}")

      :terminal ->
        print_terminal_report(data)
    end

    Output.puts("")
  end

  # ============================================================================
  # Argument Parsing
  # ============================================================================

  defp parse_args(args) do
    {opts, _rest, _invalid} = OptionParser.parse(args,
      strict: [
        checkpoint_dir: :string,
        league_state: :string,
        output: :string,
        format: :string,
        include_charts: :boolean,
        include_history: :boolean,
        theme: :string,
        help: :boolean
      ],
      aliases: [
        c: :checkpoint_dir,
        s: :league_state,
        o: :output,
        f: :format,
        h: :help
      ]
    )

    if opts[:help] do
      print_help()
      System.halt(0)
    end

    unless opts[:checkpoint_dir] || opts[:league_state] do
      Output.error("Must specify --checkpoint-dir or --league-state")
      Output.puts("")
      print_help()
      System.halt(1)
    end

    format = case opts[:format] do
      "json" -> :json
      "terminal" -> :terminal
      _ -> :html
    end

    theme = case opts[:theme] do
      "dark" -> :dark
      _ -> :light
    end

    output_ext = if format == :json, do: ".json", else: ".html"
    default_output = "league_report#{output_ext}"

    [
      checkpoint_dir: opts[:checkpoint_dir],
      league_state: opts[:league_state],
      output: opts[:output] || default_output,
      format: format,
      include_charts: opts[:include_charts] != false,
      include_history: opts[:include_history] != false,
      theme: theme
    ]
  end

  defp print_help do
    IO.puts("""
    Architecture League Report Generator

    Generate HTML or JSON reports from league results.

    USAGE:
      mix run scripts/league_report.exs [options]

    INPUT (one required):
      -c, --checkpoint-dir <path>    Load from checkpoint directory
      -s, --league-state <path>      Load from league state JSON file

    OUTPUT:
      -o, --output <path>            Output file path (default: league_report.html)
      -f, --format <type>            Output format: html, json, terminal (default: html)
      --theme <theme>                HTML theme: light, dark (default: light)
      --no-include-charts            Disable embedded charts
      --no-include-history           Exclude generation history

    EXAMPLES:
      # Generate HTML from checkpoints
      mix run scripts/league_report.exs -c checkpoints/league

      # Generate JSON for external tools
      mix run scripts/league_report.exs -c checkpoints/league -f json -o results.json

      # Dark theme report
      mix run scripts/league_report.exs -c checkpoints/league --theme dark

      # Quick terminal summary
      mix run scripts/league_report.exs -s league_state.json -f terminal
    """)
  end

  # ============================================================================
  # Data Loading
  # ============================================================================

  defp load_league_data(config) do
    cond do
      config[:league_state] ->
        load_from_state_file(config.league_state)

      config[:checkpoint_dir] ->
        load_from_checkpoint_dir(config.checkpoint_dir)

      true ->
        Output.error("No data source specified")
        System.halt(1)
    end
  end

  defp load_from_state_file(path) do
    unless File.exists?(path) do
      Output.error("League state file not found: #{path}")
      System.halt(1)
    end

    path
    |> File.read!()
    |> Jason.decode!()
    |> parse_state_data()
  end

  defp load_from_checkpoint_dir(dir) do
    unless File.dir?(dir) do
      Output.error("Checkpoint directory not found: #{dir}")
      System.halt(1)
    end

    # Try to find final state first
    final_state = Path.join(dir, "final/league_state.json")

    if File.exists?(final_state) do
      load_from_state_file(final_state)
    else
      # Scan for generation directories and build data
      load_from_generation_dirs(dir)
    end
  end

  defp load_from_generation_dirs(dir) do
    # Find all gen_N directories
    generation_dirs = Path.wildcard(Path.join(dir, "gen_*"))
    |> Enum.sort_by(fn path ->
      path
      |> Path.basename()
      |> String.replace("gen_", "")
      |> String.to_integer()
    end)

    if length(generation_dirs) == 0 do
      Output.warning("No generation checkpoints found in #{dir}")
      # Try to load individual architecture checkpoints
      load_architecture_checkpoints(dir)
    else
      # Load latest generation for current state
      latest_gen_dir = List.last(generation_dirs)
      architectures = load_architectures_from_dir(latest_gen_dir)

      # Build history from all generations
      history = Enum.map(generation_dirs, fn gen_dir ->
        gen_num = gen_dir
        |> Path.basename()
        |> String.replace("gen_", "")
        |> String.to_integer()

        gen_archs = load_architectures_from_dir(gen_dir)

        %{
          generation: gen_num,
          leaderboard: gen_archs |> Enum.sort_by(& &1.elo, :desc),
          tournament: %{matches_played: estimate_matches(gen_archs)}
        }
      end)

      %{
        architectures: architectures,
        generations: length(generation_dirs),
        history: history,
        final_leaderboard: architectures |> Enum.sort_by(& &1.elo, :desc),
        final_stats: compute_stats(architectures)
      }
    end
  end

  defp load_architecture_checkpoints(dir) do
    # Look for *.axon or *.json files
    arch_files = Path.wildcard(Path.join(dir, "*.json"))

    architectures = Enum.flat_map(arch_files, fn file ->
      case File.read(file) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} when is_map(data) ->
              case ArchitectureEntry.from_metadata(data) do
                {:ok, entry} -> [entry_to_map(entry)]
                _ -> []
              end
            _ -> []
          end
        _ -> []
      end
    end)

    %{
      architectures: architectures,
      generations: 0,
      history: [],
      final_leaderboard: architectures |> Enum.sort_by(& &1.elo, :desc),
      final_stats: compute_stats(architectures)
    }
  end

  defp load_architectures_from_dir(dir) do
    # Look for architecture metadata files
    metadata_files = Path.wildcard(Path.join(dir, "*_metadata.json"))

    Enum.flat_map(metadata_files, fn file ->
      case File.read(file) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} ->
              case ArchitectureEntry.from_metadata(data) do
                {:ok, entry} -> [entry_to_map(entry)]
                _ -> []
              end
            _ -> []
          end
        _ -> []
      end
    end)
  end

  defp parse_state_data(data) do
    architectures = (data["architectures"] || [])
    |> Enum.map(fn arch_data ->
      case ArchitectureEntry.from_metadata(arch_data) do
        {:ok, entry} -> entry_to_map(entry)
        _ -> nil
      end
    end)
    |> Enum.reject(&is_nil/1)

    history = (data["history"] || [])
    |> Enum.map(fn gen_data ->
      %{
        generation: gen_data["generation"] || 0,
        leaderboard: parse_leaderboard(gen_data["leaderboard"] || []),
        tournament: %{
          matches_played: get_in(gen_data, ["tournament", "matches_played"]) || 0
        }
      }
    end)

    %{
      architectures: architectures,
      generations: data["generations_completed"] || length(history),
      history: history,
      final_leaderboard: architectures |> Enum.sort_by(& &1.elo, :desc),
      final_stats: compute_stats(architectures)
    }
  end

  defp parse_leaderboard(entries) do
    Enum.map(entries, fn entry ->
      %{
        id: String.to_atom(entry["id"] || "unknown"),
        elo: (entry["elo"] || 1000) / 1,
        win_rate: (entry["win_rate"] || 0.5) / 1,
        wins: entry["wins"] || 0,
        losses: entry["losses"] || 0,
        draws: entry["draws"] || 0,
        games_played: entry["games_played"] || 0
      }
    end)
  end

  defp entry_to_map(entry) do
    %{
      id: entry.id,
      architecture: entry.architecture,
      elo: entry.elo,
      generation: entry.generation,
      win_rate: ArchitectureEntry.win_rate(entry),
      wins: entry.stats.wins,
      losses: entry.stats.losses,
      draws: entry.stats.draws,
      games_played: ArchitectureEntry.games_played(entry)
    }
  end

  defp compute_stats(architectures) do
    total_games = architectures
    |> Enum.map(& &1.games_played)
    |> Enum.sum()
    |> div(2)  # Each game is counted twice (once per player)

    %{
      num_architectures: length(architectures),
      matches_played: total_games
    }
  end

  defp estimate_matches(architectures) do
    n = length(architectures)
    # Round-robin pairs * estimated matches per pair
    div(n * (n - 1), 2) * 10
  end

  # ============================================================================
  # HTML Report Generation
  # ============================================================================

  defp generate_html_report(data, config) do
    theme_styles = if config.theme == :dark do
      dark_theme_css()
    else
      light_theme_css()
    end

    charts_section = if config.include_charts do
      generate_charts_section(data)
    else
      ""
    end

    history_section = if config.include_history && length(data.history) > 0 do
      generate_history_section(data.history)
    else
      ""
    end

    """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Architecture League Report</title>
      <style>
        #{theme_styles}
        #{base_css()}
      </style>
      #{if config.include_charts, do: chart_js_library(), else: ""}
    </head>
    <body>
      <div class="container">
        <header>
          <h1>Architecture League Report</h1>
          <p class="subtitle">Neural Network Architecture Competition for Melee AI</p>
        </header>

        <section class="summary">
          <h2>Summary</h2>
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-value">#{data.generations}</div>
              <div class="stat-label">Generations</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">#{data.final_stats.matches_played}</div>
              <div class="stat-label">Total Matches</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">#{data.final_stats.num_architectures}</div>
              <div class="stat-label">Architectures</div>
            </div>
            <div class="stat-card highlight">
              <div class="stat-value">#{get_champion_name(data)}</div>
              <div class="stat-label">Champion</div>
            </div>
          </div>
        </section>

        <section class="leaderboard">
          <h2>Final Leaderboard</h2>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Architecture</th>
                <th>Type</th>
                <th>Elo Rating</th>
                <th>Win Rate</th>
                <th>Games</th>
                <th>W/L/D</th>
              </tr>
            </thead>
            <tbody>
              #{generate_leaderboard_rows(data.final_leaderboard)}
            </tbody>
          </table>
        </section>

        #{charts_section}

        #{history_section}

        <section class="architecture-details">
          <h2>Architecture Details</h2>
          #{generate_architecture_details(data.architectures)}
        </section>

        <footer>
          <p>Generated by ExPhil Architecture League System</p>
          <p>#{DateTime.utc_now() |> DateTime.to_iso8601()}</p>
        </footer>
      </div>

      #{if config.include_charts, do: chart_init_script(data), else: ""}
    </body>
    </html>
    """
  end

  defp base_css do
    """
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      line-height: 1.6;
      padding: 20px;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 40px;
      padding-bottom: 20px;
      border-bottom: 2px solid var(--border-color);
    }

    h1 {
      font-size: 2.5em;
      margin-bottom: 10px;
    }

    .subtitle {
      font-size: 1.2em;
      opacity: 0.7;
    }

    h2 {
      font-size: 1.5em;
      margin: 30px 0 20px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin: 20px 0;
    }

    .stat-card {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 24px;
      text-align: center;
      box-shadow: var(--shadow);
    }

    .stat-card.highlight {
      background: var(--highlight-bg);
      color: var(--highlight-text);
    }

    .stat-value {
      font-size: 2em;
      font-weight: bold;
      margin-bottom: 8px;
    }

    .stat-label {
      font-size: 0.9em;
      opacity: 0.7;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      background: var(--card-bg);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }

    th, td {
      padding: 16px;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }

    th {
      background: var(--header-bg);
      color: var(--header-text);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.85em;
      letter-spacing: 1px;
    }

    tbody tr:hover {
      background: var(--row-hover);
    }

    .rank-1 { background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%) !important; }
    .rank-2 { background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%) !important; }
    .rank-3 { background: linear-gradient(135deg, #cd7f32 0%, #d4a574 100%) !important; }
    .rank-1, .rank-2, .rank-3 { color: #333 !important; }

    .medal {
      display: inline-block;
      width: 24px;
      height: 24px;
      line-height: 24px;
      text-align: center;
      font-size: 16px;
      margin-right: 8px;
    }

    .chart-container {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 24px;
      margin: 20px 0;
      box-shadow: var(--shadow);
    }

    .chart-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 20px;
    }

    .architecture-card {
      background: var(--card-bg);
      border-radius: 12px;
      padding: 20px;
      margin: 10px 0;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-columns: 1fr 2fr;
      gap: 20px;
    }

    .arch-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }

    .arch-type {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.85em;
      font-weight: 500;
      text-transform: uppercase;
    }

    .type-mlp { background: #e3f2fd; color: #1565c0; }
    .type-lstm { background: #f3e5f5; color: #7b1fa2; }
    .type-gru { background: #e8f5e9; color: #2e7d32; }
    .type-mamba { background: #fff3e0; color: #ef6c00; }
    .type-attention { background: #fce4ec; color: #c2185b; }
    .type-jamba { background: #e0f7fa; color: #00838f; }

    footer {
      margin-top: 60px;
      padding: 30px;
      text-align: center;
      opacity: 0.6;
      font-size: 0.9em;
      border-top: 1px solid var(--border-color);
    }

    @media (max-width: 768px) {
      .architecture-card { grid-template-columns: 1fr; }
      .chart-row { grid-template-columns: 1fr; }
    }
    """
  end

  defp light_theme_css do
    """
    :root {
      --bg-color: #f8f9fa;
      --text-color: #333;
      --card-bg: #fff;
      --border-color: #e0e0e0;
      --header-bg: #4a90d9;
      --header-text: #fff;
      --highlight-bg: #4a90d9;
      --highlight-text: #fff;
      --row-hover: #f5f5f5;
      --shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    body { background: var(--bg-color); color: var(--text-color); }
    """
  end

  defp dark_theme_css do
    """
    :root {
      --bg-color: #1a1a2e;
      --text-color: #eee;
      --card-bg: #16213e;
      --border-color: #2d2d44;
      --header-bg: #0f3460;
      --header-text: #fff;
      --highlight-bg: #e94560;
      --highlight-text: #fff;
      --row-hover: #1f2942;
      --shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    body { background: var(--bg-color); color: var(--text-color); }
    """
  end

  defp chart_js_library do
    """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    """
  end

  defp generate_charts_section(data) do
    """
    <section class="charts">
      <h2>Visualizations</h2>
      <div class="chart-row">
        <div class="chart-container">
          <h3>Elo Distribution</h3>
          <canvas id="eloChart"></canvas>
        </div>
        <div class="chart-container">
          <h3>Win Rate Comparison</h3>
          <canvas id="winRateChart"></canvas>
        </div>
      </div>
      #{if length(data.history) > 0 do
        """
        <div class="chart-container">
          <h3>Elo Progression Over Generations</h3>
          <canvas id="progressionChart"></canvas>
        </div>
        """
      else
        ""
      end}
    </section>
    """
  end

  defp chart_init_script(data) do
    labels = data.final_leaderboard |> Enum.map(& &1.id |> to_string())
    elos = data.final_leaderboard |> Enum.map(& &1.elo |> Float.round(1))
    win_rates = data.final_leaderboard |> Enum.map(& (&1.win_rate * 100) |> Float.round(1))

    progression_data = if length(data.history) > 0 do
      # Get all unique architecture IDs
      all_ids = data.architectures |> Enum.map(& &1.id) |> Enum.uniq()

      datasets = Enum.map(all_ids, fn id ->
        elo_history = Enum.map(data.history, fn gen ->
          case Enum.find(gen.leaderboard, &(&1.id == id)) do
            nil -> nil
            entry -> entry.elo
          end
        end)

        color = get_color_for_arch(id)

        """
        {
          label: '#{id}',
          data: #{Jason.encode!(elo_history)},
          borderColor: '#{color}',
          backgroundColor: '#{color}33',
          tension: 0.3,
          fill: false
        }
        """
      end)

      generations = Enum.map(data.history, & &1.generation)

      """
      new Chart(document.getElementById('progressionChart'), {
        type: 'line',
        data: {
          labels: #{Jason.encode!(generations)},
          datasets: [#{Enum.join(datasets, ",")}]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { position: 'bottom' }
          },
          scales: {
            y: { title: { display: true, text: 'Elo Rating' } },
            x: { title: { display: true, text: 'Generation' } }
          }
        }
      });
      """
    else
      ""
    end

    """
    <script>
      document.addEventListener('DOMContentLoaded', function() {
        // Elo Bar Chart
        new Chart(document.getElementById('eloChart'), {
          type: 'bar',
          data: {
            labels: #{Jason.encode!(labels)},
            datasets: [{
              label: 'Elo Rating',
              data: #{Jason.encode!(elos)},
              backgroundColor: ['#ffd700', '#c0c0c0', '#cd7f32', '#4a90d9', '#5cb85c', '#f0ad4e'],
              borderWidth: 0,
              borderRadius: 8
            }]
          },
          options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
              y: { beginAtZero: false, min: Math.min(...#{Jason.encode!(elos)}) - 50 }
            }
          }
        });

        // Win Rate Chart
        new Chart(document.getElementById('winRateChart'), {
          type: 'doughnut',
          data: {
            labels: #{Jason.encode!(labels)},
            datasets: [{
              data: #{Jason.encode!(win_rates)},
              backgroundColor: ['#ffd700', '#c0c0c0', '#cd7f32', '#4a90d9', '#5cb85c', '#f0ad4e'],
              borderWidth: 2,
              borderColor: '#fff'
            }]
          },
          options: {
            responsive: true,
            plugins: {
              legend: { position: 'bottom' },
              tooltip: {
                callbacks: {
                  label: function(context) {
                    return context.label + ': ' + context.raw + '%';
                  }
                }
              }
            }
          }
        });

        #{progression_data}
      });
    </script>
    """
  end

  defp get_color_for_arch(id) do
    id_str = to_string(id)
    cond do
      String.contains?(id_str, "mlp") -> "#1565c0"
      String.contains?(id_str, "lstm") -> "#7b1fa2"
      String.contains?(id_str, "gru") -> "#2e7d32"
      String.contains?(id_str, "mamba") -> "#ef6c00"
      String.contains?(id_str, "attention") -> "#c2185b"
      String.contains?(id_str, "jamba") -> "#00838f"
      true -> "#666666"
    end
  end

  defp generate_history_section(history) do
    """
    <section class="history">
      <h2>Generation History</h2>
      <table>
        <thead>
          <tr>
            <th>Generation</th>
            <th>Matches</th>
            <th>Leader</th>
            <th>Leader Elo</th>
            <th>Top 3</th>
          </tr>
        </thead>
        <tbody>
          #{generate_history_rows(history)}
        </tbody>
      </table>
    </section>
    """
  end

  defp generate_leaderboard_rows(leaderboard) do
    leaderboard
    |> Enum.with_index(1)
    |> Enum.map(fn {entry, rank} ->
      {rank_class, medal} = case rank do
        1 -> {"rank-1", "🥇"}
        2 -> {"rank-2", "🥈"}
        3 -> {"rank-3", "🥉"}
        _ -> {"", ""}
      end

      arch_type = Map.get(entry, :architecture, infer_architecture_type(entry.id))
      type_class = "type-#{arch_type}"

      """
      <tr class="#{rank_class}">
        <td><span class="medal">#{medal}</span>#{rank}</td>
        <td><strong>#{entry.id}</strong></td>
        <td><span class="arch-type #{type_class}">#{arch_type}</span></td>
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
    |> Enum.map(fn gen ->
      leader = List.first(gen.leaderboard) || %{id: "N/A", elo: 0}
      top3 = gen.leaderboard
      |> Enum.take(3)
      |> Enum.map(& &1.id)
      |> Enum.join(", ")

      """
      <tr>
        <td>#{gen.generation}</td>
        <td>#{gen.tournament.matches_played}</td>
        <td><strong>#{leader.id}</strong></td>
        <td>#{Float.round(leader.elo, 1)}</td>
        <td>#{top3}</td>
      </tr>
      """
    end)
    |> Enum.join("\n")
  end

  defp generate_architecture_details(architectures) do
    architectures
    |> Enum.sort_by(& &1.elo, :desc)
    |> Enum.map(fn arch ->
      arch_type = Map.get(arch, :architecture, infer_architecture_type(arch.id))
      type_class = "type-#{arch_type}"

      win_pct = Float.round(arch.win_rate * 100, 1)
      loss_pct = if arch.games_played > 0 do
        Float.round(arch.losses / arch.games_played * 100, 1)
      else
        0.0
      end

      """
      <div class="architecture-card">
        <div class="arch-info">
          <div class="arch-header">
            <h3>#{arch.id}</h3>
            <span class="arch-type #{type_class}">#{arch_type}</span>
          </div>
          <p><strong>Elo Rating:</strong> #{Float.round(arch.elo, 1)}</p>
          <p><strong>Generation:</strong> #{Map.get(arch, :generation, 0)}</p>
        </div>
        <div class="arch-stats">
          <div class="stats-grid" style="grid-template-columns: repeat(4, 1fr);">
            <div class="stat-card">
              <div class="stat-value">#{arch.games_played}</div>
              <div class="stat-label">Games</div>
            </div>
            <div class="stat-card">
              <div class="stat-value" style="color: #2e7d32;">#{arch.wins}</div>
              <div class="stat-label">Wins</div>
            </div>
            <div class="stat-card">
              <div class="stat-value" style="color: #c62828;">#{arch.losses}</div>
              <div class="stat-label">Losses</div>
            </div>
            <div class="stat-card">
              <div class="stat-value">#{win_pct}%</div>
              <div class="stat-label">Win Rate</div>
            </div>
          </div>
        </div>
      </div>
      """
    end)
    |> Enum.join("\n")
  end

  defp get_champion_name(data) do
    case List.first(data.final_leaderboard) do
      nil -> "N/A"
      entry -> to_string(entry.id)
    end
  end

  defp infer_architecture_type(id) do
    id_str = to_string(id)
    cond do
      String.contains?(id_str, "mlp") -> :mlp
      String.contains?(id_str, "lstm") -> :lstm
      String.contains?(id_str, "gru") -> :gru
      String.contains?(id_str, "mamba") -> :mamba
      String.contains?(id_str, "attention") -> :attention
      String.contains?(id_str, "jamba") -> :jamba
      true -> :unknown
    end
  end

  # ============================================================================
  # JSON Report Generation
  # ============================================================================

  defp generate_json_report(data) do
    %{
      generated_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      summary: %{
        generations: data.generations,
        total_matches: data.final_stats.matches_played,
        num_architectures: data.final_stats.num_architectures,
        champion: get_champion_name(data)
      },
      leaderboard: Enum.map(data.final_leaderboard, fn entry ->
        %{
          rank: Enum.find_index(data.final_leaderboard, &(&1 == entry)) + 1,
          id: to_string(entry.id),
          architecture: to_string(Map.get(entry, :architecture, infer_architecture_type(entry.id))),
          elo: Float.round(entry.elo, 2),
          win_rate: Float.round(entry.win_rate, 4),
          games_played: entry.games_played,
          wins: entry.wins,
          losses: entry.losses,
          draws: entry.draws
        }
      end),
      history: Enum.map(data.history, fn gen ->
        %{
          generation: gen.generation,
          matches_played: gen.tournament.matches_played,
          leaderboard: Enum.map(gen.leaderboard, fn entry ->
            %{
              id: to_string(entry.id),
              elo: Float.round(entry.elo, 2),
              win_rate: Float.round(entry.win_rate, 4)
            }
          end)
        }
      end)
    }
    |> Jason.encode!(pretty: true)
  end

  # ============================================================================
  # Terminal Report
  # ============================================================================

  defp print_terminal_report(data) do
    Output.puts("")
    Output.puts("=" |> String.duplicate(60))
    Output.puts("ARCHITECTURE LEAGUE RESULTS")
    Output.puts("=" |> String.duplicate(60))
    Output.puts("")

    Output.puts("Summary:")
    Output.puts("  Generations: #{data.generations}")
    Output.puts("  Total Matches: #{data.final_stats.matches_played}")
    Output.puts("  Architectures: #{data.final_stats.num_architectures}")
    Output.puts("")

    Output.puts("Leaderboard:")
    Output.puts("-" |> String.duplicate(60))

    data.final_leaderboard
    |> Enum.with_index(1)
    |> Enum.each(fn {entry, rank} ->
      medal = case rank do
        1 -> "🥇"
        2 -> "🥈"
        3 -> "🥉"
        _ -> "  "
      end

      Output.puts("#{medal} #{rank}. #{entry.id}")
      Output.puts("      Elo: #{Float.round(entry.elo, 1)} | " <>
                  "Win Rate: #{Float.round(entry.win_rate * 100, 1)}% | " <>
                  "Games: #{entry.games_played} (#{entry.wins}W/#{entry.losses}L/#{entry.draws}D)")
    end)

    Output.puts("")
  end
end

# Run the reporter
LeagueReporter.run(System.argv())

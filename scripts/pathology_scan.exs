# Pathology scan — Bradley's 09-05 candidates, measured with corpus
# denominators (E3's law: "too much/too little" needs an expert baseline).
#
# Metrics per game (subject port), grouped by parent dir:
#   - jab chain behavior: jab1 entries/min, jab2-per-jab1, rapid-jab-per-jab1
#     (multi-jab pathology = continuation ratios far above expert)
#   - offstage episodes/min, recovery rate (episode ends onstage/on-ledge
#     vs stock loss), away-drift fraction (offstage frames holding the
#     stick AWAY from stage center)
#   - untouched deaths/game: died without taking ANY damage in the prior
#     ~3s. Percent-at-death is NOT an SD signal (a 0% shine-spike is a
#     legit edgeguard death, Bradley 09-05); dying untouched is.
#
# Usage:
#   mix run scripts/pathology_scan.exs --port 1 'eval_runs/0905_v16a_cleanloss_live/**/*.slp'
#   mix run scripts/pathology_scan.exs --char-id 2 --limit-files 400 \
#     'replays/erickfm_ranked/FOX/extracted/*.slp'
#
# Options:
#   --port N          Pin the subject port (bot convention; overrides char)
#   --char-id N       Resolve subject port per file by CSS character id
#                     (default 2 = Fox) when --port absent
#   --limit-files N   Cap files [400]
#   --concurrency N   [8]
#   --out FILE.md     Write the report
#   --quiet
require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    strict: [port: :integer, char_id: :integer, limit_files: :integer,
             concurrency: :integer, out: :string, quiet: :boolean, max_opp_idle: :float]
  )

# AFK gate (Bradley 09-08: games where the human sat still are not a
# readable sample of the bot). Games whose OPPONENT idles more than this
# fraction of frames are excluded and counted. 1.0 = off.
max_opp_idle = opts[:max_opp_idle] || 1.0

files =
  paths
  |> Enum.flat_map(fn p -> if String.contains?(p, "*"), do: Path.wildcard(p), else: [p] end)
  |> Enum.filter(&String.ends_with?(&1, ".slp"))
  |> Enum.uniq()
  |> Enum.sort()
  |> Enum.take(opts[:limit_files] || 400)

if files == [], do: raise("no .slp matched")

Output.banner("Pathology scan")
Output.config([{"Files", length(files)}, {"Subject", opts[:port] || "char #{opts[:char_id] || 2}"}])

defmodule PathologyScan do
  @moduledoc false

  # Universal subaction states: 44 jab1, 45 jab2, 46 jab3,
  # 47/48/49 rapid-jab start/loop/end.
  @jab1 44
  @jab2 45
  @jab3 46
  @rapid [47, 48, 49]
  # Ledge/cliff action band (CliffCatch..CliffJumpQuick2): being here
  # counts as a successful return from offstage.
  @cliff_states MapSet.new(252..263)

  # Ledge |x| per stage id (competitive six; others fall back to 70).
  @ledge_x %{2 => 63.35, 3 => 87.75, 8 => 56.0, 28 => 77.27, 31 => 68.4, 32 => 85.57}

  def resolve_port(path, opts) do
    cond do
      opts[:port] ->
        {:ok, opts[:port]}

      true ->
        char = opts[:char_id] || 2

        case Peppi.metadata(path) do
          {:ok, meta} ->
            case Enum.filter(meta.players, &(&1.character == char)) do
              [%{port: p}] -> {:ok, p}
              _ -> :skip
            end

          _ ->
            :skip
        end
    end
  end

  def scan(path, opts) do
    with {:ok, port} <- resolve_port(path, opts),
         {:ok, meta} <- Peppi.metadata(path),
         {:ok, %{frames: frames}} when length(frames) > 1800 <- Peppi.parse(path) do
      ledge = Map.get(@ledge_x, meta.stage, 70.0)
      {:ok, walk(frames, port, ledge)}
    else
      _ -> :skip
    end
  end

  defp walk(frames, port, ledge) do
    init = %{
      minutes: length(frames) / 3600,
      jab1: 0, jab2: 0, jab3: 0, rapid: 0,
      prev_action: nil,
      offstage_frames: 0, away_frames: 0,
      episodes: 0, recovered: 0, died_offstage: 0,
      in_episode: false,
      deaths: 0, untouched_deaths: 0,
      prev_stock: nil, prev_pct: 0.0,
      frame_idx: 0, last_hit_frame: -100_000
    }

    init =
      Map.merge(init, %{
        grounded_frames: 0,
        facing_away_frames: 0,
        idle_frames: 0,
        opp_idle_frames: 0,
        wait_entries: 0,
        prev_wait: false
      })

    Enum.reduce(frames, init, fn frame, acc ->
      players = frame.players || %{}

      case Map.get(players, port) do
        nil ->
          acc

        pf ->
          opp = players |> Enum.find(fn {p, _} -> p != port end) |> then(fn
            {_, o} -> o
            nil -> nil
          end)

          acc |> step(pf, ledge) |> facing_step(pf, opp)
      end
    end)
  end

  # Passivity/facing (Bradley 09-07: "turns away and stands there"):
  # among grounded frames, how often the subject faces AWAY from the
  # opponent, and how often it sits in WAIT (action 14).
  defp facing_step(acc, pf, opp) do
    grounded = pf.on_ground == true
    action = trunc(pf.action || 0)

    away =
      grounded and opp != nil and is_number(pf.facing) and is_number(pf.x) and is_number(opp.x) and
        (opp.x - pf.x) * pf.facing < 0

    opp_idle = opp != nil and trunc(opp.action || 0) == 14
    in_wait = action == 14
    # WAIT entries (rising edges): the compounding hypothesis predicts the
    # bot ENTERS standing far more often than experts while its dwell per
    # entry stays expert-like (exit rate calibrated — probe_wait_exit).
    entered = in_wait and not acc.prev_wait

    %{
      acc
      | grounded_frames: acc.grounded_frames + if(grounded, do: 1, else: 0),
        facing_away_frames: acc.facing_away_frames + if(away, do: 1, else: 0),
        idle_frames: acc.idle_frames + if(in_wait, do: 1, else: 0),
        opp_idle_frames: acc.opp_idle_frames + if(opp_idle, do: 1, else: 0),
        wait_entries: acc.wait_entries + if(entered, do: 1, else: 0),
        prev_wait: in_wait
    }
  end

  defp step(acc, pf, ledge) do
    action = trunc(pf.action || 0)
    stock = pf.stock || 0
    pct = pf.percent || 0.0
    x = pf.x || 0.0
    airborne = pf.on_ground == false

    # Rising-edge jab counting
    acc =
      if action != acc.prev_action do
        case action do
          @jab1 -> %{acc | jab1: acc.jab1 + 1}
          @jab2 -> %{acc | jab2: acc.jab2 + 1}
          @jab3 -> %{acc | jab3: acc.jab3 + 1}
          a when a in @rapid ->
            if acc.prev_action in @rapid, do: acc, else: %{acc | rapid: acc.rapid + 1}
          _ -> acc
        end
      else
        acc
      end

    # Track when the subject last TOOK damage (percent rise; respawn
    # resets to 0 don't count as a rise).
    acc =
      if pct > acc.prev_pct do
        %{acc | last_hit_frame: acc.frame_idx}
      else
        acc
      end

    # Death (stock decrement). Untouched death = no damage taken in the
    # prior ~3s (180 frames) — the SD signature. Percent at death is NOT
    # one (0% shine-spikes are legit edgeguard deaths).
    died = acc.prev_stock != nil and stock < acc.prev_stock

    acc =
      if died do
        untouched = acc.frame_idx - acc.last_hit_frame > 180
        %{acc | deaths: acc.deaths + 1, untouched_deaths: acc.untouched_deaths + if(untouched, do: 1, else: 0)}
      else
        acc
      end

    # Offstage episode machine
    offstage = airborne and abs(x) > ledge + 2.0
    on_cliff = MapSet.member?(@cliff_states, action)

    acc =
      cond do
        died and acc.in_episode ->
          %{acc | in_episode: false, died_offstage: acc.died_offstage + 1}

        offstage and not acc.in_episode ->
          %{acc | in_episode: true, episodes: acc.episodes + 1}

        acc.in_episode and (on_cliff or (not offstage and not died)) ->
          %{acc | in_episode: false, recovered: acc.recovered + 1}

        true ->
          acc
      end

    # Away-drift: offstage while holding the stick further out
    away =
      offstage and
        case pf.controller do # same-frame-descriptive: stick in effect while offstage, not a target
          %{main_stick_x: sx} when is_number(sx) -> sx * sign(x) > 0.3
          _ -> false
        end

    %{
      acc
      | prev_action: action,
        prev_stock: stock,
        prev_pct: if(died, do: 0.0, else: pct),
        frame_idx: acc.frame_idx + 1,
        offstage_frames: acc.offstage_frames + if(offstage, do: 1, else: 0),
        away_frames: acc.away_frames + if(away, do: 1, else: 0)
    }
  end

  defp sign(x) when x < 0, do: -1.0
  defp sign(_), do: 1.0

  def summarize(rows) do
    n = length(rows)
    mins = rows |> Enum.map(& &1.minutes) |> Enum.sum()
    sum = fn key -> rows |> Enum.map(&Map.fetch!(&1, key)) |> Enum.sum() end

    jab1 = sum.(:jab1)
    episodes = sum.(:episodes)
    offstage_f = sum.(:offstage_frames)

    %{
      games: n,
      minutes: Float.round(mins, 1),
      jab1_per_min: ratio(jab1, mins),
      jab2_per_jab1: ratio(sum.(:jab2), jab1),
      rapid_per_jab1: ratio(sum.(:rapid), jab1),
      offstage_eps_per_min: ratio(episodes, mins),
      recovery_rate: ratio(sum.(:recovered), episodes),
      died_offstage_per_game: ratio(sum.(:died_offstage), n),
      away_drift_frac: ratio(sum.(:away_frames), offstage_f),
      deaths_per_game: ratio(sum.(:deaths), n),
      untouched_deaths_per_game: ratio(sum.(:untouched_deaths), n),
      facing_away_frac: ratio(sum.(:facing_away_frames), sum.(:grounded_frames)),
      idle_frac: ratio(sum.(:idle_frames), mins * 3600),
      wait_entries_per_min: ratio(sum.(:wait_entries), mins),
      wait_dwell_frames: ratio(sum.(:idle_frames), sum.(:wait_entries))
    }
  end

  defp ratio(_, 0), do: 0.0
  defp ratio(_, +0.0), do: 0.0
  defp ratio(a, b), do: Float.round(a / b, 3)
end

conc = opts[:concurrency] || 8

groups =
  files
  |> Task.async_stream(fn p ->
    # Group by the run dir, not Slippi's shared "2026-09-Mainline"
    # month-dir basename (which merges every arm into one group).
    group =
      p
      |> Path.dirname()
      |> Path.split()
      |> Enum.reject(&Regex.match?(~r/^20\d\d-\d\d(-Mainline)?$/, &1))
      |> List.last()

    {group, PathologyScan.scan(p, opts)}
  end,
    max_concurrency: conc,
    timeout: 120_000,
    on_timeout: :kill_task
  )
  |> Enum.reduce(%{}, fn
    {:ok, {group, {:ok, row}}}, acc ->
      opp_idle_frac = row.opp_idle_frames / max(row.minutes * 3600, 1)

      if opp_idle_frac > max_opp_idle do
        Map.update(acc, {:afk, group}, 1, &(&1 + 1))
      else
        Map.update(acc, group, [row], &[row | &1])
      end

    _, acc ->
      acc
  end)

{afk, groups} = Enum.split_with(groups, fn {k, _} -> match?({:afk, _}, k) end)

Enum.each(afk, fn {{:afk, group}, n} ->
  Output.warning("#{group}: #{n} game(s) EXCLUDED — opponent idle > #{max_opp_idle} (AFK)")
end)

groups = Map.new(groups)

header =
  "| group | games | min | jab1/min | jab2/jab1 | rapid/jab1 | offstage eps/min | recovery rate | died-offstage/game | away-drift frac | deaths/game | untouched-deaths/game | facing-away frac | idle(WAIT) frac | WAIT entries/min | WAIT dwell (frames) |"

sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

lines =
  groups
  |> Enum.sort()
  |> Enum.map(fn {group, rows} ->
    s = PathologyScan.summarize(rows)

    "| #{group} | #{s.games} | #{s.minutes} | #{s.jab1_per_min} | #{s.jab2_per_jab1} | " <>
      "#{s.rapid_per_jab1} | #{s.offstage_eps_per_min} | #{s.recovery_rate} | " <>
      "#{s.died_offstage_per_game} | #{s.away_drift_frac} | #{s.deaths_per_game} | #{s.untouched_deaths_per_game} | " <>
      "#{s.facing_away_frac} | #{s.idle_frac} | #{s.wait_entries_per_min} | #{s.wait_dwell_frames} |"
  end)

report = Enum.join([header, sep | lines], "\n")
IO.puts("\n" <> report <> "\n")

if opts[:out] do
  File.mkdir_p!(Path.dirname(opts[:out]))
  File.write!(opts[:out], "# Pathology scan — #{DateTime.utc_now() |> DateTime.to_iso8601()}\n\n" <> report <> "\n")
  Output.success("Wrote #{opts[:out]}")
end

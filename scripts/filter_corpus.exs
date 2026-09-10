# Corpus quality filter + content-hash dedupe (V2_PREP item 2).
#
# Two-tier pass over a replay corpus, emitting a FILTERED SYMLINK DIR
# that train.exs consumes via --replays, plus a manifest with a
# per-file verdict (kept files AND rejects, with reasons — GOTCHA #102's
# law: never delete the evidence, and never mutate the source corpus).
#
# Tier A (cheap, metadata + file bytes — always on):
#   - parseable metadata (corrupt .slp rejected)
#   - exactly 2 players (no teams/FFA)
#   - duration >= --min-minutes (default 1.0; slippi-ai floor)
#   - CONTENT-HASH dedupe (sha256 of file bytes): the same game filed
#     under both players' character dirs is byte-identical — first
#     path seen wins, later copies rejected as :duplicate. Paths encode
#     seating accidents; bytes are the truth (SubjectResolver's law).
#   - optional --stages competitive allowlist (FoD/PS/YS/DL/BF/FD)
#
# Tier B (--deep, full frame parse — the slippi-ai list proper):
#   - total damage across both players >= --min-damage (default 100.0;
#     rejects handwarmers/idle farms)
#   - has-winner: someone lost >= 1 stock AND final stocks differ
#     (rejects no-contest quits / timer idles)
#   - physical sanity: percents within [0, 999] (corrupt-frame guard)
#
# Usage (NOT while a training beam is live — mix rule):
#   mix run scripts/filter_corpus.exs \
#     --replays replays/erickfm_ranked/FOX \
#     --replays replays/erickfm_ranked/partners/MARTH \
#     --out replays/erickfm_ranked/v2_filtered \
#     --deep --stages competitive \
#     2>&1 | tee logs/filter_corpus_v2.log
#
# Options:
#   --replays DIR      Source dir (repeatable; recursive *.slp). Required.
#   --out DIR          Output dir for symlinks + manifest. Required.
#   --deep             Enable tier B (full parse; ~8x parallel)
#   --min-minutes F    Tier A duration floor in minutes [1.0]
#   --min-damage F     Tier B damage floor [100.0]
#   --stages LIST      "competitive" | comma-sep stage ids | "all" [all]
#   --max-files N      Limit input files (testing)
#   --concurrency N    Parallel workers [8]
#   --dry-run          Score + manifest only, no symlinks
#   --quiet            Errors/summary only
#
# Output layout under --out:
#   <name>.slp -> ../../..(source)   flat symlinks, collision-suffixed
#   manifest.jsonl                   one row per INPUT file (verdict)
#   REPORT.md                        summary table
#
# Resumable: re-running with the same --out skips input paths already
# in the manifest (append-only jsonl).

if "--quiet" in System.argv(), do: Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      replays: :keep,
      out: :string,
      deep: :boolean,
      min_minutes: :float,
      min_damage: :float,
      stages: :string,
      max_files: :integer,
      concurrency: :integer,
      dry_run: :boolean,
      quiet: :boolean
    ]
  )

replay_dirs = Keyword.get_values(opts, :replays)
if replay_dirs == [], do: raise("--replays DIR required (repeatable)")
out_dir = opts[:out] || raise("--out DIR required")
deep? = opts[:deep] || false
min_frames = trunc((opts[:min_minutes] || 1.0) * 60 * 60)
min_damage = opts[:min_damage] || 100.0
concurrency = opts[:concurrency] || 8
dry_run? = opts[:dry_run] || false

# Competitive-stage ids match the compact embedding's allowlist (CLAUDE.md):
# FoD 2, PS 3, YS 8, DL 28, BF 31, FD 32.
stage_allow =
  case opts[:stages] do
    nil -> nil
    "all" -> nil
    "competitive" -> MapSet.new([2, 3, 8, 28, 31, 32])
    list -> list |> String.split(",") |> Enum.map(&String.to_integer(String.trim(&1))) |> MapSet.new()
  end

Output.set_verbosity(if(opts[:quiet], do: 0, else: 1))
Output.banner("Corpus quality filter + dedupe")

files =
  replay_dirs
  |> Enum.flat_map(fn dir ->
    dir |> Path.expand() |> then(&Path.wildcard(Path.join(&1, "**/*.slp")))
  end)
  |> Enum.uniq()
  |> Enum.sort()
  |> then(fn fs -> if opts[:max_files], do: Enum.take(fs, opts[:max_files]), else: fs end)

if files == [], do: raise("no .slp files under #{inspect(replay_dirs)}")

File.mkdir_p!(out_dir)
manifest_path = Path.join(out_dir, "manifest.jsonl")

# Resume: skip inputs already verdicted (append-only manifest).
already =
  if File.exists?(manifest_path) do
    manifest_path
    |> File.stream!()
    |> Enum.map(&(&1 |> Jason.decode!() |> Map.fetch!("path")))
    |> MapSet.new()
  else
    MapSet.new()
  end

todo = Enum.reject(files, &MapSet.member?(already, &1))

Output.config([
  {"Sources", Enum.join(replay_dirs, ", ")},
  {"Files", "#{length(files)} (#{length(todo)} to score, #{MapSet.size(already)} resumed)"},
  {"Deep pass", deep?},
  {"Min duration", "#{min_frames} frames"},
  {"Min damage", if(deep?, do: min_damage, else: "n/a (tier A only)")},
  {"Stages", opts[:stages] || "all"},
  {"Out", out_dir <> if(dry_run?, do: " (dry-run)", else: "")}
])

defmodule CorpusFilter do
  @moduledoc false

  # Tier A + optional tier B verdict for one file. Returns
  # {:keep, info} | {:reject, reason, info}. The sha256 is computed for
  # every parseable file (dedupe happens in the serial reduce below —
  # hashing parallelizes, the first-seen decision cannot).
  def score(path, cfg) do
    with {:ok, meta} <- Peppi.metadata(path),
         :ok <- check_players(meta),
         :ok <- check_duration(meta, cfg.min_frames),
         :ok <- check_stage(meta, cfg.stage_allow),
         {:ok, sha} <- content_hash(path),
         :ok <- deep_checks(path, cfg) do
      {:keep, %{sha256: sha, stage: meta.stage, frames: meta.duration_frames}}
    else
      {:reject, reason} -> {:reject, reason, %{}}
      {:error, reason} -> {:reject, {:unparseable, format_error(reason)}, %{}}
    end
  end

  defp check_players(%{players: players}) do
    if length(players) == 2, do: :ok, else: {:reject, {:not_1v1, length(players)}}
  end

  defp check_duration(%{duration_frames: f}, min) when is_integer(f) do
    if f >= min, do: :ok, else: {:reject, {:too_short, f}}
  end

  defp check_duration(_, _), do: {:reject, {:too_short, nil}}

  defp check_stage(_meta, nil), do: :ok

  defp check_stage(%{stage: stage}, allow) do
    if MapSet.member?(allow, stage), do: :ok, else: {:reject, {:stage, stage}}
  end

  defp content_hash(path) do
    hash =
      path
      |> File.stream!(2 * 1024 * 1024)
      |> Enum.reduce(:crypto.hash_init(:sha256), &:crypto.hash_update(&2, &1))
      |> :crypto.hash_final()
      |> Base.encode16(case: :lower)

    {:ok, hash}
  rescue
    e -> {:error, e}
  end

  defp deep_checks(_path, %{deep: false}), do: :ok

  defp deep_checks(path, cfg) do
    case Peppi.parse(path) do
      {:ok, %{frames: frames}} when frames != [] ->
        stats = frame_stats(frames)

        cond do
          stats.insane -> {:reject, :physical_insanity}
          stats.total_damage < cfg.min_damage -> {:reject, {:low_damage, Float.round(stats.total_damage, 1)}}
          not stats.has_winner -> {:reject, :no_winner}
          true -> :ok
        end

      {:ok, _} ->
        {:reject, {:unparseable, "no frames"}}

      {:error, reason} ->
        {:reject, {:unparseable, format_error(reason)}}
    end
  end

  # One pass over the frame list: per-port damage taken (sum of percent
  # rises — resets to 0 on stock loss don't contribute), first/last
  # stocks, sanity bounds.
  defp frame_stats(frames) do
    init = %{last_pct: %{}, damage: 0.0, insane: false}

    acc =
      Enum.reduce(frames, init, fn frame, acc ->
        Enum.reduce(frame.players || %{}, acc, fn {port, pf}, acc ->
          pct = pf.percent || 0.0
          prev = Map.get(acc.last_pct, port, pct)
          rise = if pct > prev, do: pct - prev, else: 0.0

          %{
            acc
            | last_pct: Map.put(acc.last_pct, port, pct),
              damage: acc.damage + rise,
              insane: acc.insane or pct < 0.0 or pct > 999.0
          }
        end)
      end)

    final = List.last(frames)
    first = List.first(frames)

    stocks = fn frame ->
      Map.new(frame.players || %{}, fn {port, pf} -> {port, pf.stock || 0} end)
    end

    first_stocks = stocks.(first)
    final_stocks = stocks.(final)

    lost_stock? =
      Enum.any?(final_stocks, fn {port, s} -> s < Map.get(first_stocks, port, s) end)

    stock_vals = Map.values(final_stocks)
    decided? = length(Enum.uniq(stock_vals)) > 1

    %{
      total_damage: acc.damage,
      has_winner: lost_stock? and decided?,
      insane: acc.insane
    }
  end

  def format_error(reason), do: reason |> inspect() |> String.slice(0, 120)

  # Flat symlink name: basename, prefixed with a short source-dir tag on
  # collision (two dirs both holding "Game_X.slp").
  def link!(path, out_dir) do
    base = Path.basename(path)
    dest = Path.join(out_dir, base)

    dest =
      if File.exists?(dest) or match?({:ok, _}, File.read_link(dest)) do
        prefix = path |> Path.dirname() |> Path.basename() |> String.slice(0, 12)
        Path.join(out_dir, "#{prefix}__#{base}")
      else
        dest
      end

    case File.read_link(dest) do
      {:ok, _} -> :exists
      _ -> File.ln_s!(Path.expand(path), dest)
    end
  end
end

cfg = %{deep: deep?, min_frames: min_frames, min_damage: min_damage, stage_allow: stage_allow}

# Parallel scoring; serial dedupe + manifest append (first-seen-wins needs
# order, and jsonl appends must not interleave).
manifest = File.open!(manifest_path, [:append, :utf8])

# Seed seen-hashes from a resumed manifest so dedupe survives restarts.
seen_hashes =
  if MapSet.size(already) > 0 do
    manifest_path
    |> File.stream!()
    |> Enum.reduce(MapSet.new(), fn line, acc ->
      case Jason.decode!(line) do
        %{"verdict" => "keep", "sha256" => sha} when is_binary(sha) -> MapSet.put(acc, sha)
        _ -> acc
      end
    end)
  else
    MapSet.new()
  end

total = length(todo)
t0 = System.monotonic_time(:millisecond)

{stats, _seen} =
  todo
  |> Task.async_stream(fn path -> {path, CorpusFilter.score(path, cfg)} end,
    max_concurrency: concurrency,
    timeout: 120_000,
    on_timeout: :kill_task,
    ordered: true
  )
  |> Enum.reduce({%{kept: 0, dup: 0, rejects: %{}, done: 0}, seen_hashes}, fn
    {:ok, {path, result}}, {stats, seen} ->
      stats = %{stats | done: stats.done + 1}

      if rem(stats.done, 100) == 0 do
        rate = stats.done / max(System.monotonic_time(:millisecond) - t0, 1) * 1000
        eta = if rate > 0, do: trunc((total - stats.done) / rate), else: 0
        IO.write(:stderr, "\r  Scoring #{stats.done}/#{total} | kept #{stats.kept} | #{Float.round(rate, 1)}/s | ETA #{div(eta, 60)}m#{rem(eta, 60)}s\e[K")
      end

      case result do
        {:keep, %{sha256: sha} = info} ->
          if MapSet.member?(seen, sha) do
            IO.puts(manifest, Jason.encode!(%{path: path, verdict: "duplicate", sha256: sha}))
            {%{stats | dup: stats.dup + 1}, seen}
          else
            unless opts[:dry_run], do: CorpusFilter.link!(path, out_dir)
            IO.puts(manifest, Jason.encode!(Map.merge(%{path: path, verdict: "keep"}, info)))
            {%{stats | kept: stats.kept + 1}, MapSet.put(seen, sha)}
          end

        {:reject, reason, _info} ->
          key = reason |> then(fn
            {tag, _detail} -> tag
            tag -> tag
          end)

          IO.puts(manifest, Jason.encode!(%{path: path, verdict: "reject", reason: CorpusFilter.format_error(reason)}))
          {%{stats | rejects: Map.update(stats.rejects, key, 1, &(&1 + 1))}, seen}
      end

    {:exit, reason}, {stats, seen} ->
      IO.puts(manifest, Jason.encode!(%{path: "unknown", verdict: "reject", reason: "worker exit: #{inspect(reason)}"}))
      {%{stats | rejects: Map.update(stats.rejects, :worker_exit, 1, &(&1 + 1))}, seen}
  end)

File.close(manifest)
IO.write(:stderr, "\n")

elapsed = (System.monotonic_time(:millisecond) - t0) / 1000
rejected = stats.rejects |> Map.values() |> Enum.sum()

report = """
# Corpus filter report — #{DateTime.utc_now() |> DateTime.to_iso8601()}

Sources: #{Enum.join(replay_dirs, ", ")}
Deep pass: #{deep?} · min #{min_frames} frames#{if deep?, do: " · min #{min_damage} dmg", else: ""} · stages #{opts[:stages] || "all"}

| verdict | files |
|---|---:|
| **kept** | #{stats.kept} |
| duplicate (content hash) | #{stats.dup} |
#{stats.rejects |> Enum.sort_by(fn {_, n} -> -n end) |> Enum.map(fn {k, n} -> "| reject: #{k} | #{n} |" end) |> Enum.join("\n")}

Scored #{stats.done} files in #{Float.round(elapsed, 1)}s (#{MapSet.size(already)} resumed from a prior run).
Manifest: manifest.jsonl (one row per input file — rejects stay recorded).
"""

File.write!(Path.join(out_dir, "REPORT.md"), report)
Output.puts(report)
Output.success("Filtered corpus at #{out_dir} (#{stats.kept} kept / #{stats.dup} dups / #{rejected} rejected)")

# Situation-index CLI (DATA_FLYWHEEL_DESIGN 2026-07-23, stage B1 v2 / P5).
#
# Build an embedding index over a corpus once, then answer "find me k
# human-played instances of THIS situation" in seconds — the retrieval
# upgrade over find_situations.exs' coarse bucket-grep. The build streams
# one file at a time (peak RAM O(file) — the DATA_SCALING flatness
# property), so corpus size is disk-bound, not RAM-bound.
#
#   # Build (idempotent — rerun extends with new files only):
#   mix run scripts/situation_index.exs build --index cache/situation_index/mewtwo \
#     --char mewtwo 'corpus/archive/mewtwo/*.slp'
#
#   # Query a flubbed moment (or a gap id) against the corpus:
#   mix run scripts/situation_index.exs query --index cache/situation_index/mewtwo \
#     --like probes/newera8/r16/r13/plain/p1/Game_X.slp:4180 --k 40
#   mix run scripts/situation_index.exs query --index ... --gap g_1a2b3c4d
#
# Build options:
#   --index DIR         Index dir (required)
#   --char NAME         Autodetect subject port per file by character
#                       (skips files without it); else --port N (default 1)
#   --min-frame N       Skip frames before this (default 300)
#
# Query options:
#   --index DIR         Index dir (required)
#   --like SLP:FRAME    The moment to match (subject --port, default 1)
#   --gap ID            Take the moment from this gap-ledger entry
#                       (marks it "mined" after a successful query)
#   --k N               Results (default 40)
#   --include-self      Keep matches from the query's own replay
#   --out PATH          Situations JSON (default logs/situations_<ts>.json)
#   --manifest-out PATH Also write scenario-manifest entries
#   --gaps PATH         Gap ledger (default scenarios/gaps.json)

require Logger

alias ExPhil.Data.SituationIndex
alias ExPhil.Eval.GapLedger
alias ExPhil.Training.Output

{opts, argv, _} =
  OptionParser.parse(System.argv(),
    strict: [
      index: :string,
      char: :string,
      port: :integer,
      min_frame: :integer,
      like: :string,
      gap: :string,
      k: :integer,
      include_self: :boolean,
      out: :string,
      manifest_out: :string,
      gaps: :string,
      quiet: :boolean
    ]
  )

if opts[:quiet], do: Logger.configure(level: :warning)

{mode, rest} =
  case argv do
    ["build" | rest] -> {:build, rest}
    ["query" | rest] -> {:query, rest}
    _ -> {nil, []}
  end

unless mode && opts[:index] do
  Output.error("Usage: situation_index.exs build|query --index DIR ...")
  System.halt(1)
end

index_dir = Path.expand(opts[:index])
gaps_path = opts[:gaps] || "scenarios/gaps.json"

case mode do
  :build ->
    slp_paths =
      rest
      |> Enum.map(&Path.expand/1)
      |> Enum.flat_map(&Path.wildcard/1)
      |> Enum.uniq()

    if slp_paths == [] do
      Output.error("No replays matched")
      System.halt(1)
    end

    Output.banner("Situation Index — build")

    Output.config([
      {"Index", index_dir},
      {"Corpus", length(slp_paths)},
      {"Subject", opts[:char] || "port #{opts[:port] || 1}"},
      {"Existing", inspect(SituationIndex.stats(index_dir))}
    ])

    t0 = System.monotonic_time(:millisecond)

    progress = fn done, total, path ->
      if rem(done, 10) == 0 do
        IO.write(:stderr, "\r  #{done}/#{total} #{Path.basename(path)}\e[K")
      end
    end

    {:ok, stats} =
      SituationIndex.build(slp_paths, index_dir,
        char: opts[:char],
        port: opts[:port] || 1,
        min_frame: opts[:min_frame] || 300,
        progress: progress
      )

    IO.write(:stderr, "\r\e[K")
    wall = Float.round((System.monotonic_time(:millisecond) - t0) / 1000, 1)

    Output.puts(
      "indexed #{stats.files} file(s) / #{stats.frames} frames " <>
        "(#{stats.skipped} skipped) in #{wall}s"
    )

    # Memory-flatness evidence (DATA_SCALING step 2): peak RSS of this
    # build process — should be ~constant in corpus size (O(largest file)).
    case File.read("/proc/self/status") do
      {:ok, status} ->
        case Regex.run(~r/VmHWM:\s+(\d+) kB/, status) do
          [_, kb] ->
            Output.puts("peak RSS (VmHWM): #{Float.round(String.to_integer(kb) / 1_048_576, 2)} GB")

          _ ->
            :ok
        end

      _ ->
        :ok
    end

    Output.success("index -> #{index_dir} (#{inspect(SituationIndex.stats(index_dir))})")

  :query ->
    {slp, frame, gap_id} =
      cond do
        opts[:like] ->
          case String.split(opts[:like], ":", parts: 2) do
            [s, f] -> {Path.expand(s), String.to_integer(f), nil}
            _ -> Output.error("--like must be SLP:FRAME") && System.halt(1)
          end

        opts[:gap] ->
          ledger = GapLedger.load(gaps_path)

          case Enum.find(ledger["gaps"] || [], &(&1["id"] == opts[:gap])) do
            %{"slp" => s, "frame" => f} = _gap when is_binary(s) and is_integer(f) ->
              {Path.expand(s), f, opts[:gap]}

            _ ->
              Output.error("gap #{opts[:gap]} not found or has no (slp, frame)")
              System.halt(1)
          end

        true ->
          Output.error("query needs --like or --gap")
          System.halt(1)
      end

    Output.banner("Situation Index — query")

    Output.config([
      {"Index", inspect(SituationIndex.stats(index_dir))},
      {"Moment", "#{Path.basename(slp)}:#{frame}"},
      {"k", opts[:k] || 40}
    ])

    case SituationIndex.embed_moment(slp, frame, port: opts[:port] || 1) do
      {:ok, q} ->
        results =
          SituationIndex.query(index_dir, q,
            k: opts[:k] || 40,
            exclude_slp: if(opts[:include_self], do: nil, else: slp)
          )

        for r <- Enum.take(results, 10) do
          Output.puts("  #{Float.round(r.score, 3)}  #{Path.basename(r.slp)}:#{r.frame}")
        end

        if length(results) > 10, do: Output.puts("  ... #{length(results) - 10} more")

        ts = Calendar.strftime(NaiveDateTime.local_now(), "%Y%m%d_%H%M%S")
        out_path = opts[:out] || "logs/situations_#{ts}.json"
        File.mkdir_p!(Path.dirname(out_path))

        File.write!(
          out_path,
          Jason.encode!(
            %{
              "query" => %{"slp" => slp, "frame" => frame},
              "count" => length(results),
              "situations" =>
                Enum.map(results, fn r ->
                  %{"slp" => r.slp, "frame" => r.frame, "score" => r.score}
                end)
            },
            pretty: true
          )
        )

        Output.success("situations -> #{out_path}")

        if opts[:manifest_out] do
          entries =
            Enum.map(results, fn r ->
              %{
                "slp" => r.slp,
                "frame" => r.frame,
                "type" => "retrieved",
                "note" => "cos=#{r.score} of #{Path.basename(slp)}:#{frame}"
              }
            end)

          File.mkdir_p!(Path.dirname(opts[:manifest_out]))
          File.write!(opts[:manifest_out], Jason.encode!(%{"entries" => entries}, pretty: true))
          Output.success("manifest (#{length(entries)}) -> #{opts[:manifest_out]}")
        end

        if gap_id do
          ledger = GapLedger.load(gaps_path)
          ledger |> GapLedger.set_status(gap_id, "mined") |> GapLedger.save(gaps_path)
          Output.puts("marked gap #{gap_id} -> mined")
        end

      {:error, reason} ->
        Output.error("cannot embed query moment: #{inspect(reason)}")
        System.halt(1)
    end
end

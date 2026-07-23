defmodule ExPhil.Data.SituationIndex do
  @moduledoc """
  Embedding-based situation retrieval over replay corpora
  (DATA_FLYWHEEL_DESIGN 2026-07-23, stage B1 v2) — the richer successor to
  `find_situations.exs`' bucket-grep.

  Given a moment the bot flubbed, find the k nearest human-played
  situations in a corpus and what they did next. The corpus is embedded
  ONCE into per-file f16 shards on disk; queries brute-force cosine top-k
  over the shards (fine to ~10M frames on this GPU; HNSW only if ever
  needed).

  ## Streaming build = the data-scaling shard infrastructure

  `build/3` processes ONE file at a time: parse → embed (batched) → write
  shard → discard. Peak RAM is O(largest file), flat in corpus size — the
  memory-flatness the data-scaling analysis (DATA_SCALING_2026-07-22.md)
  requires. The shard format deliberately carries per-frame refs so the
  training side (dagger `--stream-chunk-size`) can later consume the same
  shards; a compact-targets field is the planned extension for that.

  ## Layout

      <index_dir>/manifest.json      # dim, dtype, embed fingerprint, shard list
      <index_dir>/shard_00000.bin    # term_to_binary(%{emb_bin, n, dim, frames})

  Shard `frames` are the replay's in-game frame numbers (same length as
  rows in `emb_bin`); the source .slp path lives in the manifest entry.

  Embeddings use the DEFAULT embed config (name_id 0, no prev-action) —
  a *situation* fingerprint, not a policy input. `query/3` embeds with the
  same config and asserts the manifest dim matches.
  """

  alias ExPhil.Data.Peppi
  alias ExPhil.Embeddings

  require Logger

  @embed_batch 1024
  @manifest "manifest.json"

  # ==========================================================================
  # Build
  # ==========================================================================

  @doc """
  Build (or extend) an index over `slp_paths` into `index_dir`.

  ## Options
    - `:port` — subject port (default 1)
    - `:char` — autodetect the subject port per file by character name
      substring (e.g. "mewtwo", "fox"); overrides `:port` when found, file
      skipped when the character isn't present
    - `:min_frame` — drop frames earlier than this (default 300; prefix room
      so every indexed situation is drillable)
    - `:progress` — fn(done, total, path) or nil

  Returns `{:ok, %{files: n, frames: n, skipped: n}}`. Files already in the
  manifest are skipped (idempotent extend).
  """
  def build(slp_paths, index_dir, opts \\ []) do
    File.mkdir_p!(index_dir)
    manifest = load_manifest(index_dir)
    known = MapSet.new(manifest["shards"], & &1["source_slp"])
    min_frame = Keyword.get(opts, :min_frame, 300)
    progress = Keyword.get(opts, :progress)
    total = length(slp_paths)

    {manifest, stats} =
      slp_paths
      |> Enum.with_index()
      |> Enum.reduce({manifest, %{files: 0, frames: 0, skipped: 0}}, fn {path, i},
                                                                        {manifest, stats} ->
        if progress, do: progress.(i, total, path)
        path = Path.expand(path)

        cond do
          MapSet.member?(known, path) ->
            {manifest, %{stats | skipped: stats.skipped + 1}}

          true ->
            case build_one(path, index_dir, manifest, min_frame, opts) do
              {:ok, manifest, n} ->
                {manifest, %{stats | files: stats.files + 1, frames: stats.frames + n}}

              :skip ->
                {manifest, %{stats | skipped: stats.skipped + 1}}
            end
        end
      end)

    save_manifest(index_dir, manifest)
    {:ok, stats}
  end

  defp build_one(path, index_dir, manifest, min_frame, opts) do
    with {:ok, port} <- subject_port(path, opts),
         {:ok, replay} <- Peppi.parse(path) do
      frames =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
        |> Enum.filter(fn f -> f.game_state.frame >= min_frame end)

      if frames == [] do
        :skip
      else
        states = Enum.map(frames, & &1.game_state)
        frame_nos = Enum.map(frames, & &1.game_state.frame)

        emb =
          states
          |> Enum.chunk_every(@embed_batch)
          |> Enum.map(fn chunk -> Embeddings.Game.embed_states_fast(chunk, port) end)
          |> Nx.concatenate()

        {n, dim} = Nx.shape(emb)

        emb_bin =
          emb
          |> Nx.as_type(:f16)
          |> Nx.backend_copy(Nx.BinaryBackend)
          |> Nx.to_binary()

        seq = length(manifest["shards"])
        shard_file = "shard_#{String.pad_leading(to_string(seq), 5, "0")}.bin"

        payload = %{emb_bin: emb_bin, n: n, dim: dim, frames: frame_nos}
        File.write!(Path.join(index_dir, shard_file), :erlang.term_to_binary(payload))

        entry = %{
          "file" => shard_file,
          "source_slp" => path,
          "port" => port,
          "frames" => n
        }

        manifest =
          manifest
          |> Map.put("dim", dim)
          |> Map.update!("shards", &(&1 ++ [entry]))

        {:ok, manifest, n}
      end
    else
      _ -> :skip
    end
  end

  # Subject port: explicit :port, or :char autodetect from metadata.
  defp subject_port(path, opts) do
    case Keyword.get(opts, :char) do
      nil ->
        {:ok, Keyword.get(opts, :port, 1)}

      char ->
        want = String.downcase(char)

        case Peppi.metadata(path) do
          {:ok, %{players: players}} when is_list(players) ->
            case Enum.find(players, fn p ->
                   String.contains?(String.downcase(p.character_name || ""), want)
                 end) do
              %{port: p} -> {:ok, p}
              _ -> :skip
            end

          _ ->
            :skip
        end
    end
  end

  # ==========================================================================
  # Query
  # ==========================================================================

  @doc """
  Embed the situation at `{slp, frame}` (subject `:port`, default 1) and
  return its query vector. The moment must exist in the replay.
  """
  def embed_moment(slp, frame_no, opts \\ []) do
    port = Keyword.get(opts, :port, 1)

    with {:ok, replay} <- Peppi.parse(Path.expand(slp)) do
      frame =
        replay
        |> Peppi.to_training_frames(player_port: port, opponent_port: if(port == 1, do: 2, else: 1))
        |> Enum.find(fn f -> f.game_state.frame == frame_no end)

      case frame do
        nil -> {:error, :frame_not_found}
        f -> {:ok, Embeddings.Game.embed_states_fast([f.game_state], port) |> Nx.squeeze(axes: [0])}
      end
    end
  end

  @doc """
  Brute-force cosine top-k over the index. `query_vec` is a `{dim}` tensor.
  Returns `[%{slp:, frame:, score:}]` best-first.

  Streams shard-by-shard (peak RAM O(shard)); per-shard candidates are
  merged, then TEMPORALLY SPACED — a held situation is hundreds of
  near-identical consecutive frames, and without spacing one moment floods
  the whole top-k (observed on the first live query). Per replay, a match
  within `:spacing` frames of an already-kept better match is dropped;
  shards over-fetch 4k candidates so k results survive the spacing.

  ## Options
    - `:k` — results (default 40)
    - `:spacing` — min frames between kept matches in one replay
      (default 240, the ScenarioScan curation gap; 0 disables)
    - `:exclude_slp` — drop matches from this replay (self-retrieval)
  """
  def query(index_dir, query_vec, opts \\ []) do
    k = Keyword.get(opts, :k, 40)
    spacing = Keyword.get(opts, :spacing, 240)
    exclude = Keyword.get(opts, :exclude_slp) && Path.expand(Keyword.get(opts, :exclude_slp))
    manifest = load_manifest(index_dir)
    dim = manifest["dim"]

    if dim == nil, do: raise(ArgumentError, "empty index at #{index_dir}")

    {qdim} = Nx.shape(query_vec)

    if qdim != dim,
      do: raise(ArgumentError, "query dim #{qdim} != index dim #{dim} (embed config mismatch?)")

    q = Nx.divide(query_vec, Nx.add(l2(query_vec), 1.0e-8))
    fetch = if spacing > 0, do: k * 4, else: k

    manifest["shards"]
    |> Enum.reject(fn s -> exclude && Path.expand(s["source_slp"]) == exclude end)
    |> Enum.flat_map(fn shard -> shard_topk(index_dir, shard, q, dim, fetch) end)
    |> Enum.sort_by(&(-&1.score))
    |> space_matches(spacing)
    |> Enum.take(k)
  end

  # Best-first greedy spacing: keep a match unless a better (earlier-kept)
  # match from the same replay sits within `spacing` frames.
  defp space_matches(matches, spacing) when spacing <= 0, do: matches

  defp space_matches(matches, spacing) do
    matches
    |> Enum.reduce({[], %{}}, fn m, {kept, by_slp} ->
      frames = Map.get(by_slp, m.slp, [])

      if Enum.any?(frames, fn f -> abs(f - m.frame) < spacing end) do
        {kept, by_slp}
      else
        {[m | kept], Map.put(by_slp, m.slp, [m.frame | frames])}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp shard_topk(index_dir, shard, q, dim, k) do
    payload =
      Path.join(index_dir, shard["file"])
      |> File.read!()
      |> :erlang.binary_to_term()

    n = payload.n
    kk = min(k, n)

    emb =
      payload.emb_bin
      |> Nx.from_binary(:f16, backend: Nx.BinaryBackend)
      |> Nx.reshape({n, dim})
      |> Nx.as_type(:f32)

    # Cosine: normalize rows, dot with the unit query.
    norms = emb |> Nx.pow(2) |> Nx.sum(axes: [1], keep_axes: true) |> Nx.sqrt()
    scores = emb |> Nx.divide(Nx.add(norms, 1.0e-8)) |> Nx.dot(q)

    {top_scores, top_idx} = Nx.top_k(scores, k: kk)
    frames_arr = :array.from_list(payload.frames)

    Enum.zip(Nx.to_flat_list(top_scores), Nx.to_flat_list(top_idx))
    |> Enum.map(fn {score, idx} ->
      %{
        slp: shard["source_slp"],
        frame: :array.get(idx, frames_arr),
        score: Float.round(score * 1.0, 4)
      }
    end)
  end

  defp l2(v), do: v |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt()

  # ==========================================================================
  # Manifest
  # ==========================================================================

  def load_manifest(index_dir) do
    case File.read(Path.join(index_dir, @manifest)) do
      {:ok, bin} ->
        case Jason.decode(bin) do
          {:ok, %{"shards" => _} = m} -> m
          _ -> new_manifest()
        end

      _ ->
        new_manifest()
    end
  end

  defp new_manifest, do: %{"dim" => nil, "dtype" => "f16", "shards" => []}

  defp save_manifest(index_dir, manifest) do
    File.write!(Path.join(index_dir, @manifest), Jason.encode!(manifest, pretty: true))
  end

  @doc "Index stats from the manifest."
  def stats(index_dir) do
    m = load_manifest(index_dir)

    %{
      files: length(m["shards"]),
      frames: m["shards"] |> Enum.map(& &1["frames"]) |> Enum.sum(),
      dim: m["dim"]
    }
  end
end

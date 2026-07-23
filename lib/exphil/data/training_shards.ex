defmodule ExPhil.Data.TrainingShards do
  @moduledoc """
  Streaming training data as per-file embedding shards (DATA_SCALING
  2026-07-22 / #33). The all-in-RAM drill path materializes the whole
  `{num_frames, embed_size}` f32 embedding tensor (~5.1 KB/frame) plus the
  full GameState frame list up front — capping at ~5.6M frames on this box.
  The five-char corpus needs ~10x that.

  This builds embeddings ONE source file at a time (parse → process → embed
  → f16 shard → discard), so peak RAM is O(largest file), flat in corpus
  size — the flatness `ExPhil.Data.SituationIndex` already proved for the
  situation index (1.48/1.77 GB at 65k/1.77M frames). Here the shards also
  carry the training TARGETS (packed discretized actions), per-frame
  sampling weights, and probe labels, so the whole drill can train off them
  across epochs without re-embedding.

  ## Consuming the shards

  `stream_batches/2` shuffles shard order each epoch, concatenates a rolling
  window of `:open_shards` shards into a temporary in-memory `Data` dataset,
  and drives it through the EXISTING `Data.batched_sequences/2` lazy path —
  so all the validated batch assembly, weight replication, and
  transition-weighting are reused unchanged, and window-sliding across the
  concatenated group matches the all-in-RAM path (which concatenates ALL
  files) except at group boundaries. Peak RAM = `:open_shards` shards.

  ## Shard format

      <shard_dir>/manifest.json   # embed config fingerprint + shard list + totals
      <shard_dir>/shard_00000.bin # term_to_binary(%{emb_bin, n, embed_size,
                                  #   actions_bin, weights_bin, probe_bin})

  `emb_bin` is f16 (`n × embed_size`), `actions_bin` is 6 bytes/frame
  (button bitmask + 4 axis buckets + shoulder), `weights_bin` is f32/frame,
  `probe_bin` is u8/frame (omitted when probe labels aren't requested).
  """

  import Bitwise

  alias ExPhil.Embeddings
  alias ExPhil.Training.{ConversionSampling, Data, OpenerSampling, ProbeRegularizer}

  require Logger

  @manifest "manifest.json"
  @embed_sub_batch 1024

  # ==========================================================================
  # Build (streaming, one source at a time)
  # ==========================================================================

  @doc """
  Build (or extend) shards under `shard_dir` from `specs`.

  `process_fn.(spec) :: [frame]` turns one spec into a list of SHIFTED
  training frames (`%{game_state, controller, name_id?, prev_controller?}` —
  the drill's post-`shift_actions` shape). Return `[]` to skip. The drill
  passes a closure capturing its expert/relabel/detect_port logic; keeping
  it a callback means the loading logic stays in the drill while embedding
  streams here.

  ## Options
    - `:embed_config` (required), `:window` (default 60)
    - `:use_prev_action` (default false), `:prev_action_dropout` (default 0.0)
    - `:conversion_weight`, `:opener_weight`, `:opener_lookback` — per-file
      sampling weights (composed by max), 1.0 when both nil
    - `:probe_reg?` — also store shield-family labels (default false)
    - `:spec_key` — `fn spec -> string` identity for idempotent extend
      (default: the spec itself stringified)
    - `:progress` — `fn done, total, spec -> any`

  Returns `{:ok, %{files:, frames:, sequences:, skipped:}}`.
  """
  def build(specs, shard_dir, opts) do
    File.mkdir_p!(shard_dir)
    embed_config = Keyword.fetch!(opts, :embed_config)
    window = Keyword.get(opts, :window, 60)
    process_fn = Keyword.fetch!(opts, :process_fn)
    spec_key = Keyword.get(opts, :spec_key, &inspect/1)
    progress = Keyword.get(opts, :progress)
    total = length(specs)

    manifest = load_manifest(shard_dir)
    known = MapSet.new(manifest["shards"], & &1["key"])

    {manifest, stats} =
      specs
      |> Enum.with_index()
      |> Enum.reduce({manifest, %{files: 0, frames: 0, sequences: 0, skipped: 0}}, fn {spec, i},
                                                                                      {manifest,
                                                                                       stats} ->
        if progress, do: progress.(i, total, spec)
        key = spec_key.(spec)

        if MapSet.member?(known, key) do
          {manifest, %{stats | skipped: stats.skipped + 1}}
        else
          case build_one(spec, key, shard_dir, manifest, window, embed_config, opts) do
            {:ok, manifest, n, seqs} ->
              {manifest,
               %{stats | files: stats.files + 1, frames: stats.frames + n, sequences: stats.sequences + seqs}}

            :skip ->
              {manifest, %{stats | skipped: stats.skipped + 1}}
          end
        end
      end)

    manifest =
      manifest
      |> Map.put("window", window)
      |> Map.put("total_frames", manifest["shards"] |> Enum.map(& &1["n"]) |> Enum.sum())
      |> Map.put("total_sequences", manifest["shards"] |> Enum.map(& &1["sequences"]) |> Enum.sum())

    save_manifest(shard_dir, manifest)
    {:ok, Map.put(stats, :total_frames, manifest["total_frames"])}
  end

  defp build_one(spec, key, shard_dir, manifest, window, embed_config, opts) do
    process_fn = Keyword.fetch!(opts, :process_fn)
    frames = process_fn.(spec)

    if frames == [] do
      :skip
    else
      n = length(frames)

      # Per-file sampling weights (conversion ⊔ opener), aligned to frames.
      weights = file_weights(frames, opts)

      # Probe labels (shield family) if requested.
      probe = if opts[:probe_reg?], do: ProbeRegularizer.frame_labels(frames), else: nil

      # Embed with prev-action baked (same construction as
      # precompute_frame_embeddings — boundary-aware + DAgger override).
      emb = embed_frames(frames, embed_config, opts)

      {^n, embed_size} = Nx.shape(emb)

      emb_bin =
        emb |> Nx.as_type(:f16) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_binary()

      actions_bin = frames |> Enum.map(&frame_action/1) |> pack_actions()
      weights_bin = weights |> Enum.map(&(&1 * 1.0)) |> then(&Nx.tensor(&1, type: :f32)) |> Nx.to_binary()
      probe_bin = if probe, do: :erlang.list_to_binary(probe), else: nil

      seq = length(manifest["shards"])
      file = "shard_#{String.pad_leading(to_string(seq), 5, "0")}.bin"

      payload = %{
        emb_bin: emb_bin,
        n: n,
        embed_size: embed_size,
        actions_bin: actions_bin,
        weights_bin: weights_bin,
        probe_bin: probe_bin
      }

      File.write!(Path.join(shard_dir, file), :erlang.term_to_binary(payload))

      sequences = max(n - window + 1, 0)

      entry = %{"file" => file, "key" => key, "n" => n, "sequences" => sequences}

      {:ok,
       manifest
       |> Map.put("embed_size", embed_size)
       |> Map.update!("shards", &(&1 ++ [entry])), n, sequences}
    end
  end

  # Per-file weights: replicate the drill's conversion ⊔ opener composition
  # on a single-file list (both modules already operate per-replay).
  defp file_weights(frames, opts) do
    cw = opts[:conversion_weight]
    ow = opts[:opener_weight]

    conv = if cw, do: elem(ConversionSampling.frame_weights([frames], cw), 0)
    open = if ow, do: elem(OpenerSampling.frame_weights([frames], ow, lookback: opts[:opener_lookback] || 30), 0)

    cond do
      conv && open -> OpenerSampling.combine_max(conv, open)
      true -> conv || open || List.duplicate(1.0, length(frames))
    end
  end

  # Prev-action baking, per file (frames are one contiguous replay so the
  # frame+1 boundary check is trivially satisfied except across the file's
  # own gaps; DAgger :prev_controller override honored). Mirrors
  # precompute_frame_embeddings.
  defp embed_frames(frames, embed_config, opts) do
    use_prev = opts[:use_prev_action] || false
    dropout = opts[:prev_action_dropout] || 0.0

    prevs =
      if use_prev do
        [nil | frames]
        |> Enum.zip(frames)
        |> Enum.map(fn {prev, cur} ->
          pc =
            cond do
              is_map_key(cur, :prev_controller) -> cur.prev_controller
              is_nil(prev) -> nil
              prev.game_state.frame + 1 == cur.game_state.frame -> prev.controller
              true -> nil
            end

          if pc != nil and dropout > 0.0 and :rand.uniform() < dropout, do: nil, else: pc
        end)
      else
        List.duplicate(nil, length(frames))
      end

    Enum.zip(frames, prevs)
    |> Enum.chunk_every(@embed_sub_batch)
    |> Enum.map(fn chunk ->
      {fs, ps} = Enum.unzip(chunk)
      states = Enum.map(fs, & &1.game_state)
      name_ids = Enum.map(fs, fn f -> f[:name_id] || 0 end)

      embed_opts = [config: embed_config, name_id: name_ids]
      embed_opts = if use_prev, do: embed_opts ++ [prev_controllers: ps], else: embed_opts

      Embeddings.Game.embed_states_fast(states, 1, embed_opts)
      |> Nx.backend_copy(Nx.BinaryBackend)
    end)
    |> Nx.concatenate()
  end

  defp frame_action(frame) do
    cond do
      is_map_key(frame, :action) -> frame.action
      is_map_key(frame, :controller) -> Data.controller_to_action(frame.controller)
      true -> neutral_action()
    end
  end

  defp neutral_action do
    %{buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
      main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0}
  end

  # ==========================================================================
  # Compact action packing — 6 bytes/frame, exact round-trip
  # ==========================================================================

  @buttons [:a, :b, :x, :y, :z, :l, :r, :d_up]

  defp pack_actions(actions) do
    for a <- actions, into: <<>> do
      mask =
        @buttons
        |> Enum.with_index()
        |> Enum.reduce(0, fn {b, i}, acc -> if a.buttons[b], do: acc ||| bsl(1, i), else: acc end)

      <<mask::8, a.main_x::8, a.main_y::8, a.c_x::8, a.c_y::8, a.shoulder::8>>
    end
  end

  defp unpack_actions(bin) do
    for <<mask::8, mx::8, my::8, cx::8, cy::8, sh::8 <- bin>> do
      buttons =
        @buttons
        |> Enum.with_index()
        |> Map.new(fn {b, i} -> {b, band(mask, bsl(1, i)) != 0} end)

      %{buttons: buttons, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh}
    end
  end

  # ==========================================================================
  # Consume: stream batches over a rolling window of concatenated shards
  # ==========================================================================

  @doc """
  Lazy batch stream for one epoch. Shuffles shard order (seeded per epoch),
  concatenates `:open_shards` shards at a time into a temporary in-memory
  `Data` dataset, and yields batches from the existing
  `Data.batched_sequences/2` lazy path (reusing all validated assembly).

  ## Options
    - `:shard_dir` (required)
    - `:batch_size` (default 64), `:window_size`, `:stride` (default 1)
    - `:seed` (default from time) — reshuffle shards per epoch as `42+epoch`
    - `:open_shards` (default 8) — shards concatenated per group; ↑ = better
      cross-replay mixing, more RAM (peak ≈ open_shards × shard bytes).

      **This knob changes training dynamics — measured 2026-07-23** on a
      204k-frame pool (11 shards), 3 epochs, same seed/pool:

      | config | loss |
      |--------|------|
      | `open_shards: 8` (grouped) | 0.465 |
      | all 11 shards in one group | 0.944 |
      | all-in-RAM (global shuffle) | 1.490 |

      Batches are drawn from within a group, so a small `open_shards`
      makes them near-homogeneous (one or two replays) — easier to fit,
      lower training loss, less representative gradients. More mixing
      moves monotonically toward the all-in-RAM global-shuffle baseline.
      Set `open_shards` as high as RAM allows; it is the shuffle-quality
      dial, not a free parameter. A proper cross-GROUP shuffle buffer
      (sequence-slot reservoir) would remove the trade entirely and is
      the natural next improvement — not implemented here.
    - `:transition_weight`, `:neutral_weight`, `:shuffle` (default true)
  """
  def stream_batches(manifest, opts) do
    shard_dir = Keyword.fetch!(opts, :shard_dir)
    open = Keyword.get(opts, :open_shards, 8)
    seed = Keyword.get(opts, :seed, System.system_time())
    window = Keyword.get(opts, :window_size) || manifest["window"] || 60

    shards = manifest["shards"]
    :rand.seed(:exsss, {seed, seed, seed})
    ordered = if Keyword.get(opts, :shuffle, true), do: Enum.shuffle(shards), else: shards

    ordered
    |> Enum.chunk_every(open)
    |> Stream.flat_map(fn group ->
      dataset = load_group(shard_dir, group, manifest["embed_size"], window)
      weights = group_weights(shard_dir, group)

      Data.batched_sequences(dataset,
        batch_size: Keyword.get(opts, :batch_size, 64),
        window_size: window,
        stride: Keyword.get(opts, :stride, 1),
        lazy: true,
        shuffle: Keyword.get(opts, :shuffle, true),
        drop_last: false,
        seed: seed,
        neutral_weight: Keyword.get(opts, :neutral_weight, 0.25),
        transition_weight: Keyword.get(opts, :transition_weight),
        sampling_weights: weights
      )
    end)
  end

  # Concatenate a group of shards into one temp Data dataset (f32 embeddings
  # + lightweight frames carrying :action). Windows slide across the
  # concatenation exactly as the all-in-RAM path slides across all files.
  defp load_group(shard_dir, group, embed_size, window) do
    payloads = Enum.map(group, &read_shard(shard_dir, &1))

    emb =
      payloads
      |> Enum.map(fn p ->
        Nx.from_binary(p.emb_bin, :f16, backend: Nx.BinaryBackend)
        |> Nx.reshape({p.n, embed_size})
        |> Nx.as_type(:f32)
      end)
      |> Nx.concatenate()

    frames =
      payloads
      |> Enum.flat_map(fn p ->
        p.actions_bin
        |> unpack_actions()
        |> Enum.with_index()
        |> Enum.map(fn {a, i} -> %{action: a, game_state: %{frame: i}} end)
      end)

    n = length(frames)

    %ExPhil.Training.Data{
      frames: frames,
      embedded_frames: emb,
      size: n,
      metadata: %{window_size: window},
      embed_config: nil
    }
  end

  defp group_weights(shard_dir, group) do
    group
    |> Enum.flat_map(fn entry ->
      p = read_shard(shard_dir, entry)
      Nx.from_binary(p.weights_bin, :f32, backend: Nx.BinaryBackend) |> Nx.to_list()
    end)
  end

  defp read_shard(shard_dir, entry) do
    Path.join(shard_dir, entry["file"]) |> File.read!() |> :erlang.binary_to_term()
  end

  @doc """
  Probe-reg / probe-eval subset as a REAL in-memory dataset plus aligned
  shield labels, built from a bounded set of shards (default first 8).

  Returning a `Data` dataset means `ProbeRegularizer.refit/4` and
  `ExPhil.Interp.TrainingProbes.eval/5` work UNCHANGED in streaming mode —
  they just see a smaller pool. Keeps refit O(subset), which matches what
  the all-in-RAM refit already does (it subsamples ~4096 rows anyway).

  Returns `{dataset, labels}`.
  """
  def probe_dataset(manifest, opts) do
    shard_dir = Keyword.fetch!(opts, :shard_dir)
    window = Keyword.get(opts, :window_size) || manifest["window"] || 60
    n_shards = Keyword.get(opts, :n_shards, 8)

    group = Enum.take(manifest["shards"], n_shards)
    dataset = load_group(shard_dir, group, manifest["embed_size"], window)

    labels =
      group
      |> Enum.flat_map(fn entry ->
        case read_shard(shard_dir, entry) do
          %{probe_bin: b} when is_binary(b) -> :erlang.binary_to_list(b)
          _ -> List.duplicate(0, entry["n"])
        end
      end)

    {dataset, labels}
  end

  # ==========================================================================
  # Manifest
  # ==========================================================================

  def load_manifest(shard_dir) do
    case File.read(Path.join(shard_dir, @manifest)) do
      {:ok, bin} ->
        case Jason.decode(bin) do
          {:ok, %{"shards" => _} = m} -> m
          _ -> new_manifest()
        end

      _ ->
        new_manifest()
    end
  end

  defp new_manifest, do: %{"embed_size" => nil, "window" => nil, "shards" => []}

  defp save_manifest(shard_dir, manifest),
    do: File.write!(Path.join(shard_dir, @manifest), Jason.encode!(manifest, pretty: true))

  def stats(shard_dir) do
    m = load_manifest(shard_dir)
    %{files: length(m["shards"]), frames: m["total_frames"] || 0, sequences: m["total_sequences"] || 0}
  end

  # unpack_actions is used by load_group; expose for tests.
  @doc false
  def __unpack_actions__(bin), do: unpack_actions(bin)
  @doc false
  def __pack_actions__(actions), do: pack_actions(actions)
end

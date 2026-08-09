defmodule ExPhil.Training.MmapCorpus do
  @moduledoc """
  Incrementally-built, disk-backed training corpus: embeddings on mmap,
  packed labels in RAM, per-file frame ranges for sequence windows.

  Built for the full-game IL track (fox_il_v2, 2026-08-08): the in-RAM
  pipeline cannot hold the parsed frames of a 4,461-game corpus (~50GB
  of frame structs on a 62GB box), but the actual per-frame training
  signal is tiny — a 288-dim embedding (disk) and a 6-byte action label
  (RAM). The builder streams one replay at a time (parse → embed →
  append → discard), so peak RAM is one file's frames regardless of
  corpus size.

  ## Directory layout

      corpus_dir/
        embeddings.bin   # MmapEmbeddings "EMBD" format (header patched per append)
        labels.bin       # "LBLS" header + 6 bytes/frame (see below)
        manifest.jsonl   # one JSON object per source file: path, start, frames, port
        meta.json        # embed config + build opts (written at create)

  Label packing (6 bytes/frame): button bitmask (a,b,x,y,z,l,r,d_up as
  bits 0..7), main_x, main_y, c_x, c_y, shoulder — the exact discretized
  values `Data.controller_to_action/2` produces, so batches decode to the
  same action maps the standard pipeline trains on.

  ## Why per-file ranges matter

  The in-RAM lazy path slices windows over the flat frame concat, so a
  window can straddle two games (a hidden discontinuity in every epoch).
  Here sequence starts are generated per file range, so windows never
  cross a game boundary.

  ## Usage

      # Build (one file at a time)
      writer = MmapCorpus.create_writer(dir, embed_size, meta)
      writer = MmapCorpus.append!(writer, embeddings, actions, source_path, port)
      MmapCorpus.finalize!(writer)

      # Resume: create_writer on an existing dir returns the writer
      # positioned after the last complete file; `writer.done` has the
      # already-processed paths.

      # Train
      corpus = MmapCorpus.open!(dir)
      {train_ranges, val_ranges} = MmapCorpus.split_files(corpus, 0.1)
      {stream, n} = MmapCorpus.batched_sequences(corpus, train_ranges,
        window_size: 60, stride: 30, batch_size: 64)
  """

  alias ExPhil.Training.{Data, MmapEmbeddings, Output}

  require Logger

  @embd_magic "EMBD"
  @embd_version 1
  @embd_header_size 32
  @lbls_magic "LBLS"
  @lbls_version 1
  @lbls_header_size 24
  @label_bytes 6
  @buttons [:a, :b, :x, :y, :z, :l, :r, :d_up]

  defstruct [
    :dir,
    :embd,          # MmapEmbeddings handle (reader) | fd (writer)
    :labels_fd,     # writer only
    :labels,        # reader only: full labels binary (no header)
    :files,         # [%{path:, start:, frames:, port:}] in corpus order
    :num_frames,
    :embed_size,
    :meta,          # meta.json contents (embed config etc.)
    :done           # writer only: MapSet of processed paths
  ]

  # ==========================================================================
  # Writer
  # ==========================================================================

  @doc """
  Create (or reopen for resume) an incremental corpus writer.

  `meta` is stored as meta.json on first create; on resume the stored
  meta wins and a mismatch in `embed_size` raises (a corpus built under
  one embed config must not be extended under another).
  """
  def create_writer(dir, embed_size, meta \\ %{}) do
    File.mkdir_p!(dir)
    manifest_path = Path.join(dir, "manifest.jsonl")
    meta_path = Path.join(dir, "meta.json")
    embd_path = Path.join(dir, "embeddings.bin")
    lbls_path = Path.join(dir, "labels.bin")

    if File.exists?(manifest_path) do
      reopen_writer(dir, embed_size, manifest_path, meta_path, embd_path, lbls_path)
    else
      File.write!(meta_path, Jason.encode!(Map.put(meta, :embed_size, embed_size)))
      {:ok, embd_fd} = File.open(embd_path, [:write, :read, :binary, :raw])
      :ok = :file.write(embd_fd, embd_header(0, embed_size))
      {:ok, lbls_fd} = File.open(lbls_path, [:write, :read, :binary, :raw])
      :ok = :file.write(lbls_fd, lbls_header(0))

      %__MODULE__{
        dir: dir,
        embd: embd_fd,
        labels_fd: lbls_fd,
        files: [],
        num_frames: 0,
        embed_size: embed_size,
        meta: meta,
        done: MapSet.new()
      }
    end
  end

  defp reopen_writer(dir, embed_size, manifest_path, meta_path, embd_path, lbls_path) do
    meta = meta_path |> File.read!() |> Jason.decode!()

    stored_embed = meta["embed_size"]

    if stored_embed != embed_size do
      raise ArgumentError,
            "corpus #{dir} was built with embed_size #{stored_embed}, got #{embed_size}"
    end

    files = read_manifest(manifest_path)
    num_frames = files |> Enum.map(& &1.frames) |> Enum.sum()

    # Rewrite the manifest from the parsed entries: a crash mid-append can
    # leave a torn final line (no newline), which would otherwise merge
    # with the next appended entry into one corrupt line.
    manifest_body =
      files
      |> Enum.reverse()
      |> Enum.map_join(&(Jason.encode!(&1) <> "\n"))

    File.write!(manifest_path, manifest_body)

    # Reopen data files and truncate any partial tail past the last
    # complete manifest entry (crash mid-append leaves extra bytes).
    {:ok, embd_fd} = File.open(embd_path, [:write, :read, :binary, :raw])
    embd_end = @embd_header_size + num_frames * embed_size * 4
    {:ok, _} = :file.position(embd_fd, embd_end)
    :ok = :file.truncate(embd_fd)
    :ok = patch_count(embd_fd, 8, num_frames)

    {:ok, lbls_fd} = File.open(lbls_path, [:write, :read, :binary, :raw])
    lbls_end = @lbls_header_size + num_frames * @label_bytes
    {:ok, _} = :file.position(lbls_fd, lbls_end)
    :ok = :file.truncate(lbls_fd)
    :ok = patch_count(lbls_fd, 8, num_frames)

    Output.puts("  Resuming corpus #{dir}: #{length(files)} files, #{num_frames} frames")

    %__MODULE__{
      dir: dir,
      embd: embd_fd,
      labels_fd: lbls_fd,
      files: files,
      num_frames: num_frames,
      embed_size: embed_size,
      meta: meta,
      done: MapSet.new(files, & &1.path)
    }
  end

  @doc "Has this source path already been appended? (resume support)"
  def processed?(%__MODULE__{done: done}, path), do: MapSet.member?(done, path)

  @doc """
  Append one source file's frames: `embeddings` is `{n, embed_size}`
  (any backend; copied to binary), `actions` a list of n action maps
  (from `Data.controller_to_action/2`). Headers and manifest are patched
  per call, so a crash never loses complete files.
  """
  def append!(%__MODULE__{} = w, embeddings, actions, source_path, port) do
    {n, es} = Nx.shape(embeddings)

    if es != w.embed_size do
      raise ArgumentError, "embed_size mismatch: corpus #{w.embed_size}, got #{es}"
    end

    if n != length(actions) do
      raise ArgumentError, "#{n} embeddings vs #{length(actions)} actions"
    end

    embd_bin =
      embeddings
      |> Nx.as_type({:f, 32})
      |> Nx.backend_copy(Nx.BinaryBackend)
      |> Nx.to_binary()

    lbls_bin = IO.iodata_to_binary(Enum.map(actions, &pack_label/1))

    :ok = :file.write(w.embd, embd_bin)
    :ok = :file.write(w.labels_fd, lbls_bin)

    new_total = w.num_frames + n
    :ok = patch_count(w.embd, 8, new_total)
    :ok = patch_count(w.labels_fd, 8, new_total)

    entry = %{path: source_path, start: w.num_frames, frames: n, port: port}

    File.write!(
      Path.join(w.dir, "manifest.jsonl"),
      Jason.encode!(entry) <> "\n",
      [:append]
    )

    %{w | num_frames: new_total, files: [entry | w.files], done: MapSet.put(w.done, source_path)}
  end

  @doc "Close writer file descriptors."
  def finalize!(%__MODULE__{embd: embd_fd, labels_fd: lbls_fd}) do
    :file.close(embd_fd)
    :file.close(lbls_fd)
    :ok
  end

  # ==========================================================================
  # Reader
  # ==========================================================================

  @doc """
  Open a corpus for training. Loads labels fully into RAM (6 bytes/frame
  — ~170MB for 28M frames) and mmap-opens the embeddings.
  """
  def open!(dir) do
    meta = dir |> Path.join("meta.json") |> File.read!() |> Jason.decode!()
    files = read_manifest(Path.join(dir, "manifest.jsonl")) |> Enum.reverse()
    {:ok, embd} = MmapEmbeddings.open(Path.join(dir, "embeddings.bin"))

    <<@lbls_magic, version::little-unsigned-32, num_frames::little-unsigned-64,
      label_bytes::little-unsigned-32,
      _reserved::little-unsigned-32,
      labels::binary>> = File.read!(Path.join(dir, "labels.bin"))

    if version != @lbls_version, do: raise("unsupported labels.bin version #{version}")
    if label_bytes != @label_bytes, do: raise("unexpected label width #{label_bytes}")

    if num_frames != embd.num_frames do
      raise "corpus #{dir} inconsistent: #{embd.num_frames} embedding frames vs #{num_frames} labels"
    end

    manifest_frames = files |> Enum.map(& &1.frames) |> Enum.sum()

    if manifest_frames != num_frames do
      raise "corpus #{dir} inconsistent: manifest says #{manifest_frames} frames, files hold #{num_frames}"
    end

    %__MODULE__{
      dir: dir,
      embd: embd,
      labels: labels,
      files: files,
      num_frames: num_frames,
      embed_size: embd.embed_size,
      meta: meta
    }
  end

  @doc "Close the reader's embedding file handle."
  def close(%__MODULE__{embd: %{} = embd}), do: MmapEmbeddings.close(embd)

  @doc """
  Split the corpus by FILE into train/val range lists (val = last
  `val_split` fraction of files — whole games held out, no leakage
  through overlapping windows).
  """
  def split_files(%__MODULE__{files: files}, val_split) when val_split >= 0 do
    n_val = round(length(files) * val_split)
    {train, val} = Enum.split(files, length(files) - n_val)
    {train, val}
  end

  @doc """
  Valid window start frame-indices for the given file ranges: windows lie
  entirely inside one file. Returns a flat list.
  """
  def sequence_starts(file_ranges, window_size, stride) do
    Enum.flat_map(file_ranges, fn %{start: s, frames: n} ->
      if n >= window_size do
        last = s + n - window_size
        Enum.take_while(Stream.iterate(s, &(&1 + stride)), &(&1 <= last))
      else
        []
      end
    end)
  end

  @doc """
  Lazy batch stream over sequence windows, same batch maps as
  `Data.batched_sequences/2` lazy mode: `%{states, actions, frame_weights}`.

  Opens a private fd on embeddings.bin so concurrent streams (train +
  val precompute, prefetchers) never share file positions — all reads
  are pread (positioned), the fd is just per-stream hygiene.

  ## Options
    - `:window_size` (default 60), `:stride` (default 30)
    - `:batch_size` (default 64), `:shuffle` (default true)
    - `:drop_last` (default true), `:seed`
    - `:gpu` (default true), `:neutral_weight` (default 0.25)
  """
  def batched_sequences(%__MODULE__{} = corpus, file_ranges, opts \\ []) do
    window = Keyword.get(opts, :window_size, 60)
    stride = Keyword.get(opts, :stride, 30)
    batch_size = Keyword.get(opts, :batch_size, 64)
    shuffle = Keyword.get(opts, :shuffle, true)
    drop_last = Keyword.get(opts, :drop_last, true)
    gpu = Keyword.get(opts, :gpu, true)
    neutral_weight = Keyword.get(opts, :neutral_weight, 0.25)
    transition_weight = Keyword.get(opts, :transition_weight)
    seed = Keyword.get(opts, :seed, System.system_time())

    starts = sequence_starts(file_ranges, window, stride)

    starts =
      if shuffle do
        :rand.seed(:exsss, {seed, seed, seed})
        Enum.shuffle(starts)
      else
        starts
      end

    num_batches =
      if drop_last,
        do: div(length(starts), batch_size),
        else: div(length(starts) + batch_size - 1, batch_size)

    path = Path.join(corpus.dir, "embeddings.bin")

    stream =
      starts
      |> Stream.chunk_every(batch_size)
      |> then(fn s ->
        if drop_last, do: Stream.reject(s, &(length(&1) < batch_size)), else: s
      end)
      |> Stream.transform(
        fn ->
          {:ok, fd} = File.open(path, [:read, :binary, :raw])
          fd
        end,
        fn batch_starts, fd ->
          {[build_batch(corpus, fd, batch_starts, window, gpu, neutral_weight, transition_weight)], fd}
        end,
        fn fd -> :file.close(fd) end
      )

    {stream, num_batches}
  end

  defp build_batch(corpus, fd, batch_starts, window, gpu, neutral_weight, transition_weight) do
    es = corpus.embed_size
    bytes_per_frame = es * 4
    window_bytes = window * bytes_per_frame

    # One contiguous pread per sequence window
    locs = Enum.map(batch_starts, &{@embd_header_size + &1 * bytes_per_frame, window_bytes})
    {:ok, chunks} = :file.pread(fd, locs)

    states =
      chunks
      |> IO.iodata_to_binary()
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({length(batch_starts), window, es})

    states = if gpu, do: Nx.backend_transfer(states, EXLA.Backend), else: states

    # Label of the supervised (last) frame of each window
    actions = Enum.map(batch_starts, &decode_label(corpus.labels, &1 + window - 1))
    prev_actions = Enum.map(batch_starts, &decode_label(corpus.labels, &1 + window - 2))

    action_tensors = Data.actions_to_tensors(actions)

    action_tensors =
      if gpu do
        Map.new(action_tensors, fn {k, v} -> {k, Nx.backend_transfer(v, EXLA.Backend)} end)
      else
        action_tensors
      end

    frame_weights =
      Data.compute_frame_weights(actions,
        neutral_weight: neutral_weight,
        transition_weight: transition_weight,
        prev_actions: prev_actions
      )

    frame_weights = if gpu, do: Nx.backend_transfer(frame_weights, EXLA.Backend), else: frame_weights

    %{states: states, actions: action_tensors, frame_weights: frame_weights}
  end

  @doc """
  Per-button press rates over the whole corpus, from the packed labels —
  for `button_pos_weight: :auto` without the frames list. Returns a map
  `%{a: rate, ...}` in `#{inspect(@buttons)}` order.
  """
  def button_rates(%__MODULE__{labels: labels, num_frames: n}) do
    counts =
      labels
      |> :binary.bin_to_list()
      |> Enum.take_every(@label_bytes)
      |> Enum.reduce(List.duplicate(0, 8), fn mask, acc ->
        Enum.with_index(acc, fn c, i -> c + Bitwise.band(Bitwise.bsr(mask, i), 1) end)
      end)

    @buttons
    |> Enum.zip(counts)
    |> Map.new(fn {b, c} -> {b, if(n > 0, do: c / n, else: 0.0)} end)
  end

  # ==========================================================================
  # Label packing
  # ==========================================================================

  @doc "Pack one discretized action map into #{@label_bytes} bytes."
  def pack_label(action) do
    mask =
      @buttons
      |> Enum.with_index()
      |> Enum.reduce(0, fn {btn, i}, acc ->
        if Map.get(action.buttons, btn), do: Bitwise.bor(acc, Bitwise.bsl(1, i)), else: acc
      end)

    <<mask, action.main_x, action.main_y, action.c_x, action.c_y, action.shoulder>>
  end

  @doc "Decode the label at frame `idx` back into an action map."
  def decode_label(labels, idx) do
    <<mask, mx, my, cx, cy, sh>> = binary_part(labels, idx * @label_bytes, @label_bytes)

    buttons =
      @buttons
      |> Enum.with_index()
      |> Map.new(fn {btn, i} -> {btn, Bitwise.band(Bitwise.bsr(mask, i), 1) == 1} end)

    %{buttons: buttons, main_x: mx, main_y: my, c_x: cx, c_y: cy, shoulder: sh}
  end

  # ==========================================================================
  # File format helpers
  # ==========================================================================

  defp embd_header(num_frames, embed_size) do
    <<@embd_magic, @embd_version::little-unsigned-32, num_frames::little-unsigned-64,
      embed_size::little-unsigned-64, 1::little-unsigned-32, 0::little-unsigned-32>>
  end

  defp lbls_header(num_frames) do
    <<@lbls_magic, @lbls_version::little-unsigned-32, num_frames::little-unsigned-64,
      @label_bytes::little-unsigned-32, 0::little-unsigned-32>>
  end

  # Patch a u64 count at `offset` without disturbing the append position
  defp patch_count(fd, offset, count) do
    :ok = :file.pwrite(fd, offset, <<count::little-unsigned-64>>)
  end

  defp read_manifest(path) do
    # Returns entries NEWEST-FIRST (reverse corpus order)
    path
    |> File.stream!()
    |> Enum.reduce([], fn line, acc ->
      case Jason.decode(String.trim(line)) do
        {:ok, %{"path" => p, "start" => s, "frames" => n} = e} ->
          [%{path: p, start: s, frames: n, port: e["port"]} | acc]

        _ ->
          # Torn tail line from a crash mid-write: manifest is
          # append-only JSONL, so only the last line can be damaged.
          acc
      end
    end)
  end
end

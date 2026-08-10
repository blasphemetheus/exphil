#!/usr/bin/env elixir
# Incremental corpus builder: parse → embed → append, one replay at a time.
#
# Builds an ExPhil.Training.MmapCorpus directory (embeddings.bin +
# labels.bin + manifest.jsonl) that scripts/train.exs consumes via
# --corpus. Peak RAM is one file's frames, so corpus size is unbounded
# by the box (the in-RAM pipeline caps at ~900 files / 62GB).
#
# Resumable: re-running with the same --out skips files already in the
# manifest (crash-safe — headers are patched per file append).
#
# Usage:
#   mix run scripts/build_corpus.exs \
#     --replays replays/fox_il_v1 \
#     --out cache/corpus/fox_v2 \
#     --character fox \
#     2>&1 | tee eval_runs/build_corpus_fox_v2.log
#
# Options:
#   --replays PATH      Replay directory (recursive *.slp glob) [./replays]
#   --out DIR           Corpus output directory (required)
#   --character NAME    Train-player character: per file, the port playing
#                       this character is the imitated player (files
#                       without it are skipped). Omit = port 1.
#   --max-files N       Limit files (testing)
#   --embed-batch N     GPU embedding batch size [1000]
#   --per-stage-ledge   Real per-stage edge x in the ledge-distance
#                       feature (task #25). Bakes into the embeddings —
#                       train MUST pass the same flag; recorded in meta
#   --quiet             Errors/summary only

if "--quiet" in System.argv(), do: Logger.configure(level: :warning)

alias ExPhil.Training.{Data, MmapCorpus, Output}
alias ExPhil.Data.Peppi
alias ExPhil.Embeddings

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [
      replays: :string,
      out: :string,
      character: :string,
      max_files: :integer,
      embed_batch: :integer,
      per_stage_ledge: :boolean,
      quiet: :boolean
    ]
  )

out_dir = opts[:out] || raise "--out DIR is required"
replay_dir = Path.expand(opts[:replays] || "./replays")
character = opts[:character]
embed_batch = opts[:embed_batch] || 1000

Output.set_verbosity(if(opts[:quiet], do: 0, else: 1))
Output.banner("ExPhil Corpus Builder")

# External (CSS) character ids — matches Pipeline's map (ReplayMeta carries
# external ids; internal frame-stream ids differ, GOTCHA-grade trap)
external_ids = %{
  "captainfalcon" => 0, "falcon" => 0, "dk" => 1, "donkeykong" => 1,
  "fox" => 2, "gameandwatch" => 3, "gnw" => 3, "kirby" => 4,
  "bowser" => 5, "link" => 6, "luigi" => 7, "mario" => 8,
  "marth" => 9, "mewtwo" => 10, "ness" => 11, "peach" => 12,
  "pikachu" => 13, "iceclimbers" => 14, "ics" => 14,
  "jigglypuff" => 15, "puff" => 15, "samus" => 16, "yoshi" => 17,
  "zelda" => 18, "sheik" => 19, "falco" => 20, "younglink" => 21,
  "doc" => 22, "drmario" => 22, "roy" => 23, "pichu" => 24,
  "ganondorf" => 25, "ganon" => 25
}

want_id =
  if character do
    key = character |> String.downcase() |> String.replace(~r/[^a-z0-9]/, "")
    external_ids[key] || raise "unknown character #{character}"
  end

files = Path.wildcard(Path.join(replay_dir, "**/*.slp")) |> Enum.sort()
files = if opts[:max_files], do: Enum.take(files, opts[:max_files]), else: files

if files == [], do: raise("no .slp files under #{replay_dir}")

embed_config =
  Embeddings.config(
    action_mode: :learned,
    character_mode: :learned,
    stage_mode: :one_hot_compact,
    per_stage_ledge: opts[:per_stage_ledge] || false
  )

Output.config([
  {"Replays", "#{replay_dir} (#{length(files)} files)"},
  {"Output", out_dir},
  {"Character", character || "port 1 (no filter)"},
  {"Embed batch", embed_batch}
])

neutral_action = %{
  buttons: %{a: false, b: false, x: false, y: false, z: false, l: false, r: false, d_up: false},
  main_x: 8, main_y: 8, c_x: 8, c_y: 8, shoulder: 0
}

# Per-file: pick the imitated port (the one playing --character), the
# other 2P port is the opponent. v1 lesson: a fixed player_port 1 makes
# the model imitate the OPPONENT in every game where the target
# character sits on port 2+.
pick_ports = fn meta ->
  ports = Enum.map(meta.players, & &1.port)

  cond do
    want_id == nil ->
      {1, 2}

    length(ports) != 2 ->
      nil

    true ->
      case Enum.find(meta.players, &(&1.character == want_id)) do
        nil -> nil
        p -> {p.port, Enum.find(ports, &(&1 != p.port))}
      end
  end
end

embed_file = fn frames ->
  frames
  |> Enum.chunk_every(embed_batch)
  |> Enum.map(fn chunk ->
    chunk
    |> Enum.map(& &1.game_state)
    |> Embeddings.Game.embed_states_fast(1, config: embed_config)
    |> Nx.backend_copy(Nx.BinaryBackend)
  end)
  |> Nx.concatenate(axis: 0)
end

# Writer is created lazily on the first embedded file (embed_size is
# discovered from the tensor, not hardcoded)
meta = %{
  action_mode: "learned",
  character_mode: "learned",
  stage_mode: "one_hot_compact",
  character: character,
  replay_dir: replay_dir,
  per_stage_ledge: opts[:per_stage_ledge] || false
}

start_time = System.monotonic_time(:millisecond)
total = length(files)

# Resume: open the writer up front so already-appended files are skipped
# before the (expensive) parse+embed, not after
initial_writer =
  if File.exists?(Path.join(out_dir, "manifest.jsonl")) do
    stored = Path.join(out_dir, "meta.json") |> File.read!() |> Jason.decode!()
    MmapCorpus.create_writer(out_dir, stored["embed_size"], meta)
  end

{writer, stats} =
  files
  |> Enum.with_index(1)
  |> Enum.reduce({initial_writer, %{done: 0, skipped: 0, no_char: 0, errors: 0, frames: 0}}, fn {path, idx}, {writer, stats} ->
    if writer && MmapCorpus.processed?(writer, path) do
      {writer, %{stats | skipped: stats.skipped + 1}}
    else
      result =
        try do
          with {:ok, file_meta} <- Peppi.metadata(path),
               {player_port, opp_port} when is_integer(player_port) <- pick_ports.(file_meta) || :no_char,
               {:ok, replay} <- Peppi.parse(path, player_port: player_port) do
            frames =
              Peppi.to_training_frames(replay,
                player_port: player_port,
                opponent_port: opp_port
              )

            if frames == [] do
              :empty
            else
              embeddings = embed_file.(frames)

              actions =
                Enum.map(frames, fn f ->
                  case f.controller do
                    nil -> neutral_action
                    c -> Data.controller_to_action(c)
                  end
                end)

              {:ok, embeddings, actions, player_port}
            end
          end
        rescue
          e -> {:error, e}
        end

      case result do
        {:ok, embeddings, actions, port} ->
          {n, embed_size} = Nx.shape(embeddings)

          writer =
            writer || MmapCorpus.create_writer(out_dir, embed_size, meta)

          writer =
            if MmapCorpus.processed?(writer, path) do
              writer
            else
              MmapCorpus.append!(writer, embeddings, actions, path, port)
            end

          stats = %{stats | done: stats.done + 1, frames: stats.frames + n}

          if rem(idx, 10) == 0 do
            elapsed = System.monotonic_time(:millisecond) - start_time
            rate = idx / max(elapsed / 1000, 0.001)
            eta_s = round((total - idx) / max(rate, 0.001))

            IO.write(
              :stderr,
              "\r  [#{idx}/#{total}] #{stats.frames} frames | #{Float.round(rate, 1)} files/s | ETA #{div(eta_s, 60)}m #{rem(eta_s, 60)}s\e[K"
            )
          end

          {writer, stats}

        :no_char ->
          {writer, %{stats | no_char: stats.no_char + 1}}

        :empty ->
          {writer, %{stats | errors: stats.errors + 1}}

        other ->
          Output.warning("#{Path.basename(path)}: #{inspect(other, limit: 3)}")
          {writer, %{stats | errors: stats.errors + 1}}
      end
    end
  end)

IO.write(:stderr, "\r\e[K")

if writer do
  MmapCorpus.finalize!(writer)

  elapsed_s = div(System.monotonic_time(:millisecond) - start_time, 1000)

  Output.success("Corpus complete: #{out_dir}")

  Output.config([
    {"Files appended", stats.done},
    {"Skipped (already in corpus)", stats.skipped},
    {"Skipped (no #{character || "?"})", stats.no_char},
    {"Errors/empty", stats.errors},
    {"Total frames (this run)", stats.frames},
    {"Corpus frames", writer.num_frames},
    {"Embed size", writer.embed_size},
    {"Elapsed", "#{div(elapsed_s, 60)}m #{rem(elapsed_s, 60)}s"}
  ])
else
  Output.error("No files were appended (#{stats.no_char} without character, #{stats.errors} errors)")
  System.halt(1)
end

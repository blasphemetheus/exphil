# Build a mini MmapCorpus from a MixFrames .frames export, one corpus
# "file" per snippet list. Purpose: corpus-mode temporal training mixes
# via --mix-corpus, and MmapCorpus.sequence_starts never crosses file
# boundaries — so snippets packed this way get clean GRU windows, which
# the replay-mode --mix-frames path cannot guarantee (its sequence
# builder slides over the flat frames list, stitching unrelated
# snippets into one window).
#
# The export should be mined with --keep-unlabeled so each list is
# frame-contiguous (time-skips inside a window are the same corruption
# as crossing a boundary). Lists shorter than --window yield no
# sequences and are dropped here with a count.
#
#   mix run scripts/build_snippet_corpus.exs \
#     --frames eval_runs/0810_edge_snippets_contig/snippets.frames \
#     --out cache/corpus/edge_snippets_v1 [--window 60]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Embeddings
alias ExPhil.Training.{Data, MmapCorpus, Output}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [frames: :string, out: :string, window: :integer]
  )

frames_path = opts[:frames] || raise("--frames is required")
out_dir = opts[:out] || raise("--out is required")
window = opts[:window] || 60

%{frame_lists: lists} = envelope = frames_path |> File.read!() |> :erlang.binary_to_term()

Output.banner("Snippet corpus builder")
Output.config([
  {"Source", frames_path},
  {"Expert", envelope[:expert]},
  {"Snippets", length(lists)},
  {"Window (min length)", window}
])

if File.exists?(Path.join(out_dir, "manifest.jsonl")) do
  Output.error("#{out_dir} already exists — corpora are append-only, pick a fresh dir")
  System.halt(2)
end

{usable, short} = Enum.split_with(lists, &(length(&1) >= window))

if short != [] do
  Output.warning(
    "#{length(short)} snippet(s) shorter than window #{window} dropped " <>
      "(#{Enum.sum(Enum.map(short, &length/1))} frames) — mine with a larger --pre/--post"
  )
end

if usable == [] do
  Output.error("no snippet reaches window length #{window}")
  System.halt(2)
end

embed_config =
  Embeddings.config(
    action_mode: :learned,
    character_mode: :learned,
    stage_mode: :one_hot_compact
  )

neutral_action = Data.controller_to_action(ExPhil.Bridge.ControllerState.neutral())

meta = %{
  action_mode: "learned",
  character_mode: "learned",
  stage_mode: "one_hot_compact",
  source: frames_path,
  expert: envelope[:expert],
  action_delay: envelope[:action_delay] || 0
}

writer =
  usable
  |> Enum.with_index()
  |> Enum.reduce(nil, fn {frames, idx}, writer ->
    embeddings =
      frames
      |> Enum.map(& &1.game_state)
      |> Embeddings.Game.embed_states_fast(1, config: embed_config)
      |> Nx.backend_copy(Nx.BinaryBackend)

    actions =
      Enum.map(frames, fn f ->
        case f.controller do
          nil -> neutral_action
          c -> Data.controller_to_action(c)
        end
      end)

    {_n, embed_size} = Nx.shape(embeddings)
    writer = writer || MmapCorpus.create_writer(out_dir, embed_size, meta)

    if rem(idx, 25) == 0 do
      IO.write(:stderr, "\r  [#{idx}/#{length(usable)}] snippets embedded\e[K")
    end

    MmapCorpus.append!(writer, embeddings, actions, "snippet_#{idx}", 1)
  end)

IO.write(:stderr, "\r\e[K")
MmapCorpus.finalize!(writer)

Output.success(
  "#{length(usable)} snippets / #{writer.num_frames} frames -> #{out_dir} " <>
    "(embed #{writer.embed_size})"
)

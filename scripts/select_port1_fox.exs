# Select replays whose PORT 1 player is a given character, symlink them
# into an out dir. Replay-mode training imitates a fixed player_port
# (default 1), so an unfiltered dir imitates the wrong player in every
# game where the character sits elsewhere — the fox_il_v1 port bug
# (2026-08-08: 62% of v1's files). Until replay mode gets per-file port
# detection like build_corpus.exs, train on a dir built by this script.
#
#   mix run scripts/select_port1_fox.exs --replays replays/fox_il_v1 \
#     --out replays/fox_il_v1_p1 [--character fox] [--limit 150] [--seed 42]

require Logger
Logger.configure(level: :warning)

alias ExPhil.Data.Peppi
alias ExPhil.Training.Output

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [replays: :string, out: :string, character: :string, limit: :integer, seed: :integer]
  )

replay_dir = Path.expand(opts[:replays] || raise("--replays is required"))
out_dir = Path.expand(opts[:out] || raise("--out is required"))
limit = opts[:limit]
# External (CSS-order) character id — Fox=2, same table as the
# train_character filter fix
want = String.to_atom(opts[:character] || "fox")
want_id = %{fox: 2, falco: 22, marth: 9, mewtwo: 10}[want] || raise("unknown character #{want}")

:rand.seed(:exsss, {opts[:seed] || 42, 0, 0})

files = Path.wildcard(Path.join(replay_dir, "**/*.slp"))

matches =
  files
  |> Enum.filter(fn path ->
    case Peppi.metadata(path) do
      {:ok, meta} -> Enum.any?(meta.players, fn p -> p.port == 1 and p.character == want_id end)
      _ -> false
    end
  end)
  |> Enum.shuffle()

selected = if limit, do: Enum.take(matches, limit), else: matches

File.mkdir_p!(out_dir)

for path <- selected do
  link = Path.join(out_dir, Path.basename(path))
  File.rm(link)
  File.ln_s!(path, link)
end

Output.success(
  "#{length(selected)}/#{length(matches)} port-1 #{want} replays " <>
    "(of #{length(files)} scanned) symlinked into #{out_dir}"
)

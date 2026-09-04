# Per-game style fingerprints over .slp sets (STYLE_IDENTITY.md step 1).
#
#   mix run scripts/style_fingerprint.exs "replays/.../partners/MARTH/*Fox*.slp" \
#     --subject-character Fox --out eval_runs/style/fingerprints.jsonl
#
# Emits one JSON line per game: %{path, tag, port, features: {...}} —
# `tag` from FilenameTags (nil = anonymous). Downstream: calibration
# (same-player retrieval), clustering (perceived players), matching.
#
# Port selection: the first port whose character matches
# --subject-character (external-CSS metadata space); files without that
# character are skipped.

alias ExPhil.Data.{Peppi, FilenameTags}
alias ExPhil.Interp.StyleFingerprint
alias ExPhil.Training.{Config, Output}

if "--quiet" in System.argv(), do: Logger.configure(level: :warning)

{opts, paths, _} =
  OptionParser.parse(System.argv(),
    switches: [subject_character: :string, out: :string, min_frames: :integer, concurrency: :integer, quiet: :boolean, validate_order: :boolean, all_players: :boolean]
  )

subject = opts[:subject_character] || "Fox"
out = opts[:out] || raise("--out PATH required")
min_frames = opts[:min_frames] || 1800
conc = opts[:concurrency] || 8

subject_atom = subject |> String.downcase() |> String.replace(" ", "_") |> String.to_atom()
subject_display = Config.character_name(subject_atom)

# Metadata characters are EXTERNAL-CSS ids (fox = 2; pipeline.ex
# @external_character_ids is the in-repo authority — mirror of its table).
external_ids = %{
  "captainfalcon" => 0, "falcon" => 0, "dk" => 1, "donkeykong" => 1,
  "fox" => 2, "gameandwatch" => 3, "gnw" => 3, "kirby" => 4,
  "bowser" => 5, "link" => 6, "luigi" => 7, "mario" => 8,
  "marth" => 9, "mewtwo" => 10, "ness" => 11, "peach" => 12,
  "pikachu" => 13, "iceclimbers" => 14, "jigglypuff" => 15,
  "samus" => 16, "yoshi" => 17, "zelda" => 18, "sheik" => 19,
  "falco" => 20, "younglink" => 21, "drmario" => 22, "roy" => 23,
  "pichu" => 24, "ganondorf" => 25
}

subject_key = subject |> String.downcase() |> String.replace(~r/[^a-z0-9]/, "")
subject_ext_id = external_ids[subject_key] || raise("unknown subject character #{subject}")

# --validate-order: empirically test the "filename A + B == port 1 + 2"
# convention that ditto tag resolution (FilenameTags.subject_tag/3) leans
# on. Uses NON-ditto files where the subject's port is unambiguous from
# metadata and the filename position of the subject character is known:
# agreement% is the confidence in positional ditto tags.
if opts[:validate_order] do
  files =
    paths
    |> Enum.flat_map(fn p -> if String.contains?(p, "*"), do: Path.wildcard(p), else: [p] end)
    |> Enum.filter(&String.ends_with?(&1, ".slp"))

  {agree, disagree, unusable} =
    files
    |> Task.async_stream(
      fn path ->
        entries = FilenameTags.parse(path)

        with [_, _] <- entries,
             [{_, ca}, {_, cb}] when ca != cb <- entries,
             {:ok, meta} <- Peppi.metadata(path),
             %{port: port} when port in [1, 2] <-
               Enum.find(meta.players || [], fn p -> p.character == subject_ext_id end),
             pos when not is_nil(pos) <-
               Enum.find_index(entries, fn {_t, c} -> c == String.downcase(subject_display) end) do
          if pos + 1 == port, do: :agree, else: :disagree
        else
          _ -> :unusable
        end
      end,
      max_concurrency: opts[:concurrency] || 8,
      timeout: :infinity
    )
    |> Enum.reduce({0, 0, 0}, fn
      {:ok, :agree}, {a, d, u} -> {a + 1, d, u}
      {:ok, :disagree}, {a, d, u} -> {a, d + 1, u}
      _, {a, d, u} -> {a, d, u + 1}
    end)

  total = agree + disagree
  pct = if total > 0, do: Float.round(agree * 100 / total, 1), else: 0.0
  Output.puts("order validation: #{agree}/#{total} agree (#{pct}%), #{unusable} unusable")
  if pct < 99.0 and total > 50 do
    Output.warning("position<->port convention NOT reliable — do not trust ditto tags")
  end
  System.halt(0)
end

files =
  paths
  |> Enum.flat_map(fn p -> if String.contains?(p, "*"), do: Path.wildcard(p), else: [p] end)
  |> Enum.filter(&String.ends_with?(&1, ".slp"))
  |> Enum.uniq()

if files == [], do: raise("no .slp matched")

Output.banner("Style fingerprints")
Output.config([{"Files", length(files)}, {"Subject", subject_display}, {"Out", out}])

File.mkdir_p!(Path.dirname(out))
io = File.open!(out, [:write, :utf8])

results =
  files
  |> Task.async_stream(
    fn path ->
      case Peppi.metadata(path) do
        {:ok, meta} ->
          # Default: all subject-character players — dittos yield a row per
          # port with tag nil + the candidate pair (positional tagging
          # measured 69.2% reliable -> matcher assigns 2-way instead).
          # --all-players (local corpora / identity clustering): EVERY
          # occupied port, any character; tag prefers the in-game nametag
          # (normalized from the full-width CSS charset) over filename tags.
          subjects =
            if opts[:all_players] do
              meta.players || []
            else
              Enum.filter(meta.players || [], fn p -> p.character == subject_ext_id end)
            end

          rows =
            Enum.flat_map(subjects, fn %{port: port} = player ->
              opp =
                case Enum.find(meta.players, fn p -> p.port != port end) do
                  %{port: o} -> o
                  _ -> if port == 1, do: 2, else: 1
                end

              with {:ok, replay} <- Peppi.parse(path, player_port: port),
                   frames when length(frames) >= min_frames <-
                     replay
                     |> Peppi.to_training_frames(player_port: port, opponent_port: opp)
                     |> Enum.reject(&(&1.game_state.frame < 0)) do
                states = Enum.map(frames, & &1.game_state)
                controllers = Enum.map(frames, & &1.controller)
                fp = StyleFingerprint.fingerprint(states, port, controllers)

                # Tag preference: in-game nametag (local corpora; full-width
                # normalized) -> filename bracket tag (HF convention).
                # Dittos get no filename tag (69.2%-reliable positional,
                # measured) — the candidate pair rides for 2-way matching.
                ingame = FilenameTags.normalize_tag(Map.get(player, :tag))

                tag =
                  if ingame && not FilenameTags.placeholder?(ingame),
                    do: ingame,
                    else: FilenameTags.subject_tag(path, Map.get(player, :character_name) || subject_display)

                # ditto = another player of THIS player's character in the game
                is_ditto =
                  Enum.count(meta.players, &(&1.character == player.character)) > 1

                candidates = if is_ditto, do: FilenameTags.parse(path) |> Enum.map(&elem(&1, 0)) |> Enum.reject(&is_nil/1), else: []

                # Session context for the entity model (STYLE_IDENTITY.md):
                # at locals, timestamp adjacency on a setup is strong
                # same-player evidence. File mtime ~ game time there.
                recorded_at =
                  case File.stat(path) do
                    {:ok, %{mtime: erl}} -> erl |> NaiveDateTime.from_erl!() |> NaiveDateTime.to_iso8601()
                    _ -> nil
                  end

                [
                  %{
                    path: path,
                    tag: tag,
                    port: port,
                    character: Map.get(player, :character_name),
                    ditto: is_ditto,
                    candidates: candidates,
                    recorded_at: recorded_at,
                    features: fp
                  }
                ]
              else
                _ -> []
              end
            end)

          if rows == [], do: :skip, else: {:ok, rows}

        _ ->
          :skip
      end
    end,
    max_concurrency: conc,
    timeout: :infinity
  )
  |> Enum.reduce({0, 0}, fn
    {:ok, {:ok, rows}}, {ok, skip} ->
      Enum.each(rows, &IO.write(io, Jason.encode!(&1) <> "\n"))
      if rem(ok, 50) == 0, do: IO.write(:stderr, "\r  #{ok} files fingerprinted, #{skip} skipped\e[K")
      {ok + 1, skip}

    _, {ok, skip} ->
      {ok, skip + 1}
  end)

File.close(io)
IO.write(:stderr, "\r\e[K")
{ok, skip} = results
Output.success("#{ok} fingerprints -> #{out} (#{skip} skipped)")

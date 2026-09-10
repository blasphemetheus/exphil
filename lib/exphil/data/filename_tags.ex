defmodule ExPhil.Data.FilenameTags do
  @moduledoc """
  Player-identity tags recovered from .slp FILENAMES.

  The slippi-public-dataset-v3.7 corpus is ANONYMIZED in-file: every
  player's netplay name is the placeholder `"Master Player"` (measured
  2026-09-04: 600/600 player slots across 300 random ranked-fox files).
  The only identity signal is the filename convention:

      "03_20_55 [RUDE] Captain Falcon + [INFP] Fox (PS).slp"
      "00_41_46 [SM] Falco + Fox (FD).slp"          # one side untagged
      "01_15_15 Falco + Captain Falcon (YI).slp"    # no tags
      "master-master-000551fbfa951d4cce5f3f7a.slp"  # hashed (rank-tier) name

  `subject_tag/2` extracts the bracket tag attached to the SUBJECT's
  character (the imitated player — we always know their character from
  `--train-character`). Ambiguous cases (dittos, hashed names, missing
  bracket) return `nil`, which downstream maps to name_id 0 — the
  anonymous bucket the model has always trained against.
  """

  @placeholder_tags [nil, "", "Master Player"]

  @doc "True when an in-file netplay name carries no identity signal."
  @spec placeholder?(String.t() | nil) :: boolean()
  def placeholder?(tag), do: tag in @placeholder_tags

  @doc """
  Normalize an in-game SSBM nametag to plain ASCII: the CSS name entry
  uses FULL-WIDTH characters (`ＩＮＦＰ`), which must unify with the
  ASCII bracket tags in HF filenames (`[INFP]`) — one identity, two
  encodings (measured in the D20 corpus, 2026-09-04). Maps U+FF01-FF5E
  to their ASCII forms and the ideographic space to a space; trims.
  """
  @spec normalize_tag(String.t() | nil) :: String.t() | nil
  def normalize_tag(nil), do: nil

  def normalize_tag(tag) do
    tag
    |> String.to_charlist()
    |> Enum.map(fn
      cp when cp in 0xFF01..0xFF5E -> cp - 0xFF01 + 0x21
      0x3000 -> ?\s
      cp -> cp
    end)
    |> List.to_string()
    |> String.trim()
    |> case do
      "" -> nil
      t -> t
    end
  end

  # Paren fields that are NOT player tags in the FALCO-dir "vs"
  # conventions (measured 2026-09-05): costume colors and the
  # bracket-convention stage abbreviations. Exact-case on purpose — a
  # player tag "(RED)" is all-caps, the costume is "(Red)".
  @paren_denylist ~w(Default Red Blue Green White Black Yellow Purple
                     Pink Orange Brown Grey Gray Cyan) ++
                    ["Light Blue", "Dark Blue", "Dark Green"] ++
                    ~w(BF YS PS DL FD FoD YI)

  # Display character names for tail-matching "TAG Char" sides in the
  # tournament long-form ("SOPH Marth (White)"). Longest-first so
  # "Young Link" wins over "Link", "Dr. Mario"/"Mario" likewise.
  @display_chars [
                   "Mr. Game & Watch",
                   "Captain Falcon",
                   "Ice Climbers",
                   "Donkey Kong",
                   "Game & Watch",
                   "Young Link",
                   "Jigglypuff",
                   "Dr. Mario",
                   "Ganondorf",
                   "Dr Mario",
                   "Pikachu",
                   "Mewtwo",
                   "Bowser",
                   "Falco",
                   "Kirby",
                   "Luigi",
                   "Marth",
                   "Peach",
                   "Pichu",
                   "Samus",
                   "Sheik",
                   "Yoshi",
                   "Zelda",
                   "Link",
                   "Mario",
                   "Ness",
                   "Fox",
                   "Roy"
                 ]
                 |> Enum.sort_by(&(-String.length(&1)))

  @doc """
  Parse a replay filename into its player entries, in filename order.

  Returns `[{tag_or_nil, character_name_lowercase}]` (2 entries) or `[]`
  when the name follows no known convention. Three conventions are
  recognized (all present in the HF ranked corpus, measured 09-05):

    * `"03_20_55 [RUDE] Falcon + [INFP] Fox (PS).slp"` — bracket tags,
      paren stage (FOX/MARTH/ZS dirs + 16k of FALCO)
    * `"Falco vs Link (MILO) [BF] Game_...slp"` — paren tag AFTER the
      character, bracket stage (paren costume colors / stage abbrevs
      are denylisted, exact-case)
    * `"20200101 - HNC 5 - PM 1006 - Falco (Default) vs SOPH Marth
      (White) - Dream Land N64.slp"` — tournament long-form: tag
      PREFIXES the character, parens are costumes (dropped)
  """
  @spec parse(Path.t()) :: [{String.t() | nil, String.t()}]
  def parse(path) do
    base = path |> Path.basename() |> String.replace_suffix(".slp", "")

    cond do
      String.contains?(base, " + ") -> parse_plus(base)
      String.contains?(base, " vs ") -> parse_vs(base)
      true -> []
    end
  end

  defp parse_plus(base) do
    # Drop the trailing " (STAGE)" group and any leading timestamp-ish
    # prefix (everything before the first "[" or the first letter run
    # that starts a character name — in practice: strip leading
    # digits/underscores/dots/colons/Z and whitespace).
    base = Regex.replace(~r/\s*\([^)]*\)\s*$/, base, "")
    base = Regex.replace(~r/^[\d_.:Z\s\-]*/, base, "")

    case String.split(base, " + ") do
      [_, _] = sides ->
        Enum.map(sides, fn side ->
          case Regex.run(~r/^\[([^\]]+)\]\s*(.+)$/, String.trim(side)) do
            [_, tag, char] -> {tag, String.downcase(String.trim(char))}
            nil -> {nil, String.downcase(String.trim(side))}
          end
        end)

      _ ->
        []
    end
  end

  defp parse_vs(base) do
    # Tournament long-form: the " vs " lives in one " - "-separated
    # segment; Game_-form: strip the trailing "[STAGE] Game_..." /
    # bare " Game_..." tail. Either way reduce to "SIDE vs SIDE".
    core =
      base
      |> String.split(" - ")
      |> Enum.find(base, &String.contains?(&1, " vs "))
      |> String.replace(~r/\s*\[[^\]]*\].*$/, "")
      |> String.replace(~r/\s*Game_\S*$/, "")
      |> String.trim()

    case String.split(core, " vs ") do
      [_, _] = sides -> Enum.map(sides, &parse_vs_side/1)
      _ -> []
    end
  end

  # One side of a "vs": pull paren groups (tag unless denylisted, e.g.
  # costume "(Default)"), then split any remaining "TAG Char" prefix by
  # tail-matching known character display names.
  defp parse_vs_side(side) do
    paren_tag =
      ~r/\(([^)]+)\)/
      |> Regex.scan(side, capture: :all_but_first)
      |> List.flatten()
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 in @paren_denylist))
      |> List.first()

    bare = side |> String.replace(~r/\s*\([^)]*\)/, "") |> String.trim()

    {prefix_tag, char} =
      case Enum.find(@display_chars, fn c ->
             String.downcase(bare) == String.downcase(c) or
               String.ends_with?(String.downcase(bare), " " <> String.downcase(c))
           end) do
        nil ->
          {nil, bare}

        c ->
          prefix = bare |> String.slice(0, String.length(bare) - String.length(c)) |> String.trim()
          {if(prefix == "", do: nil, else: prefix), c}
      end

    {prefix_tag || paren_tag, String.downcase(char)}
  end

  @doc """
  The bracket tag belonging to the subject player, identified by their
  character's display name (case-insensitive). Returns nil when: the
  filename has no entries, the subject character doesn't appear, BOTH
  entries are the subject character (ditto — ambiguous without a port,
  see `subject_tag/3`), or the matching entry has no tag.
  """
  @spec subject_tag(Path.t(), String.t()) :: String.t() | nil
  def subject_tag(path, subject_character_name) when is_binary(subject_character_name) do
    want = String.downcase(subject_character_name)

    case parse(path) |> Enum.filter(fn {_tag, char} -> char == want end) do
      [{tag, _}] -> tag
      _ -> nil
    end
  end

  @doc """
  Like `subject_tag/2` but DITTO-CAPABLE: when both filename entries are
  the subject character, resolve by POSITION using the subject's port —
  the `"A + B"` convention lists players in port order (P1 + P2).

  The position<->port assumption is empirically checkable on non-ditto
  tagged files (filename position of the known-port character must match)
  — `scripts/style_fingerprint.exs --validate-order` runs that check;
  see STYLE_IDENTITY.md. Ports 3/4 and out-of-range positions return nil
  (anonymous bucket), as does everything `subject_tag/2` would nil.
  """
  @spec subject_tag(Path.t(), String.t(), pos_integer()) :: String.t() | nil
  def subject_tag(path, subject_character_name, port)
      when is_binary(subject_character_name) and is_integer(port) do
    want = String.downcase(subject_character_name)
    entries = parse(path)

    case Enum.filter(entries, fn {_tag, char} -> char == want end) do
      [{tag, _}] ->
        tag

      [_, _ | _] when port in [1, 2] ->
        case Enum.at(entries, port - 1) do
          {tag, ^want} -> tag
          _ -> nil
        end

      _ ->
        nil
    end
  end
end

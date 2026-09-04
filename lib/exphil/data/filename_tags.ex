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
  Parse a replay filename into its player entries, in filename order.

  Returns `[{tag_or_nil, character_name_lowercase}]` (2 entries for the
  `A + B` convention) or `[]` when the name doesn't follow it.
  """
  @spec parse(Path.t()) :: [{String.t() | nil, String.t()}]
  def parse(path) do
    base = path |> Path.basename() |> String.replace_suffix(".slp", "")

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

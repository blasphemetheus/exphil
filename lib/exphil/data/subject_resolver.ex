defmodule ExPhil.Data.SubjectResolver do
  @moduledoc """
  THE chokepoint for "which player is the subject?" — the role-vs-port
  law (V2_PREP.md, Bradley 2026-09-04).

  A port is a SEATING ASSIGNMENT, not a property of the player: keying on
  it caused the port-1 corpus corruption (GOTCHA #107), the 69.2%
  filename-position finding, and the ditto->port-1 traps. Ports may exist
  only at the parse boundary; everything downstream speaks ROLES
  (subject/opponent), resolved here through a ladder of semantic anchors:

    1. `:explicit`  — the caller states the subject port (drills, tests).
    2. `:identity`  — netplay name / connect code match (skips
       anonymization placeholders; strongest anchor when data has it).
    3. `:character` — the subject's character (what
       `--select-character-port` does). Dittos are AMBIGUOUS by
       construction: the default is a loud error, and any tie-break must
       be explicit and shows up in provenance.

  There is deliberately NO silent default. Every result carries
  `provenance` so downstream audits can ask "how was this subject
  chosen?" — the #107 lesson: loss descending does not verify identity;
  provenance checks do. (Behavioral resolution — fingerprint matching —
  is the rung above tie-breaks for dittos; it plugs in via
  `:subject_port` once the matcher assigns, see STYLE_IDENTITY.md.)
  """

  @type resolution :: %{
          subject_port: pos_integer(),
          opponent_port: pos_integer(),
          provenance: :explicit | :identity | :character | :character_tie_break
        }

  # External-CSS character ids (metadata space; fox = 2). Single source of
  # truth — pipeline.ex and scripts previously each carried a copy.
  @external_character_ids %{
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

  @placeholder_names [nil, "", "Master Player"]

  @doc """
  Resolve subject/opponent roles from a metadata player list.

  ## Options (the ladder, tried in this order)
    - `:subject_port` — explicit port; must be occupied.
    - `:subject_identity` — netplay name/connect code (exact,
      case-insensitive); anonymization placeholders never match.
    - `:subject_character` — external-CSS id (integer) or a name/atom
      accepted by `external_character_id/1`.
    - `:ditto_tie_break` — ONLY consulted when `:subject_character`
      matches multiple players: `:error` (default), `:port1` (lowest
      matching port — the historical streaming behavior, now explicit
      and provenance-marked), or `{:port, n}`.

  Returns `{:ok, resolution}` or `{:error, reason}` — never a silent
  default.
  """
  @spec resolve([map()], keyword()) :: {:ok, resolution()} | {:error, term()}
  def resolve(players, opts) when is_list(players) do
    cond do
      port = opts[:subject_port] ->
        with {:ok, _} <- occupied(players, port),
             {:ok, opp} <- opponent_of(players, port) do
          {:ok, %{subject_port: port, opponent_port: opp, provenance: :explicit}}
        end

      identity = opts[:subject_identity] ->
        resolve_by_identity(players, identity)

      character = opts[:subject_character] ->
        resolve_by_character(players, character, opts[:ditto_tie_break] || :error)

      true ->
        {:error, :no_anchor_given}
    end
  end

  @doc """
  External-CSS character id for a character given as id, atom, or name
  ("Captain Falcon", :fox, "fox", 2 all work).
  """
  @spec external_character_id(term()) :: {:ok, non_neg_integer()} | {:error, term()}
  def external_character_id(id) when is_integer(id) and id >= 0, do: {:ok, id}

  def external_character_id(char) do
    key = char |> to_string() |> String.downcase() |> String.replace(~r/[^a-z0-9]/, "")

    case Integer.parse(key) do
      {id, ""} ->
        {:ok, id}

      _ ->
        case Map.get(@external_character_ids, key) do
          nil -> {:error, {:unknown_character, char}}
          id -> {:ok, id}
        end
    end
  end

  @doc "Known character keys (for error messages)."
  @spec known_characters() :: [String.t()]
  def known_characters, do: Map.keys(@external_character_ids) |> Enum.sort()

  # -- ladder rungs ----------------------------------------------------------

  defp resolve_by_identity(players, identity) do
    want = String.downcase(to_string(identity))

    matches =
      Enum.filter(players, fn p ->
        name = Map.get(p, :netplay_name) || Map.get(p, :display_name) || Map.get(p, :tag)
        name not in @placeholder_names and String.downcase(to_string(name)) == want
      end)

    case matches do
      [%{port: port}] ->
        with {:ok, opp} <- opponent_of(players, port) do
          {:ok, %{subject_port: port, opponent_port: opp, provenance: :identity}}
        end

      [] ->
        {:error, {:identity_not_found, identity}}

      _ ->
        {:error, {:identity_ambiguous, identity}}
    end
  end

  defp resolve_by_character(players, character, tie_break) do
    with {:ok, want_id} <- external_character_id(character) do
      case Enum.filter(players, &(&1.character == want_id)) do
        [%{port: port}] ->
          with {:ok, opp} <- opponent_of(players, port) do
            {:ok, %{subject_port: port, opponent_port: opp, provenance: :character}}
          end

        [] ->
          {:error, {:character_absent, character}}

        multiple ->
          resolve_ditto(players, multiple, tie_break)
      end
    end
  end

  defp resolve_ditto(_players, _multiple, :error), do: {:error, :ditto_ambiguous}

  defp resolve_ditto(players, multiple, :port1) do
    %{port: port} = Enum.min_by(multiple, & &1.port)

    with {:ok, opp} <- opponent_of(players, port) do
      {:ok, %{subject_port: port, opponent_port: opp, provenance: :character_tie_break}}
    end
  end

  defp resolve_ditto(players, multiple, {:port, port}) do
    if Enum.any?(multiple, &(&1.port == port)) do
      with {:ok, opp} <- opponent_of(players, port) do
        {:ok, %{subject_port: port, opponent_port: opp, provenance: :character_tie_break}}
      end
    else
      {:error, {:tie_break_port_not_a_match, port}}
    end
  end

  # -- shared ----------------------------------------------------------------

  defp occupied(players, port) do
    case Enum.find(players, &(&1.port == port)) do
      nil -> {:error, {:port_not_occupied, port}}
      p -> {:ok, p}
    end
  end

  # Opponent = the first OTHER occupied port. Multi-player free-for-alls
  # aren't a training target; taking the first other seat matches
  # Streaming.opponent_port_for/2.
  defp opponent_of(players, subject_port) do
    case Enum.find(players, &(&1.port != subject_port)) do
      nil -> {:error, :no_opponent}
      %{port: p} -> {:ok, p}
    end
  end
end

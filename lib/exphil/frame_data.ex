defmodule ExPhil.FrameData do
  @moduledoc """
  Queryable per-character frame data (task #2): hitbox active windows,
  IASA, autocancel and landing lag for the universal moves, extracted
  from the ISO's character files (`priv/frame_data/Pl??.framedata.json`,
  produced by the meleeDat2Json -> meleeFrameDataExtractor pipeline;
  verified against community values — Fox nair strong 4-7 / weak 8-31).

  Consumers: the rewind viewer's "hitbox out" gating (action_frame
  within a move's hit window, replacing the whole-animation attack
  ring), the deferred `whiff_window`/`lcancel_window` situation labels,
  and the coach's frame-advantage vocabulary.

  Keyed by character ATOM (`Melee.Enums.Character` names) and the move
  names used by `ExPhil.Options`/the viewer (:nair, :fair, :jab1,
  :dash_attack, ...). Specials are not in the extraction (subaction ids
  not enumerated yet) — queries for them return nil.
  """

  # Melee.Enums.Character atom -> Pl file code
  @char_codes %{
    captain_falcon: "Ca",
    donkey_kong: "Dk",
    fox: "Fx",
    game_and_watch: "Gw",
    mr_game_and_watch: "Gw",
    kirby: "Kb",
    bowser: "Kp",
    link: "Lk",
    luigi: "Lg",
    mario: "Mr",
    marth: "Ms",
    mewtwo: "Mt",
    ness: "Ns",
    peach: "Pe",
    pikachu: "Pk",
    ice_climbers: "Pp",
    popo: "Pp",
    nana: "Nn",
    jigglypuff: "Pr",
    samus: "Ss",
    yoshi: "Ys",
    zelda: "Zd",
    sheik: "Sk",
    falco: "Fc",
    young_link: "Cl",
    dr_mario: "Dr",
    roy: "Fe",
    pichu: "Pc",
    ganondorf: "Gn"
  }

  @doc "Full move map for a character atom, or nil when no data."
  @spec data(atom() | String.t() | nil) :: map() | nil
  def data(char) do
    case code(char) do
      nil ->
        nil

      code ->
        key = {__MODULE__, code}

        case :persistent_term.get(key, :miss) do
          :miss ->
            loaded = load(code)
            :persistent_term.put(key, loaded)
            loaded

          hit ->
            hit
        end
    end
  end

  @doc "One move's entry (map with hitFrames/iasa/landingLag/...), or nil."
  @spec move(atom() | String.t() | nil, atom() | String.t()) :: map() | nil
  def move(char, move_name) do
    case data(char) do
      nil -> nil
      moves -> Map.get(moves, to_string(move_name))
    end
  end

  @doc """
  Hit windows for a move as `[{start, stop}]` (inclusive, 1-indexed
  animation frames), or nil when unknown. `hitbox_out?/3` is the
  per-frame predicate.
  """
  @spec hit_windows(atom() | String.t() | nil, atom() | String.t()) ::
          [{integer(), integer()}] | nil
  def hit_windows(char, move_name) do
    case move(char, move_name) do
      %{"hitFrames" => hf} -> Enum.map(hf, fn %{"start" => s, "end" => e} -> {s, e} end)
      _ -> nil
    end
  end

  @doc """
  Is a hitbox active on `action_frame` of this move? Returns nil when
  the move has no data (unknown != inactive — consumers should fall
  back to their animation-level heuristic).
  """
  @spec hitbox_out?(atom() | String.t() | nil, atom() | String.t(), number()) :: boolean() | nil
  def hitbox_out?(char, move_name, action_frame) do
    case hit_windows(char, move_name) do
      nil -> nil
      windows -> Enum.any?(windows, fn {s, e} -> action_frame >= s and action_frame <= e end)
    end
  end

  # ==========================================================================

  defp code(nil), do: nil
  defp code(char) when is_binary(char), do: code(normalize_atom(char))
  defp code(char) when is_atom(char), do: Map.get(@char_codes, char)

  defp normalize_atom(s) do
    key = s |> String.downcase() |> String.replace(~r/[^a-z0-9]/, "_")

    Enum.find(Map.keys(@char_codes), fn a ->
      to_string(a) |> String.replace("_", "") == String.replace(key, "_", "")
    end)
  end

  defp load(code) do
    path = Path.join([:code.priv_dir(:exphil), "frame_data", "Pl#{code}.framedata.json"])

    case File.read(path) do
      {:ok, raw} -> Jason.decode!(raw)
      _ -> nil
    end
  end
end

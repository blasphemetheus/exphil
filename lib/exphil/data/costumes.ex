defmodule ExPhil.Data.Costumes do
  @moduledoc """
  Costume slot -> colour name, per character.

  Source of truth: the Melee decompilation's per-character
  `ft??_Init_CostumeStrings[]` tables (read from `melee-sim-light`
  `src/melee/ft/chara/*/ft*_Init.c`, 2026-09-17). Slot i loads the i-th
  `Pl<Ch><Colour>.dat`; the two-letter colour codes are the game's own:
  Nr neutral/default, Or orange, La lavender, Gr green, Bu blue, Re red,
  Ye yellow, Bk black, Wh white, Aq aqua, Pi pink, Gy grey.

  The scene talks in perceived colours ("red Fox" = slot 1 `Or`, "blue
  Fox" = slot 2 `La`), so `name/2` returns the code and `perceived/2` the
  common name. Characters missing from the decomp roster (Kirby, Pichu,
  Roy, Young Link's table lives under `ftCLink`) return `:unknown`.
  Keys are the external/CSS character names Peppi reports.
  """

  @codes %{
    "Nr" => :default, "Or" => :orange, "La" => :lavender, "Gr" => :green, "Bu" => :blue,
    "Re" => :red, "Ye" => :yellow, "Bk" => :black, "Wh" => :white, "Aq" => :aqua,
    "Pi" => :pink, "Gy" => :grey
  }

  @tables %{
    "Fox" => ~w(Nr Or La Gr),
    "Falco" => ~w(Nr Re Bu Gr),
    "Captain Falcon" => ~w(Nr Gy Re Wh Gr Bu),
    "Marth" => ~w(Nr Re Gr Bk Wh),
    "Sheik" => ~w(Nr Re Bu Gr Wh),
    "Zelda" => ~w(Nr Re Bu Gr Wh),
    "Peach" => ~w(Nr Ye Wh Bu Gr),
    "Jigglypuff" => ~w(Nr Re Bu Gr Ye),
    "Samus" => ~w(Nr Pi Bk Gr La),
    "Ice Climbers" => ~w(Nr Gr Or Re),
    "Pikachu" => ~w(Nr Re Bu Gr),
    "Yoshi" => ~w(Nr Re Bu Ye Pi Aq),
    "Luigi" => ~w(Nr Wh Aq Pi),
    "Mario" => ~w(Nr Ye Bk Bu Gr),
    "Dr. Mario" => ~w(Nr Re Bu Gr Bk),
    "Donkey Kong" => ~w(Nr Bk Re Bu Gr),
    "Ganondorf" => ~w(Nr Re Bu Gr La),
    "Bowser" => ~w(Nr Re Bu Bk),
    "Link" => ~w(Nr Re Bu Bk Wh),
    "Young Link" => ~w(Nr Re Bu Wh Bk),
    "Ness" => ~w(Nr Ye Bu Gr),
    "Mewtwo" => ~w(Nr Re Bu Gr),
    "Mr. Game & Watch" => ~w(Nr)
  }

  # What the scene calls it (YETI_SCENE_PRIORS.md): Fox's orange reads
  # as red, its lavender as blue.
  @perceived %{{"Fox", :orange} => :red, {"Fox", :lavender} => :blue}

  @doc "Colour code for `character`'s costume `slot` (`:unknown` off-table)."
  @spec name(String.t(), integer() | nil) :: atom()
  def name(character, slot) when is_integer(slot) do
    case @tables[character] do
      nil -> :unknown
      codes -> codes |> Enum.at(slot) |> then(&Map.get(@codes, &1, :unknown))
    end
  end

  def name(_, _), do: :unknown

  @doc "The colour as the scene names it (falls back to `name/2`)."
  @spec perceived(String.t(), integer() | nil) :: atom()
  def perceived(character, slot) do
    n = name(character, slot)
    Map.get(@perceived, {character, n}, n)
  end

  @doc "All known character tables (code lists in slot order)."
  @spec tables() :: %{String.t() => [String.t()]}
  def tables, do: @tables
end

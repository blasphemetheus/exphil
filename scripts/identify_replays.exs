# Quick replay identifier: characters per port, duration, stage.
# Usage: mix run --no-compile scripts/identify_replays.exs path1.slp path2.slp ...
# Internal char IDs: Fox=1, Mewtwo=16, Zelda=19, GnW=24, Popo/Nana=10/11.

alias ExPhil.Data.Peppi

names = %{
  0 => "Mario", 1 => "Fox", 2 => "CFalcon", 3 => "DK", 4 => "Kirby",
  5 => "Bowser", 6 => "Link", 7 => "Sheik", 8 => "Ness", 9 => "Peach",
  10 => "Popo(ICs)", 11 => "Nana", 12 => "Pikachu", 13 => "Samus",
  14 => "Yoshi", 15 => "Jigglypuff", 16 => "Mewtwo", 17 => "Luigi",
  18 => "Marth", 19 => "Zelda", 20 => "YLink", 21 => "DrMario",
  22 => "Falco", 23 => "Pichu", 24 => "GnW", 25 => "Ganondorf", 26 => "Roy"
}

for path <- System.argv() do
  case Peppi.parse(Path.expand(path)) do
    {:ok, replay} ->
      frames =
        replay
        |> Peppi.to_training_frames(player_port: 1, opponent_port: 2)
        |> Enum.reject(&(&1.game_state.frame < 0))

      n = length(frames)

      chars =
        case frames do
          [f | _] ->
            f.game_state.players
            |> Enum.sort()
            |> Enum.map(fn {p, pl} ->
              c = pl && trunc(pl.character || -1)
              "P#{p}=#{Map.get(names, c, "id#{c}")}"
            end)
            |> Enum.join(" vs ")

          [] ->
            "EMPTY"
        end

      IO.puts("#{Path.basename(path)}: #{chars} | #{n}f (#{Float.round(n / 3600, 1)}min)")

    {:error, e} ->
      IO.puts("#{Path.basename(path)}: PARSE ERROR #{inspect(e)}")
  end
end

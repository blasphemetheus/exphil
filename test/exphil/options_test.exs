defmodule ExPhil.OptionsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Bridge.GameState
  alias ExPhil.Bridge.Player
  alias ExPhil.Options

  defp p(action, overrides \\ []) do
    struct(
      %Player{
        x: 0.0,
        y: 0.0,
        percent: 0.0,
        stock: 4,
        action: action,
        facing: 1,
        on_ground: true,
        jumps_left: 2
      },
      overrides
    )
  end

  defp states(actions_or_players) do
    actions_or_players
    |> Enum.with_index()
    |> Enum.map(fn {ap, i} ->
      player = if is_integer(ap), do: p(ap), else: ap
      %GameState{frame: i, stage: 32, players: %{1 => player, 2 => p(14)}}
    end)
  end

  defp options(list), do: list |> states() |> Options.events(1) |> Enum.map(& &1.option)

  test "attack entries fire once, on entry" do
    assert options([14, 14, 44, 44, 44, 14]) == [:jab]
    assert options([14, 65, 65, 66, 66]) == [:aerial]
  end

  test "grab and throw with direction" do
    events = states([14, 212, 216, 220, 220]) |> Options.events(1)
    assert [%{option: :grab}, %{option: :throw, meta: %{direction: :back}}] = events
  end

  test "dashdance fires after 3 direction flips within the window" do
    seq =
      [p(20, facing: 1), p(20, facing: 1), p(14)] ++
        [p(20, facing: -1), p(14), p(20, facing: 1), p(14), p(20, facing: -1)]

    opts = seq |> states() |> Options.events(1) |> Enum.map(& &1.option)
    assert :dashdance in opts
    # each dash entry also fires :dash
    assert Enum.count(opts, &(&1 == :dash)) == 4
  end

  test "wavedash: jumpsquat -> airdodge -> grounded" do
    seq = [
      p(14),
      p(24),
      p(24),
      p(236, on_ground: false),
      p(236, on_ground: false),
      p(14, on_ground: true)
    ]

    opts = seq |> states() |> Options.events(1) |> Enum.map(& &1.option)
    assert :wavedash in opts
    refute :airdodge in opts
    refute :waveland in opts
  end

  test "waveland: airdodge lands with no recent jumpsquat" do
    seq = [
      p(25, on_ground: false),
      p(236, on_ground: false),
      p(236, on_ground: false),
      p(14, on_ground: true)
    ]

    opts = seq |> states() |> Options.events(1) |> Enum.map(& &1.option)
    assert :waveland in opts
    assert :double_jump in opts
    # the raw airdodge entry (not near a jumpsquat) also fires
    assert :airdodge in opts
  end

  test "tech and getup family" do
    assert options([38, 199, 199]) == [:tech_in_place]
    assert options([38, 200]) == [:tech_roll]
    assert options([38, 183, 184, 187]) == [:missed_tech, :getup_attack]
  end

  test "ledge options fire only out of CliffWait" do
    assert options([252, 253, 253, 256, 257]) == [:ledge_attack]
    assert options([252, 253, 260, 261]) == [:ledge_jump]
    # getup animation entered from elsewhere does not count
    assert options([14, 256]) == []
  end

  test "defense options" do
    assert options([14, 178, 179]) == [:shield_on]
    assert options([14, 235]) == [:spotdodge]
    assert options([14, 233]) == [:roll_forward]
  end

  test "frequencies aggregates" do
    freq = states([14, 44, 14, 44, 14]) |> Options.events(1) |> Options.frequencies()
    assert freq[:jab] == 2
  end
end

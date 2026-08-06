defmodule ExPhil.Data.PeppiCharacterConventionTest do
  @moduledoc """
  Pins the TWO character-id conventions the NIF emits (2026-08-06, the
  Roy->-1 fix from libmelee_ex's differential work):

    * FRAME level (PlayerFrame.character): IDENTITY over the game's
      INTERNAL enum (Fox=1, Mewtwo=16, Roy=26, unknowns clamp to 32,
      never negative). Every checkpoint trained through 2026-08 learned
      THIS numbering — silently changing it scrambles character
      embeddings for all of them, which is why this test exists.
    * GAME-START level (PlayerMeta.character): the EXTERNAL (CSS) enum
      (Fox=2, Roy=23).

  The two are different enums that happen to agree nowhere useful; code
  must never mix them (that mixture WAS the bug: internal ids fed through
  the external table — numerically identity for 0x00..=0x19, so nothing
  noticed until Roy).
  """
  use ExUnit.Case, async: true

  @fixture "test/fixtures/replays/fox_multishine_closed.slp"

  test "frame-level ids are internal-identity; meta ids are external" do
    {:ok, replay} = ExPhil.Data.Peppi.parse(@fixture)

    # Fox dittos: internal Fox = 0x01 -> frame id 1 on every port
    frame_chars =
      replay.frames
      |> Enum.take(300)
      |> Enum.flat_map(fn f -> Enum.map(f.players, fn {_port, p} -> p.character end) end)
      |> Enum.uniq()

    assert frame_chars == [1],
           "frame-level character ids drifted (#{inspect(frame_chars)}) — pre-2026-08 " <>
             "checkpoints depend on internal-identity (Fox=1); see moduledoc"

    # Game-start metas: external Fox = 0x02 -> 2
    assert Enum.map(replay.metadata.players, & &1.character) |> Enum.uniq() == [2]

    # No frame-level character may ever be negative (Roy was -1 pre-fix;
    # a negative embedding index fails silently or catastrophically).
    refute replay.frames
           |> Enum.flat_map(fn f -> Enum.map(f.players, fn {_p, pl} -> pl.character end) end)
           |> Enum.any?(&(&1 < 0))
  end
end

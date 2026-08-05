defmodule ExPhil.Replay.StampTest do
  use ExUnit.Case, async: true

  alias ExPhil.Replay.Stamp

  # A minimal .slp tail: the metadata block as Slippi writes it for a
  # local game (empty names for both players).
  defp metadata_blob do
    <<0x55, 8, "metadata", ?{, 0x55, 7, "startAt", ?S, 0x55, 20, "2026-08-05T19:32:58Z", 0x55, 9,
      "lastFrame", ?l, 0, 0, 13, 153, 0x55, 7, "players", ?{, 0x55, 1, ?1, ?{, 0x55, 5, "names",
      ?{, ?}, 0x55, 10, "characters", ?{, 0x55, 1, ?1, ?l, 0, 0, 14, 21, ?}, ?}, 0x55, 1, ?0, ?{,
      0x55, 5, "names", ?{, ?}, 0x55, 10, "characters", ?{, 0x55, 1, ?1, ?l, 0, 0, 14, 21, ?}, ?},
      ?}, 0x55, 8, "playedOn", ?S, 0x55, 16, "mainline dolphin", ?}, ?}>>
  end

  defp fake_replay do
    raw = :binary.copy(<<0x36, 0x03>>, 20)
    <<"{U", 3, "raw[$U#l", byte_size(raw)::big-unsigned-32>> <> raw <> metadata_blob()
  end

  defp temp_file(ctx) do
    path = Path.join(System.tmp_dir!(), "stamp_#{:erlang.phash2(ctx.test)}.slp")
    File.write!(path, fake_replay())
    on_exit(fn -> File.rm(path) end)
    path
  end

  test "stamps the requested port and reads back", ctx do
    path = temp_file(ctx)

    assert Stamp.names(path) == %{0 => "", 1 => "", 2 => "", 3 => ""}
    assert {:ok, 1} = Stamp.stamp_file(path, %{0 => "exph"})

    names = Stamp.names(path)
    assert names[0] == "exph"
    assert names[1] == ""
  end

  test "stamps multiple ports", ctx do
    path = temp_file(ctx)
    assert {:ok, 2} = Stamp.stamp_file(path, %{0 => "exph", 1 => "dummy"})

    names = Stamp.names(path)
    assert names[0] == "exph"
    assert names[1] == "dummy"
  end

  test "does not clobber an existing name", ctx do
    path = temp_file(ctx)
    assert {:ok, 1} = Stamp.stamp_file(path, %{0 => "exph"})
    # Second stamp finds no empty names slot for port 0 -> no change.
    assert {:ok, 0} = Stamp.stamp_file(path, %{0 => "other"})
    assert Stamp.names(path)[0] == "exph"
  end

  test "leaves the raw game data byte-identical", ctx do
    path = temp_file(ctx)
    before = File.read!(path)

    <<header::binary-size(10), len::big-unsigned-32, raw_before::binary-size(80), _::binary>> =
      before

    assert {:ok, 1} = Stamp.stamp_file(path, %{0 => "exph"})

    <<^header::binary-size(10), ^len::big-unsigned-32, raw_after::binary-size(80), _::binary>> =
      File.read!(path)

    assert raw_after == raw_before
  end

  test "the stamped replay still parses as a replay", ctx do
    path = temp_file(ctx)
    assert {:ok, 1} = Stamp.stamp_file(path, %{0 => "exph"})

    # The container header + raw length must still describe the payload.
    <<"{U", 3, "raw[$U#l", len::big-unsigned-32, rest::binary>> = File.read!(path)
    assert byte_size(rest) > len
    assert :binary.match(rest, "metadata") != :nomatch
  end

  test "refuses a file with no metadata block" do
    assert {:error, :no_metadata} = Stamp.stamp_binary(<<"{U", 3, "raw[$U#l", 0::32>>, %{0 => "x"})
  end

  test "stamp_dir walks a tree", ctx do
    dir = Path.join(System.tmp_dir!(), "stampdir_#{:erlang.phash2(ctx.test)}")
    File.mkdir_p!(Path.join(dir, "sub"))
    File.write!(Path.join(dir, "a.slp"), fake_replay())
    File.write!(Path.join([dir, "sub", "b.slp"]), fake_replay())
    on_exit(fn -> File.rm_rf!(dir) end)

    assert {2, 0} = Stamp.stamp_dir(dir, %{0 => "exph"})
    assert Stamp.names(Path.join(dir, "a.slp"))[0] == "exph"
    assert Stamp.names(Path.join([dir, "sub", "b.slp"]))[0] == "exph"
  end
end

defmodule ExPhil.Data.SlpRepairTest do
  use ExUnit.Case, async: true

  alias ExPhil.Data.{Peppi, SlpRepair}

  @fixture "test/fixtures/replays/fox_multishine_closed.slp"

  defp truncate_fixture(drop_bytes) do
    bin = File.read!(@fixture)
    tmp = Path.join(System.tmp_dir!(), "slp_trunc_#{:erlang.unique_integer([:positive])}.slp")
    File.write!(tmp, binary_part(bin, 0, byte_size(bin) - drop_bytes))
    on_exit(fn -> File.rm(tmp) end)
    tmp
  end

  test "strict parse rejects a truncated replay; repair makes it parseable" do
    # Chop off the metadata AND a partial event: an odd offset lands mid-event
    # with overwhelming probability given per-frame events every few hundred
    # bytes.
    tmp = truncate_fixture(5000 + 137)

    assert {:error, _} = Peppi.parse(tmp)

    assert {:ok, repaired, stats} = SlpRepair.repair(tmp)
    on_exit(fn -> File.rm(repaired) end)
    assert stats.events > 0
    assert stats.dropped_bytes >= 0

    assert {:ok, replay} = Peppi.parse(repaired)
    frames = Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)
    assert length(frames) > 100
  end

  test "parse_lenient falls back to repair transparently" do
    tmp = truncate_fixture(5000 + 137)
    assert {:ok, replay} = SlpRepair.parse_lenient(tmp)
    assert length(Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)) > 100
  end

  test "parse_lenient on an intact replay uses the strict path" do
    assert {:ok, replay} = SlpRepair.parse_lenient(@fixture)
    assert length(Peppi.to_training_frames(replay, player_port: 1, opponent_port: 2)) > 100
  end

  test "repair rejects a non-slippi file" do
    tmp = Path.join(System.tmp_dir!(), "not_slp_#{:erlang.unique_integer([:positive])}")
    File.write!(tmp, "definitely not a replay")
    on_exit(fn -> File.rm(tmp) end)
    assert {:error, :not_a_slippi_replay} = SlpRepair.repair(tmp)
  end
end

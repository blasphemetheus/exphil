defmodule ExPhil.Training.SnapshotOptionTest do
  use ExUnit.Case, async: true

  test "the drill snapshot guard accepts omitted, disabled, and enabled CLI options" do
    source = File.read!(Path.expand("../../../scripts/dagger_drill.exs", __DIR__))
    [_, guard] = Regex.run(~r/^\s*if (opts\[:snapshot_all\].*) do$/m, source)

    for {argv, expected} <- [
          {[], false},
          {["--no-snapshot-all"], false},
          {["--snapshot-all"], true}
        ] do
      {opts, [], []} = OptionParser.parse(argv, strict: [snapshot_all: :boolean])
      assert {^expected, _} = Code.eval_string(guard, opts: opts, loss: 1.0)
      assert {false, _} = Code.eval_string(guard, opts: opts, loss: :nonfinite_batch_loss)
    end
  end
end

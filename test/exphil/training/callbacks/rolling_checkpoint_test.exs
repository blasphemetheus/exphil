defmodule ExPhil.Training.Callbacks.RollingCheckpointTest do
  use ExUnit.Case, async: true
  alias ExPhil.Training.{Imitation, TrainingState}
  alias ExPhil.Training.Callbacks.RollingCheckpoint

  setup do
    dir = Path.join(System.tmp_dir!(), "rolling_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    trainer = Imitation.new(embed_size: 16, hidden_sizes: [8], temporal: false, precision: :f32)
    state = %TrainingState{trainer: trainer, opts: [seed: 905], epoch: 1}

    %{
      dir: dir,
      state: state,
      cb: RollingCheckpoint.init(checkpoint_path: Path.join(dir, "model.axon"), every: 2)
    }
  end

  test "saves before training and after the first update, retaining optimizer and metadata",
       ctx do
    {:cont, _, _} = RollingCheckpoint.on_train_begin(ctx.state, ctx.cb)
    {:cont, _, _} = RollingCheckpoint.on_batch_end(%{ctx.state | step: 1, batch_idx: 1}, ctx.cb)

    for slot <- ["initial", "0"] do
      data =
        File.read!(Path.join(ctx.dir, "model_resume_#{slot}.axon")) |> :erlang.binary_to_term()

      assert data.optimizer_state != nil
      assert data.policy_params != nil
      assert data.meta.seed == 905
      refute data.meta.data_cursor_restored
      assert data.meta.step == if(slot == "initial", do: 0, else: 1)
    end
  end

  test "rotates only two recovery slots and records newest step", ctx do
    RollingCheckpoint.on_train_begin(ctx.state, ctx.cb)
    for step <- 1..10, do: RollingCheckpoint.on_batch_end(%{ctx.state | step: step}, ctx.cb)

    assert Enum.sort(File.ls!(ctx.dir)) == [
             "model_resume_0.axon",
             "model_resume_1.axon",
             "model_resume_initial.axon"
           ]

    latest = File.read!(Path.join(ctx.dir, "model_resume_1.axon")) |> :erlang.binary_to_term()
    assert latest.meta.step == 10
  end

  test "write failure aborts and preserves previous recovery checkpoint", ctx do
    RollingCheckpoint.on_batch_end(%{ctx.state | step: 1}, ctx.cb)
    path = Path.join(ctx.dir, "model_resume_0.axon")
    previous = File.read!(path)
    File.mkdir_p!(path <> ".tmp")

    assert_raise RuntimeError, ~r/Recovery checkpoint failed/, fn ->
      RollingCheckpoint.on_batch_end(%{ctx.state | step: 4}, ctx.cb)
    end

    assert File.read!(path) == previous
  end

  test "invalid intervals fail before training" do
    for interval <- [0, -1, 1.5] do
      assert_raise ArgumentError, fn ->
        RollingCheckpoint.init(checkpoint_path: "x.axon", every: interval)
      end
    end
  end

  test "unsupported filename cannot overwrite the primary checkpoint" do
    assert_raise ArgumentError, ~r/must end in .axon/, fn ->
      RollingCheckpoint.init(checkpoint_path: "model.bin")
    end
  end
end

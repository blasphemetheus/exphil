defmodule ExPhil.Training.RecoveryTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Recovery

  describe "marker_path/1" do
    test "appends .incomplete suffix" do
      assert Recovery.marker_path("checkpoints/model.axon") == "checkpoints/model.axon.incomplete"
    end
  end

  describe "check_incomplete/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      checkpoint = Path.join(tmp_dir, "test_recovery_#{:erlang.unique_integer([:positive])}.axon")

      on_exit(fn ->
        File.rm(checkpoint)
        File.rm(Recovery.marker_path(checkpoint))
      end)

      %{checkpoint: checkpoint}
    end

    test "returns :ok when no marker exists", %{checkpoint: checkpoint} do
      assert Recovery.check_incomplete(checkpoint) == :ok
    end

    test "returns {:incomplete, state} when marker exists", %{checkpoint: checkpoint} do
      Recovery.mark_started(checkpoint, epochs: 10, preset: :quick)

      assert {:incomplete, state} = Recovery.check_incomplete(checkpoint)
      assert state["epochs_target"] == 10
      assert state["epochs_completed"] == 0
      assert state["preset"] == "quick"
    end
  end

  describe "mark_started/2" do
    setup do
      tmp_dir = System.tmp_dir!()
      checkpoint = Path.join(tmp_dir, "test_start_#{:erlang.unique_integer([:positive])}.axon")
      on_exit(fn -> File.rm(Recovery.marker_path(checkpoint)) end)
      %{checkpoint: checkpoint}
    end

    test "creates marker file with initial state", %{checkpoint: checkpoint} do
      assert :ok = Recovery.mark_started(checkpoint, epochs: 20, preset: :full)

      marker = Recovery.marker_path(checkpoint)
      assert File.exists?(marker)

      {:ok, content} = File.read(marker)
      {:ok, state} = Jason.decode(content)
      assert state["epochs_target"] == 20
      assert state["epochs_completed"] == 0
    end
  end

  describe "mark_epoch_complete/3" do
    setup do
      tmp_dir = System.tmp_dir!()
      checkpoint = Path.join(tmp_dir, "test_epoch_#{:erlang.unique_integer([:positive])}.axon")
      Recovery.mark_started(checkpoint, epochs: 10, preset: :quick)
      on_exit(fn -> File.rm(Recovery.marker_path(checkpoint)) end)
      %{checkpoint: checkpoint}
    end

    test "updates marker with epoch progress", %{checkpoint: checkpoint} do
      assert :ok = Recovery.mark_epoch_complete(checkpoint, 5, 0.1234)

      {:incomplete, state} = Recovery.check_incomplete(checkpoint)
      assert state["epochs_completed"] == 5
      assert_in_delta state["last_epoch_loss"], 0.1234, 0.0001
    end
  end

  describe "mark_complete/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      checkpoint = Path.join(tmp_dir, "test_complete_#{:erlang.unique_integer([:positive])}.axon")
      Recovery.mark_started(checkpoint, epochs: 10, preset: :quick)
      %{checkpoint: checkpoint}
    end

    test "removes marker file", %{checkpoint: checkpoint} do
      marker = Recovery.marker_path(checkpoint)
      assert File.exists?(marker)

      assert :ok = Recovery.mark_complete(checkpoint)
      refute File.exists?(marker)
    end

    test "returns :ok when marker doesn't exist", %{checkpoint: checkpoint} do
      # Remove marker first
      Recovery.mark_complete(checkpoint)

      # Should still return :ok
      assert :ok = Recovery.mark_complete(checkpoint)
    end
  end

  describe "get_resume_checkpoint/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      checkpoint = Path.join(tmp_dir, "test_resume_#{:erlang.unique_integer([:positive])}.axon")
      best_checkpoint = String.replace(checkpoint, ".axon", "_best.axon")

      on_exit(fn ->
        File.rm(checkpoint)
        File.rm(best_checkpoint)
      end)

      %{checkpoint: checkpoint, best: best_checkpoint}
    end

    test "returns nil when no checkpoint exists", %{checkpoint: checkpoint} do
      assert Recovery.get_resume_checkpoint(checkpoint) == nil
    end

    test "returns base checkpoint when it exists", %{checkpoint: checkpoint} do
      File.write!(checkpoint, "test")
      assert Recovery.get_resume_checkpoint(checkpoint) == checkpoint
    end

    test "prefers best checkpoint over base", %{checkpoint: checkpoint, best: best} do
      File.write!(checkpoint, "test")
      File.write!(best, "best test")
      assert Recovery.get_resume_checkpoint(checkpoint) == best
    end
  end

  describe "format_incomplete_info/1" do
    test "formats state as readable string" do
      state = %{
        "epochs_completed" => 5,
        "epochs_target" => 10,
        "started_at" => "2025-01-20T12:00:00Z",
        "last_epoch_loss" => 0.1234,
        "preset" => "quick"
      }

      info = Recovery.format_incomplete_info(state)
      assert info =~ "Incomplete training detected"
      assert info =~ "5/10 epochs"
      assert info =~ "loss: 0.1234"
      assert info =~ "preset: quick"
    end
  end
end

defmodule ExPhil.Training.CheckpointPruningTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.CheckpointPruning

  @test_dir "test/fixtures/pruning_test"

  setup do
    # Create test directory
    File.mkdir_p!(@test_dir)

    on_exit(fn ->
      File.rm_rf(@test_dir)
    end)

    :ok
  end

  describe "new/1" do
    test "creates pruner with default options" do
      pruner = CheckpointPruning.new()

      assert pruner.keep_best == 5
      assert pruner.metric == :loss
      assert pruner.checkpoints == []
    end

    test "creates pruner with custom options" do
      pruner = CheckpointPruning.new(keep_best: 3, metric: :accuracy)

      assert pruner.keep_best == 3
      assert pruner.metric == :accuracy
    end
  end

  describe "track/4" do
    test "adds checkpoint to tracking list" do
      pruner = CheckpointPruning.new()
      pruner = CheckpointPruning.track(pruner, "checkpoint_1.axon", 1.5)

      assert CheckpointPruning.count(pruner) == 1
    end

    test "tracks multiple checkpoints" do
      pruner =
        CheckpointPruning.new()
        |> CheckpointPruning.track("checkpoint_1.axon", 1.5)
        |> CheckpointPruning.track("checkpoint_2.axon", 1.2)
        |> CheckpointPruning.track("checkpoint_3.axon", 1.8)

      assert CheckpointPruning.count(pruner) == 3
    end

    test "tracks epoch with checkpoint" do
      pruner = CheckpointPruning.new()
      pruner = CheckpointPruning.track(pruner, "checkpoint.axon", 1.5, epoch: 5)

      [checkpoint] = CheckpointPruning.list(pruner)
      assert checkpoint.epoch == 5
    end
  end

  describe "list/1" do
    test "returns checkpoints sorted by loss (lower first)" do
      pruner =
        CheckpointPruning.new(metric: :loss)
        |> CheckpointPruning.track("high.axon", 2.0)
        |> CheckpointPruning.track("low.axon", 1.0)
        |> CheckpointPruning.track("mid.axon", 1.5)

      sorted = CheckpointPruning.list(pruner)

      assert Enum.at(sorted, 0).value == 1.0
      assert Enum.at(sorted, 1).value == 1.5
      assert Enum.at(sorted, 2).value == 2.0
    end

    test "returns checkpoints sorted by accuracy (higher first)" do
      pruner =
        CheckpointPruning.new(metric: :accuracy)
        |> CheckpointPruning.track("low.axon", 0.8)
        |> CheckpointPruning.track("high.axon", 0.95)
        |> CheckpointPruning.track("mid.axon", 0.9)

      sorted = CheckpointPruning.list(pruner)

      assert Enum.at(sorted, 0).value == 0.95
      assert Enum.at(sorted, 1).value == 0.9
      assert Enum.at(sorted, 2).value == 0.8
    end
  end

  describe "best/1" do
    test "returns nil for empty pruner" do
      pruner = CheckpointPruning.new()
      assert CheckpointPruning.best(pruner) == nil
    end

    test "returns best checkpoint by loss" do
      pruner =
        CheckpointPruning.new(metric: :loss)
        |> CheckpointPruning.track("high.axon", 2.0)
        |> CheckpointPruning.track("low.axon", 1.0)

      best = CheckpointPruning.best(pruner)
      assert best.path == "low.axon"
      assert best.value == 1.0
    end

    test "returns best checkpoint by accuracy" do
      pruner =
        CheckpointPruning.new(metric: :accuracy)
        |> CheckpointPruning.track("low.axon", 0.8)
        |> CheckpointPruning.track("high.axon", 0.95)

      best = CheckpointPruning.best(pruner)
      assert best.path == "high.axon"
      assert best.value == 0.95
    end
  end

  describe "needs_pruning?/1" do
    test "returns false when under limit" do
      pruner =
        CheckpointPruning.new(keep_best: 3)
        |> CheckpointPruning.track("a.axon", 1.0)
        |> CheckpointPruning.track("b.axon", 2.0)

      refute CheckpointPruning.needs_pruning?(pruner)
    end

    test "returns false when at limit" do
      pruner =
        CheckpointPruning.new(keep_best: 2)
        |> CheckpointPruning.track("a.axon", 1.0)
        |> CheckpointPruning.track("b.axon", 2.0)

      refute CheckpointPruning.needs_pruning?(pruner)
    end

    test "returns true when over limit" do
      pruner =
        CheckpointPruning.new(keep_best: 2)
        |> CheckpointPruning.track("a.axon", 1.0)
        |> CheckpointPruning.track("b.axon", 2.0)
        |> CheckpointPruning.track("c.axon", 3.0)

      assert CheckpointPruning.needs_pruning?(pruner)
    end
  end

  describe "prune/2" do
    test "keeps best checkpoints and returns deleted paths (dry run)" do
      pruner =
        CheckpointPruning.new(keep_best: 2, metric: :loss)
        |> CheckpointPruning.track("best.axon", 1.0)
        |> CheckpointPruning.track("mid.axon", 1.5)
        |> CheckpointPruning.track("worst.axon", 2.0)

      {new_pruner, deleted} = CheckpointPruning.prune(pruner, dry_run: true)

      # Should keep best 2 (best.axon and mid.axon)
      assert CheckpointPruning.count(new_pruner) == 2
      assert deleted == ["worst.axon"]

      # Check remaining checkpoints
      remaining_paths = Enum.map(CheckpointPruning.list(new_pruner), & &1.path)
      assert "best.axon" in remaining_paths
      assert "mid.axon" in remaining_paths
      refute "worst.axon" in remaining_paths
    end

    test "actually deletes files when not dry run" do
      # Create test checkpoint files
      best_path = Path.join(@test_dir, "best.axon")
      worst_path = Path.join(@test_dir, "worst.axon")

      File.write!(best_path, "best checkpoint")
      File.write!(worst_path, "worst checkpoint")

      pruner =
        CheckpointPruning.new(keep_best: 1, metric: :loss)
        |> CheckpointPruning.track(best_path, 1.0)
        |> CheckpointPruning.track(worst_path, 2.0)

      {_new_pruner, deleted} = CheckpointPruning.prune(pruner)

      assert deleted == [worst_path]
      assert File.exists?(best_path)
      refute File.exists?(worst_path)
    end

    test "deletes related files (policy.bin, config.json)" do
      # Create checkpoint with related files
      checkpoint_path = Path.join(@test_dir, "model.axon")
      policy_path = Path.join(@test_dir, "model_policy.bin")
      config_path = Path.join(@test_dir, "model_config.json")

      File.write!(checkpoint_path, "checkpoint")
      File.write!(policy_path, "policy")
      File.write!(config_path, "config")

      pruner =
        CheckpointPruning.new(keep_best: 0, metric: :loss)
        |> CheckpointPruning.track(checkpoint_path, 1.0)

      {_new_pruner, _deleted} = CheckpointPruning.prune(pruner)

      refute File.exists?(checkpoint_path)
      refute File.exists?(policy_path)
      refute File.exists?(config_path)
    end
  end

  describe "untrack/2" do
    test "removes checkpoint from tracking" do
      pruner =
        CheckpointPruning.new()
        |> CheckpointPruning.track("a.axon", 1.0)
        |> CheckpointPruning.track("b.axon", 2.0)

      pruner = CheckpointPruning.untrack(pruner, "a.axon")

      assert CheckpointPruning.count(pruner) == 1
      remaining = CheckpointPruning.list(pruner)
      assert hd(remaining).path == "b.axon"
    end
  end

  describe "prune_directory/2" do
    test "prunes old checkpoints from directory" do
      # Create test checkpoints with different timestamps
      for i <- 1..5 do
        path = Path.join(@test_dir, "model_#{i}.axon")
        File.write!(path, "checkpoint #{i}")
        # Small delay to ensure different mtimes
        Process.sleep(10)
      end

      {:ok, deleted} = CheckpointPruning.prune_directory(@test_dir, keep_best: 2)

      # Should delete 3 oldest files
      assert length(deleted) == 3

      # Check which files remain (should be newest 2)
      remaining = Path.wildcard(Path.join(@test_dir, "*.axon"))
      assert length(remaining) == 2
    end

    test "dry run doesn't delete files" do
      for i <- 1..3 do
        path = Path.join(@test_dir, "model_#{i}.axon")
        File.write!(path, "checkpoint #{i}")
      end

      {:ok, deleted} = CheckpointPruning.prune_directory(@test_dir, keep_best: 1, dry_run: true)

      # Returns paths that would be deleted
      assert length(deleted) == 2

      # But files still exist
      remaining = Path.wildcard(Path.join(@test_dir, "*.axon"))
      assert length(remaining) == 3
    end

    test "returns empty list for empty directory" do
      {:ok, deleted} = CheckpointPruning.prune_directory(@test_dir)
      assert deleted == []
    end
  end
end

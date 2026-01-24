defmodule ExPhil.Training.CheckpointTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.Checkpoint

  import ExUnit.CaptureIO

  describe "validate_embed_size/2" do
    test "returns :ok when no current_embed_size provided" do
      config = %{embed_size: 1991}
      assert :ok = Checkpoint.validate_embed_size(config, [])
    end

    test "returns :ok when sizes match" do
      config = %{embed_size: 1204}
      assert :ok = Checkpoint.validate_embed_size(config, current_embed_size: 1204)
    end

    test "returns :ok when checkpoint has no embed_size" do
      config = %{}
      assert :ok = Checkpoint.validate_embed_size(config, current_embed_size: 1204)
    end

    test "returns warning when sizes mismatch" do
      config = %{embed_size: 1991}
      assert {:warning, message} = Checkpoint.validate_embed_size(config, current_embed_size: 1204)
      assert message =~ "Embed size mismatch"
      assert message =~ "1991"
      assert message =~ "1204"
    end

    test "returns error when sizes mismatch and error_on_mismatch is true" do
      config = %{embed_size: 1991}
      result = Checkpoint.validate_embed_size(config,
        current_embed_size: 1204,
        error_on_mismatch: true
      )
      assert {:error, message} = result
      assert message =~ "Embed size mismatch"
    end
  end

  describe "get_embed_size/2" do
    test "returns embed_size from config" do
      config = %{embed_size: 1991}
      assert 1991 = Checkpoint.get_embed_size(config, 1024)
    end

    test "returns fallback when config has no embed_size" do
      config = %{}
      assert 1024 = Checkpoint.get_embed_size(config, 1024)
    end

    test "returns fallback when embed_size is nil" do
      config = %{embed_size: nil}
      assert 1024 = Checkpoint.get_embed_size(config, 1024)
    end
  end

  describe "config_diff/2" do
    test "returns empty list when configs match" do
      checkpoint_config = %{embed_size: 1204, hidden_sizes: [512, 512]}
      current_opts = [embed_size: 1204, hidden_sizes: [512, 512]]

      assert [] = Checkpoint.config_diff(checkpoint_config, current_opts)
    end

    test "returns differences for mismatched values" do
      checkpoint_config = %{embed_size: 1991, hidden_sizes: [256, 256]}
      current_opts = [embed_size: 1204, hidden_sizes: [512, 512]]

      diffs = Checkpoint.config_diff(checkpoint_config, current_opts)

      assert {:embed_size, {1991, 1204}} in diffs
      assert {:hidden_sizes, {[256, 256], [512, 512]}} in diffs
    end

    test "ignores keys not present in both" do
      checkpoint_config = %{embed_size: 1991, backbone: :mamba}
      current_opts = [embed_size: 1204]  # no backbone

      diffs = Checkpoint.config_diff(checkpoint_config, current_opts)

      assert {:embed_size, {1991, 1204}} in diffs
      refute Enum.any?(diffs, fn {k, _} -> k == :backbone end)
    end
  end

  describe "load/2 and load_policy/2" do
    setup do
      # Create temp directory for test files
      tmp_dir = Path.join(System.tmp_dir!(), "exphil_checkpoint_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(tmp_dir)
      on_exit(fn -> File.rm_rf!(tmp_dir) end)
      {:ok, tmp_dir: tmp_dir}
    end

    test "loads checkpoint file successfully", %{tmp_dir: tmp_dir} do
      checkpoint = %{
        policy_params: %{layer1: :fake_params},
        optimizer_state: %{},
        config: %{embed_size: 1204},
        step: 100,
        metrics: %{loss: [0.5]}
      }

      path = Path.join(tmp_dir, "test.axon")
      File.write!(path, :erlang.term_to_binary(checkpoint))

      assert {:ok, loaded} = Checkpoint.load(path)
      assert loaded.step == 100
      assert loaded.config.embed_size == 1204
    end

    test "loads policy file successfully", %{tmp_dir: tmp_dir} do
      export = %{
        params: %{layer1: :fake_params},
        config: %{embed_size: 1204}
      }

      path = Path.join(tmp_dir, "test_policy.bin")
      File.write!(path, :erlang.term_to_binary(export))

      assert {:ok, loaded} = Checkpoint.load_policy(path)
      assert loaded.config.embed_size == 1204
    end

    test "warns on embed size mismatch", %{tmp_dir: tmp_dir} do
      checkpoint = %{
        policy_params: %{},
        optimizer_state: %{},
        config: %{embed_size: 1991},
        step: 0,
        metrics: %{}
      }

      path = Path.join(tmp_dir, "mismatch.axon")
      File.write!(path, :erlang.term_to_binary(checkpoint))

      output = capture_io(:stderr, fn ->
        {:ok, _} = Checkpoint.load(path, current_embed_size: 1204)
      end)

      assert output =~ "Embed size mismatch" or output =~ "1991"
    end

    test "returns error for missing file" do
      assert {:error, {:file_read, :enoent}} = Checkpoint.load("/nonexistent/path.axon")
    end
  end
end

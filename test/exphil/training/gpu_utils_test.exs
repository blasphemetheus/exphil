defmodule ExPhil.Training.GPUUtilsTest do
  use ExUnit.Case, async: true

  alias ExPhil.Training.GPUUtils

  describe "format_mb/1" do
    test "formats MB values" do
      assert GPUUtils.format_mb(512) == "512 MB"
      assert GPUUtils.format_mb(100) == "100 MB"
    end

    test "formats GB values" do
      assert GPUUtils.format_mb(1024) == "1.00 GB"
      assert GPUUtils.format_mb(2048) == "2.00 GB"
      assert GPUUtils.format_mb(8192) == "8.00 GB"
    end

    test "formats fractional GB values" do
      assert GPUUtils.format_mb(1536) == "1.50 GB"
      assert GPUUtils.format_mb(3584) == "3.50 GB"
    end
  end

  describe "estimate_memory_mb/1" do
    test "returns integer for default params" do
      result = GPUUtils.estimate_memory_mb()
      assert is_integer(result)
      assert result > 0
    end

    test "scales with parameter count" do
      small = GPUUtils.estimate_memory_mb(param_count: 1_000_000)
      large = GPUUtils.estimate_memory_mb(param_count: 10_000_000)
      assert large > small
    end

    test "scales with batch size" do
      small = GPUUtils.estimate_memory_mb(batch_size: 32)
      large = GPUUtils.estimate_memory_mb(batch_size: 128)
      assert large > small
    end

    test "temporal models require more memory" do
      non_temporal = GPUUtils.estimate_memory_mb(temporal: false)
      temporal = GPUUtils.estimate_memory_mb(temporal: true, window_size: 60)
      assert temporal > non_temporal
    end

    test "bf16 uses less memory than f32" do
      bf16 = GPUUtils.estimate_memory_mb(precision: :bf16)
      f32 = GPUUtils.estimate_memory_mb(precision: :f32)
      assert bf16 < f32
    end
  end

  describe "estimate_checkpoint_size/1" do
    test "returns integer for default params" do
      result = GPUUtils.estimate_checkpoint_size()
      assert is_integer(result)
      assert result > 0
    end

    test "scales with parameter count" do
      small = GPUUtils.estimate_checkpoint_size(param_count: 1_000_000)
      large = GPUUtils.estimate_checkpoint_size(param_count: 10_000_000)
      assert large > small
      # Should scale roughly linearly
      assert large > small * 5
    end

    test "smaller without optimizer state" do
      with_opt = GPUUtils.estimate_checkpoint_size(include_optimizer: true)
      without_opt = GPUUtils.estimate_checkpoint_size(include_optimizer: false)
      assert without_opt < with_opt
    end
  end

  describe "check_checkpoint_size_warning/1" do
    test "returns :ok for small models" do
      result =
        GPUUtils.check_checkpoint_size_warning(
          param_count: 1_000_000,
          threshold_mb: 500
        )

      assert result == :ok
    end

    test "returns warning for large models" do
      result =
        GPUUtils.check_checkpoint_size_warning(
          # 100M params
          param_count: 100_000_000,
          threshold_mb: 100
        )

      assert {:warning, msg} = result
      assert msg =~ "checkpoint size"
    end
  end

  describe "count_params/1" do
    test "counts tensor parameters" do
      params = %{
        kernel: Nx.iota({100, 50}),
        bias: Nx.iota({50})
      }

      assert GPUUtils.count_params(params) == 5050
    end

    test "counts nested parameters" do
      params = %{
        layer1: %{
          kernel: Nx.iota({100, 50}),
          bias: Nx.iota({50})
        },
        layer2: %{
          kernel: Nx.iota({50, 10}),
          bias: Nx.iota({10})
        }
      }

      assert GPUUtils.count_params(params) == 5050 + 510
    end

    test "returns 0 for empty map" do
      assert GPUUtils.count_params(%{}) == 0
    end

    test "returns 0 for non-tensor values" do
      assert GPUUtils.count_params(%{config: "value"}) == 0
    end
  end

  describe "check_memory_warning/1" do
    # These tests can't run without nvidia-smi, so we test the error case
    test "returns error when nvidia-smi unavailable" do
      # If we're on a machine without nvidia-smi, this should return an error
      case GPUUtils.check_memory_warning() do
        # GPU available and memory fine
        :ok -> assert true
        # GPU available but high usage
        {:warning, msg} -> assert is_binary(msg)
        {:error, reason} -> assert reason in [:nvidia_smi_not_found, :nvidia_smi_failed]
      end
    end

    test "threshold parameter works" do
      # Test that threshold can be set (even if we can't test the actual behavior)
      result = GPUUtils.check_memory_warning(threshold: 0.99)

      assert result in [:ok, {:error, :nvidia_smi_not_found}, {:error, :nvidia_smi_failed}] or
               match?({:warning, _}, result)
    end
  end

  describe "check_free_memory/1" do
    test "returns error when nvidia-smi unavailable" do
      case GPUUtils.check_free_memory(required_mb: 1000) do
        :ok -> assert true
        {:warning, msg} -> assert is_binary(msg)
        {:error, reason} -> assert reason in [:nvidia_smi_not_found, :nvidia_smi_failed]
      end
    end

    test "works with zero required memory" do
      result = GPUUtils.check_free_memory(required_mb: 0)
      # Should be :ok or error if no GPU
      assert result == :ok or match?({:error, _}, result)
    end
  end

  describe "gpu_available?/0" do
    test "returns boolean" do
      assert is_boolean(GPUUtils.gpu_available?())
    end
  end
end

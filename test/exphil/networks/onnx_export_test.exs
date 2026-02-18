defmodule ExPhil.Networks.OnnxExportTest do
  @moduledoc """
  Tests for ONNX export and load functionality.

  These tests verify that models can be:
  1. Exported to ONNX format using AxonOnnx
  2. Loaded back using Ortex (ONNX Runtime)
  3. Produce outputs that match the original Axon model
  """
  use ExUnit.Case, async: false

  alias ExPhil.Networks.{OnnxLayers, Policy}
  alias ExPhil.Training.Utils

  @moduletag :onnx

  # Tolerance for floating point comparison (ONNX may use slightly different numerics)
  @tolerance 1.0e-4

  describe "simple MLP export/load" do
    @tag :onnx
    test "exports and loads a simple dense network" do
      # Build a simple model
      model =
        Axon.input("input", shape: {nil, 16})
        |> Axon.dense(32, activation: :relu, name: "dense1")
        |> Axon.dense(8, name: "output")

      # Initialize and get params
      template = Nx.template({1, 16}, :f32)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # Create test input
      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.uniform(key, shape: {1, 16}, type: :f32)

      # Get Axon output
      axon_output = predict_fn.(Utils.ensure_model_state(params), test_input)

      # Export to ONNX
      tmp_path = Path.join(System.tmp_dir!(), "test_mlp_#{:rand.uniform(100_000)}.onnx")

      try do
        iodata = AxonOnnx.dump(model, template, params)
        File.write!(tmp_path, IO.iodata_to_binary(iodata))

        # Load with Ortex
        ort_model = Ortex.load(tmp_path)

        # Run inference with Ortex (pass Nx tensor directly)
        {ort_output} = Ortex.run(ort_model, test_input)
        ort_tensor = Nx.backend_transfer(ort_output)

        # Compare outputs
        diff =
          Nx.subtract(axon_output, ort_tensor)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < @tolerance, "Output difference #{diff} exceeds tolerance #{@tolerance}"
      after
        File.rm(tmp_path)
      end
    end
  end

  describe "sequence_last layer export" do
    @tag :onnx
    test "exports model with OnnxLayers.sequence_last" do
      # Build a model using the ONNX-compatible sequence_last layer
      seq_len = 10
      embed_size = 32
      hidden_size = 16

      input = Axon.input("input", shape: {nil, seq_len, embed_size})

      # Use a simple approach: dense on each timestep, then sequence_last
      # (LSTM export is more complex and may have issues)
      model =
        input
        |> Axon.dense(hidden_size, name: "embed")
        |> OnnxLayers.sequence_last(name: "last_timestep")
        |> Axon.dense(8, name: "output")

      # Initialize
      template = Nx.template({1, seq_len, embed_size}, :f32)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # Test input
      key = Nx.Random.key(123)
      {test_input, _} = Nx.Random.uniform(key, shape: {1, seq_len, embed_size}, type: :f32)

      # Axon forward pass
      axon_output = predict_fn.(Utils.ensure_model_state(params), test_input)

      # Export to ONNX
      tmp_path = Path.join(System.tmp_dir!(), "test_seqlast_#{:rand.uniform(100_000)}.onnx")

      try do
        iodata = AxonOnnx.dump(model, template, params)
        File.write!(tmp_path, IO.iodata_to_binary(iodata))

        # Verify file was created
        assert File.exists?(tmp_path)
        stat = File.stat!(tmp_path)
        assert stat.size > 0

        # Load with Ortex
        ort_model = Ortex.load(tmp_path)

        # Run inference
        # Run inference with Ortex (pass Nx tensor directly)
        {ort_output} = Ortex.run(ort_model, test_input)
        ort_tensor = Nx.backend_transfer(ort_output)

        # Compare
        diff =
          Nx.subtract(axon_output, ort_tensor)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert diff < @tolerance, "sequence_last output difference #{diff} exceeds tolerance"
      after
        File.rm(tmp_path)
      end
    end
  end

  describe "policy network export" do
    @tag :onnx
    @tag :slow
    test "exports MLP policy network" do
      # Build a minimal MLP policy
      embed_size = 64

      model =
        Policy.build(
          embed_size: embed_size,
          hidden_sizes: [32, 32],
          # Disable dropout for deterministic testing
          dropout: 0.0,
          axis_buckets: 8,
          shoulder_buckets: 2
        )

      # Initialize
      template = Nx.template({1, embed_size}, :f32)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # Export to ONNX
      tmp_path = Path.join(System.tmp_dir!(), "test_policy_#{:rand.uniform(100_000)}.onnx")

      try do
        iodata = AxonOnnx.dump(model, template, params)
        File.write!(tmp_path, IO.iodata_to_binary(iodata))

        # Verify export succeeded
        assert File.exists?(tmp_path)
        stat = File.stat!(tmp_path)
        assert stat.size > 1000, "Policy ONNX file seems too small: #{stat.size} bytes"

        # Load with Ortex to verify it's valid ONNX
        _ort_model = Ortex.load(tmp_path)

        # Note: Full output comparison is complex for multi-output policy networks
        # Just verifying load succeeds is the main goal here
      after
        File.rm(tmp_path)
      end
    end
  end

  describe "round-trip consistency" do
    @tag :onnx
    test "multiple exports produce identical ONNX files" do
      # Simple model
      model =
        Axon.input("input", shape: {nil, 8})
        |> Axon.dense(4, name: "output")

      template = Nx.template({1, 8}, :f32)
      {init_fn, _} = Axon.build(model)
      params = init_fn.(template, Axon.ModelState.empty())

      # Export twice
      iodata1 = AxonOnnx.dump(model, template, params)
      iodata2 = AxonOnnx.dump(model, template, params)

      binary1 = IO.iodata_to_binary(iodata1)
      binary2 = IO.iodata_to_binary(iodata2)

      # Should be identical
      assert binary1 == binary2
    end

    @tag :onnx
    test "export preserves weights correctly" do
      # Model with known weights
      model =
        Axon.input("input", shape: {nil, 2})
        |> Axon.dense(2, name: "dense", use_bias: false)

      template = Nx.template({1, 2}, :f32)
      {init_fn, predict_fn} = Axon.build(model)

      # Use identity-like weights for easy verification
      params = init_fn.(template, Axon.ModelState.empty())
      identity_weights = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      params = put_in(params, [Access.key(:data), "dense", "kernel"], identity_weights)

      # Test input — keep a BinaryBackend copy for assertions
      # (EXLA buffers may be donated/consumed by Ortex.run)
      test_input = Nx.tensor([[3.0, 7.0]])
      test_input_copy = Nx.backend_copy(test_input, Nx.BinaryBackend)

      # Axon output (should be identity transform)
      axon_output = predict_fn.(Utils.ensure_model_state(params), test_input)
      axon_output_copy = Nx.backend_copy(axon_output, Nx.BinaryBackend)

      # Export and load
      tmp_path = Path.join(System.tmp_dir!(), "test_weights_#{:rand.uniform(100_000)}.onnx")

      try do
        iodata = AxonOnnx.dump(model, template, params)
        File.write!(tmp_path, IO.iodata_to_binary(iodata))

        ort_model = Ortex.load(tmp_path)
        # Run inference with Ortex (may consume EXLA buffers)
        {ort_output} = Ortex.run(ort_model, Nx.backend_copy(test_input, Nx.BinaryBackend))
        ort_tensor = Nx.backend_transfer(ort_output)

        # Both should output [3.0, 7.0] (identity transform)
        assert_all_close(axon_output_copy, ort_tensor)
        assert_all_close(ort_tensor, test_input_copy)
      after
        File.rm(tmp_path)
      end
    end
  end

  # Helper for floating point comparison
  defp assert_all_close(tensor1, tensor2, tolerance \\ @tolerance) do
    diff =
      Nx.subtract(tensor1, tensor2)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff < tolerance, "Tensors differ by #{diff}, exceeds tolerance #{tolerance}"
  end
end

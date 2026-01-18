#!/usr/bin/env elixir
# Export trained policy to ONNX format
#
# CURRENT STATUS: Using local axon_onnx fork with full support for:
# - Axon 0.8+ API compatibility
# - LSTM/GRU layer serialization
# - sequence_last layer for extracting final timestep
# - Multi-output models via container ops
# - Dense layers on 3D input (RNN sequence output)
#
# Fork: https://github.com/blasphemetheus/axon_onnx (runtime-fixes branch)
# PR: https://github.com/elixir-nx/axon_onnx/pull/61
#
# Usage:
#   mix run scripts/export_onnx.exs [options]
#
# Options:
#   --checkpoint PATH   - Path to checkpoint file (.axon)
#   --policy PATH       - Path to exported policy file (.bin)
#   --output PATH       - Output ONNX file path (default: model.onnx)
#
# Example:
#   mix run scripts/export_onnx.exs --policy checkpoints/imitation_latest_policy.bin --output policy.onnx

require Logger

# Check if axon_onnx is available
axon_onnx_available? = Code.ensure_loaded?(AxonOnnx)

unless axon_onnx_available? do
  IO.puts("""

  ╔════════════════════════════════════════════════════════════════╗
  ║              ExPhil ONNX Export - Status                       ║
  ╚════════════════════════════════════════════════════════════════╝

  ⚠  axon_onnx is not available (Nx 0.10+ compatibility issue)

  The axon_onnx library currently doesn't compile with Nx 0.10.
  See: https://elixirforum.com/t/error-using-axononnx-v0-4-0-undefined-function-transform-2/63326

  WORKAROUNDS:

  1. **Export weights to NumPy** (recommended):
     mix run scripts/export_numpy.exs --policy checkpoints/policy.bin --output weights.npz

     Then rebuild the model in Python:
     ```python
     import numpy as np
     import torch.nn as nn

     weights = np.load('weights.npz')
     # Build equivalent PyTorch model and load weights
     # Export via torch.onnx.export()
     ```

  2. **Pin to older Nx** (not recommended - breaks other features):
     In mix.exs, change:
       {:nx, "~> 0.9"}
       {:axon, "~> 0.7"}
       {:exla, "~> 0.9"}
     To:
       {:nx, "~> 0.6.0"}
       {:axon, "~> 0.6.0"}
       {:exla, "~> 0.6.0"}

  3. **Wait for axon_onnx update**:
     Track: https://github.com/elixir-nx/axon_onnx/issues

  Once axon_onnx is updated, uncomment it in mix.exs:
    {:axon_onnx, "~> 0.4"}

  And run this script to export to ONNX format.

  INT8 QUANTIZATION (works independently):
  If you obtain an ONNX model via another method, you can quantize it:
    python priv/python/quantize_onnx.py model.onnx model_int8.onnx

  """)
  System.halt(1)
end

alias ExPhil.Networks.Policy

# Parse command line arguments
args = System.argv()

get_arg = fn flag, default ->
  case Enum.find_index(args, &(&1 == flag)) do
    nil -> default
    idx -> Enum.at(args, idx + 1) || default
  end
end

checkpoint_path = get_arg.("--checkpoint", nil)
policy_path = get_arg.("--policy", nil)
output_path = get_arg.("--output", "model.onnx")

# Require either checkpoint or policy
unless checkpoint_path || policy_path do
  IO.puts("""

  ╔════════════════════════════════════════════════════════════════╗
  ║              ExPhil ONNX Export                                ║
  ╚════════════════════════════════════════════════════════════════╝

  Usage:
    mix run scripts/export_onnx.exs --policy checkpoints/imitation_latest_policy.bin
    mix run scripts/export_onnx.exs --checkpoint checkpoints/imitation_latest.axon

  Options:
    --checkpoint PATH   Load from full checkpoint file
    --policy PATH       Load from exported policy file
    --output PATH       Output ONNX file (default: model.onnx)

  After exporting, quantize to INT8 for faster inference:
    python priv/python/quantize_onnx.py model.onnx model_int8.onnx
  """)
  System.halt(1)
end

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil ONNX Export                                ║
╚════════════════════════════════════════════════════════════════╝
""")

# Step 1: Load policy
IO.puts("Step 1: Loading policy...")

{model, params, config} = cond do
  policy_path ->
    IO.puts("  Loading from policy: #{policy_path}")

    case File.read(policy_path) do
      {:ok, binary} ->
        export = :erlang.binary_to_term(binary)

        config = export.config
        IO.puts("  Config: #{inspect(config)}")

        # Rebuild the model architecture
        model = if config[:temporal] do
          Policy.build_temporal(
            embed_size: config[:embed_size],
            backbone: config[:backbone] || :sliding_window,
            window_size: config[:window_size] || 60,
            num_heads: config[:num_heads] || 4,
            head_dim: config[:head_dim] || 64,
            hidden_size: config[:hidden_size] || 256,
            num_layers: config[:num_layers] || 2,
            hidden_sizes: config[:hidden_sizes] || [512, 512],
            dropout: config[:dropout] || 0.1,
            axis_buckets: config[:axis_buckets] || 16,
            shoulder_buckets: config[:shoulder_buckets] || 4,
            # Mamba-specific
            state_size: config[:state_size] || 16,
            expand_factor: config[:expand_factor] || 2,
            conv_size: config[:conv_size] || 4
          )
        else
          Policy.build(
            embed_size: config[:embed_size],
            hidden_sizes: config[:hidden_sizes] || [512, 512],
            dropout: config[:dropout] || 0.1,
            axis_buckets: config[:axis_buckets] || 16,
            shoulder_buckets: config[:shoulder_buckets] || 4
          )
        end

        {model, export.params, config}

      {:error, reason} ->
        IO.puts("  Error loading policy: #{inspect(reason)}")
        System.halt(1)
    end

  checkpoint_path ->
    IO.puts("  Loading from checkpoint: #{checkpoint_path}")

    case File.read(checkpoint_path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)

        config = checkpoint.config
        IO.puts("  Config: #{inspect(config)}")

        # Rebuild the model architecture
        embed_size = config[:embed_size] || 1991

        model = if config[:temporal] do
          Policy.build_temporal(
            embed_size: embed_size,
            backbone: config[:backbone] || :sliding_window,
            window_size: config[:window_size] || 60,
            num_heads: config[:num_heads] || 4,
            head_dim: config[:head_dim] || 64,
            hidden_size: config[:hidden_size] || 256,
            num_layers: config[:num_layers] || 2,
            hidden_sizes: config[:hidden_sizes] || [512, 512],
            dropout: config[:dropout] || 0.1,
            axis_buckets: config[:axis_buckets] || 16,
            shoulder_buckets: config[:shoulder_buckets] || 4,
            # Mamba-specific
            state_size: config[:state_size] || 16,
            expand_factor: config[:expand_factor] || 2,
            conv_size: config[:conv_size] || 4
          )
        else
          Policy.build(
            embed_size: embed_size,
            hidden_sizes: config[:hidden_sizes] || [512, 512],
            dropout: config[:dropout] || 0.1,
            axis_buckets: config[:axis_buckets] || 16,
            shoulder_buckets: config[:shoulder_buckets] || 4
          )
        end

        {model, checkpoint.policy_params, config}

      {:error, reason} ->
        IO.puts("  Error loading checkpoint: #{inspect(reason)}")
        System.halt(1)
    end
end

# Extract params data if wrapped in ModelState
params_data = case params do
  %Axon.ModelState{data: data} -> data
  data when is_map(data) -> data
end

IO.puts("  Model loaded successfully")

# Step 2: Prepare for ONNX export
IO.puts("\nStep 2: Preparing model for ONNX export...")

# Determine input shape based on temporal mode
{_input_shape, input_template} = if config[:temporal] do
  embed_size = config[:embed_size] || 1991
  window_size = config[:window_size] || 60
  shape = {1, window_size, embed_size}
  IO.puts("  Input shape: #{inspect(shape)} (temporal)")
  {shape, Nx.template(shape, :f32)}
else
  embed_size = config[:embed_size] || 1991
  shape = {1, embed_size}
  IO.puts("  Input shape: #{inspect(shape)} (single-frame)")
  {shape, Nx.template(shape, :f32)}
end

# Step 3: Export to ONNX
IO.puts("\nStep 3: Exporting to ONNX format...")
IO.puts("  Output: #{output_path}")

# Ensure output directory exists
output_dir = Path.dirname(output_path)
if output_dir != "" and output_dir != "." do
  File.mkdir_p!(output_dir)
end

try do
  # AxonOnnx.dump returns iodata: dump(model, template, params)
  iodata = AxonOnnx.dump(model, input_template, params)
  onnx_binary = IO.iodata_to_binary(iodata)

  File.write!(output_path, onnx_binary)

  file_size = File.stat!(output_path).size
  size_kb = Float.round(file_size / 1024, 2)

  IO.puts("  ✓ Exported successfully (#{size_kb} KB)")
rescue
  e ->
    IO.puts("  ✗ Export failed: #{Exception.message(e)}")
    IO.puts("\n  Stack trace:")
    IO.puts(Exception.format(:error, e, __STACKTRACE__))
    IO.puts("\n  Note: AxonOnnx may not support all layer types.")
    IO.puts("  Multi-output models and some attention patterns may need custom handling.")
    System.halt(1)
end

# Step 4: Print next steps
IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Export Complete!                          ║
╚════════════════════════════════════════════════════════════════╝

ONNX model saved to: #{output_path}

Next Steps:

1. Verify the model with ONNX Runtime:
   python -c "import onnxruntime as ort; ort.InferenceSession('#{output_path}')"

2. Quantize to INT8 for 2-4x faster inference:
   python priv/python/quantize_onnx.py #{output_path} #{String.replace(output_path, ".onnx", "_int8.onnx")}

3. Run inference with ONNX Runtime:
   - Python: onnxruntime
   - Elixir: ortex (ONNX Runtime bindings)
   - Rust: ort crate

4. For Elixir inference with Ortex:
   {:ok, model} = Ortex.load("#{output_path}")
   output = Ortex.run(model, input_tensor)

Notes:
- INT8 quantization typically gives 2-4x speedup with ~1% accuracy loss
- Dynamic quantization works without calibration data
- Static quantization is more accurate but needs representative data
""")

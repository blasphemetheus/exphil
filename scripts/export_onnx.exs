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

alias ExPhil.Training.Output

# Check if axon_onnx is available
axon_onnx_available? = Code.ensure_loaded?(AxonOnnx)

unless axon_onnx_available? do
  Output.banner("ExPhil ONNX Export - Status")
  Output.warning("axon_onnx is not available (Nx 0.10+ compatibility issue)")
  Output.puts("")
  Output.puts("The axon_onnx library currently doesn't compile with Nx 0.10.")
  Output.puts("See: https://elixirforum.com/t/error-using-axononnx-v0-4-0-undefined-function-transform-2/63326")
  Output.puts("")
  Output.puts("WORKAROUNDS:")
  Output.puts("")
  Output.puts("1. **Export weights to NumPy** (recommended):")
  Output.puts("   mix run scripts/export_numpy.exs --policy checkpoints/policy.bin")
  Output.puts("")
  Output.puts("2. **Pin to older Nx** (not recommended)")
  Output.puts("")
  Output.puts("3. **Wait for axon_onnx update**")
  Output.puts("   Track: https://github.com/elixir-nx/axon_onnx/issues")
  Output.puts("")
  Output.puts("INT8 QUANTIZATION (works independently):")
  Output.puts("  python priv/python/quantize_onnx.py model.onnx model_int8.onnx")
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
  Output.banner("ExPhil ONNX Export")
  Output.puts("")
  Output.puts("Usage:")
  Output.puts("  mix run scripts/export_onnx.exs --policy checkpoints/imitation_latest_policy.bin")
  Output.puts("  mix run scripts/export_onnx.exs --checkpoint checkpoints/imitation_latest.axon")
  Output.puts("")
  Output.puts("Options:")
  Output.puts("  --checkpoint PATH   Load from full checkpoint file")
  Output.puts("  --policy PATH       Load from exported policy file")
  Output.puts("  --output PATH       Output ONNX file (default: model.onnx)")
  Output.puts("")
  Output.puts("After exporting, quantize to INT8 for faster inference:")
  Output.puts("  python priv/python/quantize_onnx.py model.onnx model_int8.onnx")
  System.halt(1)
end

Output.banner("ExPhil ONNX Export")
Output.config([
  {"Checkpoint", checkpoint_path || "none"},
  {"Policy", policy_path || "none"},
  {"Output", output_path}
])

# Step 1: Load policy
Output.step(1, 4, "Loading policy")

{model, params, config} = cond do
  policy_path ->
    Output.puts("  Loading from policy: #{policy_path}")

    case File.read(policy_path) do
      {:ok, binary} ->
        export = :erlang.binary_to_term(binary)

        config = export.config
        Output.puts("  Config: #{inspect(config)}")

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
        Output.error("Error loading policy: #{inspect(reason)}")
        System.halt(1)
    end

  checkpoint_path ->
    Output.puts("  Loading from checkpoint: #{checkpoint_path}")

    case File.read(checkpoint_path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)

        config = checkpoint.config
        Output.puts("  Config: #{inspect(config)}")

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
        Output.error("Error loading checkpoint: #{inspect(reason)}")
        System.halt(1)
    end
end

# Extract params data if wrapped in ModelState
params_data = case params do
  %Axon.ModelState{data: data} -> data
  data when is_map(data) -> data
end

Output.puts("  Model loaded successfully")

# Step 2: Prepare for ONNX export
Output.step(2, 4, "Preparing model for ONNX export")

# Determine input shape based on temporal mode
{_input_shape, input_template} = if config[:temporal] do
  embed_size = config[:embed_size] || 1991
  window_size = config[:window_size] || 60
  shape = {1, window_size, embed_size}
  Output.puts("  Input shape: #{inspect(shape)} (temporal)")
  {shape, Nx.template(shape, :f32)}
else
  embed_size = config[:embed_size] || 1991
  shape = {1, embed_size}
  Output.puts("  Input shape: #{inspect(shape)} (single-frame)")
  {shape, Nx.template(shape, :f32)}
end

# Step 3: Export to ONNX
Output.step(3, 4, "Exporting to ONNX format")
Output.puts("  Output: #{output_path}")

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

  Output.success("Exported successfully (#{size_kb} KB)")
rescue
  e ->
    Output.error("Export failed: #{Exception.message(e)}")
    Output.puts("  Stack trace:")
    Output.puts(Exception.format(:error, e, __STACKTRACE__))
    Output.puts("  Note: AxonOnnx may not support all layer types.")
    System.halt(1)
end

# Step 4: Print next steps
Output.step(4, 4, "Export complete")
Output.divider()
Output.section("Export Complete!")
Output.puts("")
Output.puts("ONNX model saved to: #{output_path}")
Output.puts("")
Output.puts("Next Steps:")
Output.puts("")
Output.puts("1. Verify the model with ONNX Runtime:")
Output.puts("   python -c \"import onnxruntime as ort; ort.InferenceSession('#{output_path}')\"")
Output.puts("")
Output.puts("2. Quantize to INT8 for 2-4x faster inference:")
Output.puts("   python priv/python/quantize_onnx.py #{output_path} #{String.replace(output_path, ".onnx", "_int8.onnx")}")
Output.puts("")
Output.puts("3. For Elixir inference with Ortex:")
Output.puts("   {:ok, model} = Ortex.load(\"#{output_path}\")")
Output.puts("   output = Ortex.run(model, input_tensor)")

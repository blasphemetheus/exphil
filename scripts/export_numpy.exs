#!/usr/bin/env elixir
# Export model weights to NumPy format for Python/ONNX conversion
#
# This is a workaround for axon_onnx incompatibility with Nx 0.10+.
# Export weights to NumPy, then rebuild the model in Python and export to ONNX.
#
# Usage:
#   mix run scripts/export_numpy.exs [options]
#
# Options:
#   --checkpoint PATH   - Path to checkpoint file (.axon)
#   --policy PATH       - Path to exported policy file (.bin)
#   --output PATH       - Output directory for NumPy files (default: exports/)
#
# Example:
#   mix run scripts/export_numpy.exs --policy checkpoints/imitation_latest_policy.bin

require Logger

alias ExPhil.Training.Output

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
output_dir = get_arg.("--output", "exports")

# Require either checkpoint or policy
unless checkpoint_path || policy_path do
  Output.banner("ExPhil NumPy Export")
  Output.puts("")
  Output.puts("Export model weights to NumPy format for Python/ONNX conversion.")
  Output.puts("")
  Output.puts("Usage:")

  Output.puts(
    "  mix run scripts/export_numpy.exs --policy checkpoints/imitation_latest_policy.bin"
  )

  Output.puts("  mix run scripts/export_numpy.exs --checkpoint checkpoints/imitation_latest.axon")
  Output.puts("")
  Output.puts("Options:")
  Output.puts("  --checkpoint PATH   Load from full checkpoint file")
  Output.puts("  --policy PATH       Load from exported policy file")
  Output.puts("  --output PATH       Output directory (default: exports/)")
  Output.puts("")
  Output.puts("After exporting, use Python to build ONNX model:")
  Output.puts("  python priv/python/build_onnx_from_numpy.py exports/")
  System.halt(1)
end

Output.banner("ExPhil NumPy Export")

Output.config([
  {"Checkpoint", checkpoint_path || "none"},
  {"Policy", policy_path || "none"},
  {"Output", output_dir}
])

# Step 1: Load policy
Output.step(1, 5, "Loading policy")

{params, config} =
  cond do
    policy_path ->
      Output.puts("  Loading from policy: #{policy_path}")

      case File.read(policy_path) do
        {:ok, binary} ->
          export = :erlang.binary_to_term(binary)
          {export.params, export.config}

        {:error, reason} ->
          Output.error("Error loading policy: #{inspect(reason)}")
          System.halt(1)
      end

    checkpoint_path ->
      Output.puts("  Loading from checkpoint: #{checkpoint_path}")

      case File.read(checkpoint_path) do
        {:ok, binary} ->
          checkpoint = :erlang.binary_to_term(binary)
          {checkpoint.policy_params, checkpoint.config}

        {:error, reason} ->
          Output.error("Error loading checkpoint: #{inspect(reason)}")
          System.halt(1)
      end
  end

Output.puts("  Config: #{inspect(config)}")

# Extract params data if wrapped in ModelState
params_data =
  case params do
    %Axon.ModelState{data: data} -> data
    data when is_map(data) -> data
  end

Output.puts("  Params loaded successfully")

# Step 2: Create output directory
Output.step(2, 5, "Creating output directory")
File.mkdir_p!(output_dir)
Output.puts("  Output: #{output_dir}/")

# Step 3: Export weights to binary format
Output.step(3, 5, "Exporting weights")

# Flatten nested params into a list of {path, tensor} pairs
# Use Y-combinator pattern for recursive anonymous function
flatten_params = fn flatten_params ->
  fn params, prefix ->
    Enum.flat_map(params, fn {key, value} ->
      path = if prefix == "", do: to_string(key), else: "#{prefix}.#{key}"

      case value do
        %Nx.Tensor{} = tensor -> [{path, tensor}]
        map when is_map(map) -> flatten_params.(flatten_params).(map, path)
        _ -> []
      end
    end)
  end
end

flat_params = flatten_params.(flatten_params).(params_data, "")

Output.puts("  Found #{length(flat_params)} weight tensors")

# Save each tensor as binary + metadata
Enum.each(flat_params, fn {path, tensor} ->
  # Convert path to safe filename
  filename = String.replace(path, ".", "_")

  # Get tensor info
  shape = Nx.shape(tensor)
  type = Nx.type(tensor)

  # Convert to f32 for compatibility
  tensor_f32 = Nx.as_type(tensor, :f32)

  # Save as raw binary (little-endian f32)
  binary = Nx.to_binary(tensor_f32)
  binary_path = Path.join(output_dir, "#{filename}.bin")
  File.write!(binary_path, binary)

  Output.puts("    #{path}: #{inspect(shape)} (#{inspect(type)})")
end)

# Step 4: Save metadata as JSON
Output.step(4, 5, "Saving metadata")

metadata = %{
  config: config,
  layers:
    Enum.map(flat_params, fn {path, tensor} ->
      %{
        name: path,
        shape: Nx.shape(tensor) |> Tuple.to_list(),
        dtype: inspect(Nx.type(tensor)),
        file: "#{String.replace(path, ".", "_")}.bin"
      }
    end)
}

metadata_json = Jason.encode!(metadata, pretty: true)
metadata_path = Path.join(output_dir, "metadata.json")
File.write!(metadata_path, metadata_json)
Output.puts("  Saved metadata to #{metadata_path}")

# Step 5: Print next steps
Output.step(5, 5, "Export complete")
Output.divider()
Output.section("Export Complete!")
Output.puts("")
Output.puts("Weights exported to: #{output_dir}/")
Output.puts("")
Output.puts("Files:")
Output.puts("  - metadata.json      Model config and layer info")
Output.puts("  - *.bin              Weight tensors (little-endian f32)")
Output.puts("")
Output.puts("Next Steps:")
Output.puts("")
Output.puts("1. Use Python to load weights and build ONNX model")
Output.puts("2. Or use the provided helper script:")
Output.puts("   python priv/python/build_onnx_from_numpy.py #{output_dir}/")
Output.puts("3. Quantize the resulting ONNX model:")
Output.puts("   python priv/python/quantize_onnx.py model.onnx model_int8.onnx")
Output.puts("")
Output.puts("Model Configuration:")
Output.puts("  Temporal: #{config[:temporal] || false}")
Output.puts("  Embed size: #{config[:embed_size] || "unknown"}")
Output.puts("  Hidden sizes: #{inspect(config[:hidden_sizes] || [512, 512])}")
Output.puts("  Axis buckets: #{config[:axis_buckets] || 16}")
Output.puts("  Shoulder buckets: #{config[:shoulder_buckets] || 4}")

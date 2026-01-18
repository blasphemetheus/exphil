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
  IO.puts("""

  ╔════════════════════════════════════════════════════════════════╗
  ║              ExPhil NumPy Export                               ║
  ╚════════════════════════════════════════════════════════════════╝

  Export model weights to NumPy format for Python/ONNX conversion.

  Usage:
    mix run scripts/export_numpy.exs --policy checkpoints/imitation_latest_policy.bin
    mix run scripts/export_numpy.exs --checkpoint checkpoints/imitation_latest.axon

  Options:
    --checkpoint PATH   Load from full checkpoint file
    --policy PATH       Load from exported policy file
    --output PATH       Output directory (default: exports/)

  After exporting, use Python to build ONNX model:
    python priv/python/build_onnx_from_numpy.py exports/
  """)
  System.halt(1)
end

IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║              ExPhil NumPy Export                               ║
╚════════════════════════════════════════════════════════════════╝
""")

# Step 1: Load policy
IO.puts("Step 1: Loading policy...")

{params, config} = cond do
  policy_path ->
    IO.puts("  Loading from policy: #{policy_path}")

    case File.read(policy_path) do
      {:ok, binary} ->
        export = :erlang.binary_to_term(binary)
        {export.params, export.config}

      {:error, reason} ->
        IO.puts("  Error loading policy: #{inspect(reason)}")
        System.halt(1)
    end

  checkpoint_path ->
    IO.puts("  Loading from checkpoint: #{checkpoint_path}")

    case File.read(checkpoint_path) do
      {:ok, binary} ->
        checkpoint = :erlang.binary_to_term(binary)
        {checkpoint.policy_params, checkpoint.config}

      {:error, reason} ->
        IO.puts("  Error loading checkpoint: #{inspect(reason)}")
        System.halt(1)
    end
end

IO.puts("  Config: #{inspect(config)}")

# Extract params data if wrapped in ModelState
params_data = case params do
  %Axon.ModelState{data: data} -> data
  data when is_map(data) -> data
end

IO.puts("  Params loaded successfully")

# Step 2: Create output directory
IO.puts("\nStep 2: Creating output directory...")
File.mkdir_p!(output_dir)
IO.puts("  Output: #{output_dir}/")

# Step 3: Export weights to binary format
IO.puts("\nStep 3: Exporting weights...")

# Flatten nested params into a list of {path, tensor} pairs
flatten_params = fn params, prefix ->
  Enum.flat_map(params, fn {key, value} ->
    path = if prefix == "", do: to_string(key), else: "#{prefix}.#{key}"
    case value do
      %Nx.Tensor{} = tensor -> [{path, tensor}]
      map when is_map(map) -> flatten_params.(map, path)
      _ -> []
    end
  end)
end

flat_params = flatten_params.(params_data, "")

IO.puts("  Found #{length(flat_params)} weight tensors")

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

  IO.puts("    #{path}: #{inspect(shape)} (#{type})")
end)

# Step 4: Save metadata as JSON
IO.puts("\nStep 4: Saving metadata...")

metadata = %{
  config: config,
  layers: Enum.map(flat_params, fn {path, tensor} ->
    %{
      name: path,
      shape: Nx.shape(tensor) |> Tuple.to_list(),
      dtype: to_string(Nx.type(tensor)),
      file: "#{String.replace(path, ".", "_")}.bin"
    }
  end)
}

metadata_json = Jason.encode!(metadata, pretty: true)
metadata_path = Path.join(output_dir, "metadata.json")
File.write!(metadata_path, metadata_json)
IO.puts("  Saved metadata to #{metadata_path}")

# Step 5: Print next steps
IO.puts("""

╔════════════════════════════════════════════════════════════════╗
║                      Export Complete!                          ║
╚════════════════════════════════════════════════════════════════╝

Weights exported to: #{output_dir}/

Files:
  - metadata.json      Model config and layer info
  - *.bin              Weight tensors (little-endian f32)

Next Steps:

1. Use Python to load weights and build ONNX model:

   import numpy as np
   import json
   import struct
   from pathlib import Path

   # Load metadata
   with open('#{output_dir}/metadata.json') as f:
       meta = json.load(f)

   # Load weights
   weights = {}
   for layer in meta['layers']:
       path = Path('#{output_dir}') / layer['file']
       data = np.frombuffer(path.read_bytes(), dtype=np.float32)
       weights[layer['name']] = data.reshape(layer['shape'])

   # Now build your PyTorch/ONNX model and load these weights

2. Or use the provided helper script:
   python priv/python/build_onnx_from_numpy.py #{output_dir}/

3. Quantize the resulting ONNX model:
   python priv/python/quantize_onnx.py model.onnx model_int8.onnx

Model Configuration:
  Temporal: #{config[:temporal] || false}
  Embed size: #{config[:embed_size] || "unknown"}
  Hidden sizes: #{inspect(config[:hidden_sizes] || [512, 512])}
  Axis buckets: #{config[:axis_buckets] || 16}
  Shoulder buckets: #{config[:shoulder_buckets] || 4}
""")

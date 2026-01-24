#!/usr/bin/env elixir
# Test the full ONNX export + INT8 quantization pipeline
#
# This script:
# 1. Creates a simple LSTM model using OnnxLayers.sequence_last
# 2. Exports it to ONNX
# 3. Quantizes to INT8
# 4. Benchmarks inference speed
#
# Usage:
#   mix run scripts/test_onnx_pipeline.exs

alias ExPhil.Networks.OnnxLayers
alias ExPhil.Training.Output

Output.banner("ONNX Export + INT8 Quantization Test")

# Configuration - using smaller sizes for faster testing
embed_size = 256
hidden_size = 64
seq_len = 10
output_size = 16

Output.step(1, 6, "Building ONNX-exportable LSTM model")
Output.config([
  {"Embed size", embed_size},
  {"Hidden size", hidden_size},
  {"Seq len", seq_len},
  {"Output size", output_size}
])

# Build model: LSTM -> sequence_last -> Dense
input = Axon.input("input", shape: {1, seq_len, embed_size})
{output_seq, _states} = Axon.lstm(input, hidden_size, name: "lstm")
model = output_seq
  |> OnnxLayers.sequence_last(name: "last_timestep")
  |> Axon.dense(output_size, name: "output")

Output.success("Model built successfully")

# Initialize parameters
template = Nx.template({1, seq_len, embed_size}, :f32)
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(template, Axon.ModelState.empty())

Output.puts("  Parameters initialized")

# Test forward pass
Output.step(2, 6, "Testing forward pass in Axon")
key = Nx.Random.key(42)
{test_input, _} = Nx.Random.uniform(key, shape: {1, seq_len, embed_size}, type: :f32)
output = predict_fn.(ExPhil.Training.Utils.ensure_model_state(params), test_input)
Output.success("Output shape: #{inspect(Nx.shape(output))}")

# Export to ONNX
Output.step(3, 6, "Exporting to ONNX")
output_dir = "/tmp/exphil_onnx_test"
File.mkdir_p!(output_dir)
onnx_path = Path.join(output_dir, "model.onnx")
int8_path = Path.join(output_dir, "model_int8.onnx")

try do
  iodata = AxonOnnx.dump(model, template, params)
  binary = IO.iodata_to_binary(iodata)
  File.write!(onnx_path, binary)

  size_kb = Float.round(byte_size(binary) / 1024, 2)
  Output.success("Exported to: #{onnx_path} (#{size_kb} KB)")
rescue
  e ->
    Output.error("FAILED: #{Exception.message(e)}")
    System.halt(1)
end

# Verify with ONNX Runtime
Output.step(4, 6, "Verifying with ONNX Runtime")
{cmd_output, exit_code} = System.cmd("python3", [
  "-c",
  """
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('#{onnx_path}', providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]
print(f"  Input: {input_name}, shape: {input_shape}")

test_input = np.random.randn(*input_shape).astype(np.float32)
outputs = sess.run(None, {input_name: test_input})
print(f"  Output shape: {outputs[0].shape}")
print("  Verification: OK")
  """
], stderr_to_stdout: true)

Output.puts(String.trim(cmd_output))

if exit_code != 0 do
  Output.error("ONNX Runtime verification failed")
  System.halt(1)
end

# Quantize to INT8
Output.step(5, 6, "Quantizing to INT8")
quantize_script = Path.join(File.cwd!(), "priv/python/quantize_onnx.py")

{cmd_output, exit_code} = System.cmd("python3", [
  quantize_script,
  onnx_path,
  int8_path
], stderr_to_stdout: true)

# Parse output for key info
cmd_output
|> String.split("\n")
|> Enum.filter(fn line ->
  String.contains?(line, "size:") or
  String.contains?(line, "Compression") or
  String.contains?(line, "successfully")
end)
|> Enum.each(&Output.puts/1)

if exit_code != 0 do
  Output.error("Quantization failed")
  Output.puts(cmd_output)
  System.halt(1)
end

# Benchmark inference
Output.step(6, 6, "Benchmarking inference speed")
{bench_output, _} = System.cmd("python3", [
  "-c",
  """
import onnxruntime as ort
import numpy as np
import time

def benchmark(model_path, name, warmup=10, iterations=100):
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    shape = [1 if d is None else d for d in inp.shape]
    x = np.random.randn(*shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {inp.name: x})

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        sess.run(None, {inp.name: x})
    avg_ms = (time.time() - start) * 1000 / iterations

    return avg_ms

fp32_ms = benchmark('#{onnx_path}', 'FP32')
int8_ms = benchmark('#{int8_path}', 'INT8')
speedup = fp32_ms / int8_ms

print(f"  FP32 inference: {fp32_ms:.2f} ms")
print(f"  INT8 inference: {int8_ms:.2f} ms")
print(f"  Speedup: {speedup:.2f}x")
  """
], stderr_to_stdout: true)

Output.puts(String.trim(bench_output))

# Summary
Output.divider()
Output.section("Test Complete!")
Output.puts("")
Output.puts("Files created:")
Output.puts("  - #{onnx_path}")
Output.puts("  - #{int8_path}")
Output.puts("")
Output.success("The ONNX + INT8 quantization pipeline is working!")
Output.puts("")
Output.puts("Next steps for ExPhil:")
Output.puts("1. Train a temporal model with LSTM backbone")
Output.puts("2. Export using: mix run scripts/export_onnx.exs --policy <path>")
Output.puts("3. Quantize using: python priv/python/quantize_onnx.py")
Output.puts("4. Use Ortex for Elixir inference or onnxruntime for Python")

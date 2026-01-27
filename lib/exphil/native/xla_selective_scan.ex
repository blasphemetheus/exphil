defmodule ExPhil.Native.XLASelectiveScan do
  @moduledoc """
  XLA Custom Call integration for selective scan.

  **STATUS: EXPERIMENTAL** - This module documents the XLA custom call approach
  but full integration requires EXLA changes or XLA FFI support.

  ## Why XLA Custom Call?

  The Rust NIF achieves ~11ms by:
  1. Copying tensors from EXLA GPU → CPU (via Nx.to_binary)
  2. Running CUDA kernel
  3. Copying results back CPU → EXLA GPU (via Nx.from_binary)

  An XLA Custom Call would achieve ~5ms by:
  1. Keeping tensors on GPU the entire time
  2. XLA directly calls our CUDA kernel
  3. No CPU involvement at all

  ## Integration Approaches

  ### Approach 1: XLA FFI (Recommended, requires EXLA changes)

  XLA 2.0+ supports Foreign Function Interface (FFI) which is cleaner than
  the legacy CustomCall. This requires:

  1. Build shared library: `cd native/xla_selective_scan && make install`
  2. Register FFI handler in EXLA's C++ code
  3. Call via `Nx.Defn.Kernel.custom_call/3`

  ### Approach 2: EXLA Plugin (Future)

  EXLA could expose a plugin API for custom kernels. This would allow:

  ```elixir
  EXLA.register_custom_kernel(:selective_scan, "libxla_selective_scan.so")

  defn fast_scan(x, dt, a, b, c) do
    EXLA.custom_call(:selective_scan, [x, dt, a, b, c],
      result_shape: Nx.shape(x),
      params: %{dt_min: 0.001, dt_max: 0.1}
    )
  end
  ```

  ### Approach 3: Direct cudarc with EXLA DeviceBuffer (Possible now)

  Access EXLA's underlying GPU buffer pointer and pass to cudarc:

  ```elixir
  # This would require EXLA to expose buffer pointers
  {:ok, x_ptr} = EXLA.DeviceBuffer.get_pointer(x)
  result_ptr = CudaKernel.selective_scan(x_ptr, dt_ptr, ...)
  result = EXLA.DeviceBuffer.from_pointer(result_ptr, shape, type)
  ```

  ## Current Recommendation

  For now, use `ExPhil.Native.SelectiveScan` (Rust NIF) which achieves:
  - 10.96ms average on RTX 4090
  - 60 FPS target met
  - No Python dependency
  - Works with current EXLA

  The ~5ms XLA custom call improvement is a future optimization when
  EXLA adds FFI or custom kernel support.

  ## Building the Kernel

  The CUDA kernel is ready at `native/xla_selective_scan/`:

  ```bash
  cd native/xla_selective_scan
  make            # Builds libxla_selective_scan.so
  make install    # Copies to priv/native/
  ```

  ## Kernel Interface

  The C interface expected by XLA CustomCall:

  ```c
  void SelectiveScan(cudaStream_t stream, void** buffers,
                     const char* opaque, size_t opaque_len);

  // buffers[0]: x      [batch, seq_len, hidden] f32
  // buffers[1]: dt     [batch, seq_len, hidden] f32
  // buffers[2]: A      [hidden, state] f32
  // buffers[3]: B      [batch, seq_len, state] f32
  // buffers[4]: C      [batch, seq_len, state] f32
  // buffers[5]: output [batch, seq_len, hidden] f32 (output)
  ```

  The `opaque` parameter contains packed ScanParams struct with dimensions.
  """

  @doc """
  Check if the XLA custom kernel library is available.
  """
  def available? do
    lib_path = Application.app_dir(:exphil, "priv/native/libxla_selective_scan.so")
    File.exists?(lib_path)
  end

  @doc """
  Get the path to the compiled kernel library.
  """
  def library_path do
    Application.app_dir(:exphil, "priv/native/libxla_selective_scan.so")
  end

  @doc """
  Pack scan parameters into opaque binary for XLA CustomCall.

  ## Parameters
  - batch: Batch size
  - seq_len: Sequence length
  - hidden: Hidden dimension
  - state: State dimension
  - dt_min: Minimum delta time (default: 0.001)
  - dt_max: Maximum delta time (default: 0.1)
  """
  def pack_params(batch, seq_len, hidden, state, dt_min \\ 0.001, dt_max \\ 0.1) do
    # Pack as little-endian: 4 int32s + 2 float32s = 24 bytes
    <<
      batch::little-signed-32,
      seq_len::little-signed-32,
      hidden::little-signed-32,
      state::little-signed-32,
      dt_min::little-float-32,
      dt_max::little-float-32
    >>
  end

  @doc """
  Placeholder for future XLA custom call integration.

  Currently falls back to the Rust NIF implementation.
  """
  def selective_scan(x, dt, a, b, c) do
    # For now, delegate to the working Rust NIF
    ExPhil.Native.SelectiveScan.scan(x, dt, a, b, c)
  end
end

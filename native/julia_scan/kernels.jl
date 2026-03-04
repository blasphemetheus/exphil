# GPU Kernels for Linear Scan
#
# Implements h = a * h + b (fused linear recurrence) in two ways:
#   1. CUDA.jl @cuda kernel — direct port of fused_linear_scan.cu
#   2. KernelAbstractions.jl — vendor-neutral (CUDA/ROCm/oneAPI)
#
# Thread layout matches CUDA C: one thread per (batch, hidden), sequential
# loop over timesteps with state in registers.

using CUDA
using KernelAbstractions

# ============================================================================
# CUDA.jl kernel (direct port of fused_linear_scan.cu)
# ============================================================================

function linear_scan_cuda_kernel!(
    output::CuDeviceArray{Float32,3},
    a_vals::CuDeviceArray{Float32,3},
    b_vals::CuDeviceArray{Float32,3},
    h0::CuDeviceArray{Float32,2},
    batch::Int32, seq_len::Int32, hidden::Int32
)
    # Thread layout: blockIdx.x = batch, threadIdx.x + blockIdx.y * blockDim.x = hidden
    b = blockIdx().x
    h = threadIdx().x + (blockIdx().y - Int32(1)) * blockDim().x

    if b > batch || h > hidden
        return nothing
    end

    # Load initial state
    h_state = h0[h, b]

    # Sequential scan over timesteps
    @inbounds for t in Int32(1):seq_len
        a = a_vals[h, t, b]
        bv = b_vals[h, t, b]
        h_state = a * h_state + bv
        output[h, t, b] = h_state
    end

    return nothing
end

"""
    linear_scan_cuda(a_vals, b_vals, h0) -> output

Run fused linear scan on GPU using CUDA.jl.

Inputs:
  a_vals: [hidden, seq_len, batch] (Julia column-major = C row-major [batch, seq_len, hidden])
  b_vals: [hidden, seq_len, batch]
  h0:     [hidden, batch]

Output:
  output: [hidden, seq_len, batch]
"""
function linear_scan_cuda(a_vals::CuArray{Float32,3}, b_vals::CuArray{Float32,3}, h0::CuArray{Float32,2})
    hidden, seq_len, batch = size(a_vals)
    output = CUDA.zeros(Float32, hidden, seq_len, batch)

    threads_per_block = min(hidden, 256)
    blocks_y = cld(hidden, threads_per_block)

    @cuda threads=threads_per_block blocks=(batch, blocks_y) linear_scan_cuda_kernel!(
        output, a_vals, b_vals, h0,
        Int32(batch), Int32(seq_len), Int32(hidden)
    )

    return output
end

# ============================================================================
# KernelAbstractions.jl kernel (vendor-neutral)
# ============================================================================

@kernel function linear_scan_ka_kernel!(
    output, a_vals, b_vals, h0,
    @Const(seq_len::Int32)
)
    # Global index maps to (hidden, batch) like the CUDA kernel
    I = @index(Global)
    hidden = size(a_vals, 1)
    batch = size(a_vals, 3)

    h_idx = ((I - 1) % hidden) + 1
    b_idx = ((I - 1) ÷ hidden) + 1

    if b_idx <= batch
        h_state = h0[h_idx, b_idx]

        @inbounds for t in Int32(1):seq_len
            a = a_vals[h_idx, t, b_idx]
            bv = b_vals[h_idx, t, b_idx]
            h_state = a * h_state + bv
            output[h_idx, t, b_idx] = h_state
        end
    end
end

"""
    linear_scan_ka(a_vals, b_vals, h0; backend=CUDABackend()) -> output

Run fused linear scan using KernelAbstractions (vendor-neutral).
"""
function linear_scan_ka(a_vals::AbstractArray{Float32,3}, b_vals::AbstractArray{Float32,3},
                        h0::AbstractArray{Float32,2}; backend=nothing)
    hidden, seq_len, batch = size(a_vals)
    output = similar(a_vals)
    fill!(output, 0.0f0)

    if backend === nothing
        backend = get_backend(a_vals)
    end

    ndrange = hidden * batch
    kernel! = linear_scan_ka_kernel!(backend, 256)
    kernel!(output, a_vals, b_vals, h0, Int32(seq_len); ndrange=ndrange)

    return output
end

# ============================================================================
# CPU reference (for validation)
# ============================================================================

"""
    linear_scan_cpu(a_vals, b_vals, h0) -> output

Sequential linear scan on CPU for reference/validation.
"""
function linear_scan_cpu(a_vals::Array{Float32,3}, b_vals::Array{Float32,3}, h0::Array{Float32,2})
    hidden, seq_len, batch = size(a_vals)
    output = zeros(Float32, hidden, seq_len, batch)

    for b in 1:batch
        for h in 1:hidden
            h_state = h0[h, b]
            for t in 1:seq_len
                h_state = a_vals[h, t, b] * h_state + b_vals[h, t, b]
                output[h, t, b] = h_state
            end
        end
    end

    return output
end

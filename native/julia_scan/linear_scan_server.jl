#!/usr/bin/env julia
# Linear Scan Server for Elixir Port Integration
#
# Msgpack over length-prefixed stdio protocol (identical to pytorch_scan_server.py).
# Long-lived process to amortize Julia's ~30s JIT cold-start.
#
# Protocol:
#   Request:  4-byte big-endian length + msgpack({op, batch, seq_len, hidden, a, b, h0})
#   Response: 4-byte big-endian length + msgpack({status, result, shape} or {status, message})
#
# Started by ExPhil.Bridge.JuliaPort GenServer.

using Pkg
Pkg.activate(@__DIR__)

using MsgPack
using CUDA

include("kernels.jl")

# ============================================================================
# Protocol I/O (matches pytorch_scan_server.py exactly)
# ============================================================================

function read_message(io::IO)
    header = read(io, 4)
    if length(header) < 4
        return nothing
    end
    len = ntoh(reinterpret(UInt32, header)[1])
    data = read(io, len)
    return MsgPack.unpack(data)
end

function write_message(io::IO, msg)
    data = MsgPack.pack(msg)
    len = UInt32(length(data))
    write(io, hton(len))
    write(io, data)
    flush(io)
end

# ============================================================================
# Request handlers
# ============================================================================

function handle_ping(msg)
    has_cuda = CUDA.functional()
    return Dict(
        "status" => "ok",
        "message" => "pong",
        "device" => has_cuda ? "cuda" : "cpu",
        "cuda_available" => has_cuda
    )
end

function handle_info(msg)
    has_cuda = CUDA.functional()
    info = Dict(
        "status" => "ok",
        "device" => has_cuda ? "cuda" : "cpu",
        "cuda_available" => has_cuda,
        "julia_version" => string(VERSION)
    )
    if has_cuda
        dev = CUDA.device()
        info["cuda_device"] = CUDA.name(dev)
        info["cuda_memory_gb"] = CUDA.totalmem(dev) / 1e9
    end
    return info
end

function handle_scan(msg)
    try
        # Extract shapes (keys may be strings or symbols)
        batch = get_val(msg, "batch")
        seq_len = get_val(msg, "seq_len")
        hidden = get_val(msg, "hidden")
        mode = get_val(msg, "mode", "cuda")

        # Extract binary data
        a_bin = get_val(msg, "a")
        b_bin = get_val(msg, "b")
        h0_bin = get_val(msg, "h0")

        # Convert to Julia arrays (column-major)
        # Elixir sends row-major [batch, seq_len, hidden], Julia needs [hidden, seq_len, batch]
        a_flat = reinterpret(Float32, Vector{UInt8}(a_bin))
        b_flat = reinterpret(Float32, Vector{UInt8}(b_bin))
        h0_flat = reinterpret(Float32, Vector{UInt8}(h0_bin))

        # Reshape: Elixir sends C-order [batch, seq_len, hidden] which in Julia column-major
        # becomes the reverse: [hidden, seq_len, batch]
        a_cpu = permutedims(reshape(a_flat, hidden, seq_len, batch), (1, 2, 3))
        b_cpu = permutedims(reshape(b_flat, hidden, seq_len, batch), (1, 2, 3))
        h0_cpu = reshape(h0_flat, hidden, batch)

        local result_cpu

        if mode == "cuda" && CUDA.functional()
            # Transfer to GPU
            a_gpu = CuArray(a_cpu)
            b_gpu = CuArray(b_cpu)
            h0_gpu = CuArray(h0_cpu)

            # Run kernel
            result_gpu = linear_scan_cuda(a_gpu, b_gpu, h0_gpu)
            CUDA.synchronize()

            # Transfer back
            result_cpu = Array(result_gpu)
        elseif mode == "ka" && CUDA.functional()
            a_gpu = CuArray(a_cpu)
            b_gpu = CuArray(b_cpu)
            h0_gpu = CuArray(h0_cpu)

            result_gpu = linear_scan_ka(a_gpu, b_gpu, h0_gpu)
            CUDA.synchronize()

            result_cpu = Array(result_gpu)
        else
            # CPU fallback
            result_cpu = linear_scan_cpu(a_cpu, b_cpu, h0_cpu)
        end

        # Convert back to bytes (same memory layout as input)
        result_bin = Vector{UInt8}(reinterpret(UInt8, vec(result_cpu)))

        return Dict(
            "status" => "ok",
            "result" => result_bin,
            "shape" => [batch, seq_len, hidden]
        )

    catch e
        return Dict(
            "status" => "error",
            "message" => sprint(showerror, e, catch_backtrace())
        )
    end
end

function handle_benchmark(msg)
    try
        batch = get_val(msg, "batch")
        seq_len = get_val(msg, "seq_len")
        hidden = get_val(msg, "hidden")
        warmup = get_val(msg, "warmup", 5)
        iterations = get_val(msg, "iterations", 30)

        # Generate random data on GPU
        if !CUDA.functional()
            return Dict("status" => "error", "message" => "CUDA not available")
        end

        a_gpu = CUDA.rand(Float32, hidden, seq_len, batch) .* 0.9f0
        b_gpu = CUDA.randn(Float32, hidden, seq_len, batch)
        h0_gpu = CUDA.zeros(Float32, hidden, batch)

        # Warmup
        for _ in 1:warmup
            linear_scan_cuda(a_gpu, b_gpu, h0_gpu)
            CUDA.synchronize()
        end

        # Timed runs (CUDA kernel)
        cuda_times = Float64[]
        for _ in 1:iterations
            t0 = time_ns()
            linear_scan_cuda(a_gpu, b_gpu, h0_gpu)
            CUDA.synchronize()
            push!(cuda_times, (time_ns() - t0) / 1e3)  # microseconds
        end

        # Timed runs (KA kernel)
        for _ in 1:warmup
            linear_scan_ka(a_gpu, b_gpu, h0_gpu)
            CUDA.synchronize()
        end

        ka_times = Float64[]
        for _ in 1:iterations
            t0 = time_ns()
            linear_scan_ka(a_gpu, b_gpu, h0_gpu)
            CUDA.synchronize()
            push!(ka_times, (time_ns() - t0) / 1e3)
        end

        sort!(cuda_times)
        sort!(ka_times)

        return Dict(
            "status" => "ok",
            "cuda_median_us" => cuda_times[div(length(cuda_times), 2) + 1],
            "cuda_min_us" => cuda_times[1],
            "cuda_max_us" => cuda_times[end],
            "ka_median_us" => ka_times[div(length(ka_times), 2) + 1],
            "ka_min_us" => ka_times[1],
            "ka_max_us" => ka_times[end]
        )

    catch e
        return Dict(
            "status" => "error",
            "message" => sprint(showerror, e, catch_backtrace())
        )
    end
end

# ============================================================================
# Helpers
# ============================================================================

function get_val(msg, key, default=nothing)
    # MsgPack may deliver keys as strings or bytes
    for k in [key, Vector{UInt8}(key)]
        if haskey(msg, k)
            val = msg[k]
            # Convert bytes to string if needed
            if val isa Vector{UInt8} && default isa Union{String, Nothing}
                return String(val)
            end
            return val
        end
    end
    return default
end

# ============================================================================
# Main server loop
# ============================================================================

function main()
    println(stderr, "Julia linear scan server starting...")

    # Force CUDA initialization (triggers JIT compilation of runtime)
    if CUDA.functional()
        println(stderr, "CUDA available: $(CUDA.name(CUDA.device()))")
        # Warmup: compile kernels
        a = CUDA.rand(Float32, 16, 4, 1)
        b = CUDA.rand(Float32, 16, 4, 1)
        h0 = CUDA.zeros(Float32, 16, 1)
        linear_scan_cuda(a, b, h0)
        linear_scan_ka(a, b, h0)
        CUDA.synchronize()
        println(stderr, "Kernel JIT warmup complete")
    else
        println(stderr, "CUDA not available, using CPU fallback")
    end

    println(stderr, "Julia linear scan server ready")

    input = stdin
    output = stdout

    while true
        msg = read_message(input)
        if msg === nothing
            break
        end

        op = get_val(msg, "op", "unknown")

        response = if op == "ping"
            handle_ping(msg)
        elseif op == "info"
            handle_info(msg)
        elseif op == "scan"
            handle_scan(msg)
        elseif op == "benchmark"
            handle_benchmark(msg)
        else
            Dict("status" => "error", "message" => "Unknown op: $op")
        end

        write_message(output, response)
    end
end

main()

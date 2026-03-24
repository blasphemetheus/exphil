{ pkgs, lib, ... }:

let
  cuda = pkgs.cudaPackages;
in
{
  # Auto-load .env secrets (gitignored)
  dotenv.enable = true;

  # Elixir + Erlang
  languages.elixir = {
    enable = true;
    package = pkgs.elixir_1_18;
  };
  languages.erlang = {
    enable = true;
    package = pkgs.erlang_27;
  };

  # Rust (for Rustler NIFs / Peppi replay parsing)
  languages.rust.enable = true;

  # Python (for libmelee bridge, Triton, benchmarks)
  languages.python = {
    enable = true;
    package = pkgs.python3.withPackages (ps: with ps; [
      msgpack
      numpy
      huggingface-hub
    ]);
  };

  # System packages
  packages = [
    pkgs.git
    pkgs.tmux
    pkgs.gcc
    pkgs.gnumake
    pkgs.cmake
    pkgs.pkg-config
    pkgs.curl
    pkgs.unzip
    pkgs.rclone

    # GPU kernel exploration
    pkgs.julia
    pkgs.futhark

    # CUDA
    cuda.cuda_nvcc
    cuda.cuda_nvrtc
    cuda.cuda_cudart
    cuda.cuda_cccl
    cuda.cudnn
    cuda.libcublas
    cuda.libcusolver
    cuda.libcufft
    cuda.libcusparse
    cuda.libcurand
    cuda.libnvjitlink
    cuda.nccl
  ];

  # Environment variables
  env = {
    ERL_AFLAGS = "-kernel shell_history enabled";
    EXLA_TARGET = "cuda";
    XLA_TARGET = "cuda12";
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cuda.cuda_nvcc}";
    ERL_FLAGS = "+P 4000000";
    # Use local nx/exla forks (required for fused CUDA kernel custom calls)
    EDIFICE_LOCAL_NX = "1";

    # Compile Edifice CUDA kernels directly into libexla.so (no symlinks needed)
    EXLA_EXTRA_CUDA_DIR = "${toString ../edifice}/native/cuda";
    EXLA_GPU_ARCH = "sm_120";
  };

  # Library path for CUDA
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    cuda.cuda_cudart
    cuda.cuda_nvrtc
    cuda.cudnn
    cuda.libcublas
    cuda.libcusolver
    cuda.libcufft
    cuda.libcusparse
    cuda.libcurand
    cuda.libnvjitlink
    cuda.nccl
  ];

  # Shell setup
  enterShell = ''
    export MIX_HOME="$PWD/.nix-mix"
    export HEX_HOME="$PWD/.nix-hex"
    mkdir -p "$MIX_HOME" "$HEX_HOME"
    export PATH="$MIX_HOME/bin:$MIX_HOME/escripts:$HEX_HOME/bin:$PATH"

    # CUDA_PATH for native kernel builds
    export CUDA_PATH="${cuda.cuda_nvcc}"

    # XLA prebuilt binaries need NVSHMEM 3.x, NVIDIA driver libs, and nvrtc compat
    export NVSHMEM_LIB="$PWD/.nvshmem/nvidia/nvshmem/lib"
    export CUDA_COMPAT="$PWD/.cuda-compat"
    export NVIDIA_DRIVER_LIB="/run/opengl-driver/lib"
    export LD_LIBRARY_PATH="$NVSHMEM_LIB:$CUDA_COMPAT:$NVIDIA_DRIVER_LIB:$LD_LIBRARY_PATH"

    # XLA wants libnvrtc-builtins.so.12.9 but nix has 12.8 — create compat symlink
    mkdir -p "$CUDA_COMPAT"
    if [ ! -f "$CUDA_COMPAT/libnvrtc-builtins.so.12.9" ]; then
      ln -sf /nix/store/f41hjldhxc1v8bwwncm8rgslazhpan17-cuda_nvrtc-12.8.93-source/lib/libnvrtc-builtins.so.12.8 "$CUDA_COMPAT/libnvrtc-builtins.so.12.9"
    fi

    # Install hex and rebar if not present
    if [ ! -f "$MIX_HOME/archives/hex-"* ] 2>/dev/null; then
      mix local.hex --force --if-missing
    fi
    if [ ! -f "$MIX_HOME/elixir/"*"/rebar3" ] 2>/dev/null; then
      mix local.rebar --force --if-missing
    fi

    echo "ExPhil dev shell ready (CUDA enabled)."
  '';
}

{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Elixir/Erlang
    elixir
    erlang

    # Rust (for NIF compilation: selective_scan, flash_attention, ortex)
    cargo
    rustc

    # Build tools
    gcc
    gnumake
    cmake
    pkg-config

    # For EXLA/XLA native compilation
    curl
    unzip
    git

    # CUDA toolkit + runtime libs (for GPU training via EXLA)
    cudaPackages.cudatoolkit
    cudaPackages.nccl
    cudaPackages.cudnn
    cudaPackages.libnvjitlink
  ];

  shellHook = ''
    export MIX_HOME="$PWD/.nix-mix"
    export HEX_HOME="$PWD/.nix-hex"
    mkdir -p "$MIX_HOME" "$HEX_HOME"
    export PATH="$MIX_HOME/bin:$MIX_HOME/escripts:$HEX_HOME/bin:$PATH"

    # CUDA/GPU support via WSL2 passthrough
    export CUDA_PATH="$(dirname $(which nvcc))/.."
    # XLA 0.10+ needs NCCL 2.27+ and NVSHMEM 3.x (newer than nixpkgs provides)
    # Downloaded from PyPI wheels: nvidia-nccl-cu12, nvidia-nvshmem-cu12
    # CUDA compat symlinks bridge minor version gaps (nixpkgs 12.8 vs XLA 12.9)
    export NCCL_LIB="$PWD/.nccl/nvidia/nccl/lib"
    export NVSHMEM_LIB="$PWD/.nvshmem/nvidia/nvshmem/lib"
    export CUDA_COMPAT="$PWD/.cuda-compat"
    export LD_LIBRARY_PATH="$NCCL_LIB:$CUDA_PATH/lib:${pkgs.cudaPackages.cudnn.lib}/lib:${pkgs.cudaPackages.libnvjitlink.lib}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$NVSHMEM_LIB:$CUDA_COMPAT:/usr/lib/wsl/lib:''${LD_LIBRARY_PATH:-}"
    export EXLA_TARGET=cuda
    export XLA_FLAGS="''${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$CUDA_PATH"

    # Install hex and rebar if not present
    if [ ! -f "$MIX_HOME/archives/hex-"* ] 2>/dev/null; then
      mix local.hex --force --if-missing
    fi
    if [ ! -f "$MIX_HOME/elixir/"*"/rebar3" ] 2>/dev/null; then
      mix local.rebar --force --if-missing
    fi

    # EXLA 0.11+ CallbackServer spawns many processes during JIT compilation.
    # Default Erlang limit (262144) is too low for the full test suite (~2700 tests).
    export ERL_FLAGS="''${ERL_FLAGS:-} +P 4000000"

    echo "ExPhil dev shell ready (CUDA enabled). Run 'mix deps.get && mix compile' to get started."
  '';
}

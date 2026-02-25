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
    export LD_LIBRARY_PATH="$CUDA_PATH/lib:${pkgs.cudaPackages.nccl}/lib:${pkgs.cudaPackages.cudnn.lib}/lib:${pkgs.cudaPackages.libnvjitlink.lib}/lib:/usr/lib/wsl/lib:''${LD_LIBRARY_PATH:-}"
    export EXLA_TARGET=cuda
    export XLA_FLAGS="''${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=$CUDA_PATH"

    # Install hex and rebar if not present
    if [ ! -f "$MIX_HOME/archives/hex-"* ] 2>/dev/null; then
      mix local.hex --force --if-missing
    fi
    if [ ! -f "$MIX_HOME/elixir/"*"/rebar3" ] 2>/dev/null; then
      mix local.rebar --force --if-missing
    fi

    echo "ExPhil dev shell ready (CUDA enabled). Run 'mix deps.get && mix compile' to get started."
  '';
}

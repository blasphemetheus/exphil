{ pkgs ? import <nixpkgs> {} }:

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
  ];

  shellHook = ''
    export MIX_HOME="$PWD/.nix-mix"
    export HEX_HOME="$PWD/.nix-hex"
    mkdir -p "$MIX_HOME" "$HEX_HOME"
    export PATH="$MIX_HOME/bin:$MIX_HOME/escripts:$HEX_HOME/bin:$PATH"

    # Install hex and rebar if not present
    if [ ! -f "$MIX_HOME/archives/hex-"* ] 2>/dev/null; then
      mix local.hex --force --if-missing
    fi
    if [ ! -f "$MIX_HOME/elixir/"*"/rebar3" ] 2>/dev/null; then
      mix local.rebar --force --if-missing
    fi

    echo "ExPhil dev shell ready. Run 'mix deps.get && mix compile' to get started."
  '';
}

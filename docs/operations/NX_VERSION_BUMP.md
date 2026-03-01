# Nx/EXLA/Axon Version Bump Guide

How to bump the Nx ecosystem deps (nx, exla, axon) on a fresh machine. The main challenge is CUDA shared libraries that XLA's prebuilt binaries expect but nixpkgs doesn't provide at the right versions.

## 1. Bump versions in mix.exs

**exphil/mix.exs:**
```elixir
{:nx, "~> 0.11"},
{:axon, "~> 0.8"},
{:exla, "~> 0.11"},
```

**edifice/mix.exs** (companion library — must stay in sync):
```elixir
{:nx, "~> 0.11"},
{:exla, "~> 0.11", optional: true},
```

Then fetch deps in both repos:
```bash
cd ~/edifice && mix deps.get
cd ~/exphil && mix deps.get
```

## 2. Fix CUDA shared library mismatches

When you first run `mix test` or `mix compile`, EXLA downloads a prebuilt XLA binary. This binary links against specific CUDA library versions that may not match what nixpkgs provides. Common errors look like:

```
** (UndefinedFunctionError) function EXLA.NIF.mlir_new_context/0 is undefined
(Could not load NIF library: libnvshmem_host.so.3: cannot open shared object file)
```

### NCCL (nvidia-nccl-cu12)

XLA links against a newer NCCL than nixpkgs provides. Symptom: `ncclCommWindowDeregister` undefined symbol, or NCCL .so not found.

```bash
cd ~/exphil
pip download nvidia-nccl-cu12 --no-deps -d /tmp/nccl-wheel
unzip /tmp/nccl-wheel/nvidia_nccl_cu12-*.whl -d .nccl/ 'nvidia/nccl/lib/*'
```

This extracts `libnccl.so.2` (and versioned symlinks) into `.nccl/nvidia/nccl/lib/`.

### NVSHMEM (nvidia-nvshmem-cu12)

XLA links against nvshmem which isn't in nixpkgs at all. Symptom: `libnvshmem_host.so.3: cannot open shared object file`.

```bash
pip download nvidia-nvshmem-cu12 --no-deps -d /tmp/nvshmem-wheel
unzip /tmp/nvshmem-wheel/nvidia_nvshmem_cu12-*.whl -d .nvshmem/ 'nvidia/nvshmem/lib/*'
```

You may also need a compat symlink if the .so version doesn't match exactly:
```bash
# Check what XLA expects vs what was extracted
ls .nvshmem/nvidia/nvshmem/lib/
# If XLA wants .so.3 but you have .so.4, symlink:
cd .nvshmem/nvidia/nvshmem/lib
ln -sf nvshmem_transport_ibrc.so.4 nvshmem_transport_ibrc.so.3
```

### CUDA minor version compat (nvrtc-builtins)

If nixpkgs has CUDA 12.8 but XLA was built against 12.9, you'll see: `libnvrtc-builtins.so.12.9: cannot open shared object file`.

```bash
mkdir -p .cuda-compat

# Find the nixpkgs version
find /nix/store -name 'libnvrtc-builtins.so.12.*' -path '*/cuda_nvrtc/*' 2>/dev/null | head -1
# e.g., /nix/store/.../lib/libnvrtc-builtins.so.12.8.93

# Symlink to the version XLA expects
ln -sf /nix/store/HASH-cuda_nvrtc-12.8.93/lib/libnvrtc-builtins.so.12.8.93 \
       .cuda-compat/libnvrtc-builtins.so.12.9
```

## 3. Update shell.nix

Add the downloaded libraries to `LD_LIBRARY_PATH` and bump the Erlang process limit:

```nix
shellHook = ''
  # ... existing CUDA_PATH etc ...

  # XLA prebuilt binaries need NCCL 2.27+ and NVSHMEM 3.x (newer than nixpkgs)
  # Downloaded from PyPI: nvidia-nccl-cu12, nvidia-nvshmem-cu12
  export NCCL_LIB="$PWD/.nccl/nvidia/nccl/lib"
  export NVSHMEM_LIB="$PWD/.nvshmem/nvidia/nvshmem/lib"
  export CUDA_COMPAT="$PWD/.cuda-compat"

  export LD_LIBRARY_PATH="$NCCL_LIB:$CUDA_PATH/lib:...:$NVSHMEM_LIB:$CUDA_COMPAT:..."

  # EXLA 0.11+ CallbackServer spawns many processes during JIT.
  # Default Erlang limit (262144) is too low.
  export ERL_FLAGS="''${ERL_FLAGS:-} +P 4000000"
'';
```

The `.nccl/`, `.nvshmem/`, and `.cuda-compat/` directories are in `.gitignore` — they must be recreated on each machine.

## 4. Verify

```bash
# Re-enter nix shell to pick up new LD_LIBRARY_PATH
exit
nix-shell

# Compile and test
mix compile
mix test
```

## 5. Checklist

- [ ] Bump versions in exphil `mix.exs`
- [ ] Bump versions in edifice `mix.exs`
- [ ] `mix deps.get` in both repos
- [ ] Download nvidia-nccl-cu12 wheel → `.nccl/`
- [ ] Download nvidia-nvshmem-cu12 wheel → `.nvshmem/`
- [ ] Create CUDA compat symlinks in `.cuda-compat/` (if minor version mismatch)
- [ ] Update `shell.nix` LD_LIBRARY_PATH and ERL_FLAGS
- [ ] Re-enter nix shell
- [ ] `mix compile` succeeds
- [ ] `mix test` passes

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `libnccl.so.2: undefined symbol: ncclCommWindowDeregister` | NCCL too old | Download newer nvidia-nccl-cu12, ensure `.nccl/` is first on LD_LIBRARY_PATH |
| `libnvshmem_host.so.3: cannot open` | NVSHMEM missing | Download nvidia-nvshmem-cu12 wheel |
| `libnvrtc-builtins.so.12.X: cannot open` | CUDA minor version gap | Create symlink in `.cuda-compat/` |
| `libstdc++.so.6: cannot open` | Missing C++ stdlib | Add `${pkgs.stdenv.cc.cc.lib}/lib` to LD_LIBRARY_PATH |
| `SystemLimitError: too many processes` | EXLA 0.11 JIT spawns | Set `ERL_FLAGS="+P 4000000"` |
| `Failed to resolve mix deps` after edifice bump | Forgot deps.get in edifice | `cd ~/edifice && mix deps.get` |

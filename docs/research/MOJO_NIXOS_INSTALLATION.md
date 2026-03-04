# Getting Mojo on NixOS/WSL2

## Current Status (March 2026)

Mojo is **not in nixpkgs** and distributed as precompiled binaries that assume FHS (Filesystem Hierarchy Standard) — which NixOS doesn't follow. This doc covers practical approaches to get Mojo running.

**Environment:** NixOS on WSL2, NVIDIA T400 4GB (Turing sm_75), CUDA 12.8 via nix packages.

## Mojo Installation Methods (2026)

The old `modular install mojo` CLI is **deprecated**. Current methods:

| Method | Command | Notes |
|--------|---------|-------|
| **Pixi** (recommended) | `pixi add mojo` | Conda-based virtual envs |
| **UV** | `uv pip install mojo --index https://whl.modular.com/nightly/simple/` | Python ecosystem |
| **Pip** | `pip install --pre mojo --extra-index-url https://whl.modular.com/nightly/simple/` | Standard pip |
| **Conda** | `conda install -c https://conda.modular.com/max-nightly/ mojo` | Traditional conda |
| **Docker** | `modular/max-nvidia-full:latest` | Full container with CUDA |

**Packages:** `mojo` (full: CLI, stdlib, LSP, debugger, REPL) or `mojo-compiler` (minimal).

**System requirements:** Linux x86-64, 8 GiB RAM, Python 3.10-3.14, C++ compiler.

## Mojo GPU Status

GPU programming is now **first-class** (introduced June 2025, stable in 26.1 release).

| Feature | Status |
|---------|--------|
| `@gpu.kernel` decorator | Stable |
| NVIDIA GPU support | Production (CUDA backend) |
| AMD GPU support | Production (HIP/ROCm backend) |
| Apple Silicon GPU | Expanding (Metal, 26.1) |
| Cross-vendor portability | Yes (same kernel for NVIDIA/AMD/Apple) |
| Min NVIDIA driver | 580+ |
| Min CUDA toolkit | 12.4+ |
| T400 compatibility | Tier 3 ("Limited") — Turing sm_75 should work |

**Mojo 1.0 targeting H1 2026.** Compiler open-sourcing planned for "conclusion of Phase 1."

---

## Approach 1: Docker (Highest Confidence)

The `modular/max-nvidia-full` image includes CUDA 12.8, Python 3.12, Ubuntu 22.04.

### Prerequisites

```bash
# Ensure nvidia-container-toolkit is available in WSL2
# This is typically handled by the NVIDIA driver installer for WSL
nvidia-smi  # Should show T400
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi  # Verify GPU passthrough
```

### Setup

```bash
# Pull image
docker pull modular/max-nvidia-full:latest

# Run interactively with GPU + mount mojo_scan directory
docker run -it --gpus all \
  -v /home/nixos/exphil/native/mojo_scan:/workspace \
  modular/max-nvidia-full:latest \
  bash

# Inside container:
pip install msgpack numpy
mojo --version
mojo run /workspace/linear_scan.mojo  # Test kernel

# Build shared library (for future NIF integration)
mojo build /workspace/linear_scan.mojo --emit shared-lib -o /workspace/liblinear_scan.so
```

### Integration with ExPhil

Option A: Run `server.py` inside Docker, connect via port:

```elixir
# In MojoPort, spawn Docker instead of python3:
Port.open({:spawn, ~s(docker run --rm -i --gpus all -v #{cwd}/native/mojo_scan:/workspace modular/max-nvidia-full:latest python3 /workspace/server.py)}, [...])
```

Option B: Build a dedicated image with deps pre-installed:

```dockerfile
# native/mojo_scan/Dockerfile
FROM modular/max-nvidia-full:latest
RUN pip install msgpack numpy
COPY server.py /app/server.py
COPY linear_scan.mojo /app/linear_scan.mojo
CMD ["python3", "/app/server.py"]
```

**Pros:** Guaranteed to work, full CUDA support, reproducible.
**Cons:** Docker overhead (~1-3s startup), multi-GB image, needs nvidia-container-toolkit.

---

## Approach 2: Conda/Pixi + nix-ld (Medium Confidence)

Install Mojo natively via pixi, using `nix-ld` for dynamic linker compatibility.

### Step 1: Enable nix-ld

Add to `/etc/nixos/configuration.nix`:

```nix
programs.nix-ld.enable = true;
programs.nix-ld.libraries = with pkgs; [
  stdenv.cc.cc.lib
  zlib
  openssl
  libffi
  ncurses
  xz
];
```

Then rebuild: `sudo nixos-rebuild switch`

### Step 2: Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
```

### Step 3: Install Mojo

```bash
cd /home/nixos/exphil/native/mojo_scan
pixi init -c https://conda.modular.com/max-nightly/ -c conda-forge
pixi add mojo
pixi shell
mojo --version
```

### Step 4: If binary fails (dynamic linker)

```bash
MOJO_BIN=$(pixi run which mojo)
patchelf --set-interpreter $(cat $NIX_CC/nix-support/dynamic-linker) "$MOJO_BIN"
```

**Pros:** Native performance, no container overhead.
**Cons:** May need iterative patchelf fixes; nix-ld requires NixOS system config change.

---

## Approach 3: buildFHSEnv in shell.nix (Medium Confidence)

Create an FHS-compatible sandbox directly in shell.nix.

### Add to shell.nix

```nix
let
  mojoFHS = pkgs.buildFHSEnv {
    name = "mojo-env";
    targetPkgs = pkgs: with pkgs; [
      stdenv.cc.cc.lib
      zlib
      openssl
      libffi
      python311
      python311Packages.pip
      python311Packages.msgpack
      python311Packages.numpy
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
    ];
    runScript = "bash";
  };
in
pkgs.mkShell {
  buildInputs = [
    mojoFHS
    # ... existing inputs ...
  ];
}
```

### Usage

```bash
nix-shell
mojo-env  # Enter FHS environment
pip install --pre mojo --extra-index-url https://whl.modular.com/nightly/simple/
mojo --version
```

**Pros:** Declarative, reproducible, no system config changes.
**Cons:** Two-layer shell (nix-shell then mojo-env), CUDA passthrough needs LD_LIBRARY_PATH setup.

---

## Approach 4: steam-run (Quick Test)

For quick "does Mojo work at all" testing:

```bash
nix-shell -p steam-run --run "steam-run bash"
# Inside: install Mojo via pip, test
pip install --pre mojo --extra-index-url https://whl.modular.com/nightly/simple/
mojo --version
```

**Pros:** One command to test.
**Cons:** Includes many unneeded libraries, not for production.

---

## Elixir Integration Paths

### Current: Port via Python (works today)

```
Elixir GenServer ←msgpack/stdio→ Python server.py ←interop→ Mojo kernel
```

The existing `MojoPort` + `server.py` already implements this. NumPy fallback works without Mojo. Bottleneck: port overhead is 97% of e2e time.

### Future: Mojo as Python extension module (near-term)

Mojo 26.1 introduced `PythonModuleBuilder`:

```mojo
from python import PythonModuleBuilder, PythonObject

@export
fn PyInit_mojo_scan() -> PythonObject:
    var m = PythonModuleBuilder("mojo_scan")
    m.def_function[do_scan]("linear_scan")
    return m.finalize()
```

Then `server.py` imports the compiled Mojo module instead of using NumPy. Same port protocol, faster kernel.

### Future: Mojo shared library as NIF (wait for 1.0)

```mojo
@export
fn linear_scan_ffi(a_ptr: UnsafePointer[Float32], ...) -> Int:
    # kernel code
    return 0
```

```bash
mojo build linear_scan.mojo --emit shared-lib -o liblinear_scan.so
```

Load via `:erlang.load_nif/2`. **Not stable yet** — Modular says export ABI is "not fully baked." Wait for Mojo 1.0 (H1 2026).

---

## Recommended Path

1. **Now:** Use Docker approach for benchmarking Mojo GPU kernels
2. **When nix-ld is configured:** Try pixi install for native performance
3. **After Mojo 1.0 (H1 2026):** Evaluate NIF integration via shared library export

## T400-Specific Notes

- Turing sm_75 is "Tier 3: Limited Compatibility" in Mojo's matrix
- 4GB VRAM is fine for small kernels (fused_linear_scan at batch=4 uses trivial memory)
- CUDA 12.8 in shell.nix meets the 12.4+ requirement
- Driver 13.1.0 exceeds the 580+ minimum

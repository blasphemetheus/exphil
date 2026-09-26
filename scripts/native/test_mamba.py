"""Native Mamba regressions. Run in devenv, with no training active.

devenv shell -- python3 scripts/native/test_mamba.py
MAMBA_MEMCHECK=/path/to/compute-sanitizer adds a full-size memcheck.
Builds in a temporary directory; does not rebuild EXLA's shared library.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CUDA = ROOT.parent / "edifice/native/cuda"


class MambaNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="mamba-regression-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.directory = Path(cls.tmp.name)
        common = ["nvcc", "-O2", "-lineinfo", "-arch=sm_120", f"-I{CUDA}"]
        cls.allocation = cls.directory / "allocation"
        cls.scan = cls.directory / "scan"
        # cudaMalloc/cudaGetDevice: the workspace is a cached grow-only
        # cudaMalloc buffer since 2026-09-26 (cudaFreeAsync only on growth).
        wraps = ["cudaGetDevice", "cudaMalloc", "cudaMemsetAsync", "cudaLaunchKernel",
                 "cudaFreeAsync", "cudaGetLastError"]
        subprocess.run(common + [str(ROOT / "scripts/native/mamba_alloc_failure_test.cu"),
            str(CUDA / "fused_selective_scan_backward.cu")]
            + [arg for name in wraps for arg in ["-Xlinker", f"--wrap={name}"]]
            + ["-o", str(cls.allocation)], check=True)
        subprocess.run(common + [str(ROOT / "scripts/native/mamba_scan_probe.cu"),
            "-o", str(cls.scan)], check=True)

    def test_allocation_failure_never_launches_kernel(self):
        subprocess.run([str(self.allocation)], check=True, timeout=30)

    def test_gradient_and_partial_thread_blocks(self):
        for shape in [(1, 1, 1), (2, 8, 7), (3, 17, 257), (128, 80, 1024)]:
            with self.subTest(shape=shape):
                subprocess.run([str(self.scan), *map(str, shape)], check=True, timeout=60)

    @unittest.skipUnless(os.environ.get("MAMBA_MEMCHECK"), "set MAMBA_MEMCHECK for sanitizer")
    def test_full_training_shape_memcheck(self):
        subprocess.run([os.environ["MAMBA_MEMCHECK"], "--tool", "memcheck",
            "--error-exitcode", "99", str(self.scan), "128", "80", "1024"],
            check=True, timeout=120)


if __name__ == "__main__":
    unittest.main(verbosity=2)

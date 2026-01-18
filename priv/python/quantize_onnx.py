#!/usr/bin/env python3
"""
ONNX INT8 Quantization Script

Quantizes an ONNX model to INT8 for faster inference with minimal accuracy loss.

Usage:
    python quantize_onnx.py input.onnx output_int8.onnx [--static --calibration-data data.npz]

Options:
    --static              Use static quantization (requires calibration data)
    --calibration-data    Path to .npz file with calibration samples
    --per-channel         Use per-channel quantization (more accurate, slightly slower)

Examples:
    # Dynamic quantization (no calibration needed, fast)
    python quantize_onnx.py model.onnx model_int8.onnx

    # Static quantization with calibration data (more accurate)
    python quantize_onnx.py model.onnx model_int8.onnx --static --calibration-data calibration.npz

Requirements:
    pip install onnxruntime onnx numpy
"""

import argparse
import sys
from pathlib import Path

try:
    import onnx
    import numpy as np
    from onnxruntime.quantization import quantize_dynamic, quantize_static
    from onnxruntime.quantization import QuantType, CalibrationDataReader
except ImportError as e:
    print(f"Error: Missing dependency - {e}")
    print("\nInstall required packages:")
    print("  pip install onnxruntime onnx numpy")
    sys.exit(1)


class NpzCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for static quantization."""

    def __init__(self, npz_path: str, input_name: str = "input"):
        self.npz_path = npz_path
        self.input_name = input_name
        self.data = np.load(npz_path)
        self.samples = self.data['samples']  # Expected shape: (N, ...)
        self.index = 0
        print(f"Loaded {len(self.samples)} calibration samples from {npz_path}")

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index:self.index+1]  # Keep batch dim
        self.index += 1
        return {self.input_name: sample.astype(np.float32)}


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX model to INT8 for faster inference"
    )
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument("output", help="Output quantized ONNX model path")
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static quantization (requires calibration data)"
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to .npz file with calibration samples"
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel quantization (more accurate)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load and validate model
    print(f"Loading model: {input_path}")
    try:
        model = onnx.load(str(input_path))
        onnx.checker.check_model(model)
        print("  Model validated successfully")
    except Exception as e:
        print(f"  Warning: Model validation failed - {e}")

    # Get input info
    input_name = model.graph.input[0].name
    print(f"  Input name: {input_name}")

    # Get model size before quantization
    original_size = input_path.stat().st_size / 1_048_576
    print(f"  Original size: {original_size:.2f} MB")

    print("\nQuantizing model...")

    if args.static:
        # Static quantization (requires calibration data)
        if not args.calibration_data:
            print("Error: Static quantization requires --calibration-data")
            print("\nTo generate calibration data, save representative inputs:")
            print("  np.savez('calibration.npz', samples=your_input_array)")
            sys.exit(1)

        calibration_reader = NpzCalibrationDataReader(
            args.calibration_data,
            input_name=input_name
        )

        quantize_static(
            model_input=str(input_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=QuantType.QInt8,
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8
        )
        print("  Used static quantization with calibration data")
    else:
        # Dynamic quantization (no calibration needed)
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            per_channel=args.per_channel,
            weight_type=QuantType.QInt8
        )
        print("  Used dynamic quantization (no calibration)")

    # Get model size after quantization
    quantized_size = output_path.stat().st_size / 1_048_576
    compression_ratio = original_size / quantized_size

    print(f"\nQuantization complete!")
    print(f"  Output: {output_path}")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}x")

    # Validate quantized model
    print("\nValidating quantized model...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output_path))
        print("  Quantized model loads successfully")

        # Get input shape for test inference
        input_shape = sess.get_inputs()[0].shape
        print(f"  Input shape: {input_shape}")

        # Try a dummy inference
        # Replace None dimensions with 1
        test_shape = [1 if dim is None else dim for dim in input_shape]
        dummy_input = np.random.randn(*test_shape).astype(np.float32)

        outputs = sess.run(None, {input_name: dummy_input})
        print(f"  Test inference succeeded: {len(outputs)} output(s)")

    except Exception as e:
        print(f"  Warning: Validation issue - {e}")

    print("""
Next steps:
1. Test the quantized model accuracy on your validation set
2. Benchmark inference speed:
   python -c "
   import onnxruntime as ort
   import numpy as np
   import time

   sess = ort.InferenceSession('%s')
   inp = sess.get_inputs()[0]
   shape = [1 if d is None else d for d in inp.shape]
   x = np.random.randn(*shape).astype(np.float32)

   # Warmup
   for _ in range(10):
       sess.run(None, {inp.name: x})

   # Benchmark
   start = time.time()
   for _ in range(100):
       sess.run(None, {inp.name: x})
   avg_ms = (time.time() - start) * 10
   print(f'Average inference: {avg_ms:.2f} ms')
   "

3. Compare with original model speed to see quantization speedup
""" % str(output_path))


if __name__ == "__main__":
    main()

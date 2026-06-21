"""
Quantize ONNX model from FP32 to INT8 for browser inference.
Usage: python scripts/quantize_onnx.py
"""
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_PATH = "scripts/model.onnx"
OUTPUT_PATH = "web-app/public/model.onnx"


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run export_onnx.py first.")
        return

    input_size = os.path.getsize(INPUT_PATH) / (1024 * 1024)
    print(f"Input model: {INPUT_PATH} ({input_size:.1f} MB)")

    print("Quantizing to INT8 (dynamic)...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    quantize_dynamic(
        model_input=INPUT_PATH,
        model_output=OUTPUT_PATH,
        weight_type=QuantType.QUInt8,
    )

    output_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    reduction = (1 - output_size / input_size) * 100
    print(f"Output model: {OUTPUT_PATH} ({output_size:.1f} MB)")
    print(f"Size reduction: {reduction:.1f}%")

    # Verify
    try:
        import onnxruntime as ort
        import numpy as np

        print("\nVerifying quantized model...")
        sess = ort.InferenceSession(OUTPUT_PATH)
        eo = np.random.randn(1, 3, 512, 512).astype(np.float32)
        sar = np.random.randn(1, 3, 512, 512).astype(np.float32)
        result = sess.run(None, {"eo": eo, "sar": sar})
        print(f"  Output shape: {result[0].shape}")
        print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
        print("  PASS: Quantized model runs correctly!")
    except Exception as e:
        print(f"  Warning: Verification failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()

"""
Export BuildingGuidedChangeDetector from PyTorch to ONNX format.
Usage: python scripts/export_onnx.py
"""
import sys
import os
import torch
import importlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model (module name starts with digit)
model_mod = importlib.import_module("05_model")
BuildingGuidedChangeDetector = model_mod.BuildingGuidedChangeDetector


class ChangeDetectorWrapper(torch.nn.Module):
    """Wrapper that only outputs change_prob for clean ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, eo, sar):
        out = self.model(eo, sar)
        return out["change_prob"]


def main():
    device = torch.device("cpu")
    
    # Load model
    print("Loading model...")
    model = BuildingGuidedChangeDetector(pretrained=False)
    ckpt = torch.load("best.pth", map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Wrap for clean single-output export
    wrapper = ChangeDetectorWrapper(model)
    wrapper.eval()

    # Dummy inputs
    eo = torch.randn(1, 3, 512, 512)
    sar = torch.randn(1, 3, 512, 512)

    # Export
    output_path = "scripts/model.onnx"
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        (eo, sar),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["eo", "sar"],
        output_names=["change_prob"],
        dynamic_axes={
            "eo": {0: "batch", 2: "height", 3: "width"},
            "sar": {0: "batch", 2: "height", 3: "width"},
            "change_prob": {0: "batch", 2: "height", 3: "width"},
        },
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported! Size: {size_mb:.1f} MB")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        print("Verifying with ONNX Runtime...")
        sess = ort.InferenceSession(output_path)
        eo_np = eo.numpy()
        sar_np = sar.numpy()
        result = sess.run(None, {"eo": eo_np, "sar": sar_np})
        print(f"  Output shape: {result[0].shape}")
        print(f"  Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")

        # Compare with PyTorch
        with torch.no_grad():
            pt_out = wrapper(eo, sar).numpy()
        mse = ((result[0] - pt_out) ** 2).mean()
        print(f"  PyTorch vs ONNX MSE: {mse:.8f}")
        assert mse < 0.001, f"MSE too high: {mse}"
        print("  PASS: ONNX output matches PyTorch!")
    except ImportError:
        print("  (onnxruntime not installed, skipping verification)")

    print("Done!")


if __name__ == "__main__":
    main()

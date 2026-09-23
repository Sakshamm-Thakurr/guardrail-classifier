"""
Export the fine-tuned HF model to ONNX (optionally quantized) for fast CPU
inference in the FastAPI service.

Usage:
    python export/export_onnx.py --model_dir train/output/best --out export/onnx
    python export/export_onnx.py --model_dir train/output/best --out export/onnx --quantize
"""
import argparse
import json
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned HF model directory")
    ap.add_argument("--out", type=str, default="export/onnx")
    ap.add_argument("--quantize", action="store_true", help="Apply dynamic INT8 quantization for lower latency")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading + converting {model_dir} -> ONNX ...")
    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    label_map_src = model_dir / "label_map.json"
    if label_map_src.exists():
        shutil.copy(label_map_src, out_dir / "label_map.json")

    if args.quantize:
        print("Applying dynamic INT8 quantization ...")
        quantizer = ORTQuantizer.from_pretrained(out_dir)
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)
        print("Quantized model written alongside fp32 ONNX graph.")

    print(f"Done. ONNX model + tokenizer written to {out_dir}")


if __name__ == "__main__":
    main()

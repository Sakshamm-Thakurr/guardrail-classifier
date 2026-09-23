"""
Head-to-head benchmark: fine-tuned guardrail classifier vs. Garak's default
keyword detector vs. PyRIT's substring scorer, all evaluated on the exact
same labeled test set.

Usage:
    # against the running FastAPI service
    python benchmark/run_benchmark.py --test_set data/processed/test.jsonl --service_url http://localhost:8000

    # or load the ONNX model directly, no service needed
    python benchmark/run_benchmark.py --test_set data/processed/test.jsonl --model_dir export/onnx

Writes results/benchmark_report.json and results/benchmark_report.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.garak_baseline import GarakBaseline
from benchmark.pyrit_baseline import PyRITBaseline
from benchmark.metrics import compute_classification_metrics, compute_latency_stats, write_report


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def label_to_binary(label: str) -> int:
    return 0 if label == "benign" else 1


class ServiceClient:
    """Calls the FastAPI /classify endpoint."""

    name = "finetuned_guardrail (via service)"

    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/classify"

    def predict(self, text: str) -> dict:
        start = time.perf_counter()
        resp = requests.post(self.url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # prefer server-reported latency, fall back to wall clock
        data.setdefault("latency_ms", round((time.perf_counter() - start) * 1000, 3))
        return data


class DirectModelClient:
    """Loads the ONNX model in-process (no HTTP round trip)."""

    name = "finetuned_guardrail (in-process)"

    def __init__(self, model_dir: str):
        import os
        os.environ["GUARDRAIL_MODEL_DIR"] = model_dir
        from service.model_loader import GuardrailClassifier
        self.clf = GuardrailClassifier(model_dir=Path(model_dir))
        if self.clf.using_fallback:
            print("[warn] No trained ONNX model found — 'ours' results below are from the "
                  "keyword fallback, NOT a trained classifier. Train + export first for real numbers.")

    def predict(self, text: str) -> dict:
        return self.clf.predict(text)


def evaluate_detector(detector, rows: list[dict]) -> dict:
    y_true, y_pred, latencies = [], [], []
    for r in rows:
        out = detector.predict(r["text"])
        y_true.append(label_to_binary(r["label"]))
        y_pred.append(1 if out["is_attack"] else 0)
        latencies.append(out["latency_ms"])
    metrics = compute_classification_metrics(y_true, y_pred)
    latency = compute_latency_stats(latencies)
    return {"metrics": metrics, "latency": latency}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set", type=str, required=True)
    ap.add_argument("--service_url", type=str, default=None,
                     help="Base URL of the running FastAPI service, e.g. http://localhost:8000")
    ap.add_argument("--model_dir", type=str, default=None,
                     help="Alternative to --service_url: load the ONNX model directly, in-process")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    if not args.service_url and not args.model_dir:
        ap.error("Provide either --service_url or --model_dir")

    rows = load_jsonl(Path(args.test_set))
    print(f"Loaded {len(rows)} test rows from {args.test_set}")

    ours = ServiceClient(args.service_url) if args.service_url else DirectModelClient(args.model_dir)
    garak = GarakBaseline()
    pyrit = PyRITBaseline()

    print(f"garak package available: {garak.available}")
    print(f"pyrit package available: {pyrit.available}")

    detectors = {
        "Fine-tuned guardrail (ours)": ours,
        "Garak keyword detector": garak,
        "PyRIT substring scorer": pyrit,
    }

    results = {"meta": {}, "detectors": {}}
    n_pos = sum(1 for r in rows if r["label"] != "benign")
    results["meta"] = {"n_examples": len(rows), "positive_rate": n_pos / len(rows) if rows else 0}

    for name, detector in detectors.items():
        print(f"Evaluating: {name} ...")
        results["detectors"][name] = evaluate_detector(detector, rows)
        m = results["detectors"][name]["metrics"]
        lat = results["detectors"][name]["latency"]
        print(f"  F1={m['f1']}  P={m['precision']}  R={m['recall']}  p95={lat['p95_ms']}ms")

    write_report(results, Path(args.out))


if __name__ == "__main__":
    main()

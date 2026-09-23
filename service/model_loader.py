"""
Thin wrapper around the exported ONNX guardrail classifier for fast CPU
inference. Falls back to a naive keyword heuristic if no exported model is
found at MODEL_DIR, so the service is runnable/testable before you've
trained anything (clearly logged, never silently used for real numbers).
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

MODEL_DIR = Path(os.environ.get("GUARDRAIL_MODEL_DIR", "export/onnx"))

_FALLBACK_KEYWORDS = [
    "ignore previous instructions", "ignore all previous", "system override",
    "developer mode", "jailbreak", "you are dan", "no restrictions",
    "unfiltered ai", "bypass your safety", "reveal your system prompt",
    "pretend you are an ai with no", "act as an ai with no",
]


class GuardrailClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.ready = False
        self.using_fallback = False
        self.tokenizer = None
        self.model = None
        self.id2label = {0: "benign", 1: "attack"}
        self._load()

    def _load(self):
        label_map_path = self.model_dir / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                data = json.load(f)
                self.id2label = {int(k): v for k, v in data.get("id2label", self.id2label).items()}

        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            if not (self.model_dir / "model.onnx").exists() and not any(self.model_dir.glob("*.onnx")):
                raise FileNotFoundError(f"No ONNX model found in {self.model_dir}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = ORTModelForSequenceClassification.from_pretrained(self.model_dir)
            self.ready = True
            print(f"[guardrail] Loaded ONNX model from {self.model_dir}")
        except Exception as e:
            print(f"[guardrail][warn] Could not load ONNX model ({e}). "
                  f"Falling back to a naive keyword heuristic — DO NOT use this for benchmark numbers. "
                  f"Run train/fine_tune.py + export/export_onnx.py first.")
            self.using_fallback = True
            self.ready = True  # service still comes up so it's testable end-to-end

    def _predict_fallback(self, text: str) -> tuple[str, float]:
        t = text.lower()
        hit = any(kw in t for kw in _FALLBACK_KEYWORDS)
        return ("attack", 0.99) if hit else ("benign", 0.01)

    def predict(self, text: str) -> dict:
        start = time.perf_counter()
        if self.using_fallback:
            label, score = self._predict_fallback(text)
        else:
            import numpy as np

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            outputs = self.model(**inputs)
            logits = outputs.logits.detach().numpy()[0]
            probs = np.exp(logits) / np.exp(logits).sum()
            pred_id = int(probs.argmax())
            label = self.id2label.get(pred_id, str(pred_id))
            score = float(probs[pred_id])
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "label": label,
            "score": score,
            "is_attack": label != "benign",
            "latency_ms": round(latency_ms, 3),
        }


_classifier: Optional[GuardrailClassifier] = None


def get_classifier() -> GuardrailClassifier:
    global _classifier
    if _classifier is None:
        _classifier = GuardrailClassifier()
    return _classifier

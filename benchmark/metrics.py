"""
F1/precision/recall + latency percentile computation and report generation
for the guardrail benchmark.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def compute_latency_stats(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


def write_report(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_lines = [
        "# Guardrail Benchmark Report",
        "",
        f"Test set size: {results['meta']['n_examples']}  |  "
        f"Positive (attack) rate: {results['meta']['positive_rate']:.2%}",
        "",
        "| Detector | Precision | Recall | F1 | Accuracy | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results["detectors"].items():
        m = r["metrics"]
        lat = r["latency"]
        md_lines.append(
            f"| {name} | {m['precision']} | {m['recall']} | {m['f1']} | {m['accuracy']} "
            f"| {lat['p50_ms']}ms | {lat['p95_ms']}ms |"
        )

    md_lines += [
        "",
        "## Confusion matrices",
        "",
    ]
    for name, r in results["detectors"].items():
        m = r["metrics"]
        md_lines += [
            f"**{name}**: TP={m['true_positive']} FP={m['false_positive']} "
            f"FN={m['false_negative']} TN={m['true_negative']}",
            "",
        ]

    md_path = out_dir / "benchmark_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {json_path} and {md_path}")

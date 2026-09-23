# Guardrail Benchmark Report

Test set size: 41  |  Positive (attack) rate: 48.78%

| Detector | Precision | Recall | F1 | Accuracy | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| Fine-tuned guardrail (ours) | 0.8 | 1.0 | 0.8889 | 0.878 | 23.057ms | 39.615ms |
| Garak keyword detector | 1.0 | 0.05 | 0.0952 | 0.5366 | 0.002ms | 0.006ms |
| PyRIT substring scorer | 0.5 | 0.05 | 0.0909 | 0.5122 | 0.002ms | 0.007ms |

## Confusion matrices

**Fine-tuned guardrail (ours)**: TP=20 FP=5 FN=0 TN=16

**Garak keyword detector**: TP=1 FP=0 FN=19 TN=21

**PyRIT substring scorer**: TP=1 FP=1 FN=19 TN=20

# Guardrail Benchmark Report

Test set size: 485  |  Positive (attack) rate: 27.84%

| Detector | Precision | Recall | F1 | Accuracy | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| Fine-tuned guardrail (ours) | 0.9664 | 0.8519 | 0.9055 | 0.9505 | 24.607ms | 41.368ms |
| Garak keyword detector | 1.0 | 0.1185 | 0.2119 | 0.7546 | 0.002ms | 0.003ms |
| PyRIT substring scorer | 0.8261 | 0.1407 | 0.2405 | 0.7526 | 0.002ms | 0.003ms |

## Confusion matrices

**Fine-tuned guardrail (ours)**: TP=115 FP=4 FN=20 TN=346

**Garak keyword detector**: TP=16 FP=0 FN=119 TN=350

**PyRIT substring scorer**: TP=19 FP=4 FN=116 TN=346

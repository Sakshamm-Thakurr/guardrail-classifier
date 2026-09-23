# Guardrail Classifier: Fine-Tuned Prompt-Injection/Jailbreak Detector

Benchmarked head-to-head against NVIDIA **Garak** and Microsoft **PyRIT** built-in
detectors on the same adversarial corpus.

This replaces a phrase-matching classifier with a fine-tuned transformer
(DistilBERT by default, ModernBERT optional) served as a low-latency FastAPI
microservice, exported to ONNX for fast CPU inference.

```
Labeled dataset (JailbreakBench + AdvBench + your own corpus)
        │
        ▼
  fine-tune classifier (train/fine_tune.py)
        │
        ▼
  export to ONNX (export/export_onnx.py)
        │
        ▼
  serve as guardrail microservice (service/main.py, FastAPI)
        │
        ▼
  benchmark vs. Garak / PyRIT on held-out test set (benchmark/run_benchmark.py)
```

## Repo layout

```
guardrail-classifier/
├── data/
│   ├── prepare_dataset.py       # downloads + merges JailbreakBench, AdvBench, benign sets
│   ├── custom_corpus.jsonl      # your 76-sample hand-labeled corpus (seed file, extend freely)
│   └── schema.md                # label schema / dataset card
├── train/
│   └── fine_tune.py             # HF Trainer fine-tuning script (DistilBERT/ModernBERT)
├── export/
│   └── export_onnx.py           # HF model -> ONNX + quantization
├── service/
│   ├── main.py                  # FastAPI guardrail microservice
│   ├── model_loader.py          # ONNXRuntime inference wrapper
│   └── schemas.py                # pydantic request/response models
├── benchmark/
│   ├── run_benchmark.py         # orchestrates the 3-way comparison
│   ├── garak_baseline.py        # wraps Garak's keyword/toxicity detectors
│   ├── pyrit_baseline.py        # wraps PyRIT's SelfAskTrueFalseScorer / substring scorer
│   └── metrics.py               # F1/precision/recall/latency computation + report generation
├── results/                     # benchmark_report.md and .json land here (gitignored data, kept structure)
├── tests/
│   └── test_service.py          # smoke tests for the API
├── .github/workflows/ci.yml     # lint + unit tests on push
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
git clone https://github.com/<you>/guardrail-classifier.git
cd guardrail-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the dataset (merges public sets with data/custom_corpus.jsonl)
python data/prepare_dataset.py --out data/processed

# 2. Fine-tune (defaults to distilbert-base-uncased, ~15 min on a single GPU,
#    ~1-2 hrs on CPU for the seed-size dataset)
python train/fine_tune.py --data data/processed --model distilbert-base-uncased --epochs 3

# 3. Export to ONNX for fast inference
python export/export_onnx.py --model_dir train/output/best --out export/onnx

# 4. Serve
uvicorn service.main:app --host 0.0.0.0 --port 8000

# 5. Benchmark against Garak + PyRIT on the same held-out set
python benchmark/run_benchmark.py --test_set data/processed/test.jsonl --service_url http://localhost:8000
```

`make all` runs steps 1–3 in sequence; see `Makefile`.

## Why this is comparable, not just "our model wins"

Garak and PyRIT are not classifiers designed to be F1-optimal out of the box —
Garak's default detectors are largely keyword/substring/toxicity-model based,
and PyRIT's scorers are meant as building blocks, not a tuned production
guardrail. Running them here as **baselines on identical inputs** is the point:
it's the same methodology a red-team/AI-safety hiring manager would want to
see — awareness of the existing tooling, not reinvention in a vacuum. The
benchmark script logs exact Garak/PyRIT versions and detector/scorer configs
used, so the comparison is reproducible and falsifiable.

## Sample result format

`benchmark/run_benchmark.py` produces `results/benchmark_report.md`, e.g.:

| Detector | Precision | Recall | F1 | p50 latency | p95 latency |
|---|---|---|---|---|---|
| Fine-tuned DistilBERT (ours) | 0.93 | 0.89 | 0.91 | 11ms | 22ms |
| Garak keyword detector | 0.71 | 0.65 | 0.68 | 2ms | 4ms |
| PyRIT SubStringScorer | 0.74 | 0.60 | 0.66 | 3ms | 5ms |

**These numbers are placeholders / targets, not measured results.** This repo
gives you the full harness — you need to actually run steps 1–5 on your machine
(with GPU access and internet access to Hugging Face Hub) to get real numbers.
Do not put unverified numbers on your resume; run the pipeline, then report
what `results/benchmark_report.md` actually says.

## Publishing to Hugging Face (for external verifiability)

```bash
huggingface-cli login
python export/push_to_hub.py --model_dir train/output/best --repo <your-username>/guardrail-distilbert
```

This makes the fine-tuned weights + eval harness independently checkable by
anyone, which is what turns "0.91 F1" from a self-reported claim into a
verifiable one.

## Resume line (once you have real numbers)

> Fine-tuned a DistilBERT prompt-injection classifier and benchmarked it
> against NVIDIA Garak and Microsoft PyRIT's built-in detectors on an
> identical 500-sample adversarial test set, achieving F1 <X> vs. Garak's
> <Y> at p95 latency <Z>ms; served via FastAPI+ONNX and published on
> Hugging Face for independent verification.

Fill in `<X>/<Y>/<Z>` with your actual `results/benchmark_report.md` numbers —
never the placeholder ones above.

## Notes on environment / what to expect

- This sandbox has no GPU and no network access to `huggingface.co`, so the
  scripts here are fully written and unit-tested for structure/logic but the
  actual fine-tuning run + real benchmark numbers must be produced on your
  own machine or in Colab. `tests/test_service.py` and the dry-run flags
  (`--dry_run`) let you sanity-check the pipeline without downloading models.
- `data/custom_corpus.jsonl` ships with a 76-row seed corpus of clearly
  synthetic, non-operational example prompts (labeled benign/injection/
  jailbreak) so the pipeline runs end-to-end immediately. Replace/extend it
  with your own labeled examples before reporting real numbers.

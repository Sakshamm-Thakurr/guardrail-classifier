# Dataset schema

Every row (in `custom_corpus.jsonl` and the processed output) is a JSON object:

```json
{"text": "<the prompt text>", "label": "benign|injection|jailbreak", "source": "custom|jailbreakbench|advbench"}
```

## Labels

- `benign` — an ordinary user request, no attack.
- `injection` — attempts to override/inject instructions into an LLM
  pipeline (e.g. "ignore previous instructions and...", hidden instructions
  inside document/tool content).
- `jailbreak` — attempts to get a model to bypass its safety training via
  roleplay, hypothetical framing, encoding tricks, persona hijacking, etc.

For binary classification (what the default training script does), `injection`
and `jailbreak` are merged into a single positive class `attack`; keep the
3-way label in the data for anyone who wants finer-grained classification
later (`--label_mode 3way` vs default `--label_mode binary` in
`fine_tune.py`).

## Public sources merged by `prepare_dataset.py`

- **JailbreakBench** (`JailbreakBench/JBB-Behaviors` on the Hugging Face Hub)
  — harmful behaviors + jailbreak prompts used for evaluation.
- **AdvBench** (`walledai/AdvBench` mirror on the Hugging Face Hub, originally
  from Zou et al. "Universal and Transferable Adversarial Attacks on Aligned
  Language Models") — harmful-behavior strings, used here as attack-class
  examples paired with benign counterparts.
- Benign negatives are sampled from a general instruction-following dataset
  (`tatsu-lab/alpaca`) to keep the classifier from just learning "long/short
  text" as a shortcut.

`prepare_dataset.py` requires network access to `huggingface.co` — if that's
blocked in your environment, run it somewhere that has Hub access, or pass
`--offline_seed_only` to build a smaller dataset from just
`custom_corpus.jsonl` (useful for testing the pipeline end-to-end).

## Your custom corpus (`custom_corpus.jsonl`)

76 rows, hand-written for this project, covering both classes plus edge
cases public datasets under-represent: multi-turn setup attacks, prompts
that *mention* jailbreaking without attempting one (hard negatives), and
injection attempts embedded inside quoted "document" content. Extend this
file with real logs from your own systems (de-identified) for the strongest,
most defensible results — public benchmark sets alone are increasingly
memorized by base models and don't reflect your actual traffic.

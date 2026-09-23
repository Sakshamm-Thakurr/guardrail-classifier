"""
Build the training/eval dataset by merging public jailbreak/injection sets
with the hand-labeled custom corpus.

Usage:
    python data/prepare_dataset.py --out data/processed
    python data/prepare_dataset.py --out data/processed --offline_seed_only

Requires network access to huggingface.co unless --offline_seed_only is set.
"""
import argparse
import json
import os
import random
from pathlib import Path

RANDOM_SEED = 42


def load_custom_corpus(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_jailbreakbench() -> list[dict]:
    """JailbreakBench/JBB-Behaviors: harmful-behavior + jailbreak prompts."""
    from datasets import load_dataset

    rows = []
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        for r in ds:
            text = r.get("Goal") or r.get("goal") or r.get("Behavior")
            if text:
                rows.append({"text": text, "label": "jailbreak", "source": "jailbreakbench"})
    except Exception as e:
        print(f"[warn] could not load JailbreakBench ({e}); skipping.")
    return rows


def load_advbench() -> list[dict]:
    """AdvBench harmful-behavior strings (Zou et al.), used as attack-class text."""
    from datasets import load_dataset

    rows = []
    try:
        ds = load_dataset("walledai/AdvBench", split="train")
        for r in ds:
            text = r.get("prompt") or r.get("goal") or r.get("text")
            if text:
                rows.append({"text": text, "label": "injection", "source": "advbench"})
    except Exception as e:
        print(f"[warn] could not load AdvBench ({e}); skipping.")
    return rows


def load_benign_negatives(n: int) -> list[dict]:
    """Sample benign instructions from Alpaca to balance the attack classes."""
    from datasets import load_dataset

    rows = []
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        idxs = random.sample(range(len(ds)), min(n, len(ds)))
        for i in idxs:
            text = ds[i]["instruction"]
            if ds[i].get("input"):
                text = f"{text}\n{ds[i]['input']}"
            rows.append({"text": text, "label": "benign", "source": "alpaca"})
    except Exception as e:
        print(f"[warn] could not load Alpaca benign set ({e}); skipping.")
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_and_write(rows: list[dict], out_dir: Path, test_frac=0.15, val_frac=0.1):
    random.Random(RANDOM_SEED).shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = rows[:n_test]
    val = rows[n_test:n_test + n_val]
    train = rows[n_test + n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(split_rows):5d} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--custom_corpus", type=str, default="data/custom_corpus.jsonl")
    ap.add_argument("--offline_seed_only", action="store_true",
                     help="Skip Hugging Face Hub downloads; build dataset from custom_corpus.jsonl only.")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(RANDOM_SEED)

    rows = load_custom_corpus(Path(args.custom_corpus))
    print(f"custom corpus: {len(rows)} rows")

    if not args.offline_seed_only:
        jbb = load_jailbreakbench()
        adv = load_advbench()
        n_attack = len(jbb) + len(adv)
        benign = load_benign_negatives(n=max(n_attack, 50))
        print(f"jailbreakbench: {len(jbb)} rows | advbench: {len(adv)} rows | benign(alpaca): {len(benign)} rows")
        rows = rows + jbb + adv + benign
    else:
        print("[info] --offline_seed_only set: using only the custom corpus.")

    rows = dedupe(rows)
    print(f"total after dedupe: {len(rows)} rows")

    split_and_write(rows, Path(args.out), test_frac=args.test_frac, val_frac=args.val_frac)


if __name__ == "__main__":
    main()

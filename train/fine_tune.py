"""
Fine-tune a small transformer (DistilBERT by default, ModernBERT optional)
on the prompt-injection/jailbreak dataset produced by data/prepare_dataset.py.

Usage:
    python train/fine_tune.py --data data/processed --model distilbert-base-uncased --epochs 3
    python train/fine_tune.py --data data/processed --model answerdotai/ModernBERT-base --epochs 3

Binary mode (default) collapses {injection, jailbreak} -> "attack" vs "benign".
Use --label_mode 3way to keep the 3-class problem.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

BINARY_LABELS = ["benign", "attack"]
THREE_WAY_LABELS = ["benign", "injection", "jailbreak"]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_binary(label: str) -> str:
    return "benign" if label == "benign" else "attack"


def build_dataset(rows: list[dict], label_mode: str, label2id: dict) -> Dataset:
    texts, labels = [], []
    for r in rows:
        lbl = r["label"] if label_mode == "3way" else to_binary(r["label"])
        texts.append(r["text"])
        labels.append(label2id[lbl])
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary" if len(set(labels)) == 2 else "macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed",
                     help="Directory containing train.jsonl / val.jsonl / test.jsonl")
    ap.add_argument("--model", type=str, default="distilbert-base-uncased",
                     help="e.g. distilbert-base-uncased or answerdotai/ModernBERT-base")
    ap.add_argument("--label_mode", type=str, default="binary", choices=["binary", "3way"])
    ap.add_argument("--output_dir", type=str, default="train/output")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--dry_run", action="store_true",
                     help="Build the pipeline (tokenizer/model/dataset) without training, to sanity-check wiring.")
    args = ap.parse_args()

    labels = THREE_WAY_LABELS if args.label_mode == "3way" else BINARY_LABELS
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    data_dir = Path(args.data)
    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "val.jsonl")
    print(f"train: {len(train_rows)} rows | val: {len(val_rows)} rows | label_mode={args.label_mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    train_ds = build_dataset(train_rows, args.label_mode, label2id)
    val_ds = build_dataset(val_rows, args.label_mode, label2id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    if args.dry_run:
        print("[dry_run] tokenizer + model + datasets built successfully. Skipping training.")
        print(f"[dry_run] sample encoded input: {train_ds[0]}")
        return

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final validation metrics:", metrics)

    best_dir = Path(args.output_dir) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    with open(best_dir / "label_map.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    print(f"Saved best model to {best_dir}")


if __name__ == "__main__":
    main()

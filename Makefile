.PHONY: install data train export serve benchmark test all clean

install:
	pip install -r requirements.txt

data:
	python data/prepare_dataset.py --out data/processed

train:
	python train/fine_tune.py --data data/processed --model distilbert-base-uncased --epochs 3

export:
	python export/export_onnx.py --model_dir train/output/best --out export/onnx --quantize

serve:
	uvicorn service.main:app --host 0.0.0.0 --port 8000

benchmark:
	python benchmark/run_benchmark.py --test_set data/processed/test.jsonl --model_dir export/onnx

test:
	pytest tests/ -v

all: data train export benchmark

clean:
	rm -rf data/processed train/output export/onnx results/*.json results/*.md

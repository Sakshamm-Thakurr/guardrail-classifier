"""
FastAPI guardrail microservice serving the ONNX-exported prompt-injection /
jailbreak classifier.

Run:
    uvicorn service.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    POST /classify        {"text": "..."}
    POST /classify_batch  {"texts": ["...", "..."]}
"""
from fastapi import FastAPI, HTTPException

from service.model_loader import get_classifier
from service.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    BatchClassifyRequest,
    BatchClassifyResponse,
    HealthResponse,
)

app = FastAPI(
    title="Guardrail Classifier",
    description="Fine-tuned transformer guardrail for prompt-injection / jailbreak detection.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    clf = get_classifier()
    return HealthResponse(status="ok", model_loaded=clf.ready and not clf.using_fallback)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    clf = get_classifier()
    try:
        result = clf.predict(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ClassifyResponse(**result)


@app.post("/classify_batch", response_model=BatchClassifyResponse)
def classify_batch(req: BatchClassifyRequest):
    clf = get_classifier()
    try:
        results = [ClassifyResponse(**clf.predict(t)) for t in req.texts]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return BatchClassifyResponse(results=results)

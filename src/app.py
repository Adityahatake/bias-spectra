"""
BiasSpectra – FastAPI Application
====================================
Backend for the premium dark-themed UI for political bias detection in Indian news.
Uses the BiasPredictor engine for inference.
"""

import sys
import os
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.predictor import BiasPredictor, BiasResult

# ─── Caching Predictors ───────────────────────────────────────
predictors = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the default NLI predictor on startup to save time later
    yield
    predictors.clear()


app = FastAPI(
    title="BiasSpectra API",
    description="Political Bias Detection for Indian News",
    lifespan=lifespan
)

def get_predictor(model_type: str) -> BiasPredictor:
    if model_type not in predictors:
         predictors[model_type] = BiasPredictor(model_type=model_type)
    return predictors[model_type]

class PredictRequest(BaseModel):
    headline: str
    model: str = "nli"

class PredictResponse(BaseModel):
    label: str
    confidence: dict
    gate: str
    reasoning: str
    is_model_prediction: bool

@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
        
    try:
        predictor = get_predictor(req.model)
        result: BiasResult = predictor.predict(req.headline)
        
        return PredictResponse(
            label=result.label,
            confidence=result.confidence,
            gate=result.gate,
            reasoning=result.reasoning,
            is_model_prediction=result.is_model_prediction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configure static file serving for the frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(frontend_dir, exist_ok=True)

app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


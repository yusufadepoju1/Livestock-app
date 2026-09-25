"""FastAPI service for livestock disease prediction.

Run:  uvicorn app.main:app --reload
Open: http://127.0.0.1:8000       (UI)
      http://127.0.0.1:8000/docs  (interactive API docs)
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .predictor import Predictor

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Livestock Disease Prediction", version="1.0.0")
predictor = Predictor()


class PredictRequest(BaseModel):
    animal: str = Field(examples=["cow"])
    age: float = Field(ge=0, le=40, description="Age in years", examples=[4])
    temperature: float = Field(ge=90, le=115, description="Body temperature in °F", examples=[103.5])
    symptoms: list[str] = Field(min_length=1, max_length=3, examples=[["chills", "fatigue", "sweats"]])


@app.get("/api/health")
def health():
    return {"status": "ok", "accuracy": predictor.meta["metrics"]["accuracy"]}


@app.get("/api/meta")
def meta():
    """Everything the UI needs to build its form: animals, symptoms, ranges, model accuracy."""
    return predictor.meta


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        return predictor.predict(req.animal, req.age, req.temperature, req.symptoms)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")

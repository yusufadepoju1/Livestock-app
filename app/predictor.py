"""Model loading and prediction logic (kept separate from FastAPI so it is easy to test)."""
import json
from itertools import permutations
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SLOTS = 3


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.pipeline = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.classes = list(self.pipeline.classes_)

    def validate(self, animal: str, symptoms: list[str]) -> None:
        if animal not in self.meta["animals"]:
            raise ValueError(f"Unknown animal '{animal}'. Choose from {self.meta['animals']}.")
        unknown = [s for s in symptoms if s not in self.meta["symptoms"]]
        if unknown:
            raise ValueError(f"Unknown symptom(s): {unknown}")
        if len(set(symptoms)) != len(symptoms):
            raise ValueError("Each symptom can only be selected once.")
        if not 1 <= len(symptoms) <= SLOTS:
            raise ValueError(f"Select between 1 and {SLOTS} symptoms.")

    def predict(self, animal: str, age: float, temperature: float, symptoms: list[str]) -> dict:
        self.validate(animal, symptoms)

        # The model was trained with three symptom slots. Symptom order carries no meaning,
        # so average over every ordering of the chosen symptoms (blank slots are ignored by
        # the encoder). This makes the answer independent of the order the user clicked in.
        padded = symptoms + [""] * (SLOTS - len(symptoms))
        rows = [
            {"Animal": animal, "Age": age, "Temperature": temperature,
             "Symptom 1": p[0], "Symptom 2": p[1], "Symptom 3": p[2]}
            for p in sorted(set(permutations(padded)))
        ]
        proba = self.pipeline.predict_proba(pd.DataFrame(rows)).mean(axis=0)

        ranked = sorted(zip(self.classes, proba), key=lambda t: t[1], reverse=True)
        top = ranked[0][0]
        result = {
            "prediction": top,
            "confidence": round(float(ranked[0][1]), 4),
            "probabilities": [{"disease": d, "probability": round(float(p), 4)} for d, p in ranked],
            "notes": [],
        }

        # Diseases that share identical symptoms in the training data can't be separated.
        for group in self.meta["confusable_groups"]:
            if top in group:
                share = sum(p for d, p in ranked if d in group)
                if share > 0.9:
                    others = [d for d in group if d != top]
                    result["notes"].append(
                        f"In the training data, {top} and {' / '.join(others)} show exactly the same "
                        "symptoms, so the model cannot reliably tell them apart. "
                        "Treat this as a shortlist and confirm with a veterinarian or lab test."
                    )

        if len(symptoms) < SLOTS:
            result["notes"].append(
                f"Only {len(symptoms)} of {SLOTS} symptoms selected. Results are most reliable with three."
            )
        lo, hi = self.meta["age_range"]
        if not lo <= age <= hi:
            result["notes"].append(f"Age {age:g} is outside the training range ({lo}-{hi} years).")
        lo, hi = self.meta["temp_range"]
        if not lo <= temperature <= hi:
            result["notes"].append(
                f"Temperature {temperature:g} is outside the training range ({lo:g}-{hi:g} °F)."
            )
        return result

"""Give pneumonia its own symptom profile.

Problem: in the original data every pneumonia row had exactly the same symptoms as
lumpy virus (depression, loss of appetite, painless lumps), so the two diseases could
not be told apart by any model.

Fix: keep every row, animal, age, temperature and disease label exactly as they were,
and re-draw only the three symptom columns of pneumonia rows from a respiratory
symptom pool. Each row gets 3 distinct symptoms, drawn at random (fixed seed), the same
way the other diseases' rows appear to have been generated.

The pool deliberately overlaps with anthrax (shortness of breath, chest discomfort),
blackleg (crackling sound, depression, loss of appetite) so pneumonia is realistic
rather than trivially separable. "cough" and "nasal discharge" are new symptoms.
"fever" is not used because body temperature is already a numeric input.

These symptom values are synthetic. Replace POOL with a veterinary-sourced list
if you have one, then re-run:

    python scripts/fix_pneumonia_symptoms.py
    python train.py

Reads data/animal_disease_dataset.original.csv (never modified) and writes
data/animal_disease_dataset.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "animal_disease_dataset.original.csv"
DST = ROOT / "data" / "animal_disease_dataset.csv"

SYMPTOM_COLS = ["Symptom 1", "Symptom 2", "Symptom 3"]
POOL = [
    "cough",
    "nasal discharge",
    "shortness of breath",
    "chest discomfort",
    "crackling sound",
    "loss of appetite",
    "depression",
]
SEED = 42

df = pd.read_csv(SRC)
mask = df["Disease"] == "pneumonia"
rng = np.random.default_rng(SEED)

new = np.array([rng.choice(POOL, size=3, replace=False) for _ in range(mask.sum())])
df.loc[mask, SYMPTOM_COLS] = new

df.to_csv(DST, index=False)
print(f"Rewrote symptoms for {int(mask.sum())} pneumonia rows -> {DST.name}")

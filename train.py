"""Train the livestock disease model (same pipeline as livestock.ipynb) and save it.

Run:  python train.py
Outputs: model/model.joblib, model/metadata.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "animal_disease_dataset.csv"
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

CATEGORICAL = ["Animal", "Symptom 1", "Symptom 2", "Symptom 3"]
NUMERICAL = ["Age", "Temperature"]
SYMPTOM_COLS = ["Symptom 1", "Symptom 2", "Symptom 3"]

df = pd.read_csv(DATA)
X, y = df.drop("Disease", axis=1), df["Disease"]

# Identical to the notebook: same split, same seed, same preprocessing, same LR settings.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numerical", StandardScaler(), NUMERICAL),
    ]
)
pipeline = Pipeline(
    [
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)
pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, average="weighted"),
    "recall": recall_score(y_test, pred, average="weighted"),
    "f1": f1_score(y_test, pred, average="weighted"),
    "train_rows": int(len(y_train)),
    "test_rows": int(len(y_test)),
}
print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

# Diseases whose symptom vocabularies are identical cannot be told apart by the model.
vocab = {
    d: frozenset(g[SYMPTOM_COLS].values.ravel()) for d, g in df.groupby("Disease")
}
groups = {}
for disease, v in vocab.items():
    groups.setdefault(v, []).append(disease)
confusable = [sorted(g) for g in groups.values() if len(g) > 1]

metadata = {
    "classes": list(pipeline.classes_),
    "animals": sorted(df["Animal"].unique()),
    "symptoms": sorted(set(df[SYMPTOM_COLS].values.ravel())),
    "age_range": [int(df["Age"].min()), int(df["Age"].max())],
    "temp_range": [float(df["Temperature"].min()), float(df["Temperature"].max())],
    "confusable_groups": confusable,
    "metrics": metrics,
}
joblib.dump(pipeline, OUT / "model.joblib")
(OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
print("Saved model/model.joblib and model/metadata.json")
print("Indistinguishable disease groups:", confusable)

# Livestock disease prediction (FastAPI)

Logistic regression from `livestock.ipynb`, served with FastAPI and a clickable demo UI.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                                        # rebuilds model/ (already included)
uvicorn app.main:app --reload
```

- UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

If your scikit-learn version differs from the one used to build `model/model.joblib`,
run `python train.py` once in your own environment so the saved model loads cleanly.

## API

`POST /api/predict`

```json
{ "animal": "cow", "age": 4, "temperature": 103.5,
  "symptoms": ["chills", "fatigue", "sweats"] }
```

Returns the top disease, confidence, all class probabilities and any notes.
`GET /api/meta` lists valid animals and symptoms, training ranges and model metrics.

## Model results (held-out 20%, same split as the notebook)

Accuracy 0.9961 | Precision 0.9962 | Recall 0.9961 | F1 0.9961

## Data note: pneumonia symptoms

In the original dataset every pneumonia row had exactly the same symptoms as lumpy virus
(depression, loss of appetite, painless lumps), so no model could tell the two apart
(accuracy was about 83.6%, and the app had to report them as a 50/50 shortlist).

This is fixed in `data/animal_disease_dataset.csv`. Only the three symptom columns of the
pneumonia rows were re-drawn (fixed seed) from a respiratory pool: cough, nasal discharge,
shortness of breath, chest discomfort, crackling sound, loss of appetite and depression.
Animal, age, temperature, the disease labels and every other disease's rows are unchanged.
The untouched original is kept as `data/animal_disease_dataset.original.csv`.

The new pneumonia symptoms are synthetic. If you have a veterinary-sourced symptom list or
real records, edit `POOL` in `scripts/fix_pneumonia_symptoms.py` (or replace the data), then run:

```bash
python scripts/fix_pneumonia_symptoms.py
python train.py
```

Because the pneumonia symptoms are synthetic, treat the high accuracy as a property of this
dataset, not as proof of real-world diagnostic performance.

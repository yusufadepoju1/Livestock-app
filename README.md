# Livestock Disease Prediction System

## AI-Based Disease Prediction for Livestock Farming

### Project Overview

This system was developed to address a practical challenge faced by a **livestock farming operation: identifying potential animal diseases early from observable health information and symptoms.**

When livestock begin showing symptoms, a farmer needs to determine how quickly the animal may require attention and whether veterinary intervention should be sought.

To support this process, I developed a **Machine Learning-based disease prediction system** that uses information about an animal and its observed symptoms to provide an initial prediction of a possible disease.

The system is designed to support the farmer's decision-making process. It does **not replace veterinary diagnosis or professional treatment.**

## The Farming Problem

In a livestock environment, identifying health problems early can be challenging.

An animal may begin showing symptoms such as:

* Changes in body temperature
* Fatigue
* Loss of appetite
* Chills
* Depression
* Other observable symptoms

When these signs appear, the farmer needs an initial indication of what may be affecting the animal so that appropriate action can be taken, including seeking professional veterinary assessment.

The problem I set out to address was:

> **How can available livestock health information be used to provide an early indication of a potential disease?**

## The Solution

I developed a Machine Learning system that allows livestock health information and observed symptoms to be entered into the application.

The system then provides:

* Predicted disease
* Prediction confidence
* Probabilities for possible diseases
* Relevant prediction notes

### System Workflow

```text
Animal Health Information
          |
          v
   Data Preprocessing
          |
          v
  Machine Learning Model
          |
          v
   Disease Prediction
          |
          v
 FastAPI Application
          |
          v
 Farmer / Farm User
```

## Machine Learning Approach

The project involved the complete Machine Learning workflow:

1. Data preparation
2. Exploratory data analysis
3. Feature preprocessing
4. Categorical feature encoding
5. Model training
6. Model comparison
7. Model evaluation
8. Model selection
9. Model serialization
10. API integration

### Models Evaluated

I evaluated multiple classification models, including:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Neural Network

Logistic Regression was selected for the deployed application based on the evaluation results obtained from the available dataset.


## Application

The trained model was integrated into a **FastAPI application** so that the prediction system could be accessed through a simple interface and API.

### Prediction Endpoint

```http
POST /api/predict
```

Example request:

```json
{
  "animal": "cow",
  "age": 4,
  "temperature": 103.5,
  "symptoms": [
    "chills",
    "fatigue",
    "sweats"
  ]
}
```

The application returns the predicted disease, confidence score, class probabilities, and relevant notes.

### API Documentation

FastAPI provides interactive documentation through:

```text
http://127.0.0.1:8000/docs
```

## Technology Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* FastAPI
* Uvicorn
* Joblib
* Jupyter Notebook

## Run the Application

### 1. Clone the repository

```bash
git clone https://github.com/yusufadepoju1/Livestock-app.git
cd Livestock-app
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Rebuild the model

The repository already contains a trained model. However, if your local scikit-learn version differs from the version used to create the saved model, rebuild it in your environment:

```bash
python train.py
```

This will recreate the model files in the `model/` directory.

### 5. Start the application

```bash
uvicorn app.main:app --reload
```

### 6. Open the application

Web interface:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Project Structure

```text
livestock-app/
│
├── app/
│   └── main.py
│
├── data/
│   ├── animal_disease_dataset.csv
│   └── animal_disease_dataset.original.csv
│
├── model/
│   └── model.joblib
│
├── scripts/
│   └── fix_pneumonia_symptoms.py
│
├── livestock.ipynb
├── train.py
├── requirements.txt
└── README.md
```

## Important Data Limitation

During development, I identified an issue in the original dataset.

The pneumonia records contained symptoms that were identical to those associated with lumpy virus. This made it difficult for the model to distinguish between the two diseases using symptoms alone.

For development purposes, the pneumonia symptom records were modified using a predefined respiratory symptom pool.

The modified symptoms are **synthetic** and have not been clinically validated.

The original dataset has been preserved separately.

Because of this limitation, the model's reported performance should be understood as performance on this particular dataset rather than evidence of clinical diagnostic accuracy.

## Moving Toward Real Farm Use

For the system to become a more reliable tool for farm operations, the next stage would involve 

Potential improvements include:

* Recording actual farm health cases
* Working with veterinary professionals to validate symptoms
* Adding vaccination history
* Adding animal breed information
* Recording previous health conditions
* Supporting additional livestock species
* Integrating farm health records
* Continuously evaluating model performance

## Project Goal

The goal of this project was to apply Machine Learning to a **real farming problem** and develop a practical tool that could provide an initial indication of potential livestock diseases from available health information.

The system is intended to support the farmer's decision-making process while keeping professional veterinary assessment as an essential part of animal healthcare.

## Disclaimer

This application is an AI/ML decision-support system and is not a veterinary diagnostic tool. Predictions should not be used as a substitute for professional veterinary examination, laboratory testing, diagnosis, or treatment.

# Diabetes Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview
This project implements a machine learning model to predict diabetes using the PIMA Diabetes Dataset. The model uses Support Vector Machine (SVM) with a linear kernel to classify whether a person is diabetic or not based on medical diagnostic measurements.

## Dataset
**Source:** PIMA Indians Diabetes Database

**Features:**
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)

**Target Variable:**
- Outcome: 0 (Non-diabetic) or 1 (Diabetic)

## Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── Diabetes_Prediction_using_ML.ipynb
├── README.md
└── requirements.txt
```

### Making Predictions
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Example input: (Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DiabetesPedigree, Age)
input_data = (1, 89, 66, 23, 94, 28.1, 0.167, 21)

# Load your trained model and scaler
# Make prediction
# ... (add your prediction code)
```

## Model Performance

- **Training Accuracy:** 78.66%
- **Test Accuracy:** 77.27%
- **Algorithm:** Support Vector Machine (Linear Kernel)

## Results
The model successfully predicts diabetes with reasonable accuracy. Further improvements could include:
- Feature engineering
- Hyperparameter tuning
- Trying different algorithms (Random Forest, XGBoost, Neural Networks)
- Handling class imbalance

## Technologies Used
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

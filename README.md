# 🩺 PCOS Early Risk Detection System using Machine Learning, FastAPI & Flutter

An AI-powered **Early Risk Prediction System for Polycystic Ovary Syndrome (PCOS)** that combines **Machine Learning**, a **FastAPI backend**, and a **Flutter-based mobile application** to provide **non-invasive, instant, and accessible PCOS risk screening**.

The system predicts early PCOS risk using a **dual-model Gradient Boosting architecture** trained on healthcare datasets and exposes predictions through a REST API integrated with a Flutter mobile application.

> **Disclaimer:** This system is designed for **early risk screening only** and **not for medical diagnosis**. Clinical consultation and validation are required.

---

## Key Features

✅ Early PCOS risk prediction using Machine Learning  
✅ Dual-model architecture to reduce cycle dominance bias  
✅ FastAPI REST API backend for real-time predictions  
✅ Flutter mobile application (Android/iOS/Web support)  
✅ Lifestyle recommendations based on prediction outcome  
✅ Sleep advisory integration  
✅ Five-tier risk stratification framework  
✅ Guest mode + authenticated user support  
✅ Prediction history support  
✅ Cloud deployment ready (Render)

---

# System Architecture

The project follows a **three-tier client-server architecture**:

```text
User
   ↓
Flutter Mobile App
   ↓ (HTTP POST / JSON)
FastAPI Backend
   ↓
Dual ML Models
   ↓
Risk Prediction
   ↓
JSON Response
   ↓
Flutter Result Screen
```

### Prediction Pipeline

1. User enters health information in Flutter App  
2. Data sent to FastAPI backend via REST API  
3. Input validated using **Pydantic schema**  
4. BMI + interaction features calculated  
5. Model A (with cycle data) generates prediction  
6. Model B (without cycle data) generates prediction  
7. Final risk score calculated:

```text
Final Probability = (Model A + Model B) / 2
```

8. Risk stage, confidence score, lifestyle recommendations generated  
9. Response returned to mobile application

---

# Machine Learning Methodology

The backend uses a **Dual Gradient Boosting Model Architecture** for better reliability and reduced feature bias.

## 🔹 Model A — Screening Model

Uses **all features**, including:

- Cycle(R/I)
- Cycle Length
- BMI
- Symptoms

### Purpose:
High clinical sensitivity for detecting potential PCOS cases.

---

## 🔹 Model B — Confirmation Model

Excludes:

- Cycle(R/I)

Focuses on:

- Hair Growth
- Skin Darkening
- Weight Gain
- BMI
- Lifestyle-related indicators

### Purpose:
Reduces over-reliance on menstrual cycle irregularity.

---

## 🔹 Final Prediction

Final risk score:

```text
P_final = (P_ModelA + P_ModelB) / 2
```

This helps reduce **single-feature dominance bias** while maintaining high recall.

---

# Dataset & Preprocessing

The final dataset contains **3,541 records**.

| Dataset | Records | Source |
|----------|---------|--------|
| Kaggle PCOS Dataset | 541 | Kerala Hospital Clinical Data |
| Rotterdam Criteria Dataset | 3000 | Synthetic Clinical Data |
| Final Combined Dataset | 3541 | Merged Dataset |

### Preprocessing Steps

- Data cleaning
- Feature engineering
- BMI calculation
- Interaction feature creation
- Missing value handling
- SMOTE oversampling (`k=5`)
- Stratified train-test split

### Engineered Features

- `Cycle_HairGrowth`
- `Cycle_WeightGain`
- `Cycle_SkinDark`

---

# Comparative Model Evaluation

Multiple classifiers were evaluated:

| Model | Accuracy | Recall | ROC-AUC |
|--------|----------|---------|----------|
| Logistic Regression | 86.2% | 72% | 0.889 |
| Random Forest | 86.2% | 75% | 0.889 |
| Gradient Boosting | 90.27% | 97% | **0.967** |

Gradient Boosting performed best and was selected.

---

# Final Model Performance

## Model A — With Cycle Data

- Accuracy: **90.27%**
- Recall: **97%**
- ROC-AUC: **0.967**

## Model B — Without Cycle Data

- Recall: **97%**
- ROC-AUC: **0.819**

### Clinical Objective:
Prioritize **high recall** to minimize false negatives in screening.

---

# Features Used

### Clinical Features

- Age (yrs)
- Cycle (R/I)
- Cycle Length (days)
- Weight Gain
- Hair Growth
- Skin Darkening
- Hair Loss
- Pimples
- Fast Food Consumption
- Regular Exercise
- BMI

### Additional Feature

- Sleep Rating (1–10)

---

# FastAPI Backend

The Machine Learning models are exposed using **FastAPI**.

### Endpoint

```http
POST /predict
```

### Sample Request

```json
{
  "Age_yrs": 22,
  "Cycle_R_I": 4,
  "Cycle_length_days": 40,
  "Weight_gain_Y_N": 1,
  "hair_growth_Y_N": 0,
  "Skin_darkening_Y_N": 0,
  "Hair_loss_Y_N": 1,
  "Pimples_Y_N": 1,
  "Fast_food_Y_N": 1,
  "Reg_Exercise_Y_N": 0,
  "Weight_kg": 68,
  "Height_cm": 150,
  "Sleep_Rating_1_10": 5
}
```

### Sample Response

```json
{
  "risk_probability": 0.81,
  "risk_stage": "High Risk",
  "prediction_confidence": "High confidence prediction",
  "model_with_cycle": 0.92,
  "model_without_cycle": 0.70,
  "lifestyle_recommendations": [],
  "sleep_advisory": "Average sleep quality..."
}
```

---

# Flutter Mobile Application

A complete **cross-platform Flutter application** was developed to make the system easily accessible.

### Main Screens

- Splash Screen
- Login Screen
- Home Screen
- Assessment Form
- Result Screen

### App Features

1. User authentication  
2. Guest mode  
3. Health assessment form  
4. Real-time API integration  
5. Circular risk gauge visualization  
6. Prediction display  
7. Recommendations section

---

# Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Gradient Boosting
- SMOTE
- Pandas
- NumPy

### Backend
- FastAPI
- Pydantic
- Uvicorn
- MongoDB

### Frontend
- Flutter
- Dart

### Deployment
- Render Cloud Platform

---

# Project Structure

```text
pcos-ml-model/
│
├── app/
│   └── main.py
│
├── ml/
│   ├── models/
│   │   ├── pcos_model.pkl
│   │   ├── pcos_model_no_cycle.pkl
│   │
│   ├── predict.py
│   ├── train_model.py
│   └── data/
│
├── lib/                # Flutter App
├── android/
├── ios/
├── web/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation & Setup

## Clone Repository

```bash
git clone https://github.com/aniteshhalderofficial-commits/pcos-ml-model.git
cd pcos-ml-model
```

---

## Backend Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Flutter App Setup

```bash
flutter pub get
flutter run
```

---

# Future Improvements

- SHAP Explainability
- Hormonal test integration
- Ultrasound-based features
- Play Store deployment
- Cloud synchronization
- Wearable device integration
- Multi-class phenotype prediction

---

# Contributors

### Anitesh Halder
**Machine Learning & Backend**

- Dataset collection & preprocessing
- Feature engineering
- Dual-model ML architecture
- Model training & evaluation
- FastAPI backend
- MongoDB integration
- API deployment on Render

### Anand Singh
**Flutter App & Integration**

- Flutter mobile application
- UI/UX implementation
- API integration
- Form handling
- Risk visualization
- Authentication & guest mode
- End-to-end app testing

---

# License

For **academic and educational purposes only**.

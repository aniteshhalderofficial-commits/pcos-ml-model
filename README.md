**PCOS Early Risk Detection System (ML + API)**

Overview:

This project is a Machine Learning-based early risk detection system for PCOS (Polycystic Ovary Syndrome).

It uses a dual-model architecture to improve both:

- Prediction performance
- Clinical reliability

The system is exposed via a FastAPI backend, making it easy to integrate with frontend applications.

---

Key Features:

- Early risk prediction using ML
- Dual-model system (reduces bias from single feature dominance)
- Lifestyle recommendations
- Sleep advisory integration
- REST API for easy integration

---

Model Architecture:

🔹 Model A — Screening Model

- Uses all features including Cycle(R/I)
- High sensitivity (detects most PCOS cases)

🔹 Model B — Confirmation Model

- Excludes Cycle(R/I)
- Focuses on:
  - Hair growth
  - Skin darkening
  - Weight gain
  - BMI

🔹 Final Prediction

The final risk is computed as:

Final Probability = (Model A + Model B) / 2

---

Features Used:

- Age (yrs)
- Cycle (R/I)
- Cycle length (days)
- Weight gain (Y/N)
- Hair growth (Y/N)
- Skin darkening (Y/N)
- Hair loss (Y/N)
- Pimples (Y/N)
- Fast food (Y/N)
- Regular exercise (Y/N)
- BMI

Additional:

- Sleep Rating (1–10) (used for advisory only)

---

Project Structure:

pcos_project/
│
├── app/
│   └── main.py              # FastAPI app
│
├── ml/
│   ├── models/
│   │   ├── pcos_model.pkl
│   │   ├── pcos_model_no_cycle.pkl
│   │
│   ├── predict.py
│   ├── train_model.py
│   │
│   ├── data/
│   │   └── pcos_combined_dataset.csv
│
├── requirements.txt
├── README.md
├── .gitignore

---

Installation & Setup: 

1. Clone Repository

git clone https://github.com/aniteshhalderofficial-commits/pcos-ml-model.git
cd pcos-ml-model

---

2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate   # Windows

---

3. Install Dependencies

pip install -r requirements.txt

---

Run the API:

uvicorn app.main:app --reload

Open in browser:

http://127.0.0.1:8000/docs

---

API Endpoint:

🔹 POST "/predict"

---

Sample Request

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

---

Sample Response

{
  "risk_probability": 0.81,
  "risk_stage": "High Risk",
  "prediction_confidence": "High confidence prediction",
  "model_with_cycle": 0.92,
  "model_without_cycle": 0.70,
  "lifestyle_recommendations": [...],
  "sleep_advisory": "Average sleep quality..."
}

---

Important Notes:

- This system is designed for early risk screening, not diagnosis
- Clinical validation is required before real-world use
- Cycle irregularity alone does not confirm PCOS

---

Model Performance (Approximately):

- Accuracy: ~0.88 – 0.91
- ROC-AUC: ~0.93 – 0.97
- High Recall (important for medical screening)

---

Tech Stack:

- Python
- Scikit-learn
- FastAPI
- Pandas / NumPy
- Uvicorn

---

Future Improvements:

- Add hormonal & ultrasound data
- Improve dataset diversity
- Deploy on cloud (Render / AWS)
- Add SHAP explainability

---

Contributors:

- ML Model & Backend: Anitesh Halder
- App Building and Integration: Anand Singh

---

License:

For academic and educational use.
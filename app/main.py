from fastapi import FastAPI
from pydantic import BaseModel
from ml.predict import predict_pcos

app = FastAPI(title="PCOS Early Risk Detection API")


# -----------------------------
# Input Schema
# -----------------------------

class PCOSInput(BaseModel):
    Age_yrs: int
    Cycle_R_I: int
    Cycle_length_days: int
    Weight_gain_Y_N: int
    hair_growth_Y_N: int
    Skin_darkening_Y_N: int
    Hair_loss_Y_N: int
    Pimples_Y_N: int
    Fast_food_Y_N: int
    Reg_Exercise_Y_N: int
    Weight_kg: float
    Height_cm: float
    Sleep_Rating_1_10: int


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(data: PCOSInput):

    if data.Height_cm <= 0:
        return {"error": "Height must be greater than zero"}

    # Calculate BMI
    bmi = data.Weight_kg / ((data.Height_cm / 100) ** 2)

    input_dict = {
        "Age (yrs)": data.Age_yrs,
        "Cycle(R/I)": data.Cycle_R_I,
        "Cycle length(days)": data.Cycle_length_days,
        "Weight gain(Y/N)": data.Weight_gain_Y_N,
        "hair growth(Y/N)": data.hair_growth_Y_N,
        "Skin darkening (Y/N)": data.Skin_darkening_Y_N,
        "Hair loss(Y/N)": data.Hair_loss_Y_N,
        "Pimples(Y/N)": data.Pimples_Y_N,
        "Fast food (Y/N)": data.Fast_food_Y_N,
        "Reg.Exercise(Y/N)": data.Reg_Exercise_Y_N,
        "BMI": round(bmi, 2),
        "Sleep Rating (1-10)": data.Sleep_Rating_1_10
    }

    result = predict_pcos(input_dict)

    return result
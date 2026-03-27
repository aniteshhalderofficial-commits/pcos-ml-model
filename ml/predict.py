import joblib
import pandas as pd
import numpy as np
import json

# -----------------------------
# Load BOTH models
# -----------------------------

model_with_cycle = joblib.load("ml/models/pcos_model.pkl")
model_no_cycle = joblib.load("ml/models/pcos_model_no_cycle.pkl")

THRESHOLD = 0.33


# -----------------------------
# Risk Categorization
# -----------------------------

def categorize_risk(prob):
    if prob < 0.2:
        return "Very Low"
    elif prob < 0.4:
        return "Early Risk"
    elif prob < 0.6:
        return "Moderate Risk"
    elif prob < 0.8:
        return "High Risk"
    else:
        return "Very High Risk"


def prediction_confidence(prob):
    if prob < 0.55:
        return "Low confidence prediction"
    elif prob < 0.75:
        return "Moderate confidence prediction"
    elif prob < 0.90:
        return "High confidence prediction"
    else:
        return "Very high confidence prediction"


# -----------------------------
# Lifestyle Suggestions
# -----------------------------

def lifestyle_suggestions(stage):
    if stage == "Very Low":
        return [
            "Maintain balanced diet",
            "Continue regular physical activity",
            "Track menstrual cycle regularly"
        ]
    elif stage in ["Early Risk", "Moderate Risk"]:
        return [
            "Reduce processed and fast food intake",
            "Increase regular exercise (at least 30 min/day)",
            "Monitor cycle irregularity",
            "Manage stress levels"
        ]
    else:
        return [
            "Consult a gynecologist",
            "Track symptoms closely",
            "Lifestyle correction strongly recommended"
        ]


# -----------------------------
# Sleep Advisory
# -----------------------------

def sleep_modifier_message(sleep_rating):
    if sleep_rating is None:
        return None

    try:
        sleep_rating = int(sleep_rating)
    except:
        return "Invalid sleep rating provided."

    if sleep_rating < 1 or sleep_rating > 10:
        return "Sleep rating must be between 1 and 10."

    if sleep_rating <= 4:
        return "Poor sleep quality may worsen insulin resistance and hormonal imbalance."
    elif 5 <= sleep_rating <= 6:
        return "Average sleep quality. Improving sleep consistency may help."
    elif 7 <= sleep_rating <= 8:
        return "Good sleep quality supports hormonal balance."
    else:
        return "Excellent sleep quality is protective."


# -----------------------------
# Probability Range
# -----------------------------

def probability_range(prob):
    lower = max(0, prob - 0.03)
    upper = min(1, prob + 0.03)
    return f"{round(lower,3)} - {round(upper,3)}"


# -----------------------------
# Prediction Function
# -----------------------------

def predict_pcos(input_data: dict):

    sleep_rating = input_data.get("Sleep Rating (1-10)")

    # Remove sleep from model input
    model_input = input_data.copy()
    model_input.pop("Sleep Rating (1-10)", None)

    # Create DataFrame
    input_df = pd.DataFrame([model_input])

    # -----------------------------
    # ADD ENGINEERED FEATURES
    # -----------------------------
    input_df["Cycle_HairGrowth"] = input_df["Cycle(R/I)"] * input_df["hair growth(Y/N)"]
    input_df["Cycle_SkinDark"] = input_df["Cycle(R/I)"] * input_df["Skin darkening (Y/N)"]
    input_df["Cycle_WeightGain"] = input_df["Cycle(R/I)"] * input_df["Weight gain(Y/N)"]

    # -----------------------------
    # MATCH MODEL FEATURES EXACTLY (IMPORTANT FIX)
    # -----------------------------
    model_features = model_with_cycle.feature_names_in_

    # Add missing columns
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure exact order
    input_df = input_df[model_features]

    # -----------------------------
    # MODEL 1 (WITH CYCLE)
    # -----------------------------
    prob_with_cycle = model_with_cycle.predict_proba(input_df)[0][1]

    # -----------------------------
    # MODEL 2 (WITHOUT CYCLE)
    # -----------------------------
    input_no_cycle = input_df.drop(columns=["Cycle(R/I)"])
    prob_no_cycle = model_no_cycle.predict_proba(input_no_cycle)[0][1]

    # -----------------------------
    # FINAL PROBABILITY
    # -----------------------------
    final_prob = (prob_with_cycle + prob_no_cycle) / 2

    prediction = 1 if final_prob >= THRESHOLD else 0
    stage = categorize_risk(final_prob)
    confidence = prediction_confidence(final_prob)

    sleep_message = sleep_modifier_message(sleep_rating)

    return {
        "risk_probability": round(float(final_prob), 3),
        "probability_range": probability_range(final_prob),
        "risk_stage": stage,
        "prediction_label": prediction,
        "prediction_confidence": confidence,
        "model_with_cycle": round(float(prob_with_cycle), 3),
        "model_without_cycle": round(float(prob_no_cycle), 3),
        "lifestyle_recommendations": lifestyle_suggestions(stage),
        "sleep_advisory": sleep_message
    }


# -----------------------------
# TEST BLOCK
# -----------------------------

if __name__ == "__main__":

    sample_input = {
        "Age (yrs)": 24,
        "Cycle(R/I)": 4,
        "Cycle length(days)": 40,
        "Weight gain(Y/N)": 1,
        "hair growth(Y/N)": 1,
        "Skin darkening (Y/N)": 1,
        "Hair loss(Y/N)": 0,
        "Pimples(Y/N)": 1,
        "Fast food (Y/N)": 1,
        "Reg.Exercise(Y/N)": 0,
        "BMI": 28.3,
        "Sleep Rating (1-10)": 5
    }

    result = predict_pcos(sample_input)
    print(json.dumps(result, indent=4))

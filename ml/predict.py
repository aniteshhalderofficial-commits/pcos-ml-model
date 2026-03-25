import joblib
import pandas as pd
import numpy as np
import json



# Load Trained Model


MODEL_PATH = "ml/models/pcos_model.pkl"
THRESHOLD = 0.33

model = joblib.load(MODEL_PATH)


# Risk Categorization
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
    

# Lifestyle Suggestions


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



# Sleep Modifier (Advisory Only)


def sleep_modifier_message(sleep_rating):
    if sleep_rating is None:
        return None

    try:
        sleep_rating = int(sleep_rating)
    except:
        return "Invalid sleep rating provided."

    # ADD THIS VALIDATION
    if sleep_rating < 1 or sleep_rating > 10:
        return "Sleep rating must be between 1 and 10."

    if sleep_rating <= 4:
        return "Poor sleep quality may worsen insulin resistance and hormonal imbalance. Improving sleep hygiene is strongly recommended."

    elif 5 <= sleep_rating <= 6:
        return "Average sleep quality. Improving sleep consistency may support hormonal balance."

    elif 7 <= sleep_rating <= 8:
        return "Good sleep quality supports metabolic and hormonal health."

    elif sleep_rating >= 9:
        return "Excellent sleep quality is protective for endocrine and metabolic stability."


# Top Contributing Factors


def get_top_factors(input_df):
    importances = model.feature_importances_

    contribution_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Contribution": input_df.iloc[0].values * importances
    })

    contribution_df["Abs"] = np.abs(contribution_df["Contribution"])
    contribution_df = contribution_df.sort_values(by="Abs", ascending=False)

    core_features = [
        "Cycle(R/I)",
        "Weight gain(Y/N)",
        "hair growth(Y/N)",
        "Skin darkening (Y/N)"
    ]

    return contribution_df[
        contribution_df["Feature"].isin(core_features)
    ].sort_values(by="Abs", ascending=False)["Feature"].tolist()

def probability_range(prob):

    lower = max(0, prob - 0.03)
    upper = min(1, prob + 0.03)

    return f"{round(lower,3)} - {round(upper,3)}"

# Main Prediction Function


def predict_pcos(input_data: dict):
    """
    input_data must contain:

    Age (yrs)
    Cycle(R/I)
    Cycle length(days)
    Weight gain(Y/N)
    hair growth(Y/N)
    Skin darkening (Y/N)
    Hair loss(Y/N)
    Pimples(Y/N)
    Fast food (Y/N)
    Reg.Exercise(Y/N)
    BMI
    Sleep Rating (1-10)  -> advisory only (not used in ML)
    """

    # Extract sleep separately
    sleep_rating = input_data.get("Sleep Rating (1-10)")

    # Remove sleep before ML prediction
    model_input = input_data.copy()
    model_input.pop("Sleep Rating (1-10)", None)

    input_df = pd.DataFrame([model_input])

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    confidence = prediction_confidence(probability)

    # Apply threshold
    prediction = 1 if probability >= THRESHOLD else 0

    # Determine stage
    stage = categorize_risk(probability)

    # Core contributing factors
    top_factors = get_top_factors(input_df)


    # Sleep advisory
    sleep_message = sleep_modifier_message(sleep_rating)

    
    return {
    "risk_probability": round(float(probability), 3),
    "probability_range": probability_range(probability),
    "risk_stage": stage,
    "prediction_label": prediction,
    "prediction_confidence": confidence,
    "top_contributing_factors": top_factors,
    "lifestyle_recommendations": lifestyle_suggestions(stage),
    "sleep_advisory": sleep_message
    }



# Test Block


if __name__ == "__main__":

    sample_input = {
        "Age (yrs)": 24,
        # Cycle(R/I) meaning
        # 2 → Mostly Non-PCOS cycles
        # 4 → Many PCOS cases observed
        # 5 → Rare extreme irregular case
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
        "Sleep Rating (1-10)": 3
    }

    result = predict_pcos(sample_input)

    print("\nPrediction Result:\n")
    print(json.dumps(result, indent=4))
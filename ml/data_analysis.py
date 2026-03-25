import pandas as pd

file_path = "ml/data/PCOS_data_without_infertility.xlsx"

df = pd.read_excel(file_path, sheet_name="Full_new")

# Clean column names
df.columns = df.columns.str.strip()

# Select required columns
selected_columns = [
    "PCOS (Y/N)",
    "Age (yrs)",
    "Weight (Kg)",
    "Height(Cm)",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]

df = df[selected_columns]

# Recalculate BMI
df["BMI"] = df["Weight (Kg)"] / ((df["Height(Cm)"] / 100) ** 2)

# Drop weight and height if you want simplicity
df.drop(columns=["Weight (Kg)", "Height(Cm)"], inplace=True)

print("Final dataset shape:", df.shape)
print("\nColumns now used:")
print(df.columns)


print("\nMissing values per column:")
print(df.isnull().sum())

# Convert Yes/No columns
print("\nUnique values in Cycle(R/I):")
print(df["Cycle(R/I)"].unique())

print("\nUnique values in Fast food (Y/N):")
print(df["Fast food (Y/N)"].unique())

df["Fast food (Y/N)"] = pd.to_numeric(df["Fast food (Y/N)"], errors="coerce")
df["Fast food (Y/N)"] = df["Fast food (Y/N)"].fillna(0)


yn_columns = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]

print("\nData Types:")
print(df.dtypes)


print("\nMissing values after encoding:")
print(df.isnull().sum())

print("\nCycle(R/I) vs Target Crosstab:")
print(pd.crosstab(df["Cycle(R/I)"], df["PCOS (Y/N)"]))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Separate features and target
X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with scaling
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------
# Random Forest Model Comparison
# -------------------------------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    class_weight="balanced",
    random_state=42
)


rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n===== Random Forest Results =====")

print("\nAccuracy:", accuracy_score(y_test, rf_pred))
print("\nROC-AUC:", roc_auc_score(y_test, rf_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))

# Gradient Boosting Model Comparison

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
gb_prob = gb.predict_proba(X_test)[:, 1]

print("\n===== Gradient Boosting Results =====")
print("\nAccuracy:", accuracy_score(y_test, gb_pred))
print("\nROC-AUC:", roc_auc_score(y_test, gb_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, gb_pred))

import numpy as np

# Use Gradient Boosting probabilities
threshold = 0.33  # lower than default 0.5

gb_pred_custom = (gb_prob >= threshold).astype(int)

print("\n===== Gradient Boosting (Threshold = 0.4) =====")
print("\nAccuracy:", accuracy_score(y_test, gb_pred_custom))
print("\nROC-AUC:", roc_auc_score(y_test, gb_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, gb_pred_custom))



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

# Example on first 5 predictions
print("\nSample Risk Categories:")
for i in range(5):
    print("Probability:", round(y_prob[i], 3),
          "→", categorize_risk(y_prob[i]))
    

import pandas as pd
import numpy as np

# Extract feature importance
feature_names = X.columns
coefficients = pipeline.named_steps["model"].coef_[0]


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute Importance": np.abs(coefficients)
})

importance_df = importance_df.sort_values(by="Absolute Importance", ascending=False)

print("\nFeature Importance (Global):\n")
print(importance_df)


def get_top_factors(input_row):
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    scaled_input = scaler.transform(
        pd.DataFrame([input_row], columns=X.columns)
    )

    contributions = scaled_input[0] * model.coef_[0]

    factor_df = pd.DataFrame({
        "Feature": X.columns,
        "Contribution": contributions
    })

    factor_df["Abs"] = np.abs(factor_df["Contribution"])
    factor_df = factor_df.sort_values(by="Abs", ascending=False)

    # Core PCOS clinical factors
    core_features = [
        "Cycle(R/I)",
        "Weight gain(Y/N)",
        "hair growth(Y/N)",
        "Skin darkening (Y/N)"
    ]

    core_df = factor_df[factor_df["Feature"].isin(core_features)]

    return core_df


# Example using first test sample
sample_input = X_test.iloc[0]
top_factors = get_top_factors(sample_input)

print("\nTop Contributing Factors For Sample:\n")
print(top_factors)


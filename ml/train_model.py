import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# Load Dataset

file_path = "ml/data/pcos_combined_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()


# Feature Selection

selected_columns = [
    "PCOS (Y/N)",
    "Age (yrs)",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
    "BMI"
]

df = df[selected_columns]


# -----------------------------
# FIX 1: Convert Cycle (reduce dominance)
# -----------------------------

df["Cycle(R/I)"] = df["Cycle(R/I)"].map({
    2: 0,
    4: 1,
    5: 1
})


# -----------------------------
# FIX 2: Interaction Features (MOST IMPORTANT)
# -----------------------------

df["Cycle_HairGrowth"] = df["Cycle(R/I)"] * df["hair growth(Y/N)"]
df["Cycle_WeightGain"] = df["Cycle(R/I)"] * df["Weight gain(Y/N)"]
df["Cycle_SkinDark"] = df["Cycle(R/I)"] * df["Skin darkening (Y/N)"]


# -----------------------------
# Target Fix
# -----------------------------

df["PCOS (Y/N)"] = df["PCOS (Y/N)"].astype(int)


# -----------------------------
# Correlation Check
# -----------------------------

print("\nFeature Correlation with Target:")
print(df.corr()["PCOS (Y/N)"].sort_values(ascending=False))


# -----------------------------
# Handle Missing Values
# -----------------------------

df["Fast food (Y/N)"] = pd.to_numeric(df["Fast food (Y/N)"], errors="coerce")
df = df.fillna(0)


# -----------------------------
# Train-Test Split
# -----------------------------

X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# SMOTE Balancing
# -----------------------------

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# -----------------------------
# Model Training (Controlled Depth)
# -----------------------------

gb_model = GradientBoostingClassifier(
    max_depth=2,
    n_estimators=150,
    learning_rate=0.05,
    random_state=42
)

gb_model.fit(X_train_res, y_train_res)


# -----------------------------
# Evaluate Model
# -----------------------------

gb_prob = gb_model.predict_proba(X_test)[:, 1]

threshold = 0.33
gb_pred = (gb_prob >= threshold).astype(int)

print("\n===== Final Model Evaluation =====")
print("\nAccuracy:", accuracy_score(y_test, gb_pred))
print("\nROC-AUC:", roc_auc_score(y_test, gb_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, gb_pred))


# -----------------------------
# Feature Importance
# -----------------------------

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": gb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)


# Plot Feature Importance
plt.figure(figsize=(8,5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Feature Importance – PCOS Model")
plt.tight_layout()
plt.savefig("ml/models/feature_importance.png", dpi=300)
plt.show()


# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, _ = roc_curve(y_test, gb_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, gb_prob):.3f}")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("ml/models/roc_curve.png", dpi=300)
plt.show()


# -----------------------------
# Save Model
# -----------------------------

os.makedirs("ml/models", exist_ok=True)
joblib.dump(gb_model, "ml/models/pcos_model.pkl")

print("\nModel saved successfully!")
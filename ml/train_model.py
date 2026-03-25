import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# -----------------------------
# Load Dataset
# -----------------------------

file_path = "ml/data/pcos_combined_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()


# -----------------------------
# Feature Selection
# -----------------------------

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
# Reduce Cycle Dominance
# -----------------------------

df["Cycle(R/I)"] = df["Cycle(R/I)"].map({
    2: 0,
    4: 1,
    5: 1
})


# -----------------------------
# Interaction Features
# -----------------------------

df["Cycle_HairGrowth"] = df["Cycle(R/I)"] * df["hair growth(Y/N)"]
df["Cycle_WeightGain"] = df["Cycle(R/I)"] * df["Weight gain(Y/N)"]
df["Cycle_SkinDark"] = df["Cycle(R/I)"] * df["Skin darkening (Y/N)"]


# -----------------------------
# Target Fix
# -----------------------------

df["PCOS (Y/N)"] = df["PCOS (Y/N)"].astype(int)


# -----------------------------
# Missing Values
# -----------------------------

df["Fast food (Y/N)"] = pd.to_numeric(df["Fast food (Y/N)"], errors="coerce")
df = df.fillna(0)


# =============================
# MODEL 1 — WITH CYCLE
# =============================

print("\n===== TRAINING MODEL WITH CYCLE =====")

X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model_with_cycle = GradientBoostingClassifier(
    max_depth=2,
    n_estimators=150,
    learning_rate=0.05,
    random_state=42
)

model_with_cycle.fit(X_train_res, y_train_res)

prob_with_cycle = model_with_cycle.predict_proba(X_test)[:, 1]
pred_with_cycle = (prob_with_cycle >= 0.33).astype(int)

print("\n--- WITH CYCLE ---")
print("Accuracy:", accuracy_score(y_test, pred_with_cycle))
print("ROC-AUC:", roc_auc_score(y_test, prob_with_cycle))
print(classification_report(y_test, pred_with_cycle))


# =============================
# MODEL 2 — WITHOUT CYCLE
# =============================

print("\n===== TRAINING MODEL WITHOUT CYCLE =====")

X_no_cycle = df.drop(columns=["PCOS (Y/N)", "Cycle(R/I)"])
y = df["PCOS (Y/N)"]

X_train_nc, X_test_nc, y_train_nc, y_test_nc = train_test_split(
    X_no_cycle, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_nc_res, y_train_nc_res = smote.fit_resample(X_train_nc, y_train_nc)

model_no_cycle = GradientBoostingClassifier(
    max_depth=2,
    n_estimators=150,
    learning_rate=0.05,
    random_state=42
)

model_no_cycle.fit(X_train_nc_res, y_train_nc_res)

prob_no_cycle = model_no_cycle.predict_proba(X_test_nc)[:, 1]
pred_no_cycle = (prob_no_cycle >= 0.33).astype(int)

print("\n--- WITHOUT CYCLE ---")
print("Accuracy:", accuracy_score(y_test_nc, pred_no_cycle))
print("ROC-AUC:", roc_auc_score(y_test_nc, prob_no_cycle))
print(classification_report(y_test_nc, pred_no_cycle))


# =============================
# ROC CURVE COMPARISON
# =============================

fpr_wc, tpr_wc, _ = roc_curve(y_test, prob_with_cycle)
fpr_nc, tpr_nc, _ = roc_curve(y_test_nc, prob_no_cycle)

plt.figure(figsize=(6,6))

plt.plot(fpr_wc, tpr_wc, label=f"With Cycle (AUC={roc_auc_score(y_test, prob_with_cycle):.3f})")
plt.plot(fpr_nc, tpr_nc, label=f"Without Cycle (AUC={roc_auc_score(y_test_nc, prob_no_cycle):.3f})")

plt.plot([0,1], [0,1], 'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("ml/models/roc_comparison.png", dpi=300)
plt.show()


# =============================
# FEATURE IMPORTANCE (WITH CYCLE MODEL)
# =============================

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model_with_cycle.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (With Cycle):\n")
print(importance_df)

plt.figure(figsize=(8,5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.title("Feature Importance – With Cycle Model")

plt.tight_layout()
plt.savefig("ml/models/feature_importance.png", dpi=300)
plt.show()

# =============================
# FEATURE IMPORTANCE (WITHOUT CYCLE MODEL)
# =============================

importance_nc_df = pd.DataFrame({
    "Feature": X_train_nc.columns,
    "Importance": model_no_cycle.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance (Without Cycle):\n")
print(importance_nc_df)


# Plot Feature Importance (WITHOUT CYCLE)
plt.figure(figsize=(8,5))

plt.barh(importance_nc_df["Feature"], importance_nc_df["Importance"])
plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.title("Feature Importance – Without Cycle Model")

plt.tight_layout()
plt.savefig("ml/models/feature_importance_no_cycle.png", dpi=300)
plt.show()

# =============================
# SAVE MODELS
# =============================

os.makedirs("ml/models", exist_ok=True)

joblib.dump(model_with_cycle, "ml/models/pcos_model.pkl")
joblib.dump(model_no_cycle, "ml/models/pcos_model_no_cycle.pkl")

print("\nBoth models saved successfully!")
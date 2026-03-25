import pandas as pd
import numpy as np

np.random.seed(42)

# Load Original PCOS Dataset

original = pd.read_excel(
    "ml/data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

original.columns = original.columns.str.strip()

# Calculate BMI
original["BMI"] = original["Weight (Kg)"] / ((original["Height(Cm)"] / 100) ** 2)

# Keep only required columns
original = original[
[
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
]


# Load Rotterdam Dataset

rotterdam = pd.read_csv("ml/data/pcos_rotterdam_balanceado.csv")

rotterdam = rotterdam.rename(columns={
    "Age": "Age (yrs)",
    "BMI": "BMI",
    "PCOS_Diagnosis": "PCOS (Y/N)"
})

# Convert menstrual irregularity to your cycle encoding
rotterdam["Cycle(R/I)"] = rotterdam["Menstrual_Irregularity"].map({
    0: 2,   # regular
    1: 4    # irregular
})


# Generate Realistic Cycle Length

rotterdam["Cycle length(days)"] = np.random.normal(30, 5, len(rotterdam))
rotterdam["Cycle length(days)"] = rotterdam["Cycle length(days)"].clip(21, 60)
rotterdam["Cycle length(days)"] = rotterdam["Cycle length(days)"].round()


# Generate Symptom Columns Using Real Distribution


symptom_columns = [
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]

for col in symptom_columns:

    distribution = original[col].value_counts(normalize=True)

    rotterdam[col] = np.random.choice(
        distribution.index,
        size=len(rotterdam),
        p=distribution.values
    )

# Keep Only Model Columns

rotterdam = rotterdam[
[
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
]

# Merge Datasets

combined = pd.concat([original, rotterdam], ignore_index=True)

print("Original dataset size:", len(original))
print("Rotterdam dataset size:", len(rotterdam))
print("Final dataset size:", len(combined))

# Save Combined Dataset


combined.to_csv("ml/data/pcos_combined_dataset.csv", index=False)

print("Dataset merged successfully.")
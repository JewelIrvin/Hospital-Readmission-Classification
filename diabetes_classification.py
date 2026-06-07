# Project# 2: Diabetes Readmission Classification
# Author: Jewel Irvin
# Goal: Predict 30-day hospital readmissions for diabetes patients

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Diabetes_Data/diabetic_data.csv')

# Get a first look at the data
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

print("=" * 60)
print("STEP 1: DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n" + "=" * 60)
print("STEP 2: TARGET VARIABLE DISTRIBUTION")
print("=" * 60)
print(df['readmitted'].value_counts())
print("\nPercentages:")
print(df['readmitted'].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("STEP 3: MISSING VALUES (marked as '?')")
print("=" * 60)
# In this dataset, missing values are coded as '?'
missing = (df == '?').sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

print("\n" + "=" * 60)
print("STEP 4: BASIC STATS FOR NUMERIC COLUMNS")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("STEP 1: CREATE BINARY TARGET")
print("=" * 60)

# We're predicting: was the patient readmitted within 30 days?
# 1 = Yes (readmitted <30 days), 0 = No (>30 or NO)
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
print(f"\nTarget distribution:")
print(df['readmitted_binary'].value_counts())


print("\n" + "=" * 60)
print("STEP 2: DROP COLUMNS WE DON'T NEED")
print("=" * 60)


columns_to_drop = [
    'encounter_id',       # just an ID
    'patient_nbr',        # just an ID
    'weight',             # 97% missing
    'payer_code',         # 40% missing, not clinical
    'medical_specialty',  # 49% missing
    'readmitted'          # replaced by readmitted_binary
]
df = df.drop(columns=columns_to_drop)
print(f"\nShape after dropping columns: {df.shape}")

print("\n" + "=" * 60)
print("STEP 3: REMOVE ROWS WITH '?' IN KEY COLUMNS")
print("=" * 60)

df = df[df['race'] != '?']
df = df[df['diag_1'] != '?']
df = df[df['diag_2'] != '?']
df = df[df['diag_3'] != '?']
print(f"Shape after removing '?' rows: {df.shape}")

print("\n" + "=" * 60)
print("STEP 4: HANDLE DUPLICATE PATIENTS")
print("=" * 60)

# Some patients have multiple visits - we'll note this but keep them for now
print(f"\nUnique records: {df.shape[0]}")

print("\n" + "=" * 60)
print("STEP 5: CHECK DATA IS CLEAN")
print("=" * 60)

print("\nAny remaining '?' values?")
remaining = (df == '?').sum()
print(remaining[remaining > 0] if remaining.sum() > 0 else "None! ✅")

print("\nFinal target distribution:")
print(df['readmitted_binary'].value_counts(normalize=True) * 100)

# Save cleaned data for next step
df.to_csv('Diabetes_Data/diabetic_data_cleaned.csv', index=False)
print("\n✅ Cleaned data saved to data/diabetic_data_cleaned.csv")
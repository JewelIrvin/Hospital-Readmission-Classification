# Project 2: Random Forest Model
# Author: Jewel Irvin
# Goal: Predict 30-day hospital readmissions for diabetes patients using Random Forest

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves plots without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ============================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================
df = pd.read_csv('Diabetes_Data/diabetic_data_cleaned.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2: GROUP DIAGNOSIS CODES INTO CATEGORIES
# ============================================================
def categorize_diagnosis(code):
    """
    Group ICD-9 diagnosis codes into broad clinical categories.
    Same grouping logic as the Logistic Regression script, so the
    two models can be compared on identical feature definitions.
    """
    try:
        code_num = float(code)
        if 390 <= code_num <= 459 or code_num == 785:
            return 'Circulatory'
        elif 460 <= code_num <= 519 or code_num == 786:
            return 'Respiratory'
        elif 520 <= code_num <= 579 or code_num == 787:
            return 'Digestive'
        elif 250 <= code_num < 251:
            return 'Diabetes'
        elif 800 <= code_num <= 999:
            return 'Injury'
        elif 710 <= code_num <= 739:
            return 'Musculoskeletal'
        elif 580 <= code_num <= 629 or code_num == 788:
            return 'Genitourinary'
        elif 140 <= code_num <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except:
        return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].apply(categorize_diagnosis)

print("Diagnosis codes grouped into 9 clinical categories")

# ============================================================
# STEP 3: PREPARE FEATURES AND SPLIT
# ============================================================
y = df['readmitted_binary']
X = df.drop(columns=['readmitted_binary'])
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

# ============================================================
# STEP 4: TRAIN RANDOM FOREST MODEL
# ============================================================
print("\nTraining Random Forest model (this may take 1-2 minutes)...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Model trained")

# ============================================================
# STEP 5: EVALUATE THE MODEL
# ============================================================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("RANDOM FOREST - MODEL PERFORMANCE")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Readmitted', 'Readmitted <30 days']))

# ============================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)
print(feature_importance.to_string(index=False))

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================

# --- Visualization 1: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Not Readmitted', 'Readmitted <30'],
    yticklabels=['Not Readmitted', 'Readmitted <30']
)
plt.title('Random Forest: Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('Diabetes_Data/rf_confusion_matrix.png', dpi=150)
plt.close()
print("Saved: rf_confusion_matrix.png")

# --- Visualization 2: ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})', linewidth=2, color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest: ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Diabetes_Data/rf_roc_curve.png', dpi=150)
plt.close()
print("Saved: rf_roc_curve.png")

# --- Visualization 3: Feature Importance Bar Chart ---
plt.figure(figsize=(9, 7))
sns.barplot(
    data=feature_importance,
    x='importance',
    y='feature',
    palette='viridis'
)
plt.title('Random Forest: Top 15 Most Important Features')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('Diabetes_Data/rf_feature_importance.png', dpi=150)
plt.close()
print("Saved: rf_feature_importance.png")

print("\nDone! All plots saved to Diabetes_Data/ folder.")
# Project 2: Logistic Regression Model
# Author: Jewel Irvin
# Goal: Predict 30-day hospital readmissions for diabetes patients using Logistic Regression
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    This reduces hundreds of unique codes into a manageable number of
    interpretable groups for modeling.
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
# STEP 3: PREPARE FEATURES (X) AND TARGET (y)
# ============================================================
y = df['readmitted_binary']
X = df.drop(columns=['readmitted_binary'])
X = pd.get_dummies(X, drop_first=True)
print(f"After one-hot encoding: {X.shape[1]} features")
 
# ============================================================
# STEP 4: SPLIT INTO TRAIN AND TEST SETS
# ============================================================
# Stratified split keeps class proportions consistent across train/test,
# which matters here because readmissions are the minority class.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")
 
# ============================================================
# STEP 5: SCALE THE FEATURES
# ============================================================
# Logistic Regression is sensitive to feature scale, so we standardize.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled with StandardScaler")
 
# ============================================================
# STEP 6: TRAIN LOGISTIC REGRESSION MODEL
# ============================================================
# class_weight='balanced' tells the model to weight the minority class
# (30-day readmissions) more heavily, since our data is imbalanced.
print("\nTraining Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)
print("Model trained")
 
# ============================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
 
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION - MODEL PERFORMANCE")
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
# STEP 8: VISUALIZATIONS
# ============================================================
 
# --- Visualization 1: Confusion Matrix Heatmap ---
# A heatmap makes it easier to see at a glance how often the model
# correctly vs. incorrectly classified each group.
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Readmitted', 'Readmitted <30'],
    yticklabels=['Not Readmitted', 'Readmitted <30']
)
plt.title('Logistic Regression: Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('Diabetes_Data/logreg_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: logreg_confusion_matrix.png")
 
# --- Visualization 2: ROC Curve ---
# The ROC curve shows the tradeoff between true positive rate
# and false positive rate at every possible classification threshold.
# The closer the curve hugs the top-left corner, the better the model.
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
 
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression: ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Diabetes_Data/logreg_roc_curve.png', dpi=150)
plt.show()
print("Saved: logreg_roc_curve.png")
 
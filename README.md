# Hospital-Readmission-Classification
# Predicting 30-Day Hospital Readmissions for Diabetic Patients

### Healthcare Machine Learning — Binary Classification

**Author:** Jewel Irvin  
**Project Type:** Healthcare Machine Learning / Clinical Risk Prediction  

---

## Overview

Hospital readmissions remain one of the most important quality and cost-related challenges in healthcare systems. Early readmissions may reflect gaps in discharge planning, medication management, chronic disease support, or access to follow-up care. Identifying patients at elevated risk before discharge can help clinical teams prioritize interventions and improve continuity of care.

This project develops and compares two machine learning models — **Logistic Regression** and **Random Forest** — to predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge. Using over 100,000 real-world hospital encounters from the UCI Diabetes 130-US Hospitals dataset, this project evaluates not only predictive performance, but also the practical and ethical implications of deploying predictive models in healthcare settings.

---

## Motivation

I chose hospital readmission because it connects directly to the type of healthcare data work I am already around. In my current role, I work with EHR systems and clinical workflows, so I see how much information exists in healthcare systems but also how hard it can be to turn that information into something useful for care teams.

What made readmission interesting to me is that it is not just a prediction problem. It represents a real patient outcome. If a patient is readmitted within 30 days, that could reflect gaps in follow-up care, discharge planning, medication management, access to resources, or chronic disease support.

For me, this project felt like a way to connect machine learning to something meaningful: helping identify patients who may need more support before they end up back in the hospital.

I also liked that it forced me to think beyond accuracy. In healthcare, being technically correct is not enough. You have to think about what the model's mistakes mean for real people.

---

## Dataset

### Source

**UCI Machine Learning Repository — Diabetes 130-US Hospitals Dataset (1999–2008)**

### Dataset Summary

| Category | Description |
|---|---|
| Original records | 101,766 hospital encounters |
| Records after cleaning | 98,053 |
| Features | 50 demographic, administrative, and clinical variables |
| Target variable | 30-day readmission status |
| Positive class prevalence | Approximately 11% |

---

## Prediction Target

The original dataset categorized readmission into three groups:

- `<30`
- `>30`
- `NO`

For this project, the problem was reframed into a binary classification task:

```text
1 = Readmitted within 30 days
0 = Not readmitted within 30 days

````

---
## Technical Approach

### What Was Hardest About Building This

The hardest part was realizing that cleaning and framing the data mattered just as much as building the model. At first, it is easy to think the main goal is just to train a model and get a good score, but the dataset made me slow down and think about missing values, confusing categories, and which columns should or should not be used.

One thing that surprised me was how messy healthcare data can be, even in a structured dataset. Some missing values were represented with question marks instead of normal null values, and a lot of the variables were categorical, which meant I had to think carefully about how to encode them.

Another challenge was defining the target. The original readmission variable had more than one category, but I reframed it as a binary problem focused on whether the patient came back within 30 days. That decision helped make the project more clinically meaningful.

---

## Data Preprocessing

Healthcare datasets often contain inconsistent formatting, missing values, and high-cardinality categorical variables. Significant preprocessing was required before modeling.

### Missing Data Handling

Several variables contained missing values represented as question marks (`?`) instead of standard null values.

Approach:

- Converted placeholder missing values into usable null representations
- Removed columns with excessive missingness
- Preserved clinically meaningful features whenever possible

Examples:

- `weight` removed because it was approximately 97% missing
- `payer_code` removed because it was approximately 40% missing

---

### Diagnosis Grouping

The dataset contained three diagnosis columns:

- `diag_1`
- `diag_2`
- `diag_3`

These columns contained hundreds of ICD-9 diagnosis codes.

To improve interpretability and reduce dimensionality, diagnosis codes were grouped into broader clinical categories such as:

- Circulatory
- Respiratory
- Diabetes
- Digestive
- Injury
- Musculoskeletal
- Neoplasms
- Genitourinary
- Other

This preprocessing step improved both model tractability and clinical interpretability.

---

### Feature Engineering

Feature engineering included:

- One-hot encoding categorical variables
- Expanding the final feature space to 103 engineered features
- Removing leakage-prone identifier columns

Identifier columns removed:

- `encounter_id`
- `patient_nbr`

These columns were removed because they identify encounters or patients but do not provide generalizable clinical signal for prediction.

---

## Modeling Strategy

Two classification models were trained and evaluated as separate experiments using a stratified 80/20 train-test split.

| Component | Logistic Regression | Random Forest |
|---|---|---|
| Scaling | StandardScaler | None |
| Class imbalance handling | `class_weight='balanced'` | `class_weight='balanced'` |
| Key hyperparameters | `max_iter=1000` | `n_estimators=100`, `max_depth=15` |

---

## Evaluation Metrics

Because the dataset is highly imbalanced, accuracy alone would not adequately measure clinical usefulness.

This project prioritized:

- **Recall** — ability to identify high-risk patients
- **F1-score** — balance between recall and precision
- **ROC-AUC** — overall discriminative performance

In healthcare settings, failing to identify a truly high-risk patient may have more serious consequences than generating additional false alerts.

---

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.6610 | 0.7871 |
| Precision | 0.1700 | 0.2007 |
| Recall | 0.5160 | 0.2973 |
| F1 Score | 0.2557 | 0.2396 |
| ROC-AUC | 0.6378 | 0.6420 |

---

## Confusion Matrices

### Logistic Regression Confusion Matrix
<img width="1050" height="750" alt="logreg_confusion_matrix" src="https://github.com/user-attachments/assets/dce0f99d-7c0a-43c4-96e3-6e1657a870ec" />

**Figure 1. Logistic Regression confusion matrix.**  
The Logistic Regression model correctly identified **1,142** patients who were readmitted within 30 days, while missing **1,071** patients who were actually readmitted. Although this model produced more false positives than Random Forest, it achieved higher recall, making it more useful in a clinical setting where the goal is to identify as many high-risk patients as possible.

---

### Random Forest Confusion Matrix
<img width="1050" height="750" alt="rf_confusion_matrix" src="https://github.com/user-attachments/assets/4728414c-6a8d-4562-b672-4c54f0dbd091" />

**Figure 2. Random Forest confusion matrix.**  
The Random Forest model correctly identified **658** patients who were readmitted within 30 days, while missing **1,555** patients who were actually readmitted. Although it achieved higher overall accuracy, its lower recall suggests that it may miss more patients who could benefit from additional follow-up or intervention.

---

## ROC Curves

### Logistic Regression ROC Curve
<img width="1050" height="750" alt="logreg_roc_curve" src="https://github.com/user-attachments/assets/764edbd0-d3d9-4be0-9e08-63d5dc72a641" />

**Figure 3. Logistic Regression ROC curve.**  
The Logistic Regression model achieved an ROC-AUC of **0.638**, indicating modest ability to distinguish between patients who were and were not readmitted within 30 days. The curve performs better than random guessing, but also shows that additional feature engineering or model refinement could improve predictive performance.

---

### Random Forest ROC Curve
<img width="1050" height="750" alt="rf_roc_curve" src="https://github.com/user-attachments/assets/514cfcc1-cb43-463b-8d37-9b7325cbbdf2" />

**Figure 4. Random Forest ROC curve.**  
The Random Forest model achieved an ROC-AUC of **0.642**, slightly higher than Logistic Regression. This suggests that Random Forest had marginally better overall discrimination, but this improvement did not translate into better recall for the readmitted class.

---

## Feature Importance

### Random Forest Feature Importance
<img width="1350" height="1050" alt="rf_feature_importance" src="https://github.com/user-attachments/assets/90d4d1ca-08a3-4b33-85f7-755413aae78c" />

**Figure 5. Random Forest top 15 feature importances.**  
The most important predictors included `number_inpatient`, `num_lab_procedures`, `discharge_disposition_id`, `num_medications`, and `time_in_hospital`. These features align with clinical intuition: patients with more prior inpatient visits, longer hospital stays, more procedures, and more complex medication regimens may face higher risk of early readmission.

---

## Operational and Ethical Tradeoffs

This project demonstrates that model selection in healthcare is not simply a technical decision.

A model with higher accuracy is not automatically more useful in clinical practice.

For example:

- A higher-recall model may identify more patients needing intervention but generate more false positives
- A higher-precision model may reduce unnecessary outreach but miss vulnerable patients

The appropriate balance depends on:

- Available clinical resources
- Care coordination capacity
- Tolerance for missed high-risk cases
- Operational workflow constraints

In this project, Random Forest achieved higher overall accuracy, but Logistic Regression identified more patients who were actually readmitted within 30 days. This makes Logistic Regression more aligned with a recall-focused healthcare use case, where missing a high-risk patient may be more harmful than generating additional false alerts.

---

## Key Findings

Several features identified by the Random Forest model aligned with existing clinical understanding of readmission risk.

Top predictive features included:

- `number_inpatient`
- `num_lab_procedures`
- `discharge_disposition_id`
- `num_medications`
- `time_in_hospital`

These variables suggest that patients with:

- More prior hospitalizations
- Longer inpatient stays
- More complex medication regimens

may face elevated risk of early readmission.

---

## Reflections

One thing I learned is that machine learning is not just about choosing a model. A lot of the important work happens before modeling, especially in how you define the problem and prepare the data.

I also learned why accuracy can be misleading, especially when one outcome happens much more often than the other. Before this project, I understood accuracy in a basic way, but this helped me see why recall and F1-score matter more in certain situations. In a healthcare setting, missing a high-risk patient can be more serious than accidentally flagging someone who may not be readmitted.

I also learned that data cleaning decisions are not just technical decisions. They affect how realistic and fair the model is. For example, handling missing values or dropping leakage-prone columns can change whether the model is actually useful or just performing well for the wrong reasons.

---

## Future Improvements

If I had additional time to improve this project, I would prioritize deeper model interpretation and subgroup evaluation.

### Subgroup Evaluation

I would evaluate whether model performance differs across groups such as:

- Age
- Race
- Gender
- Admission type

In healthcare, a model can look strong overall but still perform worse for certain patient populations. Understanding subgroup performance would help evaluate whether the model supports equitable care.

---

## Tools and Technologies

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

---

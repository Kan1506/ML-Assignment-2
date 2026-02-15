# ML Assignment 2 – Breast Cancer Classification

## Student Information
- **Name:** Sri Harsha Sattiraju  
- **Student ID:** 2024dc04136  
- **Course:** Machine Learning  
- **University:** BITS Pilani  

---

## Project Overview

This project implements and compares multiple Machine Learning models for **Breast Cancer Classification** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The objective is to classify tumors as:

- **0 → Benign**
- **1 → Malignant**

The project includes:
- Data preprocessing  
- Feature scaling  
- Model training  
- Model comparison  
- Performance evaluation  
- Confusion matrix visualization  
- Deployment using Streamlit Community Cloud  

---

## Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset contains:

- 569 samples  
- 30 numerical features  
- Features extracted from digitized images of breast mass cell nuclei  

Target Variable:
- 0 → Benign  
- 1 → Malignant  

---

## Models Implemented

The following 6 models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

All models were trained using the same train-test split for fair comparison.

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Matthews Correlation Coefficient (MCC)  
- Confusion Matrix  
- Classification Report  

---

## Model Comparison Summary

| Model               | Accuracy | Precision | Recall | F1 Score | AUC   | MCC   |
|---------------------|----------|-----------|--------|----------|-------|-------|
| Random Forest       | 0.9737   | 1.0000    | 0.9286 | 0.9630   | 0.9929| 0.9442|
| XGBoost             | 0.9737   | 1.0000    | 0.9286 | 0.9630   | 0.9940| 0.9442|
| Naive Bayes         | 0.9386   | 1.0000    | 0.8333 | 0.9091   | 0.9934| 0.8715|
| Logistic Regression | 0.9386   | 0.9730    | 0.8571 | 0.9114   | 0.9924| 0.8688|
| Decision Tree       | 0.9298   | 0.9048    | 0.9048 | 0.9048   | 0.9246| 0.8492|
| KNN                 | 0.9123   | 0.9706    | 0.7857 | 0.8684   | 0.9547| 0.8138|

---

## Observations

- Random Forest and XGBoost achieved the highest overall performance.
- Random Forest achieved the highest Accuracy, F1 Score, and MCC.
- XGBoost achieved the highest AUC.
- Tree-based ensemble methods outperformed simpler models.
- KNN showed comparatively lower recall.

**Overall Best Performing Model: Random Forest**

---

## Streamlit Web Application

The deployed Streamlit app allows:

- Model selection from sidebar  
- Uploading custom CSV test data  
- Automatic evaluation if label column (`diagnosis`) is present  
- Confusion matrix visualization  
- Display of classification report  
- Prediction probabilities  

### Live App Link
https://ml-assignment-2-ds4mdlhb77f4v3kojj4xvq.streamlit.app/

---

## GitHub Repository

Repository Link:  
https://github.com/Kan1506/ML-Assignment-2

The repository includes:
- Complete source code  
- Jupyter Notebook  
- Saved trained models  
- requirements.txt  
- runtime.txt  
- README.md  

---

## Project Structure

```text
ML-Assignment-2/
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── app.py
├── create_CVS.py
├── ML_Assignment_2.ipynb
├── requirements.txt
├── runtime.txt
├── test_features_only.csv
├── test_with_labels.csv
├── wdbc.data
├── wdbc.names
└── README.md
```

---

## How to Run the Project Locally

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

### Step 3: Open in Browser

```
http://localhost:8501
```

---

## Expected CSV Format

To compute evaluation metrics in the app:

- CSV must contain a column named `diagnosis`
- Values must be:
  - 0 / 1  
  OR  
  - B / M  

If no label column is present, the app will still generate predictions.

---

## Assignment Deliverables

- GitHub Repository (Complete Code)  
- Live Streamlit App (Community Cloud)  
- Model Comparison Table  
- Observations and Analysis  
- Screenshot of Execution on BITS Virtual Lab  

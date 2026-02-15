readme_content = """#  ML Assignment 2 – Breast Cancer Classification

##  Student Information  
- **Name:** Sri Harsha Sattiraju 
- **Student ID:** 2024dc04136
- **Course:** Machine Learning  
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)  
- **University:** BITS Pilani  

---

##  Project Overview

This project implements and compares multiple Machine Learning models for **Breast Cancer Classification** using the Wisconsin Diagnostic Dataset.

The goal is to classify tumors as:

- **0 → Benign**
- **1 → Malignant**

The project includes:

- Data preprocessing  
- Feature scaling  
- Model training  
- Evaluation metrics  
- Confusion matrix visualization  
- Streamlit web application  

---

##  Models Implemented

The following 6 models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

All models were evaluated using identical train-test splits for fair comparison.

---

##  Evaluation Metrics

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

##  Streamlit Web Application

A Streamlit app (`app.py`) is included that allows:

- Model selection from sidebar  
- Uploading custom CSV test data  
- Automatic evaluation if label column (`diagnosis`) is present  
- Confusion matrix visualization  
- Display of classification report  

---

 ##  Project Structure

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
├── test_features_only.csv
├── test_with_labels.csv
├── wdbc.data
├── wdbc.names
└── README.md


## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

### Step 3: Open in Browser

Go to:

http://localhost:8501

##  Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset contains 569 samples with 30 numerical features extracted from digitized images of breast mass cell nuclei.

Target variable:
- 0 → Benign
- 1 → Malignant



Expected CSV Format

To compute evaluation metrics in the app:
CSV must contain a column named diagnosis
Values must be either:
0 / 1
or B / M
If no label column is present, the app will still generate predictions.


Key Results

  Model Comparison Summary

| Model                | Accuracy | AUC  |
|----------------------|----------|------|
| Logistic Regression  | ~0.96+   | ~0.99 |
| Decision Tree        | ~0.93+   | ~0.94 |
| KNN                  | ~0.95+   | ~0.98 |
| Naive Bayes          | ~0.94+   | ~0.97 |
| Random Forest        | ~0.97+   | ~0.99 |
| XGBoost              | ~0.97+   | ~0.99 |

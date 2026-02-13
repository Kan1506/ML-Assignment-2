# Machine Learning Assignment – 2  
**M.Tech (AIML / DSE) – BITS Pilani**

---

## 1. Problem Statement

The objective of this assignment is to implement multiple machine learning classification models on a real-world dataset, evaluate their performance using standard evaluation metrics, and deploy the models using a Streamlit web application. This demonstrates a complete end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.

---

## 2. Dataset Description

The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset from the **UCI Machine Learning Repository**.

- Number of instances: 569  
- Number of input features: 30  
- Target variable: `diagnosis`  
  - 1 → Malignant  
  - 0 → Benign  
- Problem Type: Binary Classification  
- Missing values: None  

The dataset consists of features computed from digitized images of breast mass cell nuclei.

---

## 3. Machine Learning Models Used

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 4. Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | XX | XX | XX | XX | XX | XX |
| Decision Tree | XX | XX | XX | XX | XX | XX |
| KNN | XX | XX | XX | XX | XX | XX |
| Naive Bayes | XX | XX | XX | XX | XX | XX |
| Random Forest | XX | XX | XX | XX | XX | XX |
| XGBoost | XX | XX | XX | XX | XX | XX |

> Replace XX with your actual metric values from the comparison table.

---

## 5. Observations on Model Performance

| Model | Observation |
|-------|-------------|
| Logistic Regression | Performed strongly, indicating the dataset is largely linearly separable. |
| Decision Tree | Captured non-linear patterns but showed slight overfitting compared to ensemble models. |
| KNN | Achieved good accuracy after scaling; performance depends on choice of k. |
| Naive Bayes | Performed reasonably despite independence assumption limitations. |
| Random Forest | Demonstrated strong generalization and stable performance. |
| XGBoost | Achieved the best overall performance across most metrics. |

---

## 6. Deployment Details

The application was developed using **Streamlit** and deployed on **Streamlit Community Cloud**.

The Streamlit app includes:
- CSV file upload option  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  

---

## 7. Repository Structure
ML_Assignment_2/
│── app.py
│── requirements.txt
│── README.md
│── model/
│ ├── logistic_regression.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ └── scaler.pkl


---

## 8. Tools & Libraries Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Matplotlib  

---

## 9. Notes

- All models were implemented and executed on BITS Virtual Lab.
- GitHub repository link and live Streamlit application link are included in the final submission PDF.


# Heart-Disease-Prediction

This project builds and evaluates multiple machine learning models — Logistic Regression, Decision Tree, and Random Forest — to predict the likelihood of heart disease based on clinical features.
It uses the UCI Heart Disease dataset (or your dataset) and compares model performance to identify the most effective approach.
🚀 Project Overview
Heart disease is one of the leading causes of death worldwide.
Early prediction can help in timely diagnosis and treatment.
This project includes:
Data loading and preprocessing
Exploratory Data Analysis (EDA)
Model training
Model evaluation (accuracy, precision, recall, F1-score, ROC-AUC)
Comparison of three ML models
Saving the best model using Pickle
📂 Dataset
Typical dataset features:
Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Resting ECG Results
Maximum Heart Rate
Exercise Induced Angina
ST Depression
ST Slope
Target (1 = disease, 0 = no disease)
Dataset:
👉 UCI Heart Disease Dataset OR your custom dataset.
🛠️ Technologies Used
Python
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
Pickle
🧹 Data Preprocessing
Steps performed:
Handling missing values
Encoding categorical columns
Feature scaling using StandardScaler
Train–test split (80-20)
🤖 Models Implemented
1️⃣ Logistic Regression
Simple interpretable model
Works well for linear relationships
2️⃣ Decision Tree Classifier
Handles non-linear patterns
Easy to visualize
Can overfit
3️⃣ Random Forest Classifier
Ensemble of multiple trees
High accuracy and stability
Reduces overfitting
📊 Model Evaluation Metrics
Each model is evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC

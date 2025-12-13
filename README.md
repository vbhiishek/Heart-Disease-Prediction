# Heart-Disease-Prediction

This project builds and evaluates multiple machine learning models — Logistic Regression, Decision Tree, and Random Forest — to predict the likelihood of heart disease based on clinical features.
It uses the popular UCI Heart Disease dataset (or your dataset) and compares model performance to identify the most effective approach.
🚀 Project Overview
Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely diagnosis and treatment.
In this project, multiple ML algorithms are trained, compared, and evaluated to determine the best predictive model.
The project includes:
Data loading and preprocessing
Exploratory Data Analysis (EDA)
Feature engineering
Model training
Model comparison (accuracy & metrics)
Confusion matrix, ROC curves
Saving the best model
📂 Dataset
You can use the UCI Heart Disease Dataset or your custom dataset.
Typical features include:
Age
Sex
Chest pain type
Resting blood pressure
Cholesterol
Fasting blood sugar
ECG results
Max heart rate
Exercise-induced angina
Oldpeak
ST slope
Target (1 = Disease, 0 = No Disease)
🛠️ Technologies Used
Python
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
Pickle (for model saving)
🧹 Data Preprocessing
Steps include:
Handling missing values
Encoding categorical variables
Feature scaling
Splitting dataset into train & test sets
🤖 Models Implemented
1️⃣ Logistic Regression
Baseline model
Good for explaining feature importance
Fast and interpretable
2️⃣ Decision Tree Classifier
Non-linear model
Captures complex relationships
Easy visualization
3️⃣ Random Forest Classifier
Ensemble of decision trees
Reduces overfitting
Usually highest accuracy and robustness
📊 Model Evaluation
Each model is evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC-AUC score
A comparison table is included to identify the best performing model.
🏆 Results
Random Forest typically provides the highest accuracy and stability, while Logistic Regression is the most interpretable.
Example (your values will differ):
Model	Accuracy
Logistic Regression	85%
Decision Tree	79%
Random Forest	90%

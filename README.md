# Heart Disease Prediction using Machine Learning

This project builds and compares three machine learning models — **Logistic Regression**, **Decision Tree**, and **Random Forest** — to predict heart disease based on clinical features.

---

## 🚀 Project Overview

Heart disease is one of the leading causes of death globally.  
Machine learning can help in early detection by analyzing patient health parameters.

This project covers:

- Data loading & cleaning  
- Exploratory Data Analysis (EDA)  
- Feature selection & preprocessing  
- Training ML models  
- Model comparison  
- Saving the best-performing model  

---

## 📂 Dataset Information

The dataset typically contains:

- male															
- age  
- education  
- currentSmoker  
- cigsPerDay  
- BPMeds  
- prevalentStroke  
- prevalentHyp  
- diabetes  
- totChol 
- sysBP
- diaBP
- BMI
- heartRate
- glucose
- TenYearCHD (1 = Disease, 0 = No Disease)

Dataset Source: UCI Heart Disease Dataset

---

## 🛠️ Technologies Used

- Python  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn  
- Pickle  

---

## 🧹 Data Preprocessing

Steps performed:

- Handling missing values  
- Encoding categorical variables  
- Standardizing numerical features  
- Splitting dataset into training & testing sets  

---

## 🤖 Models Implemented

### 1️⃣ Logistic Regression  
- Interpretable, simple baseline model  
- Works well for linear decision boundaries  

### 2️⃣ Decision Tree Classifier  
- Captures non-linear relationships  
- Easy to visualize  
- Prone to overfitting  

### 3️⃣ Random Forest Classifier  
- Ensemble of many decision trees  
- High accuracy and robustness  
- Best generalization performance  

---

## 📊 Model Evaluation

Each model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC-AUC Score  

Example results:

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | 87%      |
| Decision Tree        | 80%      |
| Random Forest        | 86%      |

---

## 💾 Saving the Model

The best model is saved using **pickle**:

```python
import pickle

with open("heart_model.pkl", "wb") as f:
    pickle.dump(best_model, f) 

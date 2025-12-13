import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------------------------------------
#              Load Pretrained Models
# ------------------------------------------------

@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

# Load all models from pkl files
models = {
    "Logistic Regression": load_model("logistic_model.pkl"),
    "Random Forest": load_model("rf_model.pkl"),
    "SVM": load_model("svm_model.pkl")
}

# Load accuracy scores
model_metrics = pickle.load(open("model_metrics.pkl", "rb"))

# Load dataset for feature reference
df = pd.read_csv("dataset.csv")
features = df.drop("TenYearCHD", axis=1).select_dtypes(include=["number"]).columns.tolist()

# ------------------------------------------------
#                Streamlit UI
# ------------------------------------------------

st.set_page_config(page_title="Heart Disease Prediction (Pretrained Models)", layout="wide")

st.title("Heart Disease Prediction App (Pretrained Models)")
st.markdown("""
This version of the app uses **3 pre-trained machine learning models**, allowing you to:
- Select any model  
- View its accuracy  
- Input patient data  
- Get instant predictions  

No training required — ideal for portfolio demonstration.
""")

# ------------------------------------------------
#                Sidebar (Model Choice)
# ------------------------------------------------

st.sidebar.header("🔍 Select a Model")
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
selected_model = models[selected_model_name]

st.sidebar.write("### Model Accuracy")
st.sidebar.success(f"{selected_model_name}: {model_metrics[selected_model_name]:.4f}")

# ------------------------------------------------
#            User Input Section
# ------------------------------------------------

st.header("Enter Patient Information for Prediction")

input_data = {}
cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        default_value = float(df[feature].mean())  # use mean values as default
        input_data[feature] = st.number_input(feature, value=default_value)

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

st.markdown("---")

# ------------------------------------------------
#                    Prediction
# ------------------------------------------------

if st.button("Predict"):
    prediction = selected_model.predict(input_df)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

st.markdown("---")

st.caption("This app uses pretrained ML models for consistent performance and instant predictions.")

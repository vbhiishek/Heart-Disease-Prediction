
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import seaborn as sns
import io

st.set_page_config(layout="wide", page_title="Heart Disease Prediction", initial_sidebar_state="expanded")

@st.cache_data
def load_data(path="dataset.csv"):
    df = pd.read_csv(path)
    return df

df = load_data("dataset.csv")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data", "Exploratory Analysis", "Modeling", "Predict"])

if page == "Home":
    st.title("Heart Disease Prediction App")
    st.markdown("""
    **What this app does:**  
    - Loads the provided heart disease dataset.
    - Shows interactive EDA with plots and tables.
    - Trains simple models (Random Forest / Logistic Regression) and shows performance.
    - Lets you try live predictions from the web UI.
    """)
    st.write("Dataset shape:", df.shape)
    if st.sidebar.checkbox("Show raw data"):
        st.dataframe(df.head(100))
    st.markdown("---")
    st.write("Quick distribution of target variable (TenYearCHD)")
    fig, ax = plt.subplots()
    df['TenYearCHD'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("TenYearCHD (0 = No, 1 = Yes)")
    st.pyplot(fig)

if page == "Data":
    st.header("Dataset & Cleaning")
    st.write("Columns:", list(df.columns))
    st.write("Null counts:")
    st.write(df.isnull().sum())
    st.markdown("### Summary statistics")
    st.dataframe(df.describe().T)
    st.markdown("### Download cleaned CSV")
    todownload = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", todownload, file_name="heart_dataset_cleaned.csv", mime="text/csv")

if page == "Exploratory Analysis":
    st.header("Exploratory Analysis")
    cols = st.multiselect("Select numerical columns to plot distribution", options=df.select_dtypes(include=[np.number]).columns.tolist(), default=['age','sysBP','BMI','glucose'])
    if cols:
        for c in cols:
            fig, ax = plt.subplots(figsize=(6,2.5))
            sns.histplot(df[c].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {c}")
            st.pyplot(fig)
    st.markdown("### Correlation heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.markdown("### Relationship: Age vs Systolic BP colored by TenYearCHD")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.scatterplot(data=df, x='age', y='sysBP', hue='TenYearCHD', alpha=0.7, ax=ax)
    st.pyplot(fig)

if page == "Modeling":
    st.header("Train models and view metrics")
    target = st.selectbox("Target column", options=['TenYearCHD'], index=0)
    features = st.multiselect("Features to use (leave empty to use all numeric features)", options=[c for c in df.columns if c!=target], default=[c for c in df.select_dtypes(include=[np.number]).columns if c!=target])
    test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2)
    model_choice = st.selectbox("Choose model", ["RandomForest", "LogisticRegression"])
    if st.button("Train model"):
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if model_choice == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        st.metric("Accuracy", f"{acc:.3f}")
        st.metric("ROC AUC", f"{auc:.3f}")
        st.subheader("Classification report")
        st.text(classification_report(y_test, y_pred))
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1], '--', linewidth=0.7)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend()
        st.pyplot(fig)
        if model_choice == "RandomForest":
            st.subheader("Feature importances")
            fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(fi)

        # Save model in session state for prediction page
        st.session_state['model'] = model
        st.session_state['features'] = list(X.columns)

if page == "Predict":
    st.header("Make a prediction")
    if 'model' not in st.session_state:
        st.warning("Train a model first on the Modeling tab. Defaulting to training a quick RandomForest on numeric features...")
        # quick train with defaults
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c!='TenYearCHD']
        X = df[features].fillna(0); y = df['TenYearCHD']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_train, y_train)
        st.session_state['model'] = model; st.session_state['features'] = features

    model = st.session_state['model']
    features = st.session_state['features']

    user_input = {}
    st.subheader("Enter patient data")
    cols = st.columns(3)
    for i, f in enumerate(features):
        with cols[i%3]:
            val = st.number_input(f"{f}", value=float(df[f].median()) if f in df.columns else 0.0)
            user_input[f] = val
    input_df = pd.DataFrame([user_input])
    st.write("Input:")
    st.table(input_df.T.rename(columns={0:"value"}))

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
        st.success(f"Predicted TenYearCHD = {pred} (probability of disease = {prob:.3f} )" if prob is not None else f"Predicted TenYearCHD = {pred}")

    st.markdown("---")
    st.info("Tip: Use the Modeling tab to retrain with different feature sets or model types.")

# Footer
st.markdown('---')
st.caption("Built from your uploaded notebook & dataset. Modify /mnt/data/streamlit_app.py to customize.")

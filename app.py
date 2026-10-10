from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Maternal Satisfaction Prediction", page_icon="🤱", layout="wide")

@st.cache_resource
def load_artifacts():
    required = [
        "logistic_regression_model.joblib",
        "scaler.joblib",
        "feature_columns.joblib",
        "categorical_features_encoded.joblib",
        "deployment_metadata.joblib",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files: " + ", ".join(missing) +
            ". Follow the README instructions to export the trained notebook artifacts into the models/ folder."
        )
    model = joblib.load(MODEL_DIR / "logistic_regression_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    categorical_features = joblib.load(MODEL_DIR / "categorical_features_encoded.joblib")
    metadata = joblib.load(MODEL_DIR / "deployment_metadata.joblib")
    return model, scaler, feature_columns, categorical_features, metadata

st.title("🤱 Maternal Satisfaction Prediction")
st.caption("Maternal delivery-care satisfaction · Bahir Dar, Northwest Ethiopia")
st.warning(
    "This application is an educational research prototype, not a clinical decision tool. "
    "Predictions depend on the training data and should not replace professional judgment."
)

try:
    model, scaler, feature_columns, categorical_features, metadata = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info("The app.py file is ready, but the trained model artifacts must be exported from your Colab notebook and uploaded to the models/ folder.")
    st.stop()

raw_columns = metadata.get("raw_feature_columns", [])
category_options = metadata.get("category_options", {})
numeric_defaults = metadata.get("numeric_defaults", {})
if not raw_columns:
    st.error("The deployment metadata does not contain the original feature schema. Re-run export_deployment.py in Colab as described in README.md.")
    st.stop()

st.markdown("Enter the mother's information using the categories available in the original training dataset.")
with st.form("prediction_form"):
    inputs = {}
    numeric_columns = [c for c in raw_columns if c not in categorical_features]
    categorical_columns = [c for c in raw_columns if c in categorical_features]

    if numeric_columns:
        st.subheader("Numeric and continuous variables")
        numeric_grid = st.columns(3)
        for i, col in enumerate(numeric_columns):
            default = numeric_defaults.get(col, 0.0)
            with numeric_grid[i % 3]:
                inputs[col] = st.number_input(col, value=float(default), key=f"num_{col}")

    if categorical_columns:
        st.subheader("Categorical variables")
        cat_grid = st.columns(2)
        for i, col in enumerate(categorical_columns):
            options = category_options.get(col, [])
            if not options:
                st.error(f"No training categories were saved for '{col}'. Re-export the model artifacts.")
                st.stop()
            with cat_grid[i % 2]:
                inputs[col] = st.selectbox(col, options=options, key=f"cat_{col}")

    submitted = st.form_submit_button("Predict satisfaction", type="primary")

if submitted:
    try:
        raw_input = pd.DataFrame([{col: inputs[col] for col in raw_columns}])
        encoded = pd.get_dummies(
            raw_input,
            columns=[c for c in categorical_features if c in raw_input.columns],
            drop_first=True,
        )
        # Match the exact encoded feature names/order used during training.
        processed = encoded.reindex(columns=feature_columns, fill_value=0).copy()
        scale_columns = [c for c in getattr(scaler, "feature_names_in_", []) if c in processed.columns]
        if scale_columns:
            processed.loc[:, scale_columns] = scaler.transform(processed[scale_columns])

        prediction = int(model.predict(processed)[0])
        probability = float(model.predict_proba(processed)[0][1])
        positive_label = metadata.get("positive_label", "satisfied")
        negative_label = metadata.get("negative_label", "not satisfied")
        label = positive_label if prediction == 1 else negative_label

        st.subheader("Prediction result")
        if prediction == 1:
            st.success(f"Predicted outcome: {label}")
        else:
            st.info(f"Predicted outcome: {label}")
        st.metric("Estimated probability of satisfaction", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("This is a model-estimated probability, not a guarantee or a clinical assessment.")
        with st.expander("Review the entered information"):
            st.dataframe(raw_input, use_container_width=True)
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.caption("Check that the model artifacts were exported from the same notebook run as the trained model.")

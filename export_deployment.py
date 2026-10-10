"""Export trained objects from the active notebook namespace for Streamlit.

In Google Colab, upload this file beside the notebook and run:
    %run -i export_deployment.py
The -i option is important: it exposes the notebook's trained variables to this script.
"""
from pathlib import Path
import joblib
import pandas as pd

required_names = ["model", "scaler", "X", "df", "categorical_features_for_encoding"]
missing_names = [name for name in required_names if name not in globals()]
if missing_names:
    raise RuntimeError(
        "These notebook variables are missing: " + ", ".join(missing_names) +
        ". Run the data preparation and model training cells first, then run this exporter with %run -i."
    )

out_dir = Path("models")
out_dir.mkdir(parents=True, exist_ok=True)

# Keep artifact names aligned with app.py.
joblib.dump(model, out_dir / "logistic_regression_model.joblib")
joblib.dump(scaler, out_dir / "scaler.joblib")
feature_columns = list(X.columns)
joblib.dump(feature_columns, out_dir / "feature_columns.joblib")
categorical_features = list(categorical_features_for_encoding)
joblib.dump(categorical_features, out_dir / "categorical_features_encoded.joblib")

# Save categories from the original, pre-one-hot dataframe so the app presents valid choices.
category_options = {}
for col in categorical_features:
    if col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = list(series.cat.categories)
        else:
            values = list(series.dropna().unique())
        category_options[col] = [str(v) for v in values]

# Preserve the raw feature schema used by the notebook (excluding target and dropped columns).
raw_feature_columns = [c for c in df.columns if c != "satisfaction"]
numeric_defaults = {}
for col in raw_feature_columns:
    if col not in categorical_features and col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        numeric_defaults[col] = float(vals.median()) if not vals.empty else 0.0

# The notebook uses 0 = not satisfied and 1 = satisfied.
metadata = {
    "raw_feature_columns": raw_feature_columns,
    "category_options": category_options,
    "numeric_defaults": numeric_defaults,
    "positive_label": "satisfied",
    "negative_label": "not satisfied",
}
joblib.dump(metadata, out_dir / "deployment_metadata.joblib")

print("Deployment artifacts exported to:", out_dir.resolve())
for path in sorted(out_dir.iterdir()):
    print(f" - {path.name} ({path.stat().st_size:,} bytes)")

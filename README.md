# Maternal Satisfaction Streamlit App

Streamlit Community Cloud package for the maternal satisfaction Logistic Regression project.

## Folder structure

```text
Maternal_Satisfaction_Streamlit/
├── app.py
├── export_deployment.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── README.txt
```

## 1. Export the trained model files from Google Colab

The notebook reads `Delivery care satisfaction.sav` from your Google Drive and trains the model. The model files cannot be created from this folder alone because the SPSS dataset and trained notebook session are not included here.

1. Open your `Python_Project (3).ipynb` notebook in Google Colab.
2. Run the notebook from the data-loading/preprocessing cells through model training. Make sure the variables `df`, `model`, `scaler`, `X`, and `categorical_features_for_encoding` exist in the active session.
3. Upload `export_deployment.py` to the Colab session (left sidebar **Files** → **Upload**) or copy it into the notebook working directory.
4. Run this in a new code cell:

   ```python
   %run -i export_deployment.py
   ```

   The `-i` flag lets the exporter access variables already created by your notebook. It creates a `models/` directory with:
   - `logistic_regression_model.joblib`
   - `scaler.joblib`
   - `feature_columns.joblib`
   - `categorical_features_encoded.joblib`
   - `deployment_metadata.joblib`

5. Download the five `.joblib` files from Colab and place them in this project's `models/` folder. Keep the filenames exactly as listed.
6. Check that the folder now contains all five files before pushing the project to GitHub.

**Important:** Export artifacts from the same notebook run that trained the model. The app uses the original notebook's one-hot encoding and scaling conventions to stay consistent with the model. Do not upload the SPSS dataset unless you have confirmed that you are permitted to publish it.

## 2. Test locally (optional)

From this folder, install dependencies and launch the app:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. Upload the folder to GitHub

Create a GitHub repository, for example `maternal-satisfaction-app`. Upload the **contents** of this folder (not an extra nested copy of the folder). The repository root should show `app.py`, `requirements.txt`, `README.md`, `export_deployment.py`, and `models/`.

The repository needs the trained `.joblib` files for the deployed prediction feature to work. GitHub has file-size limits; if an artifact is too large, use an appropriate private artifact-storage solution instead of renaming or omitting it.

## 4. Deploy on Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose your repository and branch (usually `main`).
4. Set **Main file path** to exactly `app.py` if it is in the repository root.
5. Click **Deploy** and wait for the build to finish.

If `app.py` is inside a folder named `Maternal_Satisfaction_Streamlit` within the repository, the main file path must instead be `Maternal_Satisfaction_Streamlit/app.py`. Prefer uploading the contents of this package to the repository root to keep the path simple.

## Notes and limitations

- This is a research/education demonstration, not a medical decision-support system.
- Input choices are taken from the categories in the training data; values not seen in training cannot be selected in this form.
- A category that did not appear in the training split may be represented by zero-valued dummy columns when the input is encoded. For a more robust production pipeline, train and save a single scikit-learn `Pipeline` with `OneHotEncoder(handle_unknown="ignore")` and `StandardScaler` fitted only on training data.
- Review the notebook's target mapping and feature selection before reporting model performance. Do not present app predictions as validated clinical recommendations.

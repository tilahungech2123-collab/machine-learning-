# Maternal Satisfaction with Delivery Care — Streamlit App

Maternal satisfaction among vaginal and cesarean section delivery care services in Bahir Dar city health facilities, Northwest Ethiopia.

A Streamlit web app with a Logistic Regression model for predicting maternal satisfaction.

## Project Structure

```
maternal_satisfaction_app/
├── app.py                  # Main Streamlit app (4 tabs: Overview, Charts, Performance, Predict)
├── train_model.py          # Standalone training script (run locally first)
├── requirements.txt        # Python dependencies
├── data/
│   └── Delivery care satisfaction.sav   # SPSS data file (YOU MUST ADD THIS)
├── models/                 # Auto-generated model artifacts
│   ├── logistic_regression_model.joblib
│   ├── scaler.joblib
│   ├── feature_columns.joblib
│   ├── categorical_features_encoded.joblib
│   ├── y_test.joblib
│   ├── y_pred.joblib
│   ├── coef_importance.joblib
│   └── perm_importance.joblib
└── .streamlit/
    └── config.toml          # Streamlit server config
```

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add the data file
Download `Delivery care satisfaction.sav` from your Google Drive and place it in `data/`.

### 3. Train the model
```bash
python train_model.py
```
This trains the Logistic Regression model and saves all artifacts to `models/`.

### 4. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Deploy on Streamlit Community Cloud (Free)

### Prerequisites
- A free **GitHub account** (https://github.com)
- A free **Streamlit Community Cloud account** (https://share.streamlit.io — sign in with GitHub)
- The data file `Delivery care satisfaction.sav` downloaded from Google Drive

### Step 1: Train the model locally
```bash
pip install -r requirements.txt
# Place 'Delivery care satisfaction.sav' in data/
python train_model.py
```
This generates the `.joblib` model artifacts in `models/`.

### Step 2: Push to GitHub
1. Create a new **public** repository on GitHub (e.g., `maternal-satisfaction-app`)
2. Upload ALL files to the repo, including:
   - `app.py`
   - `train_model.py`
   - `requirements.txt`
   - `data/Delivery care satisfaction.sav`
   - `models/` (all .joblib files)
   - `.streamlit/config.toml`
3. Commit and push

### Step 3: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **"Create app"** (top right)
3. Select **"Yup, I have an app"**
4. Select your GitHub repository:
   - Repository: `your-username/maternal-satisfaction-app`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **"Advanced settings"**
   - Python version: select **3.12** (or 3.13 if available)
   - No secrets needed
6. Click **"Deploy"**

Your app will be live at `https://your-app-name.streamlit.app` within a few minutes.

### Streamlit Cloud Requirements Checklist
- [ ] Public GitHub repository
- [ ] `app.py` as the entrypoint file
- [ ] `requirements.txt` with all dependencies
- [ ] `data/Delivery care satisfaction.sav` included in the repo
- [ ] `models/` directory with all .joblib artifacts (from running train_model.py)
- [ ] Python 3.12+ compatible code
- [ ] No hardcoded local file paths (app uses relative paths)
- [ ] No Google Colab or Google Drive dependencies

### Important Notes
- Streamlit Community Cloud is **free** for public apps
- Apps sleep after inactivity and wake on first visit (may take 30+ seconds)
- Python version defaults to 3.12 on Streamlit Cloud
- The `.sav` data file must be in the repo (Streamlit Cloud cannot access Google Drive)
- Model artifacts (.joblib files) must be pre-generated and pushed to GitHub
- Resource limit: 1 GB RAM for free tier

Loan Approval Prediction Web App (Flask + ML)

## Overview

This project is a Flask web application that predicts loan approval based on user-provided information. It serves a trained machine learning model and includes a separate training script to build and evaluate traditional ML ensembles.

## Pipeline Configuration

### Reproducing the Deployment

This project uses GitHub Actions to deploy securely to a pre-provisioned Google VM.

The non-sensitive VM configuration is stored in the repository in `deployment/vm_config.env`.

This file contains the current demo VM values. If recreating the deployment on a different VM, update these values.

```bash
VM_HOST="34.163.29.46"              # Replace with your VM external IP
VM_USER="shannon"                   # Replace with your VM SSH username
APP_DIR="/opt/mlops/loan-approval-prediction"
APP_NAME="loan-approval-app"
IMAGE_NAME="loan-approval-app:latest"
```

These values are included in the repository to make the deployment easier to understand and recreate. They are not secrets. The IP address is specific to the current demo VM and should be changed if deploying to a different VM.

The sensitive values are excluded from the repository and must be configured as GitHub Actions secrets.

| Secret | Purpose |
|---|---|
| `VM_SSH_KEY` | Private SSH key used by GitHub Actions to connect to the Google VM |
| `ACTIONS_PUSH_TOKEN` | GitHub token used by workflows that commit model, retraining, promotion or rollback changes back to the repository |

The VM must be pre-provisioned with:

- Docker
- Docker Compose
- Git
- SSH access
- Firewall rule allowing HTTP traffic on port 80
- Repository cloned to `/opt/mlops/loan-approval-prediction`
- User permissions to run Docker

The deployment workflow is reproducible once the VM prerequisites, `deployment/vm_config.env`, and the required GitHub Actions secrets are configured.

### Workflows

| Workflow | File | Purpose |
|---|---|---|
| Deploy Loan Approval App | `.github/workflows/deploy.yml` | Deploys the Dockerised Flask app to the Google VM |
| Drift Check and Retrain | `.github/workflows/drift_check.yml` | Calls `/drift-check`, retrains if drift is detected, evaluates the candidate model and auto-promotes it if approved |
| Evidently Monitoring Report | `.github/workflows/evidently_report.yml` | Fetches live prediction logs and generates an Evidently HTML monitoring report |
| Model Package Rollback | `.github/workflows/model_package_rollback.yml` | Restores a selected model package from a previous Git commit |

### Recreating the Deployment in Another Account

To recreate the deployment:

1. Clone or fork this repository.
2. Provision a VM with Docker, Docker Compose, Git and SSH access.
3. Clone the repository on the VM at `/opt/mlops/loan-approval-prediction`.
4. Update `deployment/vm_config.env` with the new VM IP address, VM username and app path if required.
5. Add the required GitHub Actions secrets: `VM_SSH_KEY` and `ACTIONS_PUSH_TOKEN`.
6. Run the deployment workflow from `GitHub Actions → Deploy Loan Approval App → Run workflow`.

After deployment, the application can be tested using the VM host configured in `deployment/vm_config.env`.

```text
http://<VM_HOST>/health
http://<VM_HOST>/version
http://<VM_HOST>/metrics
http://<VM_HOST>/drift-check
```

For the current demo VM, `<VM_HOST>` is:

```text
34.163.29.46
```

The `/version` endpoint confirms the deployed Git commit, build time, deployment environment and active model package.

### Production Note

In this assignment, the Google VM is pre-provisioned manually. In a production environment, the VM setup would normally be automated using Infrastructure as Code tools such as Terraform, Ansible, cloud-init or a bootstrap script.



## Features

- End-to-end loan approval prediction web UI built with Flask
- Robust preprocessing with `RobustScaler` and categorical encoders saved via `joblib`
- PyTorch inference model (`model.pth`) with advanced architecture
- Optional traditional ML ensemble training (`train_ensemble.py`)
- Clear results page showing predicted decision and confidence
- Defensive input validation with helpful error messages

## Tech Stack

- Python, Flask
- PyTorch (inference model loaded from `model.pth`)
- scikit-learn (preprocessing + alternative ensemble model in `train_ensemble.py`)
- numpy, pandas, joblib, matplotlib

## Repository Structure

```
.
├── app.py                      # Flask app entrypoint
├── train_ensemble.py           # Training script for traditional ML ensemble
├── LAP.csv                     # Dataset used by the training script
├── model.pth                   # PyTorch model used by the Flask app (inference)
├── scaler.pkl                  # RobustScaler used for preprocessing
├── label_encoders.pkl          # Column-wise LabelEncoders
├── label_encoder_target.pkl    # LabelEncoder for the target variable
├── templates/
│   ├── index.html              # Form for collecting user input
│   └── results.html            # Results page
└── user_input.py               # (If used) helper for input handling
```

## Code Overview

- `app.py`: Flask routes, input validation, feature engineering, categorical encoding, scaling, PyTorch model loading, and prediction logic. Produces a confidence score for classification.
- `train_ensemble.py`: Loads `LAP.csv`, performs extensive feature engineering, trains multiple scikit-learn models, evaluates, and saves preprocessing artifacts and an ensemble checkpoint.
- `templates/`: HTML templates (`index.html`, `results.html`) for the web UI.
- `model.pth`: PyTorch model state dict used by the web app.
- `scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`: Preprocessing artifacts saved with `joblib`.

## Project Description

This application demonstrates a practical ML workflow: a trained model is packaged with its preprocessing artifacts and served via a simple Flask interface. The app collects core applicant details, derives additional predictive features, applies encoders/scaling, and performs a prediction with a confidence score. The result is presented as an eligibility decision (Yes/No) along with a confidence threshold to avoid overconfident outputs on uncertain inputs.

The repository also includes a separate training script (`train_ensemble.py`) that builds classic ML baselines (RandomForest, GradientBoosting, LogisticRegression) and an ensemble for comparison and experimentation.

## Inputs Collected

- Loan_ID, Gender, Married, Dependents, Education, Self_Employed
- ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
- Credit_History, Property_Area

Derived features include Total Income, Income-to-Loan ratio, Loan-to-Income ratio, and categorical bins for income/loan levels.

## Quickstart

1) Create a virtual environment and activate it

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Training (optional)

If you want to retrain preprocessing and an ensemble model (RandomForest, GradientBoosting, LogisticRegression), run:

```bash
python train_ensemble.py
```

This will output updated artifacts like `scaler.pkl`, `label_encoders.pkl`, and `label_encoder_target.pkl`. The script also saves `ensemble_model.pkl` for reference; the Flask app loads the PyTorch model from `model.pth` for inference.

## How it works

1) User submits the form at `/` (`templates/index.html`).
2) `app.py` validates inputs, engineers additional features, encodes categoricals, and scales numerics.
3) The preprocessed vector is passed to the PyTorch model loaded from `model.pth`.
4) The output is interpreted (classification with confidence) and shown on `/results`.

### Routes

- `/`: Home page with input form
- `/results`: Results page with prediction and confidence

## Deployment

- Production WSGI (example):
  - `pip install gunicorn waitress`
  - Gunicorn (Linux): `gunicorn -w 2 -b 0.0.0.0:8000 app:app`
  - Waitress (Windows): `waitress-serve --listen=0.0.0.0:8000 app:app`
- Cloud (Render/Heroku):
  - Set Python version and install from `requirements.txt`.
  - Ensure artifacts (`model.pth`, `scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`) are deployed.
  - Set environment variable `FLASK_SECRET` (optional) for session security.

## Environment Variables

- `FLASK_SECRET`: Optional; if set, overrides `app.secret_key` in `app.py`.

## Notes

- The Flask app expects the preprocessing artifacts (`scaler.pkl`, `label_encoders.pkl`, `label_encoder_target.pkl`) to be present in the project root.
- The dataset `LAP.csv` is used by the training script only. The web app does not require it at runtime.
- To replace the inference model, update `model.pth` and ensure its input feature ordering matches the preprocessing pipeline.

## How to publish on GitHub

1) Initialize git (inside the project folder):

```bash
git init
git add .
git commit -m "Initial commit: Loan approval prediction app"
```

2) Create a new empty repository on GitHub (no README/.gitignore). Then connect and push:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.



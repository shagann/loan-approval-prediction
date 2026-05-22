from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime


# ============================================================
# Model versioning configuration
# ============================================================

MODEL_VERSION = os.environ.get("MODEL_VERSION", "baseline-model")
MODEL_DIR = os.path.join("models", MODEL_VERSION)

MODEL_FILE = os.path.join(MODEL_DIR, "model.pth")
BEST_MODEL_FILE = os.path.join(MODEL_DIR, "best_model.pth")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENCODERS_FILE = os.path.join(MODEL_DIR, "label_encoders.pkl")
LABEL_ENCODER_TARGET_FILE = os.path.join(MODEL_DIR, "label_encoder_target.pkl")
MODEL_METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

LOG_DIR = "logs"
PREDICTION_LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")


# Check that the selected model package is complete
required_model_files = [
    MODEL_FILE,
    SCALER_FILE,
    LABEL_ENCODERS_FILE,
    LABEL_ENCODER_TARGET_FILE
]

missing_model_files = [
    file_path for file_path in required_model_files
    if not os.path.exists(file_path)
]

if missing_model_files:
    raise FileNotFoundError(
        f"Missing model artefacts for MODEL_VERSION={MODEL_VERSION}: {missing_model_files}"
    )


# ============================================================
# Flask setup
# ============================================================

app = Flask(__name__)
app.secret_key = "your-secret-key-here"


# ============================================================
# Load preprocessing objects from versioned model package
# ============================================================

robust_scaler = joblib.load(SCALER_FILE)
label_encoders = joblib.load(LABEL_ENCODERS_FILE)

try:
    le_target = joblib.load(LABEL_ENCODER_TARGET_FILE)
    is_classification = True
except Exception:
    le_target = None
    is_classification = False


# ============================================================
# Advanced Neural Network Architecture
# ============================================================

class AdvancedLoanPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], output_dim=2, dropout_rate=0.3):
        super(AdvancedLoanPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Build dynamic layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

            if i % 2 == 0:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())

            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Multiple prediction heads for ensemble effect
        self.prediction_heads = nn.ModuleList([
            nn.Linear(prev_dim, output_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Linear(prev_dim, output_dim)
        ])

        # Final prediction layer
        self.final_prediction = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(prev_dim // 2, output_dim)
        )

    def forward(self, x):
        x = self.input_bn(x)

        features = self.feature_extractor(x)

        predictions = []
        for head in self.prediction_heads:
            pred = head(features)
            predictions.append(pred)

        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)

        final_pred = self.final_prediction(features)

        return final_pred


# ============================================================
# Load model from versioned model package
# ============================================================

def load_model(input_size, output_dim):
    model = AdvancedLoanPredictor(input_dim=input_size, output_dim=output_dim)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
    model.eval()
    return model


# ============================================================
# Input validation and feature engineering
# ============================================================

def validate_and_preprocess_input(user_input):
    """Advanced input validation and preprocessing."""

    processed_input = {}

    validation_rules = {
        "Loan_ID": {"type": str, "min_length": 1, "max_length": 50},
        "Gender": {"type": str, "allowed_values": ["Male", "Female"]},
        "Married": {"type": str, "allowed_values": ["Yes", "No"]},
        "Dependents": {"type": int, "min": 0, "max": 10},
        "Education": {"type": str, "allowed_values": ["Graduate", "Not Graduate"]},
        "Self_Employed": {"type": str, "allowed_values": ["Yes", "No"]},
        "ApplicantIncome": {"type": float, "min": 0, "max": 1000000},
        "CoapplicantIncome": {"type": float, "min": 0, "max": 1000000},
        "LoanAmount": {"type": float, "min": 0, "max": 1000000},
        "Loan_Amount_Term": {"type": float, "min": 0, "max": 10000},
        "Credit_History": {"type": int, "allowed_values": [0, 1]},
        "Property_Area": {"type": str, "allowed_values": ["Urban", "Semiurban", "Rural"]}
    }

    for field, rules in validation_rules.items():
        value = user_input.get(field)

        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(f"Field '{field}' is required")

        try:
            if rules["type"] == int:
                processed_input[field] = int(value)
            elif rules["type"] == float:
                processed_input[field] = float(value)
            else:
                processed_input[field] = str(value).strip()
        except (ValueError, TypeError):
            raise ValueError(f"Field '{field}' must be of type {rules['type'].__name__}")

        if "min" in rules and processed_input[field] < rules["min"]:
            raise ValueError(f"Field '{field}' must be at least {rules['min']}")

        if "max" in rules and processed_input[field] > rules["max"]:
            raise ValueError(f"Field '{field}' must be at most {rules['max']}")

        if "min_length" in rules and len(processed_input[field]) < rules["min_length"]:
            raise ValueError(f"Field '{field}' must be at least {rules['min_length']} characters")

        if "max_length" in rules and len(processed_input[field]) > rules["max_length"]:
            raise ValueError(f"Field '{field}' must be at most {rules['max_length']} characters")

        if "allowed_values" in rules and processed_input[field] not in rules["allowed_values"]:
            raise ValueError(
                f"Field '{field}' must be one of: {', '.join(map(str, rules['allowed_values']))}"
            )

    return processed_input


def create_advanced_features(user_input):
    """Create advanced features for better prediction."""

    features = user_input.copy()

    features["Total_Income"] = features["ApplicantIncome"] + features["CoapplicantIncome"]
    features["Income_to_Loan_Ratio"] = features["Total_Income"] / (features["LoanAmount"] + 1)
    features["Loan_to_Income_Ratio"] = features["LoanAmount"] / (features["Total_Income"] + 1)
    features["Dependents_Income_Ratio"] = features["Dependents"] / (features["Total_Income"] + 1)

    if features["Total_Income"] <= 5000:
        features["Income_Category"] = "Very_Low"
    elif features["Total_Income"] <= 10000:
        features["Income_Category"] = "Low"
    elif features["Total_Income"] <= 20000:
        features["Income_Category"] = "Medium"
    elif features["Total_Income"] <= 50000:
        features["Income_Category"] = "High"
    else:
        features["Income_Category"] = "Very_High"

    if features["LoanAmount"] <= 100:
        features["Loan_Amount_Category"] = "Very_Small"
    elif features["LoanAmount"] <= 500:
        features["Loan_Amount_Category"] = "Small"
    elif features["LoanAmount"] <= 1000:
        features["Loan_Amount_Category"] = "Medium"
    elif features["LoanAmount"] <= 5000:
        features["Loan_Amount_Category"] = "Large"
    else:
        features["Loan_Amount_Category"] = "Very_Large"

    return features


# Predefined input fields
INPUT_FIELDS = {
    "Loan_ID": str,
    "Gender": str,
    "Married": str,
    "Dependents": int,
    "Education": str,
    "Self_Employed": str,
    "ApplicantIncome": float,
    "CoapplicantIncome": float,
    "LoanAmount": float,
    "Loan_Amount_Term": float,
    "Credit_History": int,
    "Property_Area": str
}


# ============================================================
# Monitoring helpers
# ============================================================

def log_prediction(user_input, prediction, confidence):
    """Save each prediction so it can be used for monitoring and drift checks."""

    os.makedirs(LOG_DIR, exist_ok=True)

    file_exists = os.path.exists(PREDICTION_LOG_FILE)

    fieldnames = [
        "timestamp",
        "Loan_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "prediction",
        "confidence",
        "app_version",
        "model_version"
    ]

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "Loan_ID": user_input.get("Loan_ID"),
        "Gender": user_input.get("Gender"),
        "Married": user_input.get("Married"),
        "Dependents": user_input.get("Dependents"),
        "Education": user_input.get("Education"),
        "Self_Employed": user_input.get("Self_Employed"),
        "ApplicantIncome": user_input.get("ApplicantIncome"),
        "CoapplicantIncome": user_input.get("CoapplicantIncome"),
        "LoanAmount": user_input.get("LoanAmount"),
        "Loan_Amount_Term": user_input.get("Loan_Amount_Term"),
        "Credit_History": user_input.get("Credit_History"),
        "Property_Area": user_input.get("Property_Area"),
        "prediction": prediction,
        "confidence": confidence,
        "app_version": os.environ.get("APP_VERSION", "local-dev"),
        "model_version": MODEL_VERSION
    }

    with open(PREDICTION_LOG_FILE, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Health/version/monitoring routes
# ============================================================

@app.route("/health")
def health():
    return {
        "status": "running",
        "service": "Loan Approval Prediction API"
    }, 200


@app.route("/version")
def version():
    return {
        "app_name": "Loan Approval Prediction API",
        "app_version": os.environ.get("APP_VERSION", "local-dev"),
        "git_commit": os.environ.get("GIT_COMMIT", "not-set"),
        "build_time": os.environ.get("BUILD_TIME", "not-set"),
        "model_version": MODEL_VERSION,
        "model_dir": MODEL_DIR,
        "model_metadata_file": MODEL_METADATA_FILE,
        "deployment_environment": os.environ.get("DEPLOYMENT_ENVIRONMENT", "local"),
        "deployment_colour": os.environ.get("DEPLOYMENT_COLOUR", "none")
    }, 200


@app.route("/metrics")
def metrics():
    """Return simple monitoring metrics from the prediction log."""

    if not os.path.exists(PREDICTION_LOG_FILE):
        return jsonify({
            "total_predictions": 0,
            "approved_predictions": 0,
            "rejected_predictions": 0,
            "average_confidence": None,
            "last_prediction_time": None,
            "model_version": MODEL_VERSION,
            "monitoring_source": "logs/predictions.csv",
            "message": "No predictions have been logged yet"
        })

    df = pd.read_csv(PREDICTION_LOG_FILE)

    if df.empty:
        return jsonify({
            "total_predictions": 0,
            "approved_predictions": 0,
            "rejected_predictions": 0,
            "average_confidence": None,
            "last_prediction_time": None,
            "model_version": MODEL_VERSION,
            "monitoring_source": "logs/predictions.csv",
            "message": "Prediction log is empty"
        })

    total_predictions = len(df)

    approved_predictions = len(
        df[df["prediction"].astype(str).str.lower().isin([
            "1", "y", "yes", "approved", "approve", "loan approved"
        ])]
    )

    rejected_predictions = total_predictions - approved_predictions

    average_confidence = None
    if "confidence" in df.columns:
        average_confidence = round(
            pd.to_numeric(df["confidence"], errors="coerce").mean(),
            4
        )

    last_prediction_time = None
    if "timestamp" in df.columns:
        last_prediction_time = df["timestamp"].iloc[-1]

    return jsonify({
        "total_predictions": int(total_predictions),
        "approved_predictions": int(approved_predictions),
        "rejected_predictions": int(rejected_predictions),
        "average_confidence": average_confidence,
        "last_prediction_time": last_prediction_time,
        "model_version": MODEL_VERSION,
        "monitoring_source": "logs/predictions.csv"
    })


@app.route("/drift-check")
def drift_check():
    """Simple drift check using recent prediction logs."""

    force_drift = request.args.get("force", "false").lower() == "true"

    if force_drift:
        return jsonify({
            "drift_detected": True,
            "drift_type": "forced_demo_drift",
            "message": "Forced drift mode enabled for demonstration",
            "drift_features": {
                "ApplicantIncome": {
                    "baseline_mean": 5400,
                    "recent_mean": 50000,
                    "percent_change": 825.93,
                    "threshold_percent": 20
                },
                "LoanAmount": {
                    "baseline_mean": 150,
                    "recent_mean": 25000,
                    "percent_change": 16566.67,
                    "threshold_percent": 20
                }
            },
            "recommendation": "Retraining should be considered"
        })

    if not os.path.exists(PREDICTION_LOG_FILE):
        return jsonify({
            "drift_detected": False,
            "message": "No prediction logs available yet",
            "recommendation": "Collect more live prediction data before checking drift",
            "monitoring_source": "logs/predictions.csv"
        })

    df = pd.read_csv(PREDICTION_LOG_FILE)

    if df.empty:
        return jsonify({
            "drift_detected": False,
            "message": "Prediction log is empty",
            "recommendation": "Collect more live prediction data before checking drift",
            "monitoring_source": "logs/predictions.csv"
        })

    baseline_stats = {
        "ApplicantIncome": 5400,
        "CoapplicantIncome": 1600,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 0.85
    }

    drift_threshold_percent = 20
    drift_features = {}
    checked_features = []

    recent_df = df.tail(20)

    for feature, baseline_mean in baseline_stats.items():
        if feature in recent_df.columns:
            checked_features.append(feature)

            recent_mean = pd.to_numeric(
                recent_df[feature],
                errors="coerce"
            ).mean()

            if pd.isna(recent_mean):
                continue

            percent_change = abs((recent_mean - baseline_mean) / baseline_mean) * 100

            if percent_change > drift_threshold_percent:
                drift_features[feature] = {
                    "baseline_mean": round(float(baseline_mean), 4),
                    "recent_mean": round(float(recent_mean), 4),
                    "percent_change": round(float(percent_change), 2),
                    "threshold_percent": drift_threshold_percent
                }

    drift_detected = len(drift_features) > 0

    recommendation = (
        "Retraining should be considered"
        if drift_detected
        else "No retraining required based on current drift rules"
    )

    return jsonify({
        "drift_detected": drift_detected,
        "checked_features": checked_features,
        "drift_features": drift_features,
        "recent_window_size": int(len(recent_df)),
        "monitoring_source": "logs/predictions.csv",
        "recommendation": recommendation
    })


# ============================================================
# Main app routes
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            user_input = {}

            for field, ftype in INPUT_FIELDS.items():
                value = request.form.get(field)

                if value is None or value.strip() == "":
                    return render_template(
                        "index.html",
                        error=f"Please fill the field: {field}",
                        inputs=request.form
                    )

                user_input[field] = value

            try:
                user_input = validate_and_preprocess_input(user_input)
            except ValueError as e:
                return render_template("index.html", error=str(e), inputs=request.form)

            user_input = create_advanced_features(user_input)

            input_df = pd.DataFrame([user_input])

            for col in input_df.columns:
                if col in label_encoders:
                    le = label_encoders[col]

                    try:
                        if col in ["Income_Category", "Loan_Amount_Category"]:
                            if input_df[col].values[0] not in le.classes_:
                                input_df[col] = le.transform(["Medium"])[0]
                            else:
                                input_df[col] = le.transform(input_df[col])
                        else:
                            input_df[col] = le.transform(input_df[col])
                    except ValueError:
                        all_categories = list(le.classes_) + [input_df[col].values[0]]
                        le.fit(all_categories)
                        input_df[col] = le.transform(input_df[col])

                elif input_df[col].dtype == "object":
                    return render_template(
                        "index.html",
                        error=f"Missing encoder for column '{col}'.",
                        inputs=request.form
                    )

            try:
                X_input = robust_scaler.transform(input_df.values.astype(np.float32))
            except Exception as e:
                return render_template(
                    "index.html",
                    error=f"Scaling error: {str(e)}",
                    inputs=request.form
                )

            X_input = torch.tensor(X_input, dtype=torch.float32)

            input_size = X_input.shape[1]
            output_dim = len(le_target.classes_) if is_classification else 1
            model = load_model(input_size, output_dim)

            with torch.no_grad():
                output = model(X_input)

                if is_classification:
                    probabilities = F.softmax(output, dim=1)
                    confidence, pred = torch.max(probabilities, 1)
                    prediction = le_target.inverse_transform([pred.item()])[0]
                    confidence_score = confidence.item()
                else:
                    prediction = output.item()
                    confidence_score = 0.8

            session["prediction"] = str(prediction) if prediction is not None else None
            session["confidence"] = confidence_score

            log_prediction(user_input, prediction, confidence_score)

            user_input_clean = {}
            for key, value in user_input.items():
                if hasattr(value, "item"):
                    user_input_clean[key] = value.item()
                else:
                    user_input_clean[key] = value

            session["user_input"] = user_input_clean

            return redirect(url_for("results"))

        except Exception as e:
            return render_template(
                "index.html",
                error=f"An error occurred: {str(e)}",
                inputs=request.form
            )

    return render_template("index.html", inputs={})


@app.route("/results")
def results():
    prediction = session.get("prediction")
    user_input = session.get("user_input")
    confidence = session.get("confidence", 0.8)

    if prediction is None or user_input is None:
        return redirect(url_for("home"))

    pred_str = str(prediction).lower()
    is_eligible = pred_str in [
        "y", "yes", "approved", "1", "true", "approve", "loan approved"
    ]

    if confidence < 0.6:
        is_eligible = False

    session.pop("prediction", None)
    session.pop("user_input", None)
    session.pop("confidence", None)

    return render_template(
        "results.html",
        prediction=prediction,
        user_input=user_input,
        is_eligible=is_eligible,
        confidence=confidence
    )


# ============================================================
# App entry point
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
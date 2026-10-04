# models.py
# -----------------------------------------------------------------------------
# Train, evaluate, compare, and persist ML models.
# Supports: Linear / Logistic Regression  and  Random Forest (Regressor /
# Classifier) selected dynamically based on the task detected in data_utils.py.
# -----------------------------------------------------------------------------

import os
import sys
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# -- 1. MODEL FACTORY ----------------------------------------------------------

def build_models(task: str = "regression") -> dict:
    if task == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
        }
    else:
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, n_jobs=-1
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
        }


# -- 2. TRAINING ---------------------------------------------------------------

def train_models(models: dict, X_train, y_train) -> dict:
    """Fit every model and return them in a dict."""
    trained = {}
    for name, model in models.items():
        print(f"  Training: {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# -- 3. EVALUATION -------------------------------------------------------------

def evaluate_models(trained: dict, X_test, y_test, task: str) -> pd.DataFrame:
    """Evaluate all models and return a tidy comparison DataFrame."""
    records = []
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        if task == "regression":
            records.append({
                "Model":   name,
                "R2 Score": round(r2_score(y_test, y_pred), 4),
                "MAE":      round(mean_absolute_error(y_test, y_pred), 4),
                "RMSE":     round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            })
        else:
            records.append({
                "Model":     name,
                "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "F1 (macro)":round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "Recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            })
    return pd.DataFrame(records)


def get_classification_report(model, X_test, y_test, label_encoder=None) -> str:
    y_pred = model.predict(X_test)
    target_names = label_encoder.classes_ if label_encoder else None
    return classification_report(y_test, y_pred, target_names=target_names, zero_division=0)


def get_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame | None:
    """Return feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return fi
    if hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coef),
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return fi
    return None


# -- 4. BEST MODEL SELECTION & PERSISTENCE -------------------------------------

def select_best(results_df: pd.DataFrame, task: str) -> str:
    """Return the name of the best model."""
    metric = "R2 Score" if task == "regression" else "Accuracy"
    return results_df.loc[results_df[metric].idxmax(), "Model"]


def save_model(model, scaler, encoders: dict, feature_cols: list,
               target_col: str, task: str, col_map: dict,
               filename: str = "best_model.pkl"):
    path = os.path.join(MODEL_DIR, filename)
    bundle = {
        "model":        model,
        "scaler":       scaler,
        "encoders":     encoders,
        "feature_cols": feature_cols,
        "target_col":   target_col,
        "task":         task,
        "col_map":      col_map,
    }
    joblib.dump(bundle, path)
    print(f"[INFO] Best model saved -> {path}")
    return path


def load_model(filename: str = "best_model.pkl") -> dict | None:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


# -- 5. PREDICTION HELPER ------------------------------------------------------

def predict_single(bundle: dict, input_dict: dict):
    """
    Make a single prediction from a dict of raw feature values.
    Handles encoding and scaling using the saved artefacts.
    Returns the raw prediction value.
    """
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    encoders = bundle["encoders"]
    features = bundle["feature_cols"]

    row = []
    for col in features:
        val = input_dict.get(col, "Unknown")
        if col in encoders:
            le = encoders[col]
            val_str = str(val).strip().title()
            if val_str in le.classes_:
                val = int(le.transform([val_str])[0])
            else:
                val = 0          # unseen label -> 0
        try:
            row.append(float(val))
        except (ValueError, TypeError):
            row.append(0.0)

    X = scaler.transform([row])
    pred = model.predict(X)[0]
    return pred

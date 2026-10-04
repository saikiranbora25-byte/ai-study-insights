# data_utils.py
# -----------------------------------------------------------------------------
# Utility functions for loading, cleaning, preprocessing, and feature
# engineering on the "How AI is Changing Life of Students (India) 2026"
# Kaggle dataset.  All column detection is DYNAMIC — the code prints column
# metadata and then adapts to whatever names are actually present.
# -----------------------------------------------------------------------------

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# -- 1. COLUMN-DISCOVERY HELPERS -----------------------------------------------

def _fuzzy_find(candidates: list[str], keywords: list[str]) -> str | None:
    """Return the first column whose lower-cased name contains ANY keyword."""
    for col in candidates:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Dynamically detect which DataFrame columns map to which semantic roles.
    Returns a dict: role -> actual_column_name  (value may be None if not found).
    """
    cols = df.columns.tolist()
    mapping = {
        "ai_tool":        _fuzzy_find(cols, ["ai tool", "tool name", "tool used", "ai_tool", "tools"]),
        "usage_hours":    _fuzzy_find(cols, ["usage hour", "hours", "time spent", "daily"]),
        "use_case":       _fuzzy_find(cols, ["use case", "purpose", "usage purpose", "primary use"]),
        "city":           _fuzzy_find(cols, ["city"]),
        "state":          _fuzzy_find(cols, ["state"]),
        "education_level":_fuzzy_find(cols, ["education level", "degree", "level of education", "study level"]),
        "academic_stream":_fuzzy_find(cols, ["stream", "branch", "field", "major", "department"]),
        "gpa_before":     _fuzzy_find(cols, ["gpa before", "cgpa before", "grade before", "marks before"]),
        "gpa_after":      _fuzzy_find(cols, ["gpa after", "cgpa after", "grade after", "marks after"]),
        "gpa_change":     _fuzzy_find(cols, ["gpa change", "cgpa change", "grade change", "improvement"]),
        "academic_impact":_fuzzy_find(cols, ["academic impact", "impact", "performance impact"]),
        "time_saved":     _fuzzy_find(cols, ["time saved", "hours saved"]),
        "ethics":         _fuzzy_find(cols, ["ethics", "ethical", "concern"]),
        "satisfaction":   _fuzzy_find(cols, ["satisfaction", "rating", "overall"]),
        "age":            _fuzzy_find(cols, ["age"]),
        "gender":         _fuzzy_find(cols, ["gender", "sex"]),
    }
    return mapping


def print_dataset_info(df: pd.DataFrame):
    """Print the diagnostic info requested in the project brief."""
    print("=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(df.columns.tolist())
    print("\n" + "=" * 60)
    print(f"SHAPE  ->  {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 60)
    print("\nDATATYPES")
    print(df.dtypes)
    print("\nMISSING VALUES")
    print(df.isnull().sum())
    print("=" * 60)


# -- 2. DATA LOADING -----------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV / Excel from *path*.  Falls back to generating a realistic
    synthetic dataset (5 000 rows) if the file is not found, so the app
    works end-to-end even without the Kaggle download.
    """
    if os.path.exists(path):
        ext = os.path.splitext(path)[-1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
        print(f"[INFO] Loaded dataset from: {path}")
    else:
        print(f"[WARN] File not found: {path} -> generating synthetic data")
        df = _generate_synthetic_data(5000)

    print_dataset_info(df)
    return df


def _generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset that mirrors the expected Kaggle
    columns so the entire pipeline works even without the real file.
    """
    rng = np.random.default_rng(42)

    ai_tools    = ["ChatGPT", "Google Gemini", "Copilot", "Claude", "Perplexity",
                   "Grammarly", "Quillbot", "Notion AI", "Wolfram Alpha", "DALL-E"]
    use_cases   = ["Assignment Help", "Exam Prep", "Research", "Coding", "Essay Writing",
                   "Language Learning", "Summarisation", "Doubt Clearing", "Project Work"]
    cities      = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune",
                   "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Bhopal"]
    states      = ["Maharashtra", "Delhi", "Karnataka", "Telangana", "Tamil Nadu",
                   "Maharashtra", "West Bengal", "Gujarat", "Rajasthan", "Uttar Pradesh",
                   "Punjab", "Madhya Pradesh"]
    edu_levels  = ["Undergraduate", "Postgraduate", "Diploma", "PhD"]
    streams     = ["Engineering", "Science", "Commerce", "Arts", "Medicine",
                   "Law", "Management", "Education"]
    genders     = ["Male", "Female", "Non-Binary"]
    impacts     = ["Highly Positive", "Positive", "Neutral", "Negative"]

    city_idx  = rng.integers(0, len(cities), n)
    tool_idx  = rng.integers(0, len(ai_tools), n)
    hours     = np.round(rng.uniform(0.5, 8.0, n), 1)
    gpa_before = np.round(rng.uniform(5.0, 9.5, n), 2)
    gpa_lift   = np.round(rng.uniform(-0.5, 1.5, n) + hours * 0.05, 2)
    gpa_after  = np.clip(np.round(gpa_before + gpa_lift, 2), 0, 10)

    df = pd.DataFrame({
        "Student_ID":         np.arange(1, n + 1),
        "Age":                rng.integers(17, 30, n),
        "Gender":             [genders[i] for i in rng.integers(0, len(genders), n)],
        "City":               [cities[i]  for i in city_idx],
        "State":              [states[i]  for i in city_idx],
        "Education_Level":    [edu_levels[i] for i in rng.integers(0, len(edu_levels), n)],
        "Academic_Stream":    [streams[i]    for i in rng.integers(0, len(streams), n)],
        "AI_Tool_Used":       [ai_tools[i]   for i in tool_idx],
        "Primary_Use_Case":   [use_cases[i]  for i in rng.integers(0, len(use_cases), n)],
        "Daily_Usage_Hours":  hours,
        "GPA_Before_AI":      gpa_before,
        "GPA_After_AI":       gpa_after,
        "GPA_Change":         np.round(gpa_after - gpa_before, 2),
        "Time_Saved_Hours_Per_Week": np.round(rng.uniform(1, 15, n), 1),
        "Academic_Impact":    [impacts[i] for i in rng.integers(0, len(impacts), n)],
        "Ethical_Concern":    rng.choice(["Yes", "No"], n),
        "Satisfaction_Score": np.round(rng.uniform(1, 5, n), 1),
    })

    # Introduce ~5 % missing values for realism
    for col in ["GPA_Before_AI", "GPA_After_AI", "Daily_Usage_Hours",
                "Time_Saved_Hours_Per_Week", "Satisfaction_Score"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


# -- 3. CLEANING & PREPROCESSING -----------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and strip whitespace from string columns."""
    df = df.copy()
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df


def preprocess_data(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Impute missing values:
      • Numeric  -> median
      • Categorical -> mode
    Then derive GPA_Change if the column is absent but both GPA cols exist.
    """
    df = df.copy()

    # Impute numeric
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Derive GPA_Change if missing
    gc = col_map.get("gpa_change")
    gb = col_map.get("gpa_before")
    ga = col_map.get("gpa_after")
    if gc is None and gb and ga:
        df["GPA_Change"] = df[ga] - df[gb]
        col_map["gpa_change"] = "GPA_Change"

    return df


# -- 4. FEATURE ENGINEERING ----------------------------------------------------

def engineer_features(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Create additional features useful for modelling."""
    df = df.copy()

    hours_col = col_map.get("usage_hours")
    if hours_col and hours_col in df.columns:
        df["Usage_Bucket"] = pd.cut(
            df[hours_col],
            bins=[0, 1, 2, 4, float("inf")],
            labels=["Low (<1h)", "Moderate (1-2h)", "High (2-4h)", "Very High (>4h)"],
        )

    gpa_b = col_map.get("gpa_before")
    gpa_a = col_map.get("gpa_after")
    if gpa_b and gpa_a and gpa_b in df.columns and gpa_a in df.columns:
        df["GPA_Improvement_Flag"] = (df[gpa_a] >= df[gpa_b]).astype(int)

    return df


# -- 5. ENCODING & SCALING -----------------------------------------------------

def encode_and_scale(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, task: str = "regression"
) -> tuple:
    """
    Encode categoricals with LabelEncoder, scale numerics with StandardScaler.
    Returns: X_train, X_test, y_train, y_test, scaler, encoders, feature_names
    """
    df_model = df[feature_cols + [target_col]].dropna().copy()

    encoders: dict[str, LabelEncoder] = {}
    for col in df_model.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    X = df_model[feature_cols].values
    y = df_model[target_col].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task == "classification" else None,
    )
    return X_train, X_test, y_train, y_test, scaler, encoders, feature_cols


# -- 6. CHOOSE TARGET & TASK ---------------------------------------------------

def determine_task(df: pd.DataFrame, col_map: dict) -> tuple[str, str]:
    """
    Decide whether to do regression or classification, and which column to
    use as the target.  Priority:
      regression   -> GPA_Change (numeric)
      classification -> Academic_Impact (categorical)
    Returns: (target_column, task_type)
    """
    gc = col_map.get("gpa_change")
    ai = col_map.get("academic_impact")

    if gc and gc in df.columns and pd.api.types.is_numeric_dtype(df[gc]):
        return gc, "regression"
    if ai and ai in df.columns:
        return ai, "classification"
    # Last resort — pick first numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        return num_cols[0], "regression"
    return df.columns[0], "classification"


def get_feature_columns(df: pd.DataFrame, col_map: dict, target_col: str) -> list[str]:
    """
    Build a sensible feature list: prefer the semantically-detected columns,
    then fall back to all non-target columns (excluding id-like columns).
    """
    preferred = [
        col_map.get("ai_tool"),
        col_map.get("usage_hours"),
        col_map.get("use_case"),
        col_map.get("city"),
        col_map.get("state"),
        col_map.get("education_level"),
        col_map.get("academic_stream"),
        col_map.get("age"),
        col_map.get("gender"),
        col_map.get("time_saved"),
        col_map.get("satisfaction"),
    ]
    # Keep only non-None and existing columns; drop target
    features = [c for c in preferred if c and c in df.columns and c != target_col]

    if not features:
        # Fall back: all columns minus target and id-like
        ignore = {target_col}
        for col in df.columns:
            if re.search(r"(id|index|unnamed)", col, re.I):
                ignore.add(col)
        features = [c for c in df.columns if c not in ignore]

    return features

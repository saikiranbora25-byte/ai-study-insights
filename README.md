# 🤖 AI Study Insights
### Analysis & Prediction of Indian Students' AI-Tool Usage and Its Academic Impact

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=saikiranbora25-byte/ai-study-insights&branch=main&mainModule=app.py)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/saikiranbora25-byte/ai-study-insights)

---

## 📌 Project Overview

This **Streamlit** web application analyses how Indian students use AI tools and predicts the academic impact using **Machine Learning**. It uses the Kaggle dataset *"How AI is Changing Life of Students (India) 2026"* (5,000 student records).

---

## 🗂️ Project Structure

```
DAV project/
├── app.py               # Main Streamlit app (7 pages)
├── data_utils.py        # Data loading, cleaning, preprocessing, feature engineering
├── models.py            # ML model training, evaluation, saving/loading
├── eda_plots.py         # 13 interactive Plotly EDA visualisations
├── requirements.txt     # Python dependencies
├── download_data.py     # Helper to download dataset via Kaggle API
├── data/
│   └── students_ai_india.csv   # ← place the Kaggle CSV here
└── models/
    └── best_model.pkl   # Auto-saved after first training run
```

---

## 📦 Requirements

- Python 3.11+
- pip (or conda)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Getting the Dataset

### Option A — Kaggle API (recommended)

1. Create a Kaggle account at https://www.kaggle.com  
2. Go to **Account → API → Create New Token** — downloads `kaggle.json`  
3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)  
4. Run the helper script:

```bash
python download_data.py
```

### Option B — Manual Download

1. Go to https://www.kaggle.com/datasets and search for *"How AI is Changing Life of Students India 2026"*  
2. Download the CSV  
3. Rename it to `students_ai_india.csv`  
4. Place it inside the `data/` folder (create it if needed)

### Option C — Synthetic Data (zero setup)

If no CSV is found in `data/`, the app **automatically generates a realistic synthetic dataset** (5 000 rows) that mirrors the expected columns. You can run the app immediately without downloading anything.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 📑 App Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, tech stack, navigation guide |
| 📂 Dataset | Raw data explorer, statistics, missing values, column map |
| 🔧 Preprocessing | Cleaning steps, feature engineering, engineered dataset preview |
| 📊 EDA | 13 interactive Plotly charts (AI tools, GPA, cities, ethics …) |
| 🤖 Model Comparison | Train & compare Linear/Logistic Regression + Random Forest |
| 🔮 Prediction | Enter details → predict GPA change or academic impact |
| 💡 Insights | Dynamic data-driven insights + recommendations |

---

## 🤖 Machine Learning

### Dynamic Task Detection

The app inspects the dataset and automatically chooses:

| Target Column | Task | Models Used |
|---|---|---|
| Numeric GPA Change / GPA After | **Regression** | Linear Regression + Random Forest Regressor |
| Categorical Academic Impact | **Classification** | Logistic Regression + Random Forest Classifier |

### Features Used (auto-detected)
`AI Tool Used`, `Daily Usage Hours`, `Primary Use Case`, `City`, `State`,
`Education Level`, `Academic Stream`, `Age`, `Gender`, `Time Saved`,
`Satisfaction Score`

### Best Model Persistence

The best model (by R² or Accuracy) is saved to `models/best_model.pkl` using **joblib** after the first training run. The Prediction page loads this file for inference.

---

## 📊 EDA Charts

1. AI Tools Distribution (Bar)  
2. Daily Usage Hours (Histogram)  
3. Usage by City (Bar — avg hours)  
4. Usage by State (Pie)  
5. Primary Use Cases (Treemap)  
6. GPA Before vs After AI (Dual Histogram)  
7. GPA Change Distribution (Violin)  
8. GPA Change by AI Tool (Box)  
9. Time Saved per Week (Bar)  
10. Ethics Concerns (Donut)  
11. Academic Impact Breakdown (Bar)  
12. Hours vs GPA Change (Scatter + trendline)  
13. Correlation Heatmap  

---

## 🔮 Prediction Page

Enter the following student details to get a prediction:

- AI Tool Used  
- Daily Usage Hours  
- Primary Use Case  
- City & State  
- Education Level  
- Academic Stream  
- Age, Gender, Satisfaction Score  

The app dynamically generates the correct input widgets based on the trained model's feature set.

---

## 📝 Notes

- **Dynamic column detection**: the code calls `detect_columns()` which fuzzy-matches column names, so it adapts automatically if the Kaggle dataset column names differ slightly from expectations.  
- On first load the app prints `df.columns`, `df.shape`, `df.dtypes`, and `df.isnull().sum()` to the console/terminal for debugging.  
- Models are cached with `@st.cache_resource` so training only happens once per session.  

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Web App | Streamlit 1.40 |
| Data Wrangling | Pandas 2.2, NumPy 1.26 |
| Visualisations | Plotly 5.24 |
| Machine Learning | Scikit-learn 1.5 |
| Model Persistence | Joblib 1.4 |

---

## 👤 Author

Built as a complete Data Analysis & Visualisation (DAV) mini-project.

---

*"AI is not replacing students — it is empowering them."*

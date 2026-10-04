# app.py
# -----------------------------------------------------------------------------
# Streamlit multi-page app:
#   Home | Dataset | Preprocessing | EDA | Model Comparison | Prediction | Insights
# Run:  streamlit run app.py
# -----------------------------------------------------------------------------

import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# -- local modules -------------------------------------------------------------
from data_utils import (
    load_data, clean_data, preprocess_data, engineer_features,
    detect_columns, encode_and_scale, determine_task, get_feature_columns,
)
from models import (
    build_models, train_models, evaluate_models,
    select_best, save_model, load_model, predict_single,
    get_feature_importance, get_classification_report, get_confusion_matrix,
)
import eda_plots as eda

# -- page config ---------------------------------------------------------------
st.set_page_config(
    page_title="AI Study Insights",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
  [data-testid="stSidebar"] {background: #12151e;}
  .main-title {
      font-size:2.6rem; font-weight:900; text-align:center;
      background: linear-gradient(90deg,#7c3aed,#2563eb,#06b6d4);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      margin-bottom:0.2rem;
  }
  .sub-title {
      font-size:1.05rem; text-align:center; color:#94a3b8;
      margin-bottom:1.5rem;
  }
  .metric-card {
      background:#1e2130; border-radius:12px; padding:1rem 1.5rem;
      border-left:4px solid #7c3aed;
  }
  .section-header {
      font-size:1.4rem; font-weight:700; color:#e2e8f0;
      border-bottom:2px solid #334155; padding-bottom:0.4rem;
      margin-top:1rem;
  }
  .badge {
      display:inline-block; background:#1e3a5f; color:#60a5fa;
      border-radius:20px; padding:2px 12px; font-size:0.8rem;
      margin:2px;
  }
  .stTabs [data-baseweb="tab"] {font-size:1rem; font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -- sidebar navigation --------------------------------------------------------
PAGES = [
    "🏠 Home",
    "📂 Dataset",
    "🔧 Preprocessing",
    "📊 EDA",
    "🤖 Model Comparison",
    "🔮 Prediction",
    "💡 Insights",
]

with st.sidebar:
    st.markdown("## 🤖 AI Study Insights")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Dataset: *How AI is Changing Life of Students (India) 2026*")
    st.caption("Models: Linear/Logistic Regression + Random Forest")

# -- data path -----------------------------------------------------------------
DATA_PATH = "data/students_ai_india.csv"

# -----------------------------------------------------------------------------
# SESSION-STATE CACHE  (compute once, reuse across pages)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & preparing data …")
def get_data():
    raw   = load_data(DATA_PATH)
    col_m = detect_columns(raw)
    clean = clean_data(raw)
    proc  = preprocess_data(clean, col_m)
    eng   = engineer_features(proc, col_m)
    return raw, clean, proc, eng, col_m


@st.cache_resource(show_spinner="Training ML models …")
def get_trained_models(data_hash: int):
    _, _, _, eng, col_m = get_data()
    target, task = determine_task(eng, col_m)
    feats        = get_feature_columns(eng, col_m, target)
    X_tr, X_te, y_tr, y_te, scaler, encoders, feat_names = encode_and_scale(
        eng, feats, target, task
    )
    mdls   = build_models(task)
    trained = train_models(mdls, X_tr, y_tr)
    results = evaluate_models(trained, X_te, y_te, task)
    best_nm  = select_best(results, task)
    best_mdl = trained[best_nm]
    path     = save_model(best_mdl, scaler, encoders, feat_names, target, task, col_m)
    return trained, results, best_nm, task, target, feats, scaler, encoders, X_te, y_te


# -----------------------------------------------------------------------------
# HELPER
# -----------------------------------------------------------------------------

def metric_card(label, value, delta=None, col=None):
    html = f"""<div class="metric-card">
        <p style="margin:0;color:#94a3b8;font-size:0.85rem">{label}</p>
        <p style="margin:0;font-size:1.6rem;font-weight:700;color:#e2e8f0">{value}</p>
        {"" if delta is None else f'<p style="margin:0;font-size:0.8rem;color:#34d399">{delta}</p>'}
    </div>"""
    target = col if col else st
    target.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-title">🤖 AI Study Insights</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Analysis & Prediction of Indian Students\' AI-Tool Usage and Its Academic Impact</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    raw, *_ = get_data()
    with col1:
        metric_card("📁 Total Records",  f"{len(raw):,}")
    with col2:
        metric_card("📋 Total Features", str(raw.shape[1]))
    with col3:
        metric_card("🏫 Dataset", "Kaggle — India 2026")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 Project Goals")
        st.markdown("""
- Understand how Indian students are using AI tools  
- Analyse usage patterns across cities, states & streams  
- Measure the impact of AI on GPA and academic performance  
- Build predictive models (Regression + Random Forest)  
- Provide an interactive prediction interface  
        """)
    with c2:
        st.markdown("### 📦 Tech Stack")
        st.markdown("""
| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Visuals | Plotly |
| ML | Scikit-learn |
| Persistence | Joblib |
        """)

    st.markdown("---")
    st.markdown("### 🗺️ App Navigation Guide")
    cols = st.columns(6)
    labels = [
        ("📂", "Dataset", "Raw data explorer"),
        ("🔧", "Preprocessing", "Cleaning & feature engineering"),
        ("📊", "EDA", "13 interactive charts"),
        ("🤖", "Model Comparison", "Train & compare models"),
        ("🔮", "Prediction", "Predict your own GPA"),
        ("💡", "Insights", "Key takeaways"),
    ]
    for col, (icon, name, desc) in zip(cols, labels):
        col.markdown(f"**{icon} {name}**  \n*{desc}*")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Dataset":
    st.markdown('<p class="section-header">📂 Dataset Explorer</p>', unsafe_allow_html=True)
    raw, clean, proc, eng, col_m = get_data()

    tab1, tab2, tab3 = st.tabs(["🔍 Preview", "📊 Statistics", "🗺️ Column Map"])

    with tab1:
        st.markdown("#### Raw Data (first 100 rows)")
        n_rows = st.slider("Rows to display", 5, min(500, len(raw)), 50)
        st.dataframe(raw.head(n_rows), use_container_width=True)
        st.caption(f"Shape: {raw.shape[0]} rows × {raw.shape[1]} columns")

    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(raw.describe(include="all").T, use_container_width=True)
        st.markdown("#### Missing Values")
        miss = raw.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["Pct"] = (miss["Missing"] / len(raw) * 100).round(2).astype(str) + " %"
        miss = miss[miss["Missing"] > 0]
        if miss.empty:
            st.success("✅ No missing values detected in the dataset.")
        else:
            st.dataframe(miss, use_container_width=True)

    with tab3:
        st.markdown("#### Detected Column Roles")
        rows = []
        for role, col_name in col_m.items():
            rows.append({"Semantic Role": role, "Detected Column": col_name or "❌ Not Found"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.markdown("#### All Column Names & Dtypes")
        dtype_df = pd.DataFrame({"Column": raw.columns, "Dtype": raw.dtypes.values})
        st.dataframe(dtype_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Preprocessing":
    st.markdown('<p class="section-header">🔧 Data Cleaning & Feature Engineering</p>',
                unsafe_allow_html=True)
    raw, clean, proc, eng, col_m = get_data()

    c1, c2, c3 = st.columns(3)
    metric_card("Raw rows",           f"{len(raw):,}",  col=c1)
    metric_card("After de-dup",       f"{len(clean):,}", col=c2)
    metric_card("Engineered features", str(eng.shape[1]), col=c3)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🧹 Cleaning Steps", "⚙️ Feature Engineering", "📋 Final Preview"])

    with tab1:
        steps = {
            "1. Duplicate Removal":     f"{len(raw) - len(clean)} rows removed",
            "2. Whitespace Stripping":  "All string columns normalised (strip + title-case)",
            "3. Numeric Imputation":    "Missing numerics -> column median",
            "4. Categorical Imputation":"Missing categoricals -> column mode",
            "5. GPA Change Derivation": "GPA_Change = GPA_After − GPA_Before (if absent)",
        }
        for step, detail in steps.items():
            st.markdown(f"**{step}** — `{detail}`")

    with tab2:
        st.markdown("""
| New Feature | Logic |
|---|---|
| `Usage_Bucket` | Cut daily hours into 4 bins: Low / Moderate / High / Very High |
| `GPA_Improvement_Flag` | 1 if GPA_After ≥ GPA_Before, else 0 |
| `GPA_Change` | Derived from GPA_After − GPA_Before if column absent |
        """)
        if "Usage_Bucket" in eng.columns:
            vc = eng["Usage_Bucket"].value_counts().reset_index()
            vc.columns = ["Bucket", "Count"]
            fig = px.bar(vc, x="Bucket", y="Count", color="Bucket",
                         title="Usage Bucket Distribution",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Engineered Dataset (first 100 rows)")
        st.dataframe(eng.head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<p class="section-header">📊 Exploratory Data Analysis</p>',
                unsafe_allow_html=True)
    raw, clean, proc, eng, col_m = get_data()

    eda_options = {
        "AI Tools Distribution":       lambda: eda.plot_ai_tools(eng, col_m),
        "Daily Usage Hours":            lambda: eda.plot_usage_hours(eng, col_m),
        "Usage by City":                lambda: eda.plot_usage_by_city(eng, col_m),
        "Usage by State":               lambda: eda.plot_usage_by_state(eng, col_m),
        "Primary Use Cases":            lambda: eda.plot_use_cases(eng, col_m),
        "GPA Before vs After AI":       lambda: eda.plot_gpa_before_after(eng, col_m),
        "GPA Change Distribution":      lambda: eda.plot_gpa_change(eng, col_m),
        "GPA Change by AI Tool":        lambda: eda.plot_gpa_by_tool(eng, col_m),
        "Time Saved per Week":          lambda: eda.plot_time_saved(eng, col_m),
        "Ethics Concerns":              lambda: eda.plot_ethics(eng, col_m),
        "Academic Impact Breakdown":    lambda: eda.plot_academic_impact(eng, col_m),
        "Hours vs GPA Change":          lambda: eda.plot_hours_vs_gpa(eng, col_m),
        "Correlation Heatmap":          lambda: eda.plot_correlation(eng),
    }

    selected = st.multiselect(
        "Choose charts to display:",
        list(eda_options.keys()),
        default=list(eda_options.keys())[:6],
    )

    if not selected:
        st.info("👆 Please select at least one chart from the list above.")
    else:
        for chart_name in selected:
            with st.expander(f"📌 {chart_name}", expanded=True):
                fig = eda_options[chart_name]()
                st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<p class="section-header">🤖 Model Training & Comparison</p>',
                unsafe_allow_html=True)
    raw, clean, proc, eng, col_m = get_data()

    with st.spinner("Training models — this may take a moment …"):
        trained, results, best_nm, task, target, feats, scaler, encoders, X_te, y_te = \
            get_trained_models(id(eng))

    st.success(f"✅ Task detected: **{task.upper()}**  |  Target column: `{target}`")
    st.info(f"🏆 Best model: **{best_nm}**")

    # Metric comparison table
    st.markdown("#### 📋 Model Performance Comparison")
    st.dataframe(results.set_index("Model"), use_container_width=True)

    # Bar chart comparison
    metric_col = "R2 Score" if task == "regression" else "Accuracy"
    fig = px.bar(
        results, x="Model", y=metric_col,
        color="Model", text=metric_col,
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title=f"Model Comparison — {metric_col}",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False,
                      plot_bgcolor="#0e1117", paper_bgcolor="#1a1d23",
                      font_color="#ffffff")
    st.plotly_chart(fig, use_container_width=True)

    # Features used
    with st.expander("📐 Features Used in Training"):
        st.write(feats)

    # Classification-specific: confusion matrix + report
    if task == "classification":
        st.markdown("---")
        st.markdown("#### 🔥 Confusion Matrix — Best Model")
        best_mdl = trained[best_nm]
        cm = get_confusion_matrix(best_mdl, X_te, y_te)
        target_le = encoders.get(target)
        labels = target_le.classes_.tolist() if target_le else None
        import plotly.figure_factory as ff
        z = cm.tolist()
        if labels:
            fig_cm = ff.create_annotated_heatmap(
                z, x=labels, y=labels,
                colorscale="Blues", showscale=True
            )
        else:
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
        fig_cm.update_layout(
            title="Confusion Matrix",
            plot_bgcolor="#0e1117", paper_bgcolor="#1a1d23", font_color="#ffffff"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("📄 Classification Report"):
            report = get_classification_report(best_mdl, X_te, y_te, target_le)
            st.text(report)

    # Feature importance
    st.markdown("---")
    st.markdown("#### 🌟 Feature Importance — Best Model")
    fi_df = get_feature_importance(trained[best_nm], feats)
    if fi_df is not None:
        fig_fi = px.bar(
            fi_df.head(15), x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis",
            title="Top Feature Importances",
        )
        fig_fi.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#1a1d23", font_color="#ffffff",
            coloraxis_showscale=False, height=420,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown('<p class="section-header">🔮 Predict Academic Impact</p>',
                unsafe_allow_html=True)
    raw, clean, proc, eng, col_m = get_data()

    with st.spinner("Loading trained model …"):
        trained, results, best_nm, task, target, feats, scaler, encoders, X_te, y_te = \
            get_trained_models(id(eng))

    bundle = load_model()
    if bundle is None:
        st.warning("Model file not found. Training now …")
        best_mdl = trained[best_nm]
        save_model(best_mdl, scaler, encoders, feats, target, task, col_m)
        bundle = load_model()

    if bundle is None:
        st.error("Could not load or save the model. Please check write permissions.")
        st.stop()

    st.info(f"Using **{best_nm}** · Task: **{task}** · Target: `{target}`")

    # -- dynamic input widgets ------------------------------------------------
    st.markdown("### 📝 Enter Student Details")
    inp = {}
    feat_cols = bundle["feature_cols"]
    enc       = bundle["encoders"]

    col_pairs = [feat_cols[i:i+2] for i in range(0, len(feat_cols), 2)]
    for pair in col_pairs:
        cols = st.columns(len(pair))
        for c_widget, feat in zip(cols, pair):
            with c_widget:
                if feat in enc:
                    options = enc[feat].classes_.tolist()
                    inp[feat] = st.selectbox(feat.replace("_", " "), options)
                else:
                    sample_vals = eng[feat].dropna() if feat in eng.columns else pd.Series([0])
                    min_v = float(sample_vals.min())
                    max_v = float(sample_vals.max())
                    mean_v = float(sample_vals.mean())
                    inp[feat] = st.number_input(
                        feat.replace("_", " "),
                        min_value=min_v, max_value=max_v,
                        value=mean_v, step=0.1,
                    )

    st.markdown("---")
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        pred = predict_single(bundle, inp)
        st.markdown("---")
        if task == "regression":
            delta_label = "📈 GPA will IMPROVE" if pred >= 0 else "📉 GPA may DECLINE"
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Predicted GPA Change", f"{pred:+.3f}", delta_label)
            with col_b:
                bar_val = min(max(pred, -2), 2)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "GPA Change"},
                    gauge={
                        "axis": {"range": [-2, 2]},
                        "bar":  {"color": "#7c3aed"},
                        "steps": [
                            {"range": [-2, 0],   "color": "#fee2e2"},
                            {"range": [0, 1],    "color": "#d1fae5"},
                            {"range": [1, 2],    "color": "#bbf7d0"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 4},
                            "thickness": 0.75,
                            "value": pred,
                        },
                    },
                ))
                fig.update_layout(paper_bgcolor="#1a1d23", font_color="#fff", height=260)
                st.plotly_chart(fig, use_container_width=True)

            if pred >= 1.0:
                st.success("🌟 Outstanding! Highly positive academic impact expected.")
            elif pred >= 0.3:
                st.success("✅ Good improvement expected in academic performance.")
            elif pred >= 0:
                st.info("ℹ️ Marginal improvement expected.")
            else:
                st.warning("⚠️ Slight decline predicted. Consider moderating AI usage or diversifying use cases.")

        else:
            # Classification
            label_le = enc.get(target)
            if label_le is not None:
                try:
                    pred_label = label_le.inverse_transform([int(round(pred))])[0]
                except Exception:
                    pred_label = str(pred)
            else:
                pred_label = str(pred)
            st.metric("Predicted Academic Impact", pred_label)
            colour_map = {
                "Highly Positive": "success",
                "Positive":        "success",
                "Neutral":         "info",
                "Negative":        "warning",
            }
            msg_fn = getattr(st, colour_map.get(pred_label, "info"))
            msg_fn(f"Predicted outcome: **{pred_label}**")

        # Show input summary
        with st.expander("📋 Input Summary"):
            st.json(inp)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.markdown('<p class="section-header">💡 Key Insights & Recommendations</p>',
                unsafe_allow_html=True)
    raw, clean, proc, eng, col_m = get_data()

    # Dynamic insight generation
    insights = []

    gc = col_m.get("gpa_change")
    if gc and gc in eng.columns:
        avg_change = eng[gc].mean()
        pct_improved = (eng[gc] > 0).mean() * 100
        insights.append(("📈 GPA Impact",
                          f"Average GPA change: **{avg_change:+.3f}**  \n"
                          f"{pct_improved:.1f}% of students showed improvement after adopting AI tools."))

    hours_col = col_m.get("usage_hours")
    if hours_col and hours_col in eng.columns:
        avg_h = eng[hours_col].mean()
        insights.append(("⏱️ Usage Hours",
                          f"Average daily AI usage: **{avg_h:.1f} hours**"))

    time_col = col_m.get("time_saved")
    if time_col and time_col in eng.columns:
        avg_saved = eng[time_col].mean()
        insights.append(("⏳ Time Saved",
                          f"On average students save **{avg_saved:.1f} hours/week** using AI tools."))

    ai_col = col_m.get("ai_tool")
    if ai_col and ai_col in eng.columns:
        top_tool = eng[ai_col].value_counts().idxmax()
        insights.append(("🥇 Top AI Tool",
                          f"The most popular AI tool is **{top_tool}**."))

    eth_col = col_m.get("ethics")
    if eth_col and eth_col in eng.columns:
        pct_eth = (eng[eth_col].astype(str).str.lower().isin(["yes", "true", "1"])).mean() * 100
        insights.append(("⚖️ Ethics Concerns",
                          f"**{pct_eth:.1f}%** of students have ethical concerns about AI tool use."))

    city_col = col_m.get("city")
    if city_col and city_col in eng.columns and hours_col and hours_col in eng.columns:
        top_city = eng.groupby(city_col)[hours_col].mean().idxmax()
        insights.append(("🏙️ Top City",
                          f"Students in **{top_city}** use AI tools the most on average."))

    if insights:
        for icon_label, text in insights:
            st.markdown(f"**{icon_label}**")
            st.markdown(f"> {text}")
            st.markdown("")
    else:
        st.info("No dynamic insights could be generated. Please check your dataset columns.")

    st.markdown("---")
    st.markdown("### 📌 General Recommendations")
    recs = [
        "🎯 Use AI tools for **Exam Prep and Research** — the highest GPA impact use cases.",
        "⏰ Keep daily AI usage between **2-4 hours** (High bucket) for optimal results.",
        "🔄 Combine AI with traditional study for **balanced learning**.",
        "⚖️ Always verify AI-generated content to avoid academic integrity issues.",
        "📚 Students in **Engineering and Science** streams show the highest AI adoption rates.",
        "🤝 Institutions should provide **AI literacy workshops** for equitable access.",
    ]
    for rec in recs:
        st.markdown(f"- {rec}")

    st.markdown("---")
    st.markdown("### 🔮 Model Performance Summary")
    with st.spinner("Loading model results …"):
        _, results, best_nm, task, *_ = get_trained_models(id(eng))
    st.dataframe(results.set_index("Model"), use_container_width=True)
    st.success(f"🏆 Best model: **{best_nm}** (Task: {task})")

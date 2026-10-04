# eda_plots.py
# -----------------------------------------------------------------------------
# All EDA visualisations returned as Plotly figures so they can be embedded
# inside Streamlit with st.plotly_chart().
# Every function accepts the DataFrame + the col_map dict and falls back
# gracefully if an expected column is absent.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = px.colors.qualitative.Vivid
BG      = "#0e1117"
PAPER   = "#1a1d23"
FONT    = "#ffffff"
_LAYOUT = dict(
    plot_bgcolor  = BG,
    paper_bgcolor = PAPER,
    font_color    = FONT,
    margin        = dict(l=30, r=30, t=50, b=30),
)


def _no_data(title: str):
    fig = go.Figure()
    fig.add_annotation(text="Column not found in dataset",
                       xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, color="#aaaaaa"))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


# -- 1. AI TOOLS ---------------------------------------------------------------

def plot_ai_tools(df: pd.DataFrame, col_map: dict):
    col = col_map.get("ai_tool")
    if col is None or col not in df.columns:
        return _no_data("AI Tools Used")
    vc = df[col].value_counts().reset_index()
    vc.columns = ["AI Tool", "Count"]
    fig = px.bar(
        vc, x="Count", y="AI Tool", orientation="h",
        color="Count", color_continuous_scale="Plasma",
        title="📊 Distribution of AI Tools Used by Students",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**_LAYOUT, coloraxis_showscale=False, height=420)
    return fig


# -- 2. USAGE HOURS DISTRIBUTION -----------------------------------------------

def plot_usage_hours(df: pd.DataFrame, col_map: dict):
    col = col_map.get("usage_hours")
    if col is None or col not in df.columns:
        return _no_data("Daily Usage Hours")
    fig = px.histogram(
        df, x=col, nbins=40, color_discrete_sequence=[PALETTE[2]],
        title="⏱️ Distribution of Daily AI Usage Hours",
        labels={col: "Daily Usage Hours"},
    )
    fig.update_layout(**_LAYOUT)
    return fig


# -- 3. USAGE BY CITY ---------------------------------------------------------

def plot_usage_by_city(df: pd.DataFrame, col_map: dict):
    city_col  = col_map.get("city")
    hours_col = col_map.get("usage_hours")
    if city_col is None or city_col not in df.columns:
        return _no_data("Usage by City")
    if hours_col and hours_col in df.columns:
        agg = df.groupby(city_col)[hours_col].mean().reset_index()
        agg.columns = ["City", "Avg Daily Hours"]
        fig = px.bar(
            agg.sort_values("Avg Daily Hours", ascending=False),
            x="City", y="Avg Daily Hours",
            color="Avg Daily Hours", color_continuous_scale="Sunset",
            title="🏙️ Average Daily AI Usage Hours by City",
        )
    else:
        vc = df[city_col].value_counts().reset_index()
        vc.columns = ["City", "Count"]
        fig = px.bar(vc, x="City", y="Count",
                     color="Count", color_continuous_scale="Sunset",
                     title="🏙️ Student Count by City")
    fig.update_layout(**_LAYOUT, coloraxis_showscale=False, xaxis_tickangle=-35)
    return fig


# -- 4. USAGE BY STATE --------------------------------------------------------

def plot_usage_by_state(df: pd.DataFrame, col_map: dict):
    state_col = col_map.get("state")
    if state_col is None or state_col not in df.columns:
        return _no_data("Usage by State")
    vc = df[state_col].value_counts().reset_index()
    vc.columns = ["State", "Count"]
    fig = px.pie(
        vc, names="State", values="Count",
        color_discrete_sequence=PALETTE,
        title="🗺️ AI Usage Distribution by State",
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(**_LAYOUT)
    return fig


# -- 5. USE CASES -------------------------------------------------------------

def plot_use_cases(df: pd.DataFrame, col_map: dict):
    col = col_map.get("use_case")
    if col is None or col not in df.columns:
        return _no_data("Use Cases")
    vc = df[col].value_counts().reset_index()
    vc.columns = ["Use Case", "Count"]
    fig = px.treemap(
        vc, path=["Use Case"], values="Count",
        color="Count", color_continuous_scale="RdBu",
        title="📚 Primary Use Cases of AI Tools",
    )
    fig.update_layout(**_LAYOUT)
    return fig


# -- 6. GPA BEFORE vs AFTER ---------------------------------------------------

def plot_gpa_before_after(df: pd.DataFrame, col_map: dict):
    gb = col_map.get("gpa_before")
    ga = col_map.get("gpa_after")
    if not gb or gb not in df.columns or not ga or ga not in df.columns:
        return _no_data("GPA Before vs After AI")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("GPA Before AI", "GPA After AI"))
    fig.add_trace(go.Histogram(x=df[gb], nbinsx=30, name="Before",
                               marker_color=PALETTE[0], opacity=0.85), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[ga], nbinsx=30, name="After",
                               marker_color=PALETTE[1], opacity=0.85), row=1, col=2)
    fig.update_layout(title_text="📈 GPA Distribution: Before vs After AI Adoption",
                      showlegend=False, **_LAYOUT)
    return fig


# -- 7. GPA CHANGE ------------------------------------------------------------

def plot_gpa_change(df: pd.DataFrame, col_map: dict):
    col = col_map.get("gpa_change")
    if col is None or col not in df.columns:
        return _no_data("GPA Change")
    fig = px.violin(
        df, y=col, box=True, points="outliers",
        color_discrete_sequence=[PALETTE[3]],
        title="🎻 GPA Change Distribution",
        labels={col: "GPA Change"},
    )
    fig.update_layout(**_LAYOUT)
    return fig


# -- 8. TIME SAVED ------------------------------------------------------------

def plot_time_saved(df: pd.DataFrame, col_map: dict):
    col = col_map.get("time_saved")
    if col is None or col not in df.columns:
        return _no_data("Time Saved")
    ai_col = col_map.get("ai_tool")
    if ai_col and ai_col in df.columns:
        agg = df.groupby(ai_col)[col].mean().reset_index()
        agg.columns = ["AI Tool", "Avg Hours Saved/Week"]
        fig = px.bar(
            agg.sort_values("Avg Hours Saved/Week", ascending=False),
            x="AI Tool", y="Avg Hours Saved/Week",
            color="Avg Hours Saved/Week", color_continuous_scale="Teal",
            title="⏳ Average Hours Saved per Week by AI Tool",
        )
    else:
        fig = px.histogram(df, x=col, nbins=30,
                           color_discrete_sequence=[PALETTE[5]],
                           title="⏳ Distribution of Time Saved per Week")
    fig.update_layout(**_LAYOUT, coloraxis_showscale=False, xaxis_tickangle=-35)
    return fig


# -- 9. ETHICS CONCERNS -------------------------------------------------------

def plot_ethics(df: pd.DataFrame, col_map: dict):
    col = col_map.get("ethics")
    if col is None or col not in df.columns:
        return _no_data("Ethics Concerns")
    vc = df[col].value_counts().reset_index()
    vc.columns = ["Response", "Count"]
    fig = px.pie(
        vc, names="Response", values="Count",
        color_discrete_sequence=PALETTE,
        title="⚖️ Ethical Concerns About AI Tool Usage",
        hole=0.4,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(**_LAYOUT)
    return fig


# -- 10. ACADEMIC IMPACT ------------------------------------------------------

def plot_academic_impact(df: pd.DataFrame, col_map: dict):
    col = col_map.get("academic_impact")
    if col is None or col not in df.columns:
        return _no_data("Academic Impact")
    vc = df[col].value_counts().reset_index()
    vc.columns = ["Impact", "Count"]
    fig = px.bar(
        vc, x="Impact", y="Count",
        color="Impact", color_discrete_sequence=PALETTE,
        title="🎓 Academic Impact Levels",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**_LAYOUT, showlegend=False)
    return fig


# -- 11. GPA CHANGE BY AI TOOL (BOX) -----------------------------------------

def plot_gpa_by_tool(df: pd.DataFrame, col_map: dict):
    gc  = col_map.get("gpa_change")
    ait = col_map.get("ai_tool")
    if not gc or gc not in df.columns or not ait or ait not in df.columns:
        return _no_data("GPA Change by AI Tool")
    fig = px.box(
        df, x=ait, y=gc,
        color=ait, color_discrete_sequence=PALETTE,
        title="🔬 GPA Change by AI Tool",
        labels={gc: "GPA Change", ait: "AI Tool"},
    )
    fig.update_layout(**_LAYOUT, showlegend=False, xaxis_tickangle=-35)
    return fig


# -- 12. CORRELATION HEATMAP --------------------------------------------------

def plot_correlation(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return _no_data("Correlation Heatmap")
    corr = num.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r",
        title="🔗 Feature Correlation Heatmap",
    )
    fig.update_layout(**_LAYOUT)
    return fig


# -- 13. USAGE HOURS vs GPA CHANGE (SCATTER) ----------------------------------

def plot_hours_vs_gpa(df: pd.DataFrame, col_map: dict):
    h  = col_map.get("usage_hours")
    gc = col_map.get("gpa_change")
    at = col_map.get("ai_tool")
    if not h or h not in df.columns or not gc or gc not in df.columns:
        return _no_data("Hours vs GPA Change")
    kwargs = dict(color=at) if at and at in df.columns else {}
    fig = px.scatter(
        df, x=h, y=gc, opacity=0.5,
        color_discrete_sequence=PALETTE,
        trendline="ols",
        title="📉 Daily Usage Hours vs GPA Change",
        labels={h: "Daily Usage Hours", gc: "GPA Change"},
        **kwargs,
    )
    fig.update_layout(**_LAYOUT)
    return fig

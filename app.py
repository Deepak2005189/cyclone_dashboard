"""
Cyclone Preheater Abnormal Operation Detection  -  Streamlit dashboard
Algo8 assignment: "Using Python and any algorithm of your choice, highlight
time periods where abnormal operation can be observed."

Run:  streamlit run app.py
All heavy computation lives in analysis.py; this file is only the user interface.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as an

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Cyclone Preheater Abnormal Operation Detection",
    page_icon="🏭",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV = DATA_DIR / "Data (1).csv"

SENSITIVITY_OPTIONS = {"0.1%": 0.001, "0.5%": 0.005, "1%": 0.01, "2%": 0.02, "5%": 0.05}
DURATION_OPTIONS = {"15 minutes": 15, "30 minutes": 30, "60 minutes": 60, "120 minutes": 120}
MAX_SHADED_PERIODS = 500   # shaded rectangles drawn on the timeline (markers show all)

# Colour palette (industrial monitoring look)
C_NORMAL = "#4cc9f0"
C_ANOM = "#ff4d4d"
C_ISOLATED = "#ffb703"
C_PERIOD = "#ff8a00"
C_SHADE = "rgba(255,138,0,0.30)"
C_THRESHOLD = "#ffd166"
C_SCORE = "#7aa2ff"

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
.hero {background: linear-gradient(120deg,#0f1b33 0%,#16294d 60%,#3a2410 100%);
       border: 1px solid #263a63; border-left: 6px solid #ff8a00; border-radius: 10px;
       padding: 20px 26px; margin-bottom: 14px;}
.hero h1 {margin: 0; font-size: 1.9rem; letter-spacing: .3px; color: #f4f7fd;}
.hero p {margin: 4px 0 0 0; color: #9fb3d9; font-size: 1.0rem;}
.kpi {background: #121c33; border: 1px solid #243455; border-top: 3px solid #ff8a00;
      border-radius: 8px; padding: 12px 16px; min-height: 104px;}
.kpi-label {font-size: .74rem; text-transform: uppercase; letter-spacing: 1.1px; color: #8fa5cf;}
.kpi-value {font-size: 1.75rem; font-weight: 700; color: #ffffff; line-height: 1.25;}
.kpi-value.small {font-size: 1.05rem; padding-top: 6px;}
.kpi-sub {font-size: .78rem; color: #8fa5cf;}
.sec {font-size: 1.15rem; font-weight: 700; color: #f4f7fd; margin: 26px 0 6px 0;
      padding-bottom: 4px; border-bottom: 1px solid #243455;}
.sec span {color: #ff8a00; margin-right: 8px;}
.note {background:#101a30; border-left:4px solid #4cc9f0; padding:10px 14px;
       border-radius:6px; color:#c9d6f2; font-size:.93rem;}
</style>
"""


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def kpi(label: str, value: str, sub: str = "", small: bool = False) -> str:
    cls = "kpi-value small" if small else "kpi-value"
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="{cls}">{value}</div><div class="kpi-sub">{sub}</div></div>')


def section(number: int, title: str) -> None:
    st.markdown(f'<div class="sec"><span>{number:02d}</span>{title}</div>', unsafe_allow_html=True)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    """st.dataframe stretched to full width on both old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def fmt_ts(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %Y %H:%M")


def style_figure(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=dict(text=title, x=0.01) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=25, t=55 if title else 25, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    fig.update_xaxes(gridcolor="#1f2d4d")
    fig.update_yaxes(gridcolor="#1f2d4d")
    return fig


# --------------------------------------------------------------------------
# Cached heavy steps (data -> model -> scores -> CSV bytes)
# Arguments starting with "_" are not hashed by Streamlit; `data_key` is the cache key.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning the dataset ...")
def cached_load(path: str | None, file_bytes: bytes | None, mtime: float):
    """load_data() + clean_data(); `mtime` only invalidates the cache when the file changes."""
    source = io.BytesIO(file_bytes) if file_bytes is not None else path
    raw = an.load_data(source)
    return an.clean_data(raw)


@st.cache_resource(show_spinner="Training Isolation Forest on the full dataset ...")
def cached_model(data_key: str, _clean: pd.DataFrame):
    """The model is fitted ONCE per dataset - never when a chart or filter changes."""
    X, medians = an.prepare_features(_clean)
    model, raw_scores = an.train_model(X)
    return {"model": model, "raw_scores": raw_scores, "medians": medians}


@st.cache_resource(show_spinner="Scoring records ...")
def cached_scored(data_key: str, contamination: float, _clean: pd.DataFrame, _bundle: dict):
    """Apply the chosen sensitivity to the cached model scores (fast, no re-training)."""
    score, flag, offset = an.detect_anomalies(_bundle["raw_scores"], contamination)
    scored = _clean.copy()
    scored["raw_score"] = _bundle["raw_scores"]
    scored["anomaly_score"] = score
    scored["is_anomaly"] = flag
    return scored, offset


@st.cache_data(show_spinner=False)
def cached_csv_clean(data_key: str, _clean: pd.DataFrame) -> bytes:
    return an.to_csv_bytes(_clean)


@st.cache_data(show_spinner=False)
def cached_csv_scored(data_key: str, contamination: float, _scored: pd.DataFrame) -> bytes:
    cols = [an.TIME_COL] + an.FEATURES + ["anomaly_score", "is_anomaly"]
    return an.to_csv_bytes(_scored[cols])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def build_timeline_figure(view: pd.DataFrame, periods: pd.DataFrame, interval: float) -> go.Figure:
    """Anomaly score over time with threshold line + shaded abnormal periods."""
    idx = an.downsample_indices(view["anomaly_score"].to_numpy(), 8000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["time"].iloc[idx], y=view["anomaly_score"].iloc[idx], mode="lines",
        name="Anomaly score", line=dict(color=C_SCORE, width=1),
        hovertemplate="%{x|%d %b %Y %H:%M}<br>Anomaly score: %{y:.4f}<extra></extra>",
    ))

    if len(periods):
        step = pd.Timedelta(minutes=interval)
        shaded = periods.head(MAX_SHADED_PERIODS)
        fig.update_layout(shapes=[
            dict(type="rect", xref="x", yref="paper", y0=0, y1=1,
                 x0=s.isoformat(), x1=(e + step).isoformat(),
                 fillcolor=C_SHADE, line=dict(width=0), layer="below")
            for s, e in zip(shaded["Start Time"], shaded["End Time"])
        ])
        custom = [
            [int(r), fmt_ts(e), an.fmt_duration(d), int(n), f"{a:.4f}", f"{m:.4f}"]
            for r, e, d, n, a, m in zip(
                periods["Rank"], periods["End Time"], periods["Duration"], periods["Records"],
                periods["Average Anomaly Score"], periods["Maximum Anomaly Score"])
        ]
        fig.add_trace(go.Scattergl(
            x=periods["Start Time"], y=periods["Maximum Anomaly Score"], mode="markers",
            name="Abnormal period (start)",
            marker=dict(color=C_PERIOD, size=9, symbol="diamond", line=dict(color="white", width=0.6)),
            customdata=custom,
            hovertemplate=("<b>Abnormal period #%{customdata[0]}</b><br>Start: %{x|%d %b %Y %H:%M}"
                           "<br>End: %{customdata[1]}<br>Duration: %{customdata[2]}"
                           "<br>Records: %{customdata[3]}<br>Avg score: %{customdata[4]}"
                           "<br>Max score: %{customdata[5]}<extra></extra>"),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=C_THRESHOLD,
                  annotation_text="Anomaly threshold (score = 0)", annotation_position="top left")
    style_figure(fig, 470, "Anomaly Score Timeline")
    fig.update_yaxes(title="Anomaly score (higher = more abnormal)")
    fig.update_xaxes(title="Time", rangeslider=dict(visible=True, thickness=0.07))
    return fig


def build_variable_figure(view: pd.DataFrame, col: str, member: np.ndarray) -> go.Figure:
    """One sensor over time: normal line + highlighted anomalous observations."""
    y = view[col].to_numpy()
    idx = an.downsample_indices(y, 6000)
    status = np.where(view["is_anomaly"].to_numpy(), "Anomalous", "Normal")
    score = view["anomaly_score"].to_numpy()
    hover = ("%{x|%d %b %Y %H:%M}<br>Value: %{y:.3f}<br>Anomaly score: %{customdata:.4f}"
             "<br>Status: %{text}<extra></extra>")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["time"].iloc[idx], y=y[idx], mode="lines", name="Normal data",
        line=dict(color=C_NORMAL, width=1), customdata=score[idx], text=status[idx],
        hovertemplate=hover,
    ))
    anom = view["is_anomaly"].to_numpy() & ~np.isnan(y)
    for mask, name, color, size in (
        (anom & member, "Anomalous (in sustained period)", C_ANOM, 5),
        (anom & ~member, "Anomalous (isolated)", C_ISOLATED, 4),
    ):
        if mask.any():
            fig.add_trace(go.Scattergl(
                x=view["time"][mask], y=y[mask], mode="markers", name=name,
                marker=dict(color=color, size=size), customdata=score[mask], text=status[mask],
                hovertemplate=hover,
            ))
    style_figure(fig, 430, an.PRETTY[col])
    fig.update_xaxes(title="Time", rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title=an.PRETTY[col])
    return fig


def build_top_events_figure(periods: pd.DataFrame) -> go.Figure:
    top = periods.head(10)
    labels = [f"#{r}  {fmt_ts(s)} → {fmt_ts(e)}"
              for r, s, e in zip(top["Rank"], top["Start Time"], top["End Time"])]
    custom = [[fmt_ts(s), fmt_ts(e), an.fmt_duration(d), f"{a:.4f}", f"{m:.4f}"]
              for s, e, d, a, m in zip(top["Start Time"], top["End Time"], top["Duration"],
                                       top["Average Anomaly Score"], top["Maximum Anomaly Score"])]
    fig = go.Figure(go.Bar(
        x=top["Average Anomaly Score"], y=labels, orientation="h",
        marker=dict(color=top["Average Anomaly Score"], colorscale="YlOrRd", showscale=False),
        customdata=custom,
        hovertemplate=("Start: %{customdata[0]}<br>End: %{customdata[1]}<br>Duration: %{customdata[2]}"
                       "<br>Average score: %{customdata[3]}<br>Maximum score: %{customdata[4]}<extra></extra>"),
    ))
    style_figure(fig, 440, "Top 10 Most Severe Abnormal Periods")
    fig.update_yaxes(title="Abnormal Period", autorange="reversed")   # rank 1 at the top
    fig.update_xaxes(title="Average Anomaly Score")
    return fig


def build_event_figure(view: pd.DataFrame, period: pd.Series, pad_records: int, interval: float) -> go.Figure:
    """Six stacked, x-synchronised charts around one abnormal period."""
    a = max(0, int(period["start_pos"]) - pad_records)
    b = min(len(view), int(period["end_pos"]) + 1 + pad_records)
    win = view.iloc[a:b]
    x0 = pd.Timestamp(period["Start Time"]).to_pydatetime()
    x1 = (pd.Timestamp(period["End Time"]) + pd.Timedelta(minutes=interval)).to_pydatetime()

    fig = make_subplots(rows=len(an.FEATURES), cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=[an.PRETTY[f] for f in an.FEATURES])
    for i, col in enumerate(an.FEATURES, start=1):
        y = win[col].to_numpy()
        idx = an.downsample_indices(y, 3000)
        fig.add_trace(go.Scatter(
            x=win["time"].iloc[idx], y=y[idx], mode="lines", showlegend=False,
            line=dict(color=C_NORMAL, width=1.3),
            hovertemplate="%{x|%d %b %Y %H:%M}<br>%{y:.3f}<extra>" + an.PRETTY[col] + "</extra>",
        ), row=i, col=1)
        flagged = win["is_anomaly"].to_numpy() & ~np.isnan(y)
        if flagged.any():
            fig.add_trace(go.Scatter(
                x=win["time"][flagged], y=y[flagged], mode="markers", showlegend=False,
                marker=dict(color=C_ANOM, size=4), hoverinfo="skip",
            ), row=i, col=1)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=C_SHADE, line_width=0, layer="below", row=i, col=1)

    fig.update_layout(template="plotly_dark", height=1000, hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=60, r=25, t=45, b=40))
    fig.update_xaxes(gridcolor="#1f2d4d")
    fig.update_yaxes(gridcolor="#1f2d4d")
    for ann in fig.layout.annotations:           # subplot titles
        ann.update(font=dict(size=13, color="#c9d6f2"), x=0, xanchor="left")
    return fig


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def executive_interpretation(periods: pd.DataFrame, n_anom: int, rate: float, min_minutes: int) -> str:
    """Auto-generated, data-driven summary (no hard-coded dates)."""
    if not len(periods):
        return (f"No sustained abnormal operating period (≥ {min_minutes} minutes) was detected for the "
                f"selected settings. {n_anom:,} individual records ({rate:.2f}%) were statistically unusual "
                "but did not persist long enough to form a period. Try a higher sensitivity or a shorter "
                "minimum duration to examine shorter deviations.")
    top, lng = an.most_severe_period(periods), an.longest_period(periods)
    text = (f"{len(periods):,} sustained abnormal operating period(s) were detected. The highest-severity "
            f"period occurred between {fmt_ts(top['Start Time'])} and {fmt_ts(top['End Time'])}, lasting "
            f"{an.fmt_duration(top['Duration'])}. The longest period ran from {fmt_ts(lng['Start Time'])} to "
            f"{fmt_ts(lng['End Time'])} ({an.fmt_duration(lng['Duration'])}). ")
    return text + ("These are statistically unusual multivariate operating conditions that require "
                   "operational investigation; they are not confirmed equipment failures.")


# --------------------------------------------------------------------------
# Dashboard sections
# --------------------------------------------------------------------------
def section_summary(view, periods, min_minutes, start_d, end_d):
    n = len(view)
    n_anom = int(view["is_anomaly"].sum())
    rate = 100.0 * n_anom / n if n else 0.0

    cols = st.columns(5)
    cards = [
        ("Total Records", f"{n:,}", "5-minute process samples" if abs(INTERVAL - 5) < 0.01 else f"{INTERVAL:g}-minute samples"),
        ("Analysis Period", f"{start_d:%d %b %Y} → {end_d:%d %b %Y}", f"{(end_d - start_d).days + 1:,} days"),
        ("Anomalous Records", f"{n_anom:,}", "statistically unusual records"),
        ("Abnormal Periods", f"{len(periods):,}", f"sustained ≥ {min_minutes} min"),
        ("Anomaly Rate", f"{rate:.2f}%", "share of records flagged"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(kpi(label, value, sub, small=(label == "Analysis Period")), unsafe_allow_html=True)

    section(1, "Executive Summary")
    top, lng = an.most_severe_period(periods), an.longest_period(periods)
    c1, c2 = st.columns(2)
    if top is not None:
        c1.markdown(kpi("Most Severe Abnormal Period",
                        f"{fmt_ts(top['Start Time'])} → {fmt_ts(top['End Time'])}",
                        f"Duration {an.fmt_duration(top['Duration'])} · avg score {top['Average Anomaly Score']:.3f}",
                        small=True), unsafe_allow_html=True)
        c2.markdown(kpi("Longest Abnormal Period",
                        f"{fmt_ts(lng['Start Time'])} → {fmt_ts(lng['End Time'])}",
                        f"Duration {an.fmt_duration(lng['Duration'])} · {int(lng['Records'])} records",
                        small=True), unsafe_allow_html=True)
    else:
        c1.markdown(kpi("Most Severe Abnormal Period", "None detected", "", small=True), unsafe_allow_html=True)
        c2.markdown(kpi("Longest Abnormal Period", "None detected", "", small=True), unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f'<div class="note">{executive_interpretation(periods, n_anom, rate, min_minutes)}</div>',
                unsafe_allow_html=True)


def section_timeline(view, periods, interval):
    section(2, "Anomaly Score Timeline")
    st.plotly_chart(build_timeline_figure(view, periods, interval), key="timeline")
    extra = ""
    if len(periods) > MAX_SHADED_PERIODS:
        extra = f" Shading is drawn for the {MAX_SHADED_PERIODS} most severe periods; diamonds mark all {len(periods):,}."
    st.caption("Orange shaded bands = sustained abnormal periods. Diamonds mark each period's start at its "
               "peak score (periods are narrow when viewing years of data - zoom or drag the range slider)." + extra)


def section_periods(periods):
    section(3, "Detected Abnormal Operating Periods")
    st.markdown("**Detected Abnormal Operating Periods**")
    if not len(periods):
        st.info("No sustained abnormal periods for the current settings.")
        return
    table = periods[["Rank", "Start Time", "End Time", "Duration", "Records", "duration_min",
                     "Average Anomaly Score", "Maximum Anomaly Score"]].copy()
    table["Duration"] = table["Duration"].map(an.fmt_duration)
    table = table.rename(columns={"duration_min": "Duration (min)"})
    show_table(table, column_config={
        "Start Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "End Time": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        "Duration (min)": st.column_config.NumberColumn(format="%.0f"),
        "Average Anomaly Score": st.column_config.NumberColumn(format="%.4f"),
        "Maximum Anomaly Score": st.column_config.NumberColumn(format="%.4f"),
    })
    st.caption("Click a column header to sort; use the table toolbar (top-right) to search or download.")
    st.download_button("Download Abnormal Periods CSV",
                       an.to_csv_bytes(an.periods_for_export(periods)),
                       file_name="abnormal_periods.csv", mime="text/csv", key="dl_periods_main")


def section_top_events(periods):
    section(4, "Top Abnormal Events")
    if not len(periods):
        st.info("No abnormal periods to rank.")
        return
    st.plotly_chart(build_top_events_figure(periods), key="top_events")


def section_variables(view, member):
    section(5, "Process Variable Overview")
    col = st.selectbox("Process variable", an.FEATURES, format_func=lambda c: an.PRETTY[c], key="var_select")
    st.plotly_chart(build_variable_figure(view, col, member), key="var_chart")
    st.caption("Blue line = full-history signal (min/max downsampled for display only). "
               "Red = anomalous records inside sustained periods; amber = isolated anomalous records.")


def section_event(view, periods, interval):
    section(6, "Sensor Behavior During Selected Event")
    if not len(periods):
        st.info("Select settings that produce at least one abnormal period to inspect an event.")
        return
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox(
        "Abnormal period", list(range(len(periods))), key="event_select",
        format_func=lambda i: (f"#{int(periods.iloc[i]['Rank'])} · {fmt_ts(periods.iloc[i]['Start Time'])} → "
                               f"{fmt_ts(periods.iloc[i]['End Time'])} · "
                               f"{an.fmt_duration(periods.iloc[i]['Duration'])}"))
    hours = c2.slider("Context (hours either side)", 0.5, 24.0, 3.0, 0.5, key="event_ctx")
    period = periods.iloc[choice]
    pad = int(round(hours * 60 / interval))
    st.plotly_chart(build_event_figure(view, period, pad, interval), key="event_chart")

    # Statistics for the selected period vs. the whole selected range
    seg = view.iloc[int(period["start_pos"]): int(period["end_pos"]) + 1]
    rows = []
    for f in an.FEATURES:
        overall_mean, overall_std = view[f].mean(), view[f].std()
        m = seg[f].mean()
        dev = (m - overall_mean) / overall_std if overall_std and not np.isnan(overall_std) else np.nan
        rows.append({"Sensor": an.PRETTY[f], "Mean": m, "Minimum": seg[f].min(), "Maximum": seg[f].max(),
                     "Std Dev": seg[f].std(), "Overall Mean": overall_mean, "Mean vs Overall (σ)": dev})
    st.markdown("**Sensor statistics during the selected abnormal period**")
    show_table(pd.DataFrame(rows).round(3))


def section_sensitivity(view, bundle, min_records, interval):
    section(7, "Model Sensitivity Validation")
    st.caption("Same trained model, different decision thresholds - shows the findings are not tied to one arbitrary cut-off.")
    labels = st.multiselect("Sensitivity levels to compare", list(SENSITIVITY_OPTIONS),
                            default=["0.5%", "1%", "2%"], key="sens_levels")
    if not labels:
        st.info("Select at least one sensitivity level.")
        return
    levels = [SENSITIVITY_OPTIONS[l] for l in labels]
    table = an.sensitivity_summary(view["time"], view["raw_score"].to_numpy(), bundle["raw_scores"],
                                   levels, min_records, interval)
    left, right = st.columns([3, 2])
    with left:
        show_table(table)
    with right:
        fig = go.Figure(go.Bar(x=table["Sensitivity"], y=table["Sustained Abnormal Periods"],
                               marker_color=C_PERIOD, text=table["Sustained Abnormal Periods"]))
        style_figure(fig, 250, "Sustained abnormal periods")
        st.plotly_chart(fig, key="sens_chart")


def section_quality(clean, q):
    section(8, "Data Quality")
    c = st.columns(6)
    c[0].metric("Rows (raw file)", f"{q['rows_raw']:,}")
    c[1].metric("Rows (after cleaning)", f"{q['rows_clean']:,}")
    c[2].metric("Columns (file / used)", f"{q['columns_raw']} / {q['columns_used']}")
    c[3].metric("Missing values", f"{q['total_missing']:,}")
    c[4].metric("Invalid values", f"{q['total_invalid']:,}")
    c[5].metric("Sampling interval", f"{q['interval_minutes']:g} min")
    st.write(f"**Time range:** {fmt_ts(q['time_min'])} → {fmt_ts(q['time_max'])}  ·  "
             f"**Gaps in record:** {q['n_gaps']:,} (longest {an.fmt_duration(q['max_gap'])})  ·  "
             f"**Rows removed:** {q['rows_bad_timestamp']:,} unreadable timestamp, "
             f"{q['rows_duplicate_timestamp']:,} duplicate timestamp")

    df = pd.DataFrame({
        "Sensor": [an.PRETTY[f] for f in an.FEATURES],
        "Missing": [q["missing_by_sensor"][f] for f in an.FEATURES],
        "Invalid (non-numeric)": [q["invalid_by_sensor"][f] for f in an.FEATURES],
    })
    df["Total NaN used for imputation"] = df["Missing"] + df["Invalid (non-numeric)"]
    left, right = st.columns([2, 3])
    with left:
        show_table(df)
    with right:
        fig = go.Figure()
        fig.add_bar(x=df["Sensor"], y=df["Missing"], name="Missing", marker_color=C_NORMAL)
        fig.add_bar(x=df["Sensor"], y=df["Invalid (non-numeric)"], name="Invalid", marker_color=C_ANOM)
        fig.update_layout(barmode="stack")
        style_figure(fig, 300, "Missing / invalid values per sensor")
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, key="missing_chart")
    st.markdown('<div class="note">Invalid/non-numeric sensor values were converted to missing values and '
                'median imputation was used for model input. The original values are preserved in the '
                'cleaned-dataset download; nothing was removed except rows with an unreadable or duplicated '
                'timestamp (counts shown above).</div>', unsafe_allow_html=True)


def section_downloads(clean, scored, q, contamination, periods):
    section(9, "Downloads")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download Cleaned Dataset", cached_csv_clean(q["data_key"], clean),
                       file_name="cleaned_dataset.csv", mime="text/csv", key="dl_clean")
    c2.download_button("Download Anomaly-Scored Dataset",
                       cached_csv_scored(q["data_key"], contamination, scored),
                       file_name="anomaly_scored_dataset.csv", mime="text/csv", key="dl_scored")
    c3.download_button("Download Abnormal Periods CSV",
                       an.to_csv_bytes(an.periods_for_export(periods)),
                       file_name="abnormal_periods.csv", mime="text/csv", key="dl_periods_footer")
    st.caption("Cleaned and scored files cover the full dataset (all dates); the periods file reflects "
               "the current date range and settings.")


def section_about():
    with st.expander("About the Analysis"):
        st.markdown("""
**Why Isolation Forest was selected**  
The dataset does not contain labelled abnormal/normal operating states. Therefore, an unsupervised anomaly
detection method was selected. Isolation Forest identifies observations that are unusual compared with the
dominant multivariate operating pattern. All six process variables are modelled together, because abnormal
operation may appear as a combined change across temperatures and drafts rather than in one sensor alone.

**How to read the result**  
- *Normal* = operating state similar to the learned normal pattern.  
- *Anomaly* = unusual multivariate operating state.  
- *Anomaly score* = −decision_function(X); higher means more abnormal; score > 0 is flagged.

**Abnormal periods**  
Individual anomalous observations were grouped chronologically. Because measurements are recorded every
5 minutes, six consecutive anomalous observations correspond to a sustained 30-minute abnormal period.
A time gap larger than 1.5× the sampling interval ends a group. Periods are ranked by average anomaly score.

**Sensitivity**  
The forest is trained once on the full dataset. The sensitivity setting (expected anomaly share) sets the
score threshold exactly as scikit-learn's `contamination` parameter does, so changing it is instant.

**Limitations**  
The detected periods represent statistically unusual operating conditions based on the six provided process
variables. They should not be interpreted as confirmed equipment failures because no ground-truth failure
labels or root-cause information were provided. Findings require operational investigation.
""")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
INTERVAL = an.DEFAULT_INTERVAL_MINUTES   # replaced with the measured interval once data is loaded


def create_dashboard() -> None:
    global INTERVAL
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Cyclone Preheater Abnormal Operation Detection</h1>'
                '<p>Multivariate anomaly detection and abnormal operating period analysis</p></div>',
                unsafe_allow_html=True)

    # ---- data source -------------------------------------------------------
    st.sidebar.markdown("### 🏭 Controls")
    with st.sidebar.expander("Data source", expanded=False):
        uploaded = st.file_uploader("Use a different CSV (optional)", type="csv")
        st.caption(f"Default file: `data/{DEFAULT_CSV.name}`")

    path, file_bytes, mtime = None, None, 0.0
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
    else:
        candidates = [DEFAULT_CSV] if DEFAULT_CSV.exists() else sorted(DATA_DIR.glob("*.csv"))
        if not candidates:
            st.error(f"No CSV found. Place the dataset at `{DEFAULT_CSV}` (or upload it from the sidebar) "
                     "and refresh.")
            st.stop()
        path, mtime = str(candidates[0]), candidates[0].stat().st_mtime

    try:
        clean, q = cached_load(path, file_bytes, mtime)
    except Exception as exc:        # clear message instead of a stack trace
        st.error(f"Could not load the dataset: {exc}")
        st.stop()
    if clean.empty:
        st.error("The dataset has no rows with a readable timestamp.")
        st.stop()
    INTERVAL = q["interval_minutes"]

    # ---- sidebar form (applies on 'Run Analysis') -------------------------------
    dmin, dmax = q["time_min"].date(), q["time_max"].date()
    defaults = dict(start=dmin, end=dmax, sens="0.5%", dur="30 minutes")
    if st.session_state.get("data_key") != q["data_key"]:      # new dataset -> reset filters
        st.session_state["applied"] = defaults.copy()
        st.session_state["data_key"] = q["data_key"]
    applied = st.session_state["applied"]

    with st.sidebar.form("controls"):
        st.markdown("**Date Range**")
        start_in = st.date_input("Start Date", value=max(dmin, min(applied["start"], dmax)),
                                 min_value=dmin, max_value=dmax)
        end_in = st.date_input("End Date", value=max(dmin, min(applied["end"], dmax)),
                               min_value=dmin, max_value=dmax)
        sens_in = st.select_slider("Anomaly Sensitivity", options=list(SENSITIVITY_OPTIONS),
                                   value=applied["sens"])
        dur_in = st.selectbox("Minimum Abnormal Duration", list(DURATION_OPTIONS),
                              index=list(DURATION_OPTIONS).index(applied["dur"]))
        submitted = st.form_submit_button("Run Analysis", type="primary")
    if submitted:
        if start_in > end_in:
            st.sidebar.error("Start Date must not be after End Date.")
        else:
            applied = dict(start=start_in, end=end_in, sens=sens_in, dur=dur_in)
            st.session_state["applied"] = applied
    st.sidebar.caption("Higher sensitivity % = more records flagged. The model is trained once on the "
                       "full dataset; filters only change the view and threshold.")

    contamination = SENSITIVITY_OPTIONS[applied["sens"]]
    min_minutes = DURATION_OPTIONS[applied["dur"]]
    min_records = an.minutes_to_records(min_minutes, INTERVAL)

    # ---- analysis (model cached; scoring/grouping are fast) -----------------------
    bundle = cached_model(q["data_key"], clean)
    scored, _offset = cached_scored(q["data_key"], contamination, clean, bundle)

    start_ts = pd.Timestamp(applied["start"])
    end_ts = pd.Timestamp(applied["end"]) + pd.Timedelta(days=1)
    times = scored["time"]
    i0, i1 = int(times.searchsorted(start_ts, side="left")), int(times.searchsorted(end_ts, side="left"))
    view = scored.iloc[i0:i1].reset_index(drop=True)
    if view.empty:
        st.warning("No records fall inside the selected date range.")
        st.stop()

    periods = an.build_abnormal_periods(view["time"], view["anomaly_score"].to_numpy(),
                                        view["is_anomaly"].to_numpy(), min_records, INTERVAL)
    member = an.mark_period_membership(len(view), periods)

    # ---- page sections (answer "where are the abnormal periods?" first) ---------------
    section_summary(view, periods, min_minutes, applied["start"], applied["end"])
    section_timeline(view, periods, INTERVAL)
    section_periods(periods)
    section_top_events(periods)
    section_variables(view, member)
    section_event(view, periods, INTERVAL)
    section_sensitivity(view, bundle, min_records, INTERVAL)
    section_quality(clean, q)
    section_downloads(clean, scored, q, contamination, periods)
    section_about()


create_dashboard()

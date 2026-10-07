"""
analysis.py - core data + anomaly-detection logic (no Streamlit code here).

Pipeline
--------
load_data()              read the CSV exactly as text (nothing is altered yet)
clean_data()             robust timestamp parsing + numeric conversion + data-quality report
prepare_features()       median imputation -> model input matrix
train_model()            fit Isolation Forest once, score every record
detect_anomalies()       turn raw scores into anomaly_score / is_anomaly for a chosen sensitivity
build_abnormal_periods() group consecutive anomalies into sustained abnormal periods

Nothing in this file contains hard-coded dates or results: every abnormal
period is calculated from the supplied dataset.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TIME_COL = "time"

FEATURES: List[str] = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Outlet_Gas_draft",
    "Cyclone_cone_draft",
    "Cyclone_Inlet_Draft",
    "Cyclone_Material_Temp",
]

# Friendly names + units used in charts/tables.
PRETTY: Dict[str, str] = {
    "Cyclone_Inlet_Gas_Temp": "Cyclone Inlet Gas Temperature",
    "Cyclone_Gas_Outlet_Temp": "Cyclone Gas Outlet Temperature",
    "Cyclone_Outlet_Gas_draft": "Cyclone Outlet Gas Draft",
    "Cyclone_cone_draft": "Cyclone Cone Draft",
    "Cyclone_Inlet_Draft": "Cyclone Inlet Draft",
    "Cyclone_Material_Temp": "Cyclone Material Temperature",
}

DEFAULT_CONTAMINATION = 0.005
DEFAULT_MIN_MINUTES = 30
DEFAULT_INTERVAL_MINUTES = 5.0
RANDOM_STATE = 42

# Text values that mean "no reading" (-> missing). Any OTHER non-numeric text
# (e.g. "#VALUE!", "Error", "I/O Timeout") is counted as an *invalid* value.
NA_TOKENS = {"", "nan", "na", "n/a", "null", "none"}

# Timestamp formats tried in order. The two formats named in the brief come first.
TIME_FORMATS = [
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
]

PERIOD_COLUMNS = [
    "Rank", "Start Time", "End Time", "Duration", "Records",
    "Average Anomaly Score", "Maximum Anomaly Score",
]


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------
def load_data(source) -> pd.DataFrame:
    """Read the CSV as raw text. `source` can be a path or a file-like object.

    Everything is read as string so that invalid sensor values and mixed
    timestamp formats survive untouched and can be handled explicitly later.
    """
    raw = pd.read_csv(
        source,
        dtype=str,
        keep_default_na=False,   # keep blanks as "" so we can count them ourselves
        low_memory=False,
        skipinitialspace=True,
        encoding="utf-8-sig",    # tolerates an Excel byte-order mark
    )
    n_cols_original = raw.shape[1]
    raw.columns = [str(c).strip() for c in raw.columns]

    # Case-insensitive column matching so "Time" / "time " both work.
    lookup = {c.lower(): c for c in raw.columns}
    needed = [TIME_COL] + FEATURES
    missing = [c for c in needed if c.lower() not in lookup]
    if missing:
        raise ValueError(
            f"The CSV is missing expected column(s): {missing}. "
            f"Columns found: {list(raw.columns)}"
        )
    raw = raw.rename(columns={lookup[c.lower()]: c for c in needed})[needed].copy()
    raw.attrs["n_columns_original"] = n_cols_original
    return raw


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def parse_timestamps(raw_time: pd.Series) -> pd.Series:
    """Parse timestamps that may be in several formats (dd-mm-yyyy, m/d/yyyy, ISO ...).

    Each known format is tried only on the rows the previous formats could not
    parse, so a mixed column is handled row-by-row. Anything still unparsed
    falls back to pandas' flexible parser; whatever remains becomes NaT.
    """
    s = raw_time.astype(str).str.strip()
    out = None
    for fmt in TIME_FORMATS:
        todo = s if out is None else s[out.isna()]
        if len(todo) == 0:
            break
        parsed = pd.to_datetime(todo, format=fmt, errors="coerce")
        if out is None:
            out = parsed
        else:
            out = out.fillna(parsed.reindex(out.index))
    if out is None:
        out = pd.to_datetime(s, errors="coerce")

    leftover = out.isna() & s.ne("")
    if leftover.any():
        try:
            fallback = pd.to_datetime(s[leftover], errors="coerce", format="mixed")
        except (ValueError, TypeError):          # older pandas without format="mixed"
            fallback = pd.to_datetime(s[leftover], errors="coerce")
        out = out.fillna(fallback.reindex(out.index))
    return pd.to_datetime(out)


def clean_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Return (cleaned_df, data_quality_report).

    * timestamps parsed robustly, rows sorted chronologically
    * sensor values converted to numbers; non-numeric -> NaN (never silently dropped)
    * rows are only removed if the timestamp is unreadable or exactly duplicated,
      and both counts are reported
    * the untouched raw frame is NOT modified (kept by the caller for transparency)
    """
    n_raw = len(raw)
    time = parse_timestamps(raw[TIME_COL])

    numeric = pd.DataFrame(index=raw.index)
    miss_flags = pd.DataFrame(index=raw.index)
    inval_flags = pd.DataFrame(index=raw.index)
    for col in FEATURES:
        s = raw[col].astype(str).str.strip()
        is_missing = s.str.lower().isin(NA_TOKENS)
        num = pd.to_numeric(s.mask(is_missing), errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan)
        numeric[col] = num.astype("float64")
        miss_flags[col] = is_missing
        inval_flags[col] = num.isna() & ~is_missing     # text that is not a number

    df = numeric.copy()
    df.insert(0, TIME_COL, time)

    bad_time = df[TIME_COL].isna()
    n_bad_time = int(bad_time.sum())
    df = df.loc[~bad_time].sort_values(TIME_COL, kind="stable")

    dup = df.duplicated(subset=TIME_COL, keep="first")
    n_dup = int(dup.sum())
    df = df.loc[~dup]

    kept = df.index
    df = df.reset_index(drop=True)

    # Sampling interval (median step) and gaps in the record
    diffs = df[TIME_COL].diff().dropna()
    interval = DEFAULT_INTERVAL_MINUTES
    n_gaps, max_gap = 0, pd.Timedelta(0)
    if len(diffs):
        med = diffs.median().total_seconds() / 60.0
        if med > 0:
            interval = float(med)
        gap_mask = diffs > pd.Timedelta(minutes=interval * 1.5)
        n_gaps = int(gap_mask.sum())
        max_gap = diffs.max()

    quality = {
        "rows_raw": int(n_raw),
        "rows_bad_timestamp": n_bad_time,
        "rows_duplicate_timestamp": n_dup,
        "rows_clean": int(len(df)),
        "columns_raw": int(raw.attrs.get("n_columns_original", raw.shape[1])),
        "columns_used": int(df.shape[1]),
        "missing_by_sensor": {c: int(miss_flags.loc[kept, c].sum()) for c in FEATURES},
        "invalid_by_sensor": {c: int(inval_flags.loc[kept, c].sum()) for c in FEATURES},
        "rows_with_any_nan": int(df[FEATURES].isna().any(axis=1).sum()),
        "time_min": df[TIME_COL].min() if len(df) else pd.NaT,
        "time_max": df[TIME_COL].max() if len(df) else pd.NaT,
        "interval_minutes": interval,
        "n_gaps": n_gaps,
        "max_gap": max_gap,
    }
    quality["total_missing"] = int(sum(quality["missing_by_sensor"].values()))
    quality["total_invalid"] = int(sum(quality["invalid_by_sensor"].values()))
    # Fingerprint used as a cache key by the dashboard.
    h = int(pd.util.hash_pandas_object(df, index=False).sum())
    quality["data_key"] = f"{len(df)}-{h}"
    return df, quality


# --------------------------------------------------------------------------
# 3. Features + model
# --------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Median-impute the six sensors. Returns (X, medians). `df` is not modified."""
    X = df[FEATURES].copy()
    medians = X.median()
    if medians.isna().any():
        empty = list(medians[medians.isna()].index)
        raise ValueError(f"Sensor column(s) contain no numeric values at all: {empty}")
    return X.fillna(medians), medians


def train_model(
    X: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> Tuple[IsolationForest, np.ndarray]:
    """Fit Isolation Forest on all six variables together and score every record.

    Returns (model, raw_scores) where raw_scores = model.score_samples(X)
    (lower = more abnormal). Fitting is done once; the sensitivity setting only
    changes the decision threshold (see detect_anomalies).
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    values = X.to_numpy()
    model.fit(values)
    return model, model.score_samples(values)


def threshold_offset(raw_scores: np.ndarray, contamination: float) -> float:
    """Decision offset for a contamination level.

    This is exactly what scikit-learn does inside IsolationForest.fit():
    offset_ = percentile(score_samples(X), 100 * contamination).
    """
    return float(np.percentile(raw_scores, 100.0 * contamination))


def detect_anomalies(
    raw_scores: np.ndarray, contamination: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (anomaly_score, is_anomaly, offset).

    anomaly_score = -decision_function(X) = offset - score_samples(X)
    is_anomaly    = (model.predict(X) == -1) = (anomaly_score > 0)
    Higher anomaly_score = more abnormal.
    """
    offset = threshold_offset(raw_scores, contamination)
    anomaly_score = offset - raw_scores
    return anomaly_score, anomaly_score > 0, offset


# --------------------------------------------------------------------------
# 4. Abnormal periods
# --------------------------------------------------------------------------
def minutes_to_records(minutes: float, interval_minutes: float) -> int:
    """Number of consecutive records that make up `minutes` (e.g. 30 min @ 5 min = 6)."""
    return max(1, int(math.ceil(round(minutes / interval_minutes, 6))))


def _empty_periods() -> pd.DataFrame:
    return pd.DataFrame({
        "Rank": pd.Series(dtype="int64"),
        "Start Time": pd.to_datetime(pd.Series(dtype="datetime64[ns]")),
        "End Time": pd.to_datetime(pd.Series(dtype="datetime64[ns]")),
        "Duration": pd.to_timedelta(pd.Series(dtype="timedelta64[ns]")),
        "Records": pd.Series(dtype="int64"),
        "Average Anomaly Score": pd.Series(dtype="float64"),
        "Maximum Anomaly Score": pd.Series(dtype="float64"),
        "start_pos": pd.Series(dtype="int64"),
        "end_pos": pd.Series(dtype="int64"),
        "duration_min": pd.Series(dtype="float64"),
    })


def build_abnormal_periods(
    times,
    anomaly_score: np.ndarray,
    is_anomaly: np.ndarray,
    min_records: int,
    interval_minutes: float,
) -> pd.DataFrame:
    """Group consecutive anomalous records into sustained abnormal periods.

    * records are "consecutive" if the time step between them is <= 1.5 x the
      sampling interval (so a data gap never gets silently bridged)
    * only groups with >= `min_records` records are kept
    * Duration = (End - Start) + one sampling interval, so 6 records = 30 min
    * periods are ranked by severity (average anomaly score, then maximum)

    `start_pos` / `end_pos` are positions in the arrays passed in (inclusive).
    """
    flag = np.asarray(is_anomaly, dtype=bool)
    score = np.asarray(anomaly_score, dtype=float)
    n = len(flag)
    if n == 0 or not flag.any():
        return _empty_periods()

    t = pd.DatetimeIndex(times).values.astype("datetime64[ns]")
    gap_ns = np.timedelta64(int(round(interval_minutes * 1.5 * 60 * 1e9)), "ns")

    # A new run starts where the flag changes or where there is a time gap.
    brk = np.ones(n, dtype=bool)
    if n > 1:
        brk[1:] = (flag[1:] != flag[:-1]) | ((t[1:] - t[:-1]) > gap_ns)
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], n)           # exclusive
    lengths = ends - starts

    keep = flag[starts] & (lengths >= min_records)
    if not keep.any():
        return _empty_periods()

    sums = np.add.reduceat(score, starts)
    maxs = np.maximum.reduceat(score, starts)
    s_k, e_k, len_k = starts[keep], ends[keep], lengths[keep]

    start_t, end_t = t[s_k], t[e_k - 1]
    one_step = np.timedelta64(int(round(interval_minutes * 60 * 1e9)), "ns")
    duration = (end_t - start_t) + one_step

    periods = pd.DataFrame({
        "Start Time": pd.to_datetime(start_t),
        "End Time": pd.to_datetime(end_t),
        "Duration": pd.to_timedelta(duration),
        "Records": len_k.astype("int64"),
        "Average Anomaly Score": sums[keep] / len_k,
        "Maximum Anomaly Score": maxs[keep],
        "start_pos": s_k.astype("int64"),
        "end_pos": (e_k - 1).astype("int64"),
    })
    periods = periods.sort_values(
        ["Average Anomaly Score", "Maximum Anomaly Score"], ascending=False, kind="stable"
    ).reset_index(drop=True)
    periods.insert(0, "Rank", np.arange(1, len(periods) + 1, dtype="int64"))
    periods["duration_min"] = periods["Duration"].dt.total_seconds() / 60.0
    return periods


def fmt_duration(td: pd.Timedelta) -> str:
    """Timedelta -> '1d 3h 5m' / '45m'."""
    minutes = int(round(pd.Timedelta(td).total_seconds() / 60.0))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins or not parts:
        parts.append(f"{mins}m")
    return " ".join(parts)


def most_severe_period(periods: pd.DataFrame):
    return periods.iloc[0] if len(periods) else None


def longest_period(periods: pd.DataFrame):
    """Longest period (ties broken by severity because rows are rank-ordered)."""
    return periods.loc[periods["Duration"].idxmax()] if len(periods) else None


def mark_period_membership(n: int, periods: pd.DataFrame) -> np.ndarray:
    """Boolean array: True where a record belongs to a sustained abnormal period."""
    member = np.zeros(n, dtype=bool)
    for s, e in zip(periods["start_pos"].to_numpy(), periods["end_pos"].to_numpy()):
        member[s:e + 1] = True
    return member


def periods_for_export(periods: pd.DataFrame) -> pd.DataFrame:
    """Public table exactly as requested for the CSV download."""
    fmt = "%Y-%m-%d %H:%M:%S"
    return pd.DataFrame({
        "Rank": periods["Rank"],
        "Start Time": periods["Start Time"].dt.strftime(fmt),
        "End Time": periods["End Time"].dt.strftime(fmt),
        "Duration": periods["Duration"].map(fmt_duration),
        "Number of Records": periods["Records"],
        "Average Anomaly Score": periods["Average Anomaly Score"].round(6),
        "Maximum Anomaly Score": periods["Maximum Anomaly Score"].round(6),
    })


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode("utf-8")


# --------------------------------------------------------------------------
# 5. Sensitivity comparison + plotting helper
# --------------------------------------------------------------------------
def sensitivity_summary(
    times,
    raw_scores_view: np.ndarray,
    raw_scores_full: np.ndarray,
    levels: Sequence[float],
    min_records: int,
    interval_minutes: float,
) -> pd.DataFrame:
    """Re-evaluate several contamination levels on the same (date-filtered) data."""
    rows = []
    for c in levels:
        offset = threshold_offset(raw_scores_full, c)     # threshold from the full history
        score = offset - raw_scores_view
        flag = score > 0
        p = build_abnormal_periods(times, score, flag, min_records, interval_minutes)
        longest = longest_period(p)
        rows.append({
            "Sensitivity": f"{c * 100:g}%",
            "Anomalous Records": int(flag.sum()),
            "Sustained Abnormal Periods": int(len(p)),
            "Longest Abnormal Period": fmt_duration(longest["Duration"]) if longest is not None else "-",
        })
    return pd.DataFrame(rows)


def downsample_indices(y, max_points: int = 6000) -> np.ndarray:
    """Min/max bucket downsampling for display only.

    Keeps the lowest and highest value of every bucket so spikes are never lost.
    The anomaly calculations always use the full data; this only reduces how many
    points the browser has to draw.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= max_points or max_points < 4:
        return np.arange(n)
    n_buckets = max_points // 2
    size = n // n_buckets
    usable = size * n_buckets
    block = y[:usable].reshape(n_buckets, size)
    lo = np.where(np.isnan(block), np.inf, block).argmin(axis=1)
    hi = np.where(np.isnan(block), -np.inf, block).argmax(axis=1)
    base = np.arange(n_buckets) * size
    return np.unique(np.concatenate([base + lo, base + hi, np.arange(usable, n)]))

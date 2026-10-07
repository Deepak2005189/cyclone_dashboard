# Cyclone Preheater Abnormal Operation Detection

Multivariate anomaly detection and abnormal operating period analysis (Algo8 assignment).

## 1. Project objective
> "Using Python and any algorithm of your choice, highlight time periods where abnormal operation can be observed."

The dashboard answers one question immediately: **where are the abnormal periods?**
It uses an unsupervised model (Isolation Forest) and reports sustained abnormal *time periods*, not just single records.

## 2. Dataset description
`data/Data (1).csv` - ~370k records, one every 5 minutes, ~3 years.

| Column | Meaning |
|---|---|
| `time` | Timestamp (mixed `dd-mm-yyyy HH:MM` and `m/d/yyyy HH:MM` supported) |
| `Cyclone_Inlet_Gas_Temp` | Temperature of hot gas entering the cyclone |
| `Cyclone_Gas_Outlet_Temp` | Temperature of hot gas leaving the cyclone |
| `Cyclone_Outlet_Gas_draft` | Draft/pressure at the cyclone gas outlet |
| `Cyclone_cone_draft` | Draft/pressure at the cyclone cone section |
| `Cyclone_Inlet_Draft` | Draft/pressure at the cyclone inlet |
| `Cyclone_Material_Temp` | Material temperature at the cyclone outlet |

There is no normal/abnormal label, so supervised learning is not possible.

## 3. Data preparation
1. CSV is read as raw text (nothing altered on load).
2. Timestamps are parsed row-by-row against several known formats (then a flexible fallback); rows are sorted chronologically. Rows are removed **only** if the timestamp is unreadable or exactly duplicated - both counts are shown.
3. The six sensors are converted to numbers; non-numeric text becomes NaN (counted as *invalid*; blanks are counted as *missing*).
4. Median imputation is applied to build the model input. Original values are preserved in the cleaned-dataset download.
5. No sub-sampling or random removal happens before detection. Chart downsampling is display-only.

## 4. Algorithm
`IsolationForest(n_estimators=200, contamination=0.005, random_state=42, n_jobs=-1)` on all six variables together.

* `anomaly_score = -decision_function(X)` (higher = more abnormal; score > 0 is flagged)
* `is_anomaly = predict(X) == -1`
* The forest is trained **once**. Changing the sensitivity only moves the threshold, using the same formula scikit-learn applies internally (`offset_ = percentile(score_samples, 100 x contamination)`), so results are identical to re-fitting with that contamination but instant.

## 5. Why Isolation Forest
Unlabelled data -> unsupervised method. Isolation Forest isolates records that differ from the dominant multivariate operating pattern, scales to 370k rows, needs no feature scaling, and can flag *combinations* of sensor values that are individually unremarkable.

## 6. Abnormal-period methodology
1. Flag individual anomalous records.
2. Group consecutive anomalous records (a time step > 1.5x the sampling interval breaks a group, so data gaps are never bridged).
3. Keep groups of at least the minimum duration (default 30 min = 6 records at 5 min; options 15/30/60/120 min).
4. For each period: start, end, duration (= end - start + one sample interval), record count, average and maximum anomaly score.
5. Rank by severity (average score, then maximum).

## 7. How to run
```bash
cd cyclone_dashboard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# make sure the CSV is at: data/Data (1).csv
streamlit run app.py
```
Open http://localhost:8501. The first load takes a short while (parsing + training); afterwards everything is cached.
If the CSV is not in `data/`, upload it from the sidebar ("Data source").

## 8. Output description
Page order: KPI cards -> Executive Summary -> **Anomaly Score Timeline** -> **Detected Abnormal Operating Periods** table -> Top 10 events -> Process variable overview -> Sensor behaviour for a selected event -> Sensitivity validation -> Data quality -> Downloads -> About.

Sidebar: date range, sensitivity (0.1/0.5/1/2/5 %), minimum duration (15/30/60/120 min), **Run Analysis**.

Downloads: cleaned dataset, anomaly-scored dataset (`time`, six sensors, `anomaly_score`, `is_anomaly`), abnormal periods CSV (`Rank, Start Time, End Time, Duration, Number of Records, Average Anomaly Score, Maximum Anomaly Score`).

Files: `app.py` (UI), `analysis.py` (all data/ML logic, no Streamlit), `.streamlit/config.toml` (dark theme).

## 9. Limitations
* Detected periods are **statistically unusual multivariate operating conditions**, not confirmed equipment failures - no failure labels or root-cause data were provided. They require operational investigation.
* The share of data flagged is set by the sensitivity setting; the model ranks how unusual records are, it does not know what is "unsafe".
* Planned changes of operating mode (start-ups, shutdowns, load changes) may also appear as unusual and should be reviewed by a process engineer.
* The date filter changes the view and period grouping; the model itself is always trained on the full dataset.

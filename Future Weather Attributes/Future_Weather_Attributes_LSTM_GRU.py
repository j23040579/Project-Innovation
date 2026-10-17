
# %% [markdown]
# # Future weather attributes — compare LSTM and GRU
# 
# All **37 physical weather attributes remain in every experiment**. Wind direction is represented by sine and cosine, giving **38 weather channels**. Calendar and station metadata are shared across architectures. Attribute-removal experiments are not part of this notebook.
# 
# Set `MODEL_CHOICE` to `"LSTM"`, `"GRU"`, or `"BOTH"`. The default compares both using matching training settings. Optional settings compare lookback lengths, recurrent layer sizes, learning rates or dropout while keeping the attributes fixed. Both models use the same sampled training windows, preprocessing and evaluation dates. Matching widths does not give equal parameter counts; these are logged.
# 
# **Start:** set `DATA_PATH`, then **Restart Kernel and Run All**. The default runs up to 30 epochs per model on 160,000 common training windows. Add settings or seeds only when you intend to train the extra runs. Do not run all cells again merely to view saved output.
# 
# **Reading accuracy:** this is continuous weather regression. Compare per-attribute MAE/RMSE and aggregate normalized RMSE (lower is better), not a classification accuracy percentage. Validation selects the model before test results are computed. No future accuracy is reported until actual observations exist.
# 
# Outputs are separated into model files, experiment logs and future CSVs. Model reuse checks the exact settings, full feature contract, dataset, software versions and shared windows. Existing LSTM-only artifacts are not silently reused. Use a **new `EXPERIMENT_ID`** when changing the model choices, settings, seeds, periods, budget or evaluation/forecast options; this keeps different comparison results from mixing in one export. Repeating the same request reuses its saved models unless `FORCE_RETRAIN=True`.


# %% [markdown]
# ## 1. Settings
# 
# Change `MODEL_CHOICE` to compare architectures. Add an `EXPERIMENT_SETTINGS` entry to test another training setup without removing attributes. `SEEDS` controls repeated runs.


# %%
from pathlib import Path

_WEATHER_STARTUP_COMPLETE = False
ROOT = Path.cwd()
if (ROOT / "Future Weather Attributes").is_dir():
    ROOT = ROOT / "Future Weather Attributes"
EXPERIMENT_ID = "lstm_gru_all_attributes"  # change this for a different comparison
DATA_PATH = ROOT / "dataset/LSTM_processed/japan_daily_lstm_processed.csv"
if not DATA_PATH.is_file() and DATA_PATH.with_name(DATA_PATH.name + ".csv").is_file():
    DATA_PATH = DATA_PATH.with_name(DATA_PATH.name + ".csv")

# Run either architecture, or compare both with exactly the same attributes.
MODEL_CHOICE = "BOTH"                 # "LSTM", "GRU", or "BOTH"
SEEDS = [42]                          # e.g. [42, 52, 62] for repeated runs
EXPERIMENT_SETTINGS = [
    {"name": "baseline", "lookback": 30, "recurrent_units": [96, 64],
     "dense_units": 96, "learning_rate": 0.001, "dropout": 0.0},
    # Optional: compare another training setting, with EVERY attribute retained.
    # {"name": "longer_history", "lookback": 60, "recurrent_units": [96, 64],
    #  "dense_units": 96, "learning_rate": 0.001, "dropout": 0.0},
]
EPOCHS = 30
BATCH_SIZE = 128
MAX_TRAIN_WINDOWS = 160_000
SAMPLING_SEED = 42                     # same sampled windows for ALL runs/seeds
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 3
TRAIN_IF_MISSING = True
FORCE_RETRAIN = False                  # use a new EXPERIMENT_ID to keep old runs
MAX_FORWARD_FILL_DAYS = 3

TRAIN_START = "2021-01-01"
TRAIN_END = "2025-09-30"
VAL_START = "2025-10-01"
VAL_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-01-31"
RUN_EVALUATION = True                  # heldout test for each requested run
HORIZON_DAYS = [1, 7, 14, 30, 90, 365]  # cumulative recursive scores, where covered

# Validation uses one continuous recursive forecast, VAL_START through VAL_END.
# Its mean normalized RMSE selects the model/settings before test results exist.
# Future forecasts use that selected configuration and the FIRST listed seed.
RUN_FORECAST = True
FORECAST_START = "2026-02-01"
FORECAST_END = "2027-01-31"             # None = one complete year from start
FORECAST_PREFECTURE = None
FORECAST_TOWN = None
FORECAST_BATCH_SIZE = 256
PROGRESS_EVERY = 30
EXPORT_LOG_ARCHIVE = True

MODEL_ROOT = ROOT / "trained_models/future_weather" / EXPERIMENT_ID
FORECAST_DIR = ROOT / "Forecast" / EXPERIMENT_ID
LOG_ROOT = ROOT / "experiment_logs" / EXPERIMENT_ID
LOG_DIR = LOG_ROOT / "shared"
print("Project:", ROOT.resolve())
print("Dataset:", DATA_PATH.resolve())
print("Models:", MODEL_CHOICE, "| Settings:", len(EXPERIMENT_SETTINGS), "| Seeds:", SEEDS)


# %% [markdown]
# ## 2. Imports and validation


# %%
if "MODEL_ROOT" not in globals():
    raise RuntimeError("Run section 1 (Settings) first. Use Restart Kernel and Run All.")

# Install once if needed:
# %pip install numpy pandas matplotlib scikit-learn tensorflow joblib
import copy, hashlib, json, re, shutil, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from IPython.display import display

MODEL_CHOICE = MODEL_CHOICE.upper()
if MODEL_CHOICE not in {"LSTM", "GRU", "BOTH"}:
    raise ValueError('MODEL_CHOICE must be "LSTM", "GRU", or "BOTH".')
MODEL_TYPES = ["LSTM", "GRU"] if MODEL_CHOICE == "BOTH" else [MODEL_CHOICE]
def positive_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0
if not all(positive_integer(x) for x in [EPOCHS, BATCH_SIZE, FORECAST_BATCH_SIZE, EARLY_STOPPING_PATIENCE, LR_PATIENCE]):
    raise ValueError("Epochs, batch sizes and callback patience must be positive integers.")
if MAX_TRAIN_WINDOWS is not None and not positive_integer(MAX_TRAIN_WINDOWS):
    raise ValueError("MAX_TRAIN_WINDOWS must be a positive integer or None.")
if not isinstance(MAX_FORWARD_FILL_DAYS, int) or MAX_FORWARD_FILL_DAYS < 0:
    raise ValueError("MAX_FORWARD_FILL_DAYS must be a nonnegative integer.")
if not SEEDS or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in SEEDS) or len(SEEDS) != len(set(SEEDS)):
    raise ValueError("SEEDS must be a nonempty list of unique nonnegative integers.")
if not HORIZON_DAYS or not all(positive_integer(x) for x in HORIZON_DAYS):
    raise ValueError("HORIZON_DAYS must contain positive integers.")
if not EXPERIMENT_SETTINGS:
    raise ValueError("Provide at least one experiment setting.")
names = []
required_settings = {"name", "lookback", "recurrent_units", "dense_units", "learning_rate", "dropout"}
for setting in EXPERIMENT_SETTINGS:
    if set(setting) != required_settings:
        raise ValueError(f"Each setting needs exactly {sorted(required_settings)}; attributes cannot be removed.")
    if not isinstance(setting["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", setting["name"]):
        raise ValueError("Use letters, numbers, underscores or hyphens in setting names.")
    if not positive_integer(setting["lookback"]) or not positive_integer(setting["dense_units"]):
        raise ValueError("Lookback and dense units must be positive integers.")
    if not isinstance(setting["recurrent_units"], (list, tuple)) or not setting["recurrent_units"] or not all(positive_integer(x) for x in setting["recurrent_units"]):
        raise ValueError("recurrent_units must be a nonempty list of positive integers.")
    if not np.isfinite(setting["learning_rate"]) or setting["learning_rate"] <= 0 or not 0 <= setting["dropout"] < 1:
        raise ValueError("Use a positive learning rate and dropout in [0, 1).")
    names.append(setting["name"])
if len(names) != len(set(names)):
    raise ValueError("Experiment setting names must be unique.")
dates = list(map(pd.Timestamp, [TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END]))
if any(pd.isna(x) or x.tzinfo is not None or x != x.normalize() for x in dates):
    raise ValueError("Split dates must be finite timezone-free daily dates.")
if not (dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]):
    raise ValueError("Training, validation and test dates must be ordered and disjoint.")
if not DATA_PATH.is_file():
    raise FileNotFoundError("Set DATA_PATH to the complete historical weather CSV.")
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
DATA_SHA256 = file_sha256(DATA_PATH)
request = {"schema_version": "full_attributes_lstm_gru_v1", "dataset_sha256": DATA_SHA256,
           "model_types": MODEL_TYPES, "settings": EXPERIMENT_SETTINGS, "seeds": SEEDS,
           "epochs": EPOCHS, "batch_size": BATCH_SIZE, "max_train_windows": MAX_TRAIN_WINDOWS,
           "sampling_seed": SAMPLING_SEED, "max_forward_fill_days": MAX_FORWARD_FILL_DAYS,
           "early_stopping_patience": EARLY_STOPPING_PATIENCE, "lr_patience": LR_PATIENCE,
           "splits": [TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END],
           "run_evaluation": RUN_EVALUATION, "run_forecast": RUN_FORECAST,
           "horizon_days": HORIZON_DAYS, "forecast_start": FORECAST_START, "forecast_end": FORECAST_END,
           "forecast_prefecture": FORECAST_PREFECTURE, "forecast_town": FORECAST_TOWN,
           "forecast_batch_size": FORECAST_BATCH_SIZE,
           "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
request = json.loads(json.dumps(request))
request_path = LOG_ROOT / "experiment_request.json"
if request_path.is_file():
    if json.loads(request_path.read_text(encoding="utf-8")) != request:
        raise ValueError("This EXPERIMENT_ID already contains a different comparison. Choose a new EXPERIMENT_ID to keep results separate.")
elif LOG_ROOT.exists() and any(LOG_ROOT.iterdir()):
    raise ValueError("This log folder contains untracked old outputs. Choose a new EXPERIMENT_ID.")
for folder in [MODEL_ROOT, FORECAST_DIR, LOG_ROOT, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
# A rerun is not complete merely because an earlier completion marker exists.
for name in ["completion.json", "selection_frozen.json"]:
    previous = LOG_ROOT / name
    if previous.is_file():
        history_dir = LOG_ROOT / "execution_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous, history_dir / name)
        previous.unlink()
_WEATHER_STARTUP_COMPLETE = True


# %% [markdown]
# ## 3. Load the complete weather attribute contract
# 
# Inputs may be repaired from training-derived fallbacks. Original observations remain separate for target masks and evaluation. Missing entire columns are an error; no attribute is dropped.


# %%
if not globals().get("_WEATHER_STARTUP_COMPLETE", False):
    raise RuntimeError("Run sections 1 (Settings) and 2 (Imports) before this cell. Use Restart Kernel and Run All. Settings define the output directories.")

DEFAULT_WEATHER = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "snowfall_sum",
    "rain_sum",
    "precipitation_hours",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "sunshine_duration",
    "daylight_duration",
    "dew_point_2m_mean",
    "dew_point_2m_max",
    "surface_pressure_mean",
    "surface_pressure_max",
    "surface_pressure_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_max",
    "relative_humidity_2m_min",
    "dew_point_2m_min",
    "cloud_cover_mean",
    "cloud_cover_max",
    "cloud_cover_min",
    "pressure_msl_mean",
    "pressure_msl_max",
    "pressure_msl_min",
    "wind_speed_10m_mean",
    "wind_speed_10m_min",
    "wind_gusts_10m_mean",
    "wind_gusts_10m_min",
    "river_discharge",
    "river_discharge_p75",
    "river_discharge_mean",
    "river_discharge_median",
    "river_discharge_max",
    "wind_direction_10m_dominant"
]
ANGLE = "wind_direction_10m_dominant"
# The complete contract is fixed for every model and experiment setting.
features = [c for c in DEFAULT_WEATHER if c != ANGLE] + ["wind_direction_sin", "wind_direction_cos"]
PHYSICAL_FEATURES = DEFAULT_WEATHER.copy()
EXPECTED_FEATURES = tuple(features)
assert len(PHYSICAL_FEATURES) == 37 and len(features) == 38

def load_data(path, ordered_features):
    data = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    required = {"prefecture", "town", "date", "latitude", "longitude"}
    if not required.issubset(data):
        raise ValueError(f"Historical data requires {sorted(required)}.")
    if "elevation" not in data:
        data["elevation"] = data["openmeteo_elevation"] if "openmeteo_elevation" in data else np.nan
    data["date"] = pd.to_datetime(data.date, errors="raise").dt.normalize()
    if data[["prefecture", "town", "date"]].isna().any().any():
        raise ValueError("Missing town/date keys; restore the station identifier/date in the source CSV.")
    if data.duplicated(["prefecture", "town", "date"]).any():
        raise ValueError("Duplicate town/date records.")
    data = data.sort_values(["prefecture", "town", "date"]).reset_index(drop=True)
    keys = [data.prefecture, data.town]
    metadata = ["latitude", "longitude", "elevation"]
    numeric = list(dict.fromkeys(list(ordered_features) + metadata + [ANGLE, "wind_direction_sin", "wind_direction_cos"]))
    for name in numeric:
        if name in data:
            data[name] = pd.to_numeric(data[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Use encoded observations only if the raw angle column was not supplied.
    # When the supplied raw angle is missing, helpers cannot establish observed truth.
    raw_angle_column_present = ANGLE in data
    if not raw_angle_column_present:
        data[ANGLE] = np.nan
    if not raw_angle_column_present and {"wind_direction_sin", "wind_direction_cos"}.issubset(data):
        valid_pair = np.isfinite(data[["wind_direction_sin", "wind_direction_cos"]]).all(axis=1)
        valid_pair &= np.hypot(data.wind_direction_sin, data.wind_direction_cos) > 1e-8
        decoded = np.degrees(np.arctan2(data.wind_direction_sin, data.wind_direction_cos)) % 360
        data[ANGLE] = data[ANGLE].where(data[ANGLE].notna(), decoded.where(valid_pair))
    data["wind_direction_sin"] = np.sin(np.deg2rad(data[ANGLE]))
    data["wind_direction_cos"] = np.cos(np.deg2rad(data[ANGLE]))
    missing = set(ordered_features) - set(data)
    if missing:
        raise ValueError(f"Historical weather lacks required model columns: {sorted(missing)}")
    report_columns = list(dict.fromkeys(list(ordered_features) + metadata))
    bad = data[report_columns].isna()
    counts = bad.sum().rename("missing_or_nonfinite_cells").rename_axis("column").reset_index()
    counts["total_rows"] = len(data)
    counts.to_csv(LOG_DIR / "data_quality_counts.csv", index=False)
    town_counts = bad.groupby(keys).sum().reset_index()
    town_counts.to_csv(LOG_DIR / "data_quality_by_town.csv", index=False)
    sample = data.loc[bad.any(axis=1), ["prefecture", "town", "date"] + report_columns].head(20)
    sample.to_csv(LOG_DIR / "data_quality_examples.csv", index=False, date_format="%Y-%m-%d")
    if bad.any().any():
        print("Missing / nonfinite inputs (details also saved in experiment_logs):")
        display(counts[counts.missing_or_nonfinite_cells > 0])

    # Use only static metadata known by the relevant historical cutoff.
    metadata_cutoff = pd.Timestamp(TRAIN_END)
    for name in metadata:
        known = data[name].where(data.date <= metadata_cutoff).groupby(keys).transform("first")
        if name == "elevation" and known.isna().any() and "openmeteo_elevation" in data:
            alternate = pd.to_numeric(data.openmeteo_elevation, errors="coerce").replace([np.inf, -np.inf], np.nan)
            alt_known = alternate.where(data.date <= metadata_cutoff).groupby(keys).transform("first")
            alt_min = alternate.groupby(keys).transform("min")
            alt_max = alternate.groupby(keys).transform("max")
            if ((alt_max - alt_min > 1e-5) & known.isna()).any():
                raise ValueError("Alternate station elevation changes over time; restore consistent elevation metadata.")
            known = known.fillna(alt_known)
        spread = data[name].groupby(keys).transform("max") - data[name].groupby(keys).transform("min")
        if (spread > 1e-5).any():
            towns = data.loc[spread > 1e-5, ["prefecture", "town"]].drop_duplicates().head(10).to_dict("records")
            raise ValueError(f"Conflicting static {name} for stations: {towns}. See data_quality_by_town.csv.")
        if known.isna().any():
            towns = data.loc[known.isna(), ["prefecture", "town"]].drop_duplicates().head(10).to_dict("records")
            raise ValueError(f"No known {name} by {metadata_cutoff.date()} for stations: {towns}. Restore these metadata in the CSV; coordinates are not invented.")
        data[name] = known  # canonicalize small numerical jitter after checking consistency
    if not data.latitude.between(-90, 90).all() or not data.longitude.between(-180, 180).all():
        raise ValueError("Invalid latitude/longitude bounds; see data_quality_examples.csv.")

    # Preserve original observed weather, including missing cells, for targets and metrics.
    observed = data.copy(deep=True)
    repaired = data.copy(deep=True)
    physical = [c for c in ordered_features if c not in {"wind_direction_sin", "wind_direction_cos"}]
    if "wind_direction_sin" in ordered_features or "wind_direction_cos" in ordered_features:
        physical = list(dict.fromkeys(physical + [ANGLE]))
    train_rows = observed.date.between(TRAIN_START, TRAIN_END)
    if not train_rows.any():
        raise ValueError("No training-period rows available for missing-weather fallbacks.")
    fallback_features = observed.loc[train_rows, ordered_features].median().to_dict()
    fallback_source = "training-period median"
    physical_fallbacks = {}
    repairs = []
    for name in physical:
        raw = observed[name]
        previous = raw.groupby(keys).ffill()
        previous_date = observed.date.where(raw.notna()).groupby(keys).ffill()
        age = (observed.date - previous_date).dt.days
        short_gap = raw.isna() & age.between(0, MAX_FORWARD_FILL_DAYS)
        repaired[name] = raw.where(~short_gap, previous)
        remaining = repaired[name].isna()
        fallback = fallback_features.get(name, np.nan)
        if name == ANGLE and {"wind_direction_sin", "wind_direction_cos"}.issubset(ordered_features):
            sine = observed.loc[train_rows, "wind_direction_sin"].mean()
            cosine = observed.loc[train_rows, "wind_direction_cos"].mean()
            fallback = float(np.degrees(np.arctan2(sine, cosine)) % 360) if np.isfinite([sine, cosine]).all() and np.hypot(sine, cosine) > 1e-8 else np.nan
        source_name = fallback_source
        physical_fallbacks[name] = float(fallback) if np.isfinite(fallback) else None
        if remaining.any():
            if not np.isfinite(fallback):
                raise ValueError(f"No finite training-derived fallback for {name}. Restore observed training values or the original scaler; no zero placeholder is inserted.")
            repaired.loc[remaining, name] = fallback
        repairs.append({"feature": name, "forward_filled_cells": int(short_gap.sum()),
                        "training_fallback_cells": int(remaining.sum()), "fallback_source": source_name})
    repaired["wind_direction_sin"] = np.sin(np.deg2rad(repaired[ANGLE]))
    repaired["wind_direction_cos"] = np.cos(np.deg2rad(repaired[ANGLE]))
    if not np.isfinite(repaired[list(ordered_features) + metadata].to_numpy()).all():
        raise ValueError("Unrepaired model inputs remain; inspect data_quality_counts.csv.")
    repaired.attrs["input_repair_fallbacks"] = physical_fallbacks
    repair_report = pd.DataFrame(repairs)
    repair_report.to_csv(LOG_DIR / "input_repair_counts.csv", index=False)
    print("Input repairs:", int(repair_report.forward_filled_cells.sum()), "short-gap cells and",
          int(repair_report.training_fallback_cells.sum()), "fallback cells.")
    print("Fallback sources:", sorted(set(repair_report.loc[repair_report.training_fallback_cells > 0, "fallback_source"])))
    print("Original CSV remains unchanged; missing observations stay missing for evaluation.")
    return repaired, observed

df, observed_df = load_data(DATA_PATH, features)
print(f"{len(df):,} historical rows | {df.groupby(['prefecture','town']).ngroups} towns | {len(features)} model features")
print("Available dates:", df.date.min().date(), "through", df.date.max().date())
print("All 37 weather attributes are retained (38 channels with circular direction encoding).")
display(df.head(3))


# %% [markdown]
# ## 4. Shared forecast helpers
# 
# The same physical bounds, circular direction encoding and recursive feedback are applied to both models.


# %%
"""Batched daily recursive weather forecasting; usable without TensorFlow imports."""

import numpy as np
import pandas as pd

ANGLE = "wind_direction_10m_dominant"
DIRECTION_SIN = "wind_direction_sin"
DIRECTION_COS = "wind_direction_cos"
NONNEG = (
    "rain_", "precipitation_", "snowfall_", "sunshine_duration",
    "daylight_duration", "shortwave_radiation_sum", "wind_speed_",
    "wind_gusts_", "river_discharge",
)
PERCENT = ("relative_humidity_", "cloud_cover_")


def calendar(dates):
    dates = pd.DatetimeIndex(dates)
    angle = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25
    return np.column_stack((np.sin(angle), np.cos(angle))).astype("float32")


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _day_timestamp(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f"{name} must be a timezone-free daily date")
    return result


def _as_finite_matrix(values, expected_shape, name):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape}, expected {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _bound_weather(raw, features):
    """Apply the same physical output constraints before output and feedback."""
    bounded = raw.copy()
    positions = {name: index for index, name in enumerate(features)}
    for name, column in positions.items():
        if name.startswith(NONNEG):
            bounded[:, column] = np.maximum(0, bounded[:, column])
        if name.startswith(PERCENT):
            bounded[:, column] = np.clip(bounded[:, column], 0, 100)
        if name == "precipitation_hours":
            bounded[:, column] = np.clip(bounded[:, column], 0, 24)
        if name in ("sunshine_duration", "daylight_duration"):
            bounded[:, column] = np.clip(bounded[:, column], 0, 86400)
    # Project any available min/mean/max family onto increasing order.
    families = {}
    for name in features:
        for suffix in ("_min", "_mean", "_max"):
            if name.endswith(suffix):
                families.setdefault(name[:-len(suffix)], {})[suffix] = positions[name]
    for family in families.values():
        ordered = [family[s] for s in ("_min", "_mean", "_max") if s in family]
        if len(ordered) > 1:
            bounded[:, ordered] = np.sort(bounded[:, ordered], axis=1)
    if "sunshine_duration" in positions and "daylight_duration" in positions:
        bounded[:, positions["sunshine_duration"]] = np.minimum(
            bounded[:, positions["sunshine_duration"]], bounded[:, positions["daylight_duration"]]
        )
    if DIRECTION_SIN in positions and DIRECTION_COS in positions:
        angle = np.arctan2(bounded[:, positions[DIRECTION_SIN]], bounded[:, positions[DIRECTION_COS]])
        bounded[:, positions[DIRECTION_SIN]] = np.sin(angle)
        bounded[:, positions[DIRECTION_COS]] = np.cos(angle)
    if ANGLE in positions:
        bounded[:, positions[ANGLE]] %= 360
    return bounded


def recursive_forecast(
    history_df, model, scaler, loc_scaler, features, lookback,
    start, end, prefecture=None, town=None, batch_size=128, progress_every=30,
):
    """Forecast every selected town/day in the inclusive start..end interval.

    Only rows strictly before start may initialize the model. Every selected
    station must have a complete consecutive lookback ending on start - 1 day.
    model must be callable as model([past, target_calendar], training=False).
    Observations on/after start are never read into forecast state.
    """
    lookback = _positive_integer(lookback, "lookback")
    batch_size = _positive_integer(batch_size, "batch_size")
    if progress_every == 0 and isinstance(progress_every, (int, np.integer)) and not isinstance(progress_every, (bool, np.bool_)):
        progress_every = None
    if progress_every is not None:
        progress_every = _positive_integer(progress_every, "progress_every")
    start = _day_timestamp(start, "start")
    end = _day_timestamp(end, "end")
    if end < start:
        raise ValueError("end must be on or after start")
    features = list(features)
    if not features or any(not isinstance(name, str) for name in features) or len(set(features)) != len(features):
        raise ValueError("features must be nonempty unique names, in saved scaler order")
    metadata = ["prefecture", "town", "date", "latitude", "longitude", "elevation"]
    if set(features).intersection(metadata + ["year", "month", "day"]):
        raise ValueError("Saved weather target features cannot contain station/calendar metadata")
    absent = set(metadata) - set(history_df.columns)
    if absent:
        raise ValueError(f"History lacks required station metadata: {sorted(absent)}")
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any() or df["date"].dt.tz is not None:
        raise ValueError("History dates must be finite timezone-free daily dates")
    if not df["date"].eq(df["date"].dt.normalize()).all():
        raise ValueError("History contains timestamps that are not daily midnight dates")
    if prefecture is not None:
        df = df.loc[df["prefecture"] == prefecture].copy()
    if town is not None:
        df = df.loc[df["town"] == town].copy()
    if df.empty:
        raise ValueError("No selected towns in history input")
    if df[["prefecture", "town"]].isna().any().any():
        raise ValueError("Station prefecture/town keys cannot be missing")
    selected_keys = set(df[["prefecture", "town"]].itertuples(index=False, name=None))
    # Inspect only keys/dates above; future weather/metadata never enter seed state.
    df = df.loc[df["date"] < start].copy()
    historical_keys = set(df[["prefecture", "town"]].itertuples(index=False, name=None))
    missing_seeds = selected_keys - historical_keys
    if missing_seeds:
        labels = ", ".join(f"{pref}/{town_name}" for pref, town_name in sorted(missing_seeds))
        raise ValueError(f"Missing historical seed for selected towns: {labels}; no observations before forecast start")
    if df.duplicated(["prefecture", "town", "date"]).any():
        raise ValueError("Duplicate historical station/date records")
    direction_features = {DIRECTION_SIN, DIRECTION_COS}.intersection(features)
    if direction_features and len(direction_features) != 2:
        raise ValueError("Wind direction requires both saved sine and cosine features")
    if direction_features and ANGLE in df.columns:
        # Encode from the raw angle rather than trusting potentially stale helper columns.
        degrees = pd.to_numeric(df[ANGLE], errors="raise")
        radians = np.deg2rad(degrees)
        df[DIRECTION_SIN] = np.sin(radians)
        df[DIRECTION_COS] = np.cos(radians)
    absent = set(features) - set(df.columns)
    if absent:
        raise ValueError(f"History lacks saved model features: {sorted(absent)}")
    for name in features + ["latitude", "longitude", "elevation"]:
        df[name] = pd.to_numeric(df[name], errors="raise")

    histories, calendars, locations, stations, previous_pressures = [], [], [], [], []
    for (pref, town_name), g in df.groupby(["prefecture", "town"], sort=False):
        g = g.sort_values("date")
        label = f"{pref}/{town_name}"
        if len(g) < lookback:
            raise ValueError(f"{label}: need {lookback} historical seed days, found {len(g)}")
        if g["date"].iloc[-1] != start - pd.Timedelta(days=1):
            raise ValueError(f"{label}: history must end the day before forecast start")
        seed = g.tail(lookback)
        expected = pd.date_range(start - pd.Timedelta(days=lookback), periods=lookback, freq="D")
        if not np.array_equal(seed["date"].to_numpy(), expected.to_numpy()):
            raise ValueError(f"{label}: nonconsecutive or incomplete historical seed days")
        metadata_values = g[["latitude", "longitude", "elevation"]].to_numpy(dtype="float64")
        if not np.isfinite(metadata_values).all():
            raise ValueError(f"{label}: nonfinite station metadata")
        if not np.allclose(metadata_values, metadata_values[-1], rtol=0, atol=1e-6):
            raise ValueError(f"{label}: station coordinates/elevation must remain static")
        lat, lon, elevation = metadata_values[-1]
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"{label}: invalid latitude/longitude")
        seed_raw = _as_finite_matrix(seed[features].to_numpy(), (lookback, len(features)), f"{label} weather seed")
        histories.append(_as_finite_matrix(scaler.transform(seed_raw), seed_raw.shape, f"{label} scaled weather seed"))
        calendars.append(calendar(seed["date"]))
        locations.append(metadata_values[-1])
        stations.append((pref, town_name, float(lat), float(lon), float(elevation)))
        if "surface_pressure_mean" in features:
            previous_pressures.append(float(seed_raw[-1, features.index("surface_pressure_mean")]))
    station_count = len(stations)
    history = np.stack(histories)
    histcal = np.stack(calendars)
    locations_scaled = _as_finite_matrix(
        loc_scaler.transform(np.asarray(locations)), (station_count, 3), "Scaled station metadata"
    )
    loc_windows = np.broadcast_to(locations_scaled[:, None, :], (station_count, lookback, 3))
    days = pd.date_range(start, end, freq="D")
    feature_positions = {name: i for i, name in enumerate(features)}
    previous_pressures = np.asarray(previous_pressures, dtype="float32")
    frames = []
    for step, day in enumerate(days, start=1):
        targetcal = calendar([day])
        targetcal_batch = np.broadcast_to(targetcal, (station_count, 2)).copy()
        scaled_predictions = []
        for offset in range(0, station_count, batch_size):
            stop = min(offset + batch_size, station_count)
            past = np.concatenate([history[offset:stop], histcal[offset:stop], loc_windows[offset:stop]], axis=2)
            output = model([past, targetcal_batch[offset:stop]], training=False)
            scaled_predictions.append(_as_finite_matrix(output, (stop-offset, len(features)), "Model prediction"))
        scaled_prediction = np.concatenate(scaled_predictions, axis=0)
        physical = _as_finite_matrix(scaler.inverse_transform(scaled_prediction), scaled_prediction.shape, "Physical prediction")
        bounded = _bound_weather(physical, features)
        delta = None
        if "surface_pressure_mean" in feature_positions:
            pressures = bounded[:, feature_positions["surface_pressure_mean"]]
            delta = pressures - previous_pressures
            previous_pressures = pressures.copy()
            if "pressure_change" in feature_positions:
                bounded[:, feature_positions["pressure_change"]] = delta
        records = []
        for index, (pref, town_name, lat, lon, elevation) in enumerate(stations):
            record = {
                "prefecture": pref, "town": town_name, "date": day.strftime("%Y-%m-%d"),
                "latitude": lat, "longitude": lon, "elevation": elevation,
                "year": day.year, "month": day.month, "day": day.day,
            }
            record.update({name: float(bounded[index, feature_positions[name]]) for name in features if name not in {DIRECTION_SIN, DIRECTION_COS}})
            if direction_features:
                sine = bounded[index, feature_positions[DIRECTION_SIN]]
                cosine = bounded[index, feature_positions[DIRECTION_COS]]
                record[ANGLE] = float(np.degrees(np.arctan2(sine, cosine)) % 360)
            if delta is not None:
                record["pressure_change"] = float(delta[index])
            records.append(record)
        frames.append(pd.DataFrame.from_records(records))
        # Feed the bounded physical predictions through the ORIGINAL saved scaler order.
        next_scaled = _as_finite_matrix(scaler.transform(bounded), bounded.shape, "Bounded scaled feedback")
        history = np.concatenate([history[:, 1:, :], next_scaled[:, None, :]], axis=1)
        histcal = np.concatenate([histcal[:, 1:, :], targetcal_batch[:, None, :]], axis=1)
        if progress_every and (step % progress_every == 0 or step == len(days)):
            print(f"Forecasted {step}/{len(days)} days for {station_count} towns; through {day.date()}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    expected_rows = station_count * len(days)
    if len(result) != expected_rows or result.duplicated(["prefecture", "town", "date"]).any():
        raise AssertionError("Forecast output does not have exactly one row per selected town/day")
    counts = result.groupby(["prefecture", "town"], sort=False)["date"].nunique()
    if not counts.eq(len(days)).all():
        raise AssertionError("Forecast date coverage differs between towns")
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    if not np.isfinite(result[numeric_columns].to_numpy(dtype="float64")).all():
        raise ValueError("Forecast output contains nonfinite numeric values")
    result.attrs["forecast_start"] = str(start.date())
    result.attrs["forecast_end"] = str(end.date())
    result.attrs["forecast_days"] = len(days)
    result.attrs["town_count"] = station_count
    return result


# %% [markdown]
# ## 5. Shared preprocessing and chronological windows
# 
# When lookbacks differ, the largest configured lookback determines common eligible target dates. Each model then reads its own requested amount of past history for those same dates. Train/validation targets must be fully observed; test errors use each attribute's observed cells.


# %%
def build_groups(data, feature_scaler, location_scaler):
    result = []
    for (prefecture, town), part in data.groupby(["prefecture", "town"], sort=False):
        dates = pd.DatetimeIndex(part.date)
        if not np.all(np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1):
            raise ValueError(f"Nonconsecutive historical dates: {prefecture}/{town}.")
        locations = part[["latitude", "longitude", "elevation"]].to_numpy(dtype="float32")
        if not np.allclose(locations, locations[0], atol=1e-5, rtol=0):
            raise ValueError(f"Station coordinates changed: {prefecture}/{town}.")
        values = feature_scaler.transform(part[features].to_numpy(dtype="float32")).astype("float32")
        observed_targets = observed_df.loc[part.index, features].to_numpy(dtype="float32")
        complete_target = np.isfinite(observed_targets).all(axis=1)
        location = location_scaler.transform(locations[[0]])[0].astype("float32")
        result.append(dict(prefecture=prefecture, town=town, dates=dates, values=values,
                           calendar=calendar(dates), location=location, complete_target=complete_target))
    return result

def build_index(groups, first, last, minimum_history):
    pairs = []
    for group_id, group in enumerate(groups):
        targets = np.flatnonzero((group["dates"] >= pd.Timestamp(first)) & (group["dates"] <= pd.Timestamp(last)))
        pairs.extend((group_id, int(t)) for t in targets if t >= minimum_history)
    return np.asarray(pairs, dtype="int32").reshape(-1, 2)

def input_batch(pairs, groups, lookback):
    xx, target_calendar, yy = [], [], []
    for group_id, t in pairs:
        group = groups[int(group_id)]
        xx.append(np.concatenate([group["values"][t-lookback:t], group["calendar"][t-lookback:t],
                    np.broadcast_to(group["location"], (lookback, 3))], axis=1))
        target_calendar.append(group["calendar"][t])
        yy.append(group["values"][t])
    return np.stack(xx), np.stack(target_calendar), np.stack(yy)

# Fit preprocessing once on the training period, then share it across all runs.
train_rows = df.date.between(TRAIN_START, TRAIN_END)
if not train_rows.any():
    raise ValueError("No rows in the training period.")
scaler = StandardScaler().fit(df.loc[train_rows, features].to_numpy(dtype="float32"))
loc_scaler = StandardScaler().fit(df.loc[train_rows, ["latitude", "longitude", "elevation"]].drop_duplicates().to_numpy())
groups = build_groups(df, scaler, loc_scaler)
COMMON_LOOKBACK = max(s["lookback"] for s in EXPERIMENT_SETTINGS)
train_idx = build_index(groups, TRAIN_START, TRAIN_END, COMMON_LOOKBACK)
val_idx = build_index(groups, VAL_START, VAL_END, COMMON_LOOKBACK)
val_eval_idx = val_idx.copy()  # keep partial observations for per-attribute evaluation
test_idx = build_index(groups, TEST_START, TEST_END, COMMON_LOOKBACK)
target_quality = []
for split_name in ["train_idx", "val_idx"]:
    indices = globals()[split_name]
    complete = np.asarray([groups[int(g)]["complete_target"][int(t)] for g, t in indices], dtype=bool)
    globals()[split_name] = indices[complete]
    target_quality.append({"split": split_name, "candidate_windows": len(indices),
                           "missing_target_windows": int((~complete).sum()), "retained_windows": int(complete.sum())})
pd.DataFrame(target_quality).to_csv(LOG_DIR / "target_window_quality.csv", index=False)
if min(len(train_idx), len(val_idx)) == 0 or (RUN_EVALUATION and len(test_idx) == 0):
    raise ValueError("The configured dates do not provide the required train/validation/test windows.")
sample_rng = np.random.default_rng(SAMPLING_SEED)
if MAX_TRAIN_WINDOWS is not None and len(train_idx) > MAX_TRAIN_WINDOWS:
    train_idx = train_idx[sample_rng.choice(len(train_idx), MAX_TRAIN_WINDOWS, replace=False)]
np.savez_compressed(LOG_DIR / "shared_window_indices.npz", train=train_idx, validation=val_idx,
                    validation_evaluation=val_eval_idx, test=test_idx)
def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
WINDOW_HASHES = {"train": array_sha256(train_idx), "validation": array_sha256(val_idx),
                 "validation_evaluation": array_sha256(val_eval_idx), "test": array_sha256(test_idx)}
joblib.dump({"scaler": scaler, "loc_scaler": loc_scaler}, LOG_DIR / "shared_scalers.joblib")
normalizers = {}
for feature in PHYSICAL_FEATURES:
    values = observed_df.loc[train_rows, feature].to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError(f"No observed training values for {feature}.")
    scale = 180.0 if feature == ANGLE else float(np.std(finite, ddof=0))
    # A constant attribute is still scored; one physical unit is its explicit scale.
    normalizers[feature] = {"scale": scale if scale > 1e-12 else 1.0,
                            "rule": "half-circle (degrees)" if feature == ANGLE else "training population std" if scale > 1e-12 else "one physical unit (constant training feature)"}
pd.DataFrame([{"feature": f, **v} for f, v in normalizers.items()]).to_csv(LOG_DIR / "score_normalizers.csv", index=False)
plan = {"schema_version": "full_attributes_lstm_gru_v1", "dataset_sha256": DATA_SHA256,
        "model_types": MODEL_TYPES, "settings": EXPERIMENT_SETTINGS, "seeds": SEEDS,
        "physical_attributes": PHYSICAL_FEATURES, "encoded_features": features,
        "sampling_seed": SAMPLING_SEED, "common_minimum_history": COMMON_LOOKBACK,
        "window_sha256": WINDOW_HASHES, "window_counts": {"train": len(train_idx), "validation": len(val_idx),
                                                          "validation_evaluation": len(val_eval_idx), "test": len(test_idx)},
        "splits": {"train": [TRAIN_START, TRAIN_END], "validation": [VAL_START, VAL_END], "test": [TEST_START, TEST_END]},
        "selection": "minimum mean-over-attributes recursive validation RMSE / training std; mean over listed seeds",
        "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "future_accuracy": None}
(LOG_ROOT / "comparison_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
print("Shared windows:", plan["window_counts"])
print("All runs use the same 37 attributes, scalers, target dates and sampled windows.")


# %% [markdown]
# ## 6. LSTM / GRU model factory and training
# 
# Only recurrent cell type changes within a matched setting. Every run gets an independent shuffle stream reset to its seed, model file, history, parameter count and configuration. Early stopping chooses the epoch using one-step scaled validation MSE.


# %%
def build_weather_model(model_type, setting):
    """Only the recurrent cell type changes between matched LSTM and GRU runs."""
    layer_type = {"LSTM": tf.keras.layers.LSTM, "GRU": tf.keras.layers.GRU}[model_type]
    lookback = setting["lookback"]
    past = tf.keras.layers.Input((lookback, len(features) + 5), name="past_weather_calendar_location")
    target = tf.keras.layers.Input((2,), name="target_day_calendar")
    z = past
    for index, units in enumerate(setting["recurrent_units"]):
        z = layer_type(units, return_sequences=index < len(setting["recurrent_units"]) - 1,
                       dropout=setting["dropout"], name=f"{model_type.lower()}_{index + 1}")(z)
    z = tf.keras.layers.Concatenate()([z, target])
    z = tf.keras.layers.Dense(setting["dense_units"], activation="relu")(z)
    output = tf.keras.layers.Dense(len(features), name="next_day_weather")(z)
    result = tf.keras.Model([past, target], output, name=f"weather_{model_type.lower()}")
    result.compile(optimizer=tf.keras.optimizers.Adam(setting["learning_rate"]), loss="mse", metrics=["mae"])
    return result

def check_model_contract(model, model_type, lookback):
    if tuple(features) != EXPECTED_FEATURES or len(features) != 38:
        raise ValueError("Every run must retain the complete weather feature contract.")
    expected = [(None, lookback, len(features) + 5), (None, 2)]
    if model.input_shape != expected or model.output_shape != (None, len(features)):
        raise ValueError("Saved model shapes do not match the full weather feature contract.")
    recurrent = [layer.__class__.__name__ for layer in model.layers if isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU))]
    if not recurrent or any(name != model_type for name in recurrent):
        raise ValueError(f"Saved model architecture is not {model_type}: {recurrent}.")

def make_window_datasets(lookback, seed):
    # Independent stream per run: LSTM and GRU see identical epoch shuffle orders.
    shuffle_rng = np.random.default_rng(seed)
    def dataset(indices, shuffle):
        def generate():
            order = indices.copy()
            if shuffle:
                shuffle_rng.shuffle(order)
            for offset in range(0, len(order), BATCH_SIZE):
                xx, cc, yy = input_batch(order[offset:offset + BATCH_SIZE], groups, lookback)
                yield (xx, cc), yy
        signature = ((tf.TensorSpec((None, lookback, len(features) + 5), tf.float32),
                      tf.TensorSpec((None, 2), tf.float32)), tf.TensorSpec((None, len(features)), tf.float32))
        result = tf.data.Dataset.from_generator(generate, output_signature=signature)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        batches = (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE
        result = result.apply(tf.data.experimental.assert_cardinality(batches))
        return result.with_options(options).prefetch(1)
    return dataset(train_idx, True), dataset(val_idx, False)

def make_forecast_call(local_model, lookback):
    # A fresh closure per model avoids reusing a traced function from another run.
    @tf.function(input_signature=[tf.TensorSpec((None, lookback, len(features) + 5), tf.float32),
                                  tf.TensorSpec((None, 2), tf.float32)])
    def batch(past, target):
        return local_model([past, target], training=False)
    def call(inputs, training=False):
        return batch(tf.convert_to_tensor(inputs[0], tf.float32), tf.convert_to_tensor(inputs[1], tf.float32))
    return call

def requested_contract(model_type, setting, seed):
    return {"schema_version": plan["schema_version"], "model_type": model_type,
            "setting": setting, "seed": seed, "features": features,
            "physical_attributes": PHYSICAL_FEATURES, "dataset_sha256": DATA_SHA256,
            "window_sha256": WINDOW_HASHES, "sampling_seed": SAMPLING_SEED,
            "splits": plan["splits"], "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "max_train_windows": MAX_TRAIN_WINDOWS, "max_forward_fill_days": MAX_FORWARD_FILL_DAYS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE, "lr_patience": LR_PATIENCE,
            "input_repair_fallbacks": df.attrs.get("input_repair_fallbacks", {}),
            "versions": plan["versions"]}

def train_or_load(model_type, setting, seed):
    run_id = f"{model_type.lower()}/{setting['name']}/seed_{seed}"
    model_dir, log_dir = MODEL_ROOT / run_id, LOG_ROOT / run_id
    for folder in [model_dir, log_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    contract = requested_contract(model_type, setting, seed)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    paths = [model_dir / n for n in ["model.keras", "scalers.joblib", "config.json"]]
    reuse = all(p.is_file() for p in paths) and not FORCE_RETRAIN
    if not reuse and any(p.exists() for p in paths) and not FORCE_RETRAIN:
        raise FileNotFoundError(f"Incomplete artifacts in {model_dir}. Choose a new EXPERIMENT_ID or FORCE_RETRAIN=True.")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    if reuse:
        saved = json.loads(paths[2].read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint or saved.get("contract") != json.loads(json.dumps(contract)):
            raise ValueError(f"Saved {run_id} has different attributes/settings/data. Use a new EXPERIMENT_ID or FORCE_RETRAIN=True.")
        saved_scales = joblib.load(paths[1])
        for key, expected in [("scaler", scaler), ("loc_scaler", loc_scaler)]:
            for field in ["mean_", "scale_", "var_"]:
                if not np.array_equal(getattr(saved_scales[key], field), getattr(expected, field)):
                    raise ValueError(f"Saved {run_id} preprocessing differs from the shared training scalers.")
        local_model = tf.keras.models.load_model(paths[0], compile=False)
        training = saved["training"]
        print("Loaded:", run_id)
    else:
        if not TRAIN_IF_MISSING and not FORCE_RETRAIN:
            raise FileNotFoundError(f"No complete saved model for {run_id}. Enable TRAIN_IF_MISSING.")
        local_model = build_weather_model(model_type, setting)
        train_data, val_data = make_window_datasets(setting["lookback"], seed)
        print("Training:", run_id, "| Parameters:", local_model.count_params(), flush=True)
        start_time = time.perf_counter()
        history = local_model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=LR_PATIENCE),
            tf.keras.callbacks.CSVLogger(str(log_dir / "training_history.csv"))], verbose=2)
        elapsed = time.perf_counter() - start_time
        training = {"epochs_completed": len(history.history["loss"]),
                    "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
                    "best_validation_scaled_mse": float(np.min(history.history["val_loss"])),
                    "training_seconds": elapsed, "parameter_count": local_model.count_params(),
                    "train_windows": len(train_idx), "validation_windows": len(val_idx)}
        history_frame = pd.DataFrame(history.history)
        ax = history_frame[["loss", "val_loss"]].rename_axis("epoch_zero_based").plot(figsize=(8, 3), title=run_id)
        ax.set_ylabel("Scaled MSE")
        plt.tight_layout()
        plt.savefig(log_dir / "training_loss.png", dpi=120, bbox_inches="tight")
        plt.show()
        local_model.save(paths[0])
        joblib.dump({"scaler": scaler, "loc_scaler": loc_scaler}, paths[1])
        saved = {"fingerprint": fingerprint, "contract": contract, "training": training}
        paths[2].write_text(json.dumps(saved, indent=2), encoding="utf-8")
    check_model_contract(local_model, model_type, setting["lookback"])
    run = {"run_id": run_id, "model_type": model_type, "setting_name": setting["name"],
           "setting": copy.deepcopy(setting), "seed": seed, "reused_model": reuse,
           "model_dir": str(model_dir), "log_dir": str(log_dir), "fingerprint": fingerprint,
           "training": training, "window_sha256": WINDOW_HASHES}
    (log_dir / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return local_model, run


# %% [markdown]
# ## 7. Common evaluation functions
# 
# MAE/RMSE/bias use original units and observed truth. Direction uses circular error. One-step persistence uses the preceding observation; recursive persistence repeats the last observation before the forecast origin. The horizon table summarizes all days from 1 through h.


# %%
def physical_rows(scaled_predictions):
    raw = _bound_weather(scaler.inverse_transform(np.asarray(scaled_predictions)), features)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite one-step model predictions; inspect the saved model/scalers, not the observation mask.")
    rows = []
    for values in raw:
        values = dict(zip(features, map(float, values)))
        if "wind_direction_sin" in values and "wind_direction_cos" in values:
            values["wind_direction_10m_dominant"] = float(np.degrees(np.arctan2(values.pop("wind_direction_sin"), values.pop("wind_direction_cos"))) % 360)
        rows.append(values)
    return pd.DataFrame(rows)

def metric_tables(predicted, actual, previous):
    join = ["prefecture", "town", "date"]
    aligned = predicted.merge(actual, on=join, suffixes=("_predicted", "_actual"), validate="one_to_one")
    if len(aligned) != len(predicted):
        raise ValueError("Some forecast dates lack matching observed weather; metrics cannot silently drop targets.")
    baseline_keys = join if "date" in previous else ["prefecture", "town"]
    aligned = aligned.merge(previous, on=baseline_keys, how="left", validate="one_to_one" if "date" in previous else "many_to_one")
    physical_features = [c for c in features if c not in {"wind_direction_sin", "wind_direction_cos"}]
    if "wind_direction_sin" in features:
        physical_features += [ANGLE]
    def summarize(frame, horizon=None):
        result=[]
        for feature in physical_features:
            observed=frame[feature + "_actual"].to_numpy()
            estimate=frame[feature + "_predicted"].to_numpy()
            baseline=frame[feature + "_baseline"].to_numpy()
            if not np.isfinite(estimate).all():
                raise ValueError(f"Nonfinite model predictions for {feature}; these cannot be scored as missing actuals.")
            valid=np.isfinite(observed)
            paired=valid & np.isfinite(baseline)
            if not valid.any():
                result.append({"feature":feature,"MAE":np.nan,"RMSE":np.nan,"bias":np.nan,
                               "persistence_MAE":np.nan,"persistence_RMSE":np.nan,"MAE_skill_vs_persistence":np.nan,
                               "count":0,"missing_actual_count":len(frame),"paired_baseline_count":0,"horizon_days":horizon})
                continue
            error=estimate-observed; baseline_error=baseline-observed
            if feature == ANGLE:
                error=(error+180)%360-180; baseline_error=(baseline_error+180)%360-180
            mae=float(np.abs(error[valid]).mean())
            baseline_mae=float(np.abs(baseline_error[paired]).mean()) if paired.any() else np.nan
            paired_mae=float(np.abs(error[paired]).mean()) if paired.any() else np.nan
            result.append({"feature":feature, "MAE":mae, "RMSE":float(np.sqrt(np.mean(error[valid]**2))),
                           "bias":float(np.mean(error[valid])), "persistence_MAE":baseline_mae,
                           "persistence_RMSE":float(np.sqrt(np.mean(baseline_error[paired]**2))) if paired.any() else np.nan,
                           "MAE_skill_vs_persistence": np.nan if not np.isfinite(baseline_mae) or baseline_mae == 0 else 1-paired_mae/baseline_mae,
                           "count":int(valid.sum()), "missing_actual_count":int((~valid).sum()),
                           "paired_baseline_count":int(paired.sum()), "horizon_days":horizon})
        return pd.DataFrame(result)
    metrics=summarize(aligned)
    horizons = None
    if "horizon_day" in aligned:
        maximum = int(aligned.horizon_day.max())
        horizons = pd.concat([summarize(aligned[aligned.horizon_day <= h], h)
                              for h in sorted(set(HORIZON_DAYS + [maximum])) if h <= maximum], ignore_index=True)
    return metrics, horizons, aligned

def evaluate_period(forecast_call, lookback, first, last, indices, output_dir):
    """Score identical physical attributes, dates and truth masks for every model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for offset in range(0, len(indices), FORECAST_BATCH_SIZE):
        pairs = indices[offset:offset + FORECAST_BATCH_SIZE]
        xx, cc, _ = input_batch(pairs, groups, lookback)
        values = physical_rows(forecast_call([xx, cc]))
        values["prefecture"] = [groups[int(g)]["prefecture"] for g, t in pairs]
        values["town"] = [groups[int(g)]["town"] for g, t in pairs]
        values["date"] = [groups[int(g)]["dates"][int(t)] for g, t in pairs]
        chunks.append(values)
    if not chunks:
        raise ValueError("No one-step windows in the requested evaluation period.")
    one_step = pd.concat(chunks, ignore_index=True)
    # Architecture comparison always covers all stations. Future-output filters
    # do not change the validation/test sample or the selected model.
    recursive = recursive_forecast(df, forecast_call, scaler, loc_scaler, features, lookback, first, last,
                                   batch_size=FORECAST_BATCH_SIZE, progress_every=None)
    recursive["date"] = pd.to_datetime(recursive["date"])
    recursive["horizon_day"] = (recursive.date - pd.Timestamp(first)).dt.days + 1
    join = ["prefecture", "town", "date"]
    keys = recursive[join]
    if len(one_step) != len(keys) or len(one_step.merge(keys, on=join, validate="one_to_one")) != len(keys):
        raise ValueError("One-step and recursive evaluation must cover the same stations and dates.")
    actual = observed_df[observed_df.date.between(first, last)][join + PHYSICAL_FEATURES]
    previous = observed_df[observed_df.date < pd.Timestamp(first)].sort_values("date").groupby(["prefecture", "town"], as_index=False).tail(1)
    previous = previous[["prefecture", "town"] + PHYSICAL_FEATURES].rename(columns={c: c + "_baseline" for c in PHYSICAL_FEATURES})
    daily_baseline = observed_df[join + PHYSICAL_FEATURES].copy()
    daily_baseline["date"] += pd.Timedelta(days=1)
    daily_baseline = daily_baseline.rename(columns={c: c + "_baseline" for c in PHYSICAL_FEATURES})
    one_metrics, _, one_predictions = metric_tables(one_step, actual, daily_baseline)
    recursive_metrics, horizon_metrics, recursive_predictions = metric_tables(recursive, actual, previous)
    results = {"one_step": one_metrics, "recursive": recursive_metrics}
    summary = {}
    for mode, table in results.items():
        if set(table.feature) != set(PHYSICAL_FEATURES) or (table["count"] == 0).any():
            raise ValueError("Every weather attribute needs observed values for a complete model comparison.")
        table["normalizer"] = table.feature.map(lambda f: normalizers[f]["scale"])
        table["normalized_RMSE"] = table.RMSE / table.normalizer
        table.to_csv(output_dir / f"{mode}_metrics.csv", index=False)
        summary[f"{mode}_mean_normalized_RMSE"] = float(table.normalized_RMSE.mean())
    horizon_metrics["normalizer"] = horizon_metrics.feature.map(lambda f: normalizers[f]["scale"])
    horizon_metrics["normalized_RMSE"] = horizon_metrics.RMSE / horizon_metrics.normalizer
    horizon_metrics.to_csv(output_dir / "recursive_horizon_metrics.csv", index=False)
    one_predictions.to_csv(output_dir / "one_step_historical_predictions.csv", index=False, date_format="%Y-%m-%d")
    recursive_predictions.to_csv(output_dir / "recursive_historical_predictions.csv", index=False, date_format="%Y-%m-%d")
    summary.update({"start": first, "end": last, "prediction_rows": len(recursive_predictions),
                    "physical_attributes": len(PHYSICAL_FEATURES), "recursive_horizons": sorted(horizon_metrics.horizon_days.unique().astype(int).tolist())})
    (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# %% [markdown]
# ## 8. Train both models and freeze the validation selection
# 
# Each validation recursive forecast starts on `VAL_START` and continues through `VAL_END` without refreshing observed history. The configuration ranking averages normalized RMSE across all attributes and then across seeds. The first configured seed represents the selected configuration for future forecasts.


# %%
# Train/load each requested run and compare VALIDATION only at this stage.
runs, validation_rows = [], []
for setting in EXPERIMENT_SETTINGS:
    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            local_model, run = train_or_load(model_type, setting, seed)
            forecast_call = make_forecast_call(local_model, setting["lookback"])
            validation_summary = evaluate_period(forecast_call, setting["lookback"], VAL_START, VAL_END,
                                                 val_eval_idx, Path(run["log_dir"]) / "validation")
            run["validation"] = validation_summary
            runs.append(run)
            validation_rows.append({"run_id": run["run_id"], "model_type": model_type,
                                    "setting_name": setting["name"], "seed": seed, **run["training"],
                                    **validation_summary})
            print(run["run_id"], "recursive validation normalized RMSE:",
                  validation_summary["recursive_mean_normalized_RMSE"], flush=True)

validation_comparison = pd.DataFrame(validation_rows)
validation_comparison.to_csv(LOG_ROOT / "validation_comparison.csv", index=False)
score_column = "recursive_mean_normalized_RMSE"
validation_ranking = validation_comparison.groupby(["model_type", "setting_name"], as_index=False).agg(
    validation_mean_normalized_RMSE=(score_column, "mean"),
    validation_seed_std=(score_column, "std"),
    seed_count=("seed", "count"), mean_parameter_count=("parameter_count", "mean"),
    mean_training_seconds=("training_seconds", "mean"))
validation_ranking = validation_ranking.sort_values(
    ["validation_mean_normalized_RMSE", "model_type", "setting_name"], kind="stable").reset_index(drop=True)
validation_ranking.insert(0, "rank", np.arange(1, len(validation_ranking) + 1))
validation_ranking.to_csv(LOG_ROOT / "validation_configuration_ranking.csv", index=False)
chosen = validation_ranking.iloc[0]
# Seed selection is declared in advance; do not cherry-pick the luckiest seed.
selected_run = next(r for r in runs if r["model_type"] == chosen.model_type
                    and r["setting_name"] == chosen.setting_name and r["seed"] == SEEDS[0])
selection = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
             "criterion": plan["selection"], "selected_model_type": chosen.model_type,
             "selected_setting": chosen.setting_name, "representative_seed": SEEDS[0],
             "forecast_run_id": selected_run["run_id"],
             "validation_mean_normalized_RMSE": float(chosen.validation_mean_normalized_RMSE),
             "new_test_metrics_used_for_selection": False,
             "note": "Single-seed results do not establish a statistically reliable model difference." if len(SEEDS) == 1 else
                     "Seed spread measures initialization variability, not independent weather years."}
(LOG_ROOT / "selection_frozen.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
(LOG_ROOT / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
print("Validation configuration ranking (lower error is better):")
display(validation_ranking)
print("Selected for future forecasts:", selected_run["run_id"])
ax = validation_ranking.set_index(["model_type", "setting_name"])["validation_mean_normalized_RMSE"].plot.bar(
    figsize=(9, 4), title="LSTM / GRU validation comparison — all attributes")
ax.set_ylabel("Mean normalized recursive RMSE (lower is better)")
plt.tight_layout()
plt.savefig(LOG_ROOT / "validation_comparison.png", dpi=120, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 9. Compare both models on heldout test weather
# 
# The validation selection is fixed. Test results provide a final comparison and never rerank it. Architecture comparison covers all stations; future-output station filters do not alter these scores.


# %%
def load_run_forecast(run):
    tf.keras.backend.clear_session()
    local_model = tf.keras.models.load_model(Path(run["model_dir"]) / "model.keras", compile=False)
    check_model_contract(local_model, run["model_type"], run["setting"]["lookback"])
    return local_model, make_forecast_call(local_model, run["setting"]["lookback"])

test_comparison, test_feature_metrics = pd.DataFrame(), pd.DataFrame()
if RUN_EVALUATION:
    if not (LOG_ROOT / "selection_frozen.json").is_file():
        raise RuntimeError("Complete validation and freeze the selection before testing.")
    test_rows, feature_tables = [], []
    for run in runs:
        local_model, forecast_call = load_run_forecast(run)
        summary = evaluate_period(forecast_call, run["setting"]["lookback"], TEST_START, TEST_END,
                                   test_idx, Path(run["log_dir"]) / "test")
        test_rows.append({"run_id": run["run_id"], "model_type": run["model_type"],
                          "setting_name": run["setting_name"], "seed": run["seed"], **summary})
        for mode in ["one_step", "recursive"]:
            table = pd.read_csv(Path(run["log_dir"]) / "test" / f"{mode}_metrics.csv")
            table.insert(0, "mode", mode)
            table.insert(0, "run_id", run["run_id"])
            feature_tables.append(table)
    test_comparison = pd.DataFrame(test_rows)
    test_feature_metrics = pd.concat(feature_tables, ignore_index=True)
    test_comparison.to_csv(LOG_ROOT / "test_comparison.csv", index=False)
    test_feature_metrics.to_csv(LOG_ROOT / "test_all_attribute_metrics.csv", index=False)
    print("Heldout test comparison; the validation selection remains fixed:")
    display(test_comparison)
    display(test_feature_metrics[test_feature_metrics.feature.isin(
        ["temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean", "rain_sum"])])
else:
    print("Heldout test skipped by RUN_EVALUATION=False; validation comparison is complete.")


# %% [markdown]
# ## 10. View LSTM / GRU historical comparisons


# %%
if RUN_EVALUATION:
    # Show both architectures on the same dates. The shaded/actual support is
    # determined by observed truth, never by which model gives a smaller error.
    for feature in ["temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean", "wind_speed_10m_max", "rain_sum"]:
        fig, ax = plt.subplots(figsize=(10, 4))
        actual_added = False
        for run in runs:
            frame = pd.read_csv(Path(run["log_dir"]) / "test/recursive_historical_predictions.csv", parse_dates=["date"])
            columns = ["date", feature + "_actual", feature + "_predicted"]
            observed_plot = frame[columns].copy()
            observed_plot.loc[~np.isfinite(observed_plot[feature + "_actual"]), feature + "_predicted"] = np.nan
            daily = observed_plot.groupby("date")[[feature + "_actual", feature + "_predicted"]].mean()
            if not actual_added:
                ax.plot(daily.index, daily[feature + "_actual"], label="Observed", color="black", linewidth=2)
                actual_added = True
            ax.plot(daily.index, daily[feature + "_predicted"], label=run["run_id"])
        ax.set_title(f"Heldout recursive comparison — {feature} (station average)")
        ax.set_ylabel(feature)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(LOG_ROOT / f"test_comparison_{feature}.png", dpi=120, bbox_inches="tight")
        plt.show()


# %% [markdown]
# ## 11. Forecast with the model selected on validation
# 
# The default forecast covers 2026-02-01 through 2027-01-31. Both models are compared historically; the selected model produces the future CSV with all 37 physical attributes. To forecast with a particular architecture, run that architecture alone using `MODEL_CHOICE`. A short historical test does not establish annual forecast accuracy.


# %%
if RUN_FORECAST:
    local_model, forecast_call = load_run_forecast(selected_run)
    selected = df
    if FORECAST_PREFECTURE is not None:
        selected = selected[selected.prefecture == FORECAST_PREFECTURE]
    if FORECAST_TOWN is not None:
        selected = selected[selected.town == FORECAST_TOWN]
    if selected.empty:
        raise ValueError("No stations match the future forecast filters.")
    start = pd.Timestamp(FORECAST_START).normalize() if FORECAST_START is not None else selected.date.max() + pd.Timedelta(days=1)
    end = pd.Timestamp(FORECAST_END).normalize() if FORECAST_END is not None else start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    future_df = recursive_forecast(df, forecast_call, scaler, loc_scaler, features,
        selected_run["setting"]["lookback"], start, end, prefecture=FORECAST_PREFECTURE,
        town=FORECAST_TOWN, batch_size=FORECAST_BATCH_SIZE, progress_every=PROGRESS_EVERY)
    future_df["date"] = pd.to_datetime(future_df["date"])
    if set(PHYSICAL_FEATURES) - set(future_df) or not np.isfinite(future_df[PHYSICAL_FEATURES].to_numpy()).all():
        raise ValueError("Every one of the 37 physical weather attributes must be present and finite.")
    path = FORECAST_DIR / "future_daily_weather_attributes.csv"
    future_df.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", float_format="%.2f")
    towns = future_df.groupby(["prefecture", "town"]).ngroups
    coverage = {"run_id": selected_run["run_id"], "start": str(start.date()), "end": str(end.date()),
                "days_per_town": (end - start).days + 1, "towns": towns,
                "actual_rows": len(future_df), "physical_attributes": len(PHYSICAL_FEATURES),
                "all_dates_verified": True, "future_accuracy": None,
                "note": "Historical test errors do not establish year-ahead future accuracy."}
    (LOG_ROOT / "forecast_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print("Forecast model selected on validation:", selected_run["run_id"])
    print("Saved:", path, "| Rows:", len(future_df))
    display(future_df.head())
else:
    print("Future forecast skipped by RUN_FORECAST=False.")


# %% [markdown]
# ## 12. Preview future weather


# %%
if RUN_FORECAST:
    first=future_df[["prefecture","town"]].drop_duplicates().iloc[0]
    preview=future_df[(future_df.prefecture==first.prefecture)&(future_df.town==first.town)].set_index("date")
    for feature in ["temperature_2m_mean","pressure_msl_mean","rain_sum"]:
        if feature in preview:
            preview[feature].plot(figsize=(10,3),title=f"{first.prefecture}/{first.town} — {feature}")
            plt.ylabel(feature)
            plt.tight_layout()
            plt.savefig(LOG_ROOT/f"annual_forecast_{feature}.png",dpi=120,bbox_inches="tight")
            plt.show()


# %% [markdown]
# ## 13. Export comparison logs and a results report
# 
# The log ZIP contains all run settings, epoch logs, observed-vs-predicted tables, attribute metrics, horizon metrics, rankings and plots. Keras models and scalers remain in `trained_models`. The report uses actual results produced by this execution.


# %%
def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])

report = ["# Future Weather Attributes: LSTM and GRU Comparison", "",
          "## Model training", "",
          f"Models requested: {', '.join(MODEL_TYPES)}. Every run retains all 37 physical weather attributes "
          "(38 encoded channels because wind direction uses sine and cosine), historical calendar and static station metadata. "
          "The target-day calendar is supplied separately.", "",
          f"Training: {TRAIN_START} to {TRAIN_END}; validation: {VAL_START} to {VAL_END}; test: {TEST_START} to {TEST_END}. "
          f"Shared sampled training windows: {len(train_idx):,}; maximum epochs: {EPOCHS}; batch size: {BATCH_SIZE}; seeds: {SEEDS}.", "",
          "Training settings are varied without removing attributes. Both cell types receive identical shared scaler fits, "
          "window samples and per-seed shuffle streams. Equal layer widths give different parameter counts in LSTM and GRU; "
          "those counts are recorded. Checkpoints minimize one-step scaled validation MSE, with early stopping and learning-rate reduction.", "",
          "```json", json.dumps(EXPERIMENT_SETTINGS, indent=2), "```", "",
          "## Model evaluation", "",
          "MAE, RMSE and bias are reported in each attribute's original units. Wind-direction errors use the shortest angular distance. "
          "Persistence is scored on paired observed rows. Missing inputs use training-derived repairs; missing observed targets "
          "are never substituted with repaired values for scoring. Incomplete targets are excluded from training and checkpoint validation.", "",
          "One-step evaluation refreshes observed history each day. Recursive evaluation starts once at the beginning of the period "
          "and then feeds predictions back. Horizon tables are cumulative through day h. "
          "The ranking uses the mean of per-attribute recursive RMSE divided by observed training standard deviation "
          "(180 degrees for direction; one physical unit for a constant training feature), then averages across the listed seeds. "
          "This is regression, so classification accuracy percentages are not used.", "",
          "### Validation comparison", "", markdown_table(validation_ranking), "",
          "## Selected model", "",
          f"Selected architecture: **{selection['selected_model_type']}**; setting: **{selection['selected_setting']}**. "
          f"Validation mean normalized RMSE: {selection['validation_mean_normalized_RMSE']:.6f}. "
          f"Future forecast seed: {selection['representative_seed']} (the first configured seed). "
          "Selection was frozen before this notebook computed test results.", ""]
if RUN_EVALUATION:
    report += ["### Heldout test comparison", "", markdown_table(test_comparison[["run_id", "one_step_mean_normalized_RMSE", "recursive_mean_normalized_RMSE"]]), "",
               "Per-attribute errors for every run and both evaluation modes are exported in `test_all_attribute_metrics.csv`. "
               "The heldout test is for final comparison and does not change the validation selection.", ""]
report += ["## Interpretation and limits", "", selection["note"],
           "Validation uses one continuous recursive period, not repeated independent backtests. "
           "Test dates are already known from the earlier project work and are not a newly blind benchmark. "
           "Historical errors over these periods do not verify an unobserved annual forecast. "
           "Training durations are observations from this machine, not controlled speed benchmarks.", "",
           "## Retained weather attributes", "", ", ".join(PHYSICAL_FEATURES), ""]
(LOG_ROOT / "model_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
completion = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_count": len(runs),
              "models": MODEL_TYPES, "all_attributes_retained": True,
              "test_completed": RUN_EVALUATION, "forecast_completed": RUN_FORECAST,
              "selected_run_id": selected_run["run_id"]}
(LOG_ROOT / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
print("Comparison tables and report:", LOG_ROOT)
if EXPORT_LOG_ARCHIVE:
    archive_dir = ROOT / "experiment_exports"
    archive_dir.mkdir(exist_ok=True)
    archive_path = shutil.make_archive(str(archive_dir / f"{EXPERIMENT_ID}_comparison_logs"), "zip", LOG_ROOT)
    print("All comparison logs, predictions and plots:", archive_path)
print("Saved model artifacts:", MODEL_ROOT)

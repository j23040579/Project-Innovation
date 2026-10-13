"""Forecast weather through an inclusive date using the supplied notebook's artifacts.

The notebook saves ``model.keras``, ``scalers.joblib`` and ``config.json`` in
``trained_models/future_weather/<experiment>``. This module loads those artifacts;
it never trains a replacement model. Latitude and longitude here identify weather
stations. They remain fixed and are separate from predicted typhoon coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ANGLE = "wind_direction_10m_dominant"
KEY = ["prefecture", "town", "date"]
NONNEG = (
    "rain_", "precipitation_", "snowfall_", "sunshine_duration",
    "daylight_duration", "shortwave_radiation_sum", "wind_speed_",
    "wind_gusts_", "river_discharge",
)
PERCENT = ("relative_humidity_", "cloud_cover_")
STATIC = (
    "location_id", "prefecture", "town", "latitude", "longitude", "elevation",
    "utc_offset_seconds", "openmeteo_latitude", "openmeteo_longitude",
    "openmeteo_elevation",
)


def _day(value: Any) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise ValueError("A valid requested end date is required (YYYY-MM-DD).")
    if day.tzinfo is not None:
        day = day.tz_localize(None)
    return day.normalize()


def _validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(KEY) - set(frame.columns)
    if missing:
        raise ValueError(f"Daily weather is missing columns: {sorted(missing)}")
    result = frame.copy()
    if result.empty:
        raise ValueError("Daily weather contains no rows.")
    try:
        dates = pd.to_datetime(result["date"], errors="raise")
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        result["date"] = dates.dt.normalize()
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Daily weather dates must be valid, consistently formatted dates.") from exc
    if result[KEY].isna().any().any():
        raise ValueError("Daily weather contains missing prefecture, town or date keys.")
    for column in ("prefecture", "town"):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Daily weather contains an empty {column} key.")
    duplicates = result.duplicated(KEY, keep=False)
    if duplicates.any():
        example = result.loc[duplicates, KEY].iloc[0].to_dict()
        raise ValueError(f"Duplicate (prefecture, town, date) weather rows: {example}")
    return result.sort_values(KEY).reset_index(drop=True)


def read_daily(path: str | Path) -> pd.DataFrame:
    """Read a daily CSV; dates become normalized pandas timestamps.

    Keys must be unique for each weather station and day. Missing auxiliary
    values such as the first historical pressure difference are permitted here;
    model input values are validated only when forecasting is necessary.
    """
    return _validate_daily(pd.read_csv(path, low_memory=False))


def _calendar(dates: Any) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    angle = 2 * np.pi * (dates.dayofyear.to_numpy() - 1) / 365.25
    return np.column_stack((np.sin(angle), np.cos(angle))).astype("float32")


def _load_weather_artifacts(model_dir: str | Path):
    folder = Path(model_dir)
    needed = [folder / name for name in ("config.json", "scalers.joblib", "model.keras")]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Future weather forecasting requires the trained weather notebook artifacts. "
            "Missing: " + ", ".join(missing) + ". Run the weather notebook's save-model "
            "cell and supply that directory. The forecast CSV and trajectory_lstm.keras "
            "cannot replace the weather model and its original scalers."
        )
    config = json.loads(needed[0].read_text(encoding="utf-8"))
    features = config.get("features")
    if not isinstance(features, list) or not features or any(not isinstance(x, str) for x in features):
        raise ValueError("Weather config.json must contain an ordered, nonempty features list.")
    if len(set(features)) != len(features):
        raise ValueError("Weather config.json contains duplicate features.")
    lookback = config.get("lookback")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Weather config.json lookback must be a positive integer.")
    import joblib

    scales = joblib.load(needed[1])
    if not isinstance(scales, dict) or not {"scaler", "loc_scaler"}.issubset(scales):
        raise ValueError("Weather scalers.joblib must contain 'scaler' and 'loc_scaler'.")
    # TensorFlow is deliberately optional until a new forecast is actually needed.
    import tensorflow as tf

    model = tf.keras.models.load_model(needed[2], compile=False)
    return config, scales, model


def _feature_frame(frame: pd.DataFrame, features: list[str], name: str) -> pd.DataFrame:
    values = frame.copy()
    if "wind_direction_sin" in features or "wind_direction_cos" in features:
        if ANGLE in values:
            direction = np.deg2rad(pd.to_numeric(values[ANGLE], errors="coerce"))
            values["wind_direction_sin"] = np.sin(direction)
            values["wind_direction_cos"] = np.cos(direction)
    missing = set(features) - set(values)
    if missing:
        raise ValueError(f"{name}: historical weather lacks model features: {sorted(missing)}")
    values = values[features].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError(f"{name}: the weather lookback contains missing or nonfinite model inputs.")
    return values


def ensure_weather_through(
    history_df: pd.DataFrame,
    model_dir: str | Path | None,
    end_date: Any,
    prefecture: str | None = None,
    town: str | None = None,
) -> pd.DataFrame:
    """Return selected history plus daily forecasts through ``end_date`` inclusive.

    Each town starts after its own latest input date and requires a continuous
    historical series. Different town end dates are supported. All towns needing
    the same next day are predicted in one model call. The returned frame keeps
    historical rows, including any already later than the requested end date.
    ``weather_source`` distinguishes provided inputs from newly forecast rows;
    existing source annotations are preserved. Inputs without source metadata
    are labeled ``provided`` because a CSV alone does not establish observation
    versus prior forecast provenance.

    No artifacts or TensorFlow are required when every selected station already
    covers the requested date. Interior historical gaps always raise an explicit
    error; extrapolation must not silently stand in for missing observations.
    """
    end = _day(end_date)
    history = _validate_daily(history_df)
    if prefecture is not None:
        history = history.loc[history.prefecture.eq(prefecture)]
    if town is not None:
        history = history.loc[history.town.eq(town)]
    if history.empty:
        raise ValueError(f"No weather stations match prefecture={prefecture!r}, town={town!r}.")
    history = history.copy().reset_index(drop=True)
    if "weather_source" not in history:
        history["weather_source"] = "provided"
    else:
        history["weather_source"] = history["weather_source"].fillna("provided")
    grouped = list(history.groupby(["prefecture", "town"], sort=False))
    pending = []
    for key, group in grouped:
        dates = pd.DatetimeIndex(group.date)
        if end < dates[0]:
            raise ValueError(
                f"{key[0]}/{key[1]}: requested date {end.date()} precedes the first "
                f"available historical date {dates[0].date()}; forward forecasting cannot backfill it."
            )
        if len(dates) > 1 and not np.all(np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1):
            raise ValueError(
                f"{key[0]}/{key[1]}: historical weather has missing daily dates. "
                "Repair the historical series before forecasting; gaps are not skipped."
            )
        if dates[-1] < end:
            pending.append((key, group))
    if not pending:
        return history.sort_values(KEY).reset_index(drop=True)
    if model_dir is None:
        raise FileNotFoundError("A weather --model-dir containing model.keras, scalers.joblib and config.json is required to extend weather history.")
    config, scales, model = _load_weather_artifacts(model_dir)
    features = config["features"]
    lookback = config["lookback"]
    scaler, loc_scaler = scales["scaler"], scales["loc_scaler"]
    states = []
    for (pref, place), group in pending:
        name = f"{pref}/{place}"
        if len(group) < lookback:
            raise ValueError(f"{name}: need {lookback} consecutive historical days; found {len(group)}.")
        tail = group.tail(lookback).copy()
        if "elevation" not in tail:
            if "openmeteo_elevation" in tail:
                tail["elevation"] = tail["openmeteo_elevation"]
            else:
                raise ValueError(f"{name}: missing elevation (or openmeteo_elevation) station metadata.")
        elif "openmeteo_elevation" in tail:
            tail["elevation"] = tail["elevation"].fillna(tail["openmeteo_elevation"])
        location_cols = ["latitude", "longitude", "elevation"]
        missing = set(location_cols) - set(tail)
        if missing:
            raise ValueError(f"{name}: missing static weather station coordinates: {sorted(missing)}")
        locations = tail[location_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(locations).all():
            raise ValueError(f"{name}: station coordinates/elevation must be finite.")
        if not np.allclose(locations, locations[-1], atol=1e-6, rtol=0):
            raise ValueError(f"{name}: weather station coordinates/elevation vary inside the lookback. Use station coordinates, not cyclone trajectory coordinates.")
        if not (-90 <= locations[-1, 0] <= 90 and -180 <= locations[-1, 1] <= 180):
            raise ValueError(f"{name}: station latitude/longitude are outside geographic bounds.")
        raw = _feature_frame(tail, features, name).to_numpy(dtype="float32")
        scaled = np.asarray(scaler.transform(raw), dtype="float32")
        if scaled.shape != raw.shape or not np.isfinite(scaled).all():
            raise ValueError(f"{name}: weather scaler returned an invalid lookback.")
        loc = np.asarray(loc_scaler.transform(locations[[-1]]), dtype="float32")[0]
        if loc.shape != (3,) or not np.isfinite(loc).all():
            raise ValueError(f"{name}: location scaler must return three finite values.")
        last = tail.iloc[-1]
        static = {column: last[column] for column in STATIC if column in tail}
        static.update(latitude=float(locations[-1, 0]), longitude=float(locations[-1, 1]), elevation=float(locations[-1, 2]))
        previous_pressure = None
        if "surface_pressure_mean" in features:
            previous_pressure = float(raw[-1, features.index("surface_pressure_mean")])
        states.append(dict(history=scaled, histcal=_calendar(tail.date), loc=loc,
                           next_day=_day(last.date) + pd.Timedelta(days=1),
                           static=static, previous_pressure=previous_pressure))
    rows = []
    start = min(state["next_day"] for state in states)
    for day in pd.date_range(start, end, freq="D"):
        active = [state for state in states if state["next_day"] == day]
        if not active:
            continue
        target = np.repeat(_calendar([day]), len(active), axis=0)
        batch = np.stack([
            np.concatenate([state["history"], state["histcal"],
                            np.broadcast_to(state["loc"], (lookback, 3))], axis=1)
            for state in active
        ]).astype("float32")
        predicted = np.asarray(model.predict([batch, target], verbose=0))
        if predicted.shape != (len(active), len(features)) or not np.isfinite(predicted).all():
            raise ValueError(f"Weather model returned invalid predictions for {day.date()}: expected {(len(active), len(features))}, received {predicted.shape}.")
        physical = np.asarray(scaler.inverse_transform(predicted), dtype="float64")
        if physical.shape != predicted.shape or not np.isfinite(physical).all():
            raise ValueError(f"Weather inverse scaler returned invalid values for {day.date()}.")
        for state, prediction in zip(active, physical):
            values = {feature: float(value) for feature, value in zip(features, prediction)}
            for feature in features:
                if feature.startswith(NONNEG):
                    values[feature] = max(0.0, values[feature])
                if feature.startswith(PERCENT):
                    values[feature] = float(np.clip(values[feature], 0, 100))
            if "wind_direction_sin" in values and "wind_direction_cos" in values:
                angle = float(np.degrees(np.arctan2(values["wind_direction_sin"], values["wind_direction_cos"])) % 360)
                values[ANGLE] = angle
                values["wind_direction_sin"] = float(np.sin(np.deg2rad(angle)))
                values["wind_direction_cos"] = float(np.cos(np.deg2rad(angle)))
            row = dict(state["static"], date=day, year=day.year, month=day.month,
                       day=day.day, weather_source="forecast", **values)
            if "split" in history:
                row["split"] = "forecast"
            if state["previous_pressure"] is not None:
                row["pressure_change"] = values["surface_pressure_mean"] - state["previous_pressure"]
                state["previous_pressure"] = values["surface_pressure_mean"]
            rows.append(row)
            next_scaled = np.asarray(scaler.transform(np.array([[values[name] for name in features]], dtype="float32")), dtype="float32")
            if next_scaled.shape != (1, len(features)) or not np.isfinite(next_scaled).all():
                raise ValueError(f"Weather scaler returned an invalid recursive state for {day.date()}.")
            state["history"] = np.concatenate((state["history"][1:], next_scaled), axis=0)
            state["histcal"] = np.concatenate((state["histcal"][1:], _calendar([day])), axis=0)
            state["next_day"] = day + pd.Timedelta(days=1)
    result = pd.concat([history, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return _validate_daily(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Historical daily weather CSV.")
    parser.add_argument("--model-dir", type=Path, help="Saved weather model/scalers/config directory; needed only for new dates.")
    parser.add_argument("--end-date", required=True, help="Inclusive requested endpoint, YYYY-MM-DD.")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV containing selected history plus new forecasts.")
    parser.add_argument("--prefecture")
    parser.add_argument("--town")
    args = parser.parse_args(argv)
    output = args.out.resolve()
    if output == args.input.resolve():
        parser.error("--out must not overwrite the input weather CSV.")
    if args.model_dir is not None and (output == args.model_dir.resolve() or args.model_dir.resolve() in output.parents):
        parser.error("--out must be outside the model artifact directory.")
    try:
        result = ensure_weather_through(read_daily(args.input), args.model_dir, args.end_date, args.prefecture, args.town)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".weather-", suffix=".csv", delete=False, encoding="utf-8-sig", newline="") as handle:
                temporary = Path(handle.name)
                result.to_csv(handle, index=False, date_format="%Y-%m-%d")
            os.replace(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    forecast_count = int(result.weather_source.eq("forecast").sum())
    print(f"Saved {len(result):,} daily station rows to {output}; {forecast_count:,} rows marked forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

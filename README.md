# Daily weather → reconstructed position → nearest Japanese prefecture → risk

The LSTM predicts latitude/longitude. Japan's `jp.shp` converts the predicted position into `nearest_prefecture`. Random Forest, CART, Gradient Boosting and SVM classify weather plus that prefecture. The output maps the original `Moderate` class to `MEDIUM`; risk classification does not use an LSTM.

`typhoon_pipeline.py` implements training, reconstruction, validated merging, and inference. `weather_forecast.py` extends the saved weather model to an inclusive requested end date. Neither script edits input files or replaces the supplied models/notebooks.

## Installation

Use Python 3.12 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
```

## 1. Train location reconstruction and risk models

Run from the directory containing these scripts. Substitute your file paths:

```bash
python typhoon_pipeline.py train-risk \
  --labels /path/to/Typhoon/data/processed/jma_day3_to_landfall_with_weather_prefecture.csv \
  --gis /path/to/Typhoon/data/gis/jp_shp/jp.shp \
  --start-date 2021-01-01 --end-date 2025-01-31 \
  --epochs 30 --lookback 1 --folds 3 \
  --out trained_daily_pipeline
```

The command saves:

- `location/model.keras`, `location/preprocessing.joblib`, `location/config.json`: a **new weather-only coordinate model**, with no latitude/longitude inputs.
- Four `*_model.joblib` classifiers, `metrics.json`, and per-class reports.
- `training_reconstructed.csv` and `testing_reconstructed.csv`: weather, predicted coordinates, reconstructed `nearest_prefecture`, and original event risk labels.

Whole storms (`intl_id`) are held out together. Training classifier inputs use grouped out-of-fold coordinate predictions. Its heldout storms are excluded from both location-model and classifier training. The saved deployment location model is trained on the risk-training storms, not the evaluation storms.

For just a standalone location model, run the same flags with `train-location` instead of `train-risk`, omitting `--gis` and `--folds`. This evaluates a separate holdout, then refits its deployment model on all supplied location labels.

The default **lookback of 1** retains Low, Moderate and High examples in the supplied four-day landfall sequences. A one-timestep LSTM is a weather-to-position regression and does not learn multi-day temporal memory. To use `--lookback 3`, supply enough earlier daily weather in each labelled sequence: the supplied four-day data alone then produces only Day −1/Day 0 targets, both High, and cannot train the three-class risk model. All windows include target-day weather, never future days. Event IDs keep different storm sequences separate; town weather uses town identifiers instead.

## 2. Reconstruct and merge historical daily weather

```bash
python typhoon_pipeline.py prepare \
  --weather /path/to/japan_daily_lstm_processed.csv \
  --location-model trained_daily_pipeline/location \
  --gis /path/to/Typhoon/data/gis/jp_shp/jp.shp \
  --start-date 2021-01-01 --end-date 2025-01-31 \
  --out reconstructed_historical_weather.csv
```

Output includes the daily weather, reconstructed `latitude`, `longitude`, `nearest_prefecture`, distance to prefecture, and provenance. Original station/observed coordinates are preserved as `input_latitude` and `input_longitude`; station `prefecture` and `town` remain intact. `year/month/day` are derived from the date and `elevation` is mapped to `openmeteo_elevation` when required.

The merge uses `(prefecture, town, date)` for town weather or `(landfall_event_id, date)` for labelled cyclone sequences. `--series-columns` can supply another explicit key. Date alone is accepted only when there is exactly one series and one row per date. Multiple towns or simultaneous storms cannot be mixed by date alone. Longer lookbacks require preceding history before the requested start; incomplete requested windows fail instead of silently disappearing.

This output has **no new risk training labels**. Only a labelled event dataset can train the classifier. Do not assign Low/Medium/High to ordinary weather dates without observed target labels.

## 3. Request a date: reuse weather or extend the forecast, then classify

```bash
python typhoon_pipeline.py prepare \
  --weather /path/to/japan_daily_lstm_processed.csv \
  --weather-model /path/to/trained_models/future_weather/baseline_v1 \
  --location-model trained_daily_pipeline/location \
  --risk-model trained_daily_pipeline/random_forest_model.joblib \
  --gis /path/to/Typhoon/data/gis/jp_shp/jp.shp \
  --request-date 2026-12-31 \
  --prefecture aichi --town Minamichita \
  --out daily_risk_2026-12-31.csv
```

A cached requested date does not load the weather model. A future date requires:

```text
weather-model-directory/
  model.keras
  scalers.joblib       # contains scaler and loc_scaler
  config.json         # exact feature order and lookback
```

These are the artifacts saved by cell 22 of `Future_Weather_Attributes_LSTM.ipynb`. **They were not included in the uploads.** `trajectory_lstm.keras` is not the weather model. The original historical `japan_daily_lstm_processed.csv` was also not uploaded. Supply these paths to generate beyond the seven-day CSV. A sufficiently long forecast cache can also be extended, with its `weather_source` retained, but it is already predicted history rather than observations.

Forecasting recursively fills every missing day through the requested date, keeping each town's static weather-station position unchanged. The separate location model then reconstructs a cyclone-like position from the weather. It does not recursively invent prior cyclone coordinates. A request returns just that date, with `risk_prediction` LOW/MEDIUM/HIGH and per-class probabilities. Omit town filters to get one result per town on the date; select a town for a single daily row.

A missing date inside historical weather is an input-data gap, not a future forecast. Restore the missing observations instead of interpolating silently. Dates before the available history cannot be extrapolated backward. Missing weather or missing elevation does not become an invented zero.

## Why the weather CSV stops at 2026-02-07

The notebook's settings cell now says `FORECAST_DAYS = 365`, but its saved diagnostic output says **`Forecast days: 7`**. The uploaded CSV confirms the run: **251 towns × 7 dates = 1,757 rows**, from February 1 through February 7, 2026. Changing the cell text did not update that saved run. Rerun the settings cell and the forecast cell using the saved weather weights/scalers, or use:

```bash
python weather_forecast.py \
  --input /path/to/japan_daily_lstm_processed.csv \
  --model-dir /path/to/trained_models/future_weather/baseline_v1 \
  --end-date 2026-12-31 --out weather_through_2026.csv
```

February 1 through December 31 inclusive is **334 days**. 365 days from February 1 ends January 31, 2027. The script uses the actual end date, so there is no fixed seven-day or one-year cutoff. It batches towns per forecast day, preserves the model's exact feature order, re-encodes wind direction, seeds first-day `pressure_change` from prior history, and checks forecast coverage.

## What these data can establish

The existing coordinate model requires observed latitude/longitude among its inputs; it cannot perform the requested weather-only reconstruction. New training is necessary, rather than renaming its inputs.

January 2021–January 2025 in the uploaded Typhoon archive contains **75 labelled event rows across 42 dates, 19 landfall events and 10 storms**, not a full daily coordinate target series. The classifier labels describe days before **confirmed landfall** (Day −3 Low, −2 Moderate, −1/0 High). They do not establish daily typhoon occurrence risk on nonstorm dates. Weather at a fixed town does not uniquely identify a moving cyclone's position; a model trained on storm-location weather is being applied to a different input population when used on town weather. The scripts preserve this distinction; evaluating that application requires correctly aligned daily cyclone targets and both storm and nonstorm examples. Treat current outputs as experimental landfall-associated classes, not validated daily warnings. Recursive annual weather predictions also accumulate error.

## Validation

```bash
python -m unittest discover -s tests -v
```

The implementation was exercised on the supplied 75-row date-filtered event data with **two epochs as a smoke test**, producing grouped reconstruction features and all four trained classifiers. The supplied seven-day town CSV was used for a cached-date reconstruction/classification run. Test-model accuracy is not a claim that the daily forecasting application is accurate. The absent real weather weights/history prevented a real annual forecast; recursive endpoint and batching behavior are tested using controlled fake models.

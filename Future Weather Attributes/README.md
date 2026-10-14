# Future weather attributes: annual LSTM forecast

Open `Future_Weather_Attributes_LSTM.ipynb`, edit its settings cell, and run from top to bottom in your Python notebook environment.

```text
Future Weather Attributes/
├── Data/
│   └── LSTM_processed/japan_daily_lstm_processed.csv
├── Forecast/
│   └── baseline_v1/future_daily_weather_attributes.csv
├── experiment_logs/
│   └── baseline_v1/...
├── trained_models/
│   └── future_weather/baseline_v1/
│       ├── model.keras
│       ├── scalers.joblib
│       └── config.json
├── Future_Weather_Attributes_LSTM.ipynb
└── README.md
```

The ZIP includes folder placeholders and the revised notebook. Merge it with your existing `Future Weather Attributes` folder so your `Data` and saved model files stay in place. The uploaded notebook does not contain those historical data or model artifacts.

## One year of prediction

Default settings produce **2026-02-01 through 2027-01-31**, 365 dates per town. `FORECAST_END=None` computes a complete calendar year from `FORECAST_START`, including leap-year days where applicable. To forecast through December 31, 2026 instead, set `FORECAST_END="2026-12-31"`; February 1 through December 31 is 334 days.

All selected towns are forecast together for each date, in chunks of `FORECAST_BATCH_SIZE=256`. With 251 towns, the annual run uses 365 batched network calls instead of 91,615 separate town/day calls. TensorFlow inference is compiled once. The notebook verifies one row per town/date and complete date coverage before writing the CSV.

Forecasting uses only history before the start date. Every selected town must have a consecutive seed window ending the day before that start. Missing towns/dates raise an explicit error instead of silently disappearing. Wind direction is decoded to degrees, bounded weather predictions are fed back recursively, min/mean/max families stay ordered, and the first forecast pressure difference uses the last historical pressure.

The future CSV formats floating-point values to **two decimal places**. Recursive prediction and metric calculations retain full precision; dates and integer calendar fields retain their formats.

The output keeps fixed weather-station latitude/longitude and supplies the 29 raw weather attributes required by the weather-only Typhoon notebook. In that notebook, set `PREDICTION_FILE` to this future weather CSV.

## Reuse the existing model

The original weather architecture, ordered features, model directory and scaler keys are preserved. A complete `model.keras`, `scalers.joblib`, `config.json` set is loaded automatically. `FORCE_RETRAIN=False` avoids repeating the original training. When artifacts are missing, `TRAIN_IF_MISSING=True` trains once using the configured historical splits. Partial artifact sets are rejected before anything is overwritten; restore them, choose another experiment ID, or explicitly retrain.

The original splits remain: train through September 30, 2025; validation October–December 2025; heldout test January 2026. Training normalizes using training rows only, precomputes station transforms, and builds batches without materializing every large window at once. Scalers are never refitted merely to generate forecasts.

## Forecast and logs have separate purposes

`Forecast/<EXPERIMENT_ID>/` stores the future CSV only. `experiment_logs/<EXPERIMENT_ID>/` stores:

- Training history and a training/validation loss plot when training runs.
- `one_step_metrics.csv`: per-feature MAE, RMSE, bias, baseline error and skill on historical test dates using observed prior weather.
- `recursive_test_metrics.csv`: the same measures for an uninterrupted recursive historical backtest.
- `recursive_horizon_metrics.csv`: errors over the first 1, 7, 14, 31 days and the available final horizon.
- Historical actual/predicted tables and comparison plots.
- `experiment_config.json`, annual preview plots, and `forecast_coverage.json` with town/date/row counts.

Wind direction errors use shortest angular differences. MAE/RMSE have each weather attribute's units; a single percentage classification accuracy is unsuitable for these continuous outputs. Positive skill means improvement over persistence, negative skill means worse. One-step persistence uses the previous observed day's weather; recursive persistence holds the starting observation fixed.

If `Forecast_outputs` exists, the notebook moves its old future CSVs into `Forecast` and its other files into `experiment_logs`, preserving name collisions with a `_legacyN` suffix. It removes the emptied old directory. Existing CSV/PNG metrics in the selected model directory are moved to the logs when their destination is free. Model weights, scalers and configuration stay in `trained_models`.

## Evaluation limits and validation

January 2026 one-step and recursive backtests measure different forecast behaviors. Neither is an annual accuracy measurement. A full-year historical evaluation needs an independent earlier checkpoint and observations for that entire heldout year; the notebook rejects test dates that overlap model fitting or validation. Real future-year accuracy becomes measurable when actual observations arrive.

The previous notebook saved output for seven days even though its settings text said 365; the settings and forecast were not rerun together. The revision derives an explicit end date and verifies all dates, eliminating that ambiguous fixed-day runtime setting.

No real annual CSV was generated during this revision because the historical weather file and saved weather model were not uploaded. Validation used 14 contract checks, including all 251 towns across 365 dates (91,615 rows), and an end-to-end synthetic notebook run that trained/saved/reloaded the original LSTM shape, produced historical metrics, migrated old folders, and generated a 365-day forecast. Synthetic scores are not included in the user notebook. Its outputs are cleared so actual metrics are generated from your own files when you run it.

## Startup troubleshooting

Use **Restart Kernel and Run All**, starting with Settings, then Imports, then feature loading. `MODEL_DIR` is created in Settings; saved execution counts do not define variables in a fresh kernel. Out-of-order execution now produces explicit instructions. Both `Data/LSTM_processed/japan_daily_lstm_processed.csv` and the double-extension `japan_daily_lstm_processed.csv.csv` are accepted; the canonical name is preferred when both exist.

## Missing and nonfinite historical inputs

The loader now reports missing values, infinities and invalid numeric cells by column and town before handling them. Diagnostic files go to `experiment_logs/<EXPERIMENT_ID>/`:

- `data_quality_counts.csv`: affected columns and cell counts.
- `data_quality_by_town.csv`: counts for each town.
- `data_quality_examples.csv`: up to 20 affected rows.
- `input_repair_counts.csv`: cells forward-filled or given training-derived input fallbacks.
- `target_window_quality.csv`: incomplete target windows excluded from training/validation.

The original source CSV is not edited and no calendar rows are removed. Weather input gaps up to `MAX_FORWARD_FILL_DAYS=3` use earlier observations from the same town, measured by calendar age. Longer/leading gaps use training-period medians for a new model, its saved input fallback parameters on reuse, or the original training scaler's means for older saved models. Wind direction is repaired as a circular quantity, with raw angle and sine/cosine kept consistent. Saved scalers are not refitted. New models record their input repair settings so reuse reproduces them.

Station latitude, longitude and elevation are restored only from known consistent same-town metadata available by the relevant historical cutoff. Minor floating-point jitter is canonicalized. Existing `openmeteo_elevation` can restore otherwise unknown elevation. Entirely unknown/conflicting station metadata or a weather feature with no finite training fallback raises a specific error naming the field/towns; no guessed coordinates or arbitrary zero values are supplied.

Original weather observations are kept separately from repaired model inputs. Incomplete multifeature **training and validation targets** are excluded, with counts reported. Test dates remain available for feature-wise metrics, which score only finite actual observations. Missing actuals have separate counts; skill against persistence uses paired observed samples. Plots average actual and predicted weather over the same observed towns. A feature with no observed test values has `count=0` and undefined metrics. Nonfinite model outputs still stop evaluation explicitly.

If the notebook still stops, send `data_quality_counts.csv` and the specific new error. These small reports identify what needs fixing without transferring the full historical CSV.

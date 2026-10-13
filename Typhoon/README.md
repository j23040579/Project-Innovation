# Weather-only pre-typhoon classification

The active workflow is **`training/train_typhoon_models.ipynb`**. Run it in Jupyter or VS Code from top to bottom. Settings and model parameters are in the first cell.

```text
Typhoon/
├── data/
│   ├── raw/
│   └── processed/
│       └── jma_day3_to_landfall_with_weather.csv
├── preprocessing/
├── testing/
├── training/
│   ├── experiment_logs/
│   ├── trained_models/
│   └── train_typhoon_models.ipynb
└── README.md
```

Your existing data, preprocessing and testing files stay in their folders. Data and generated outputs are excluded from Git. The earlier prefecture notebook remains as an unused reference; it is not needed for this workflow.

## Training

The notebook reads `data/processed/jma_day3_to_landfall_with_weather.csv` directly. It uses all available history by default, with optional `START_DATE` and `END_DATE` settings. The provided file has **588 rows across 102 storms** after selecting the three days before landfall. Landfall-day records are excluded.

Only the selected weather attributes enter the models. Nearest prefecture, coordinates, elevation, dates, storm attributes, IDs, and day labels are excluded from the feature matrix. Wind direction is encoded as sine/cosine inside each saved model. Missing weather cells are imputed using the training data.

Profile labels are derived from the known event's `days_before_landfall`:

- **Low:** Day −3 weather.
- **Medium:** Day −2 weather.
- **High:** Day −1 weather.

Random Forest, CART, Gradient Boosting and SVM are compared using a holdout of whole storms. Every event and weather row belonging to the same storm stays on one side of the split. Tables, confusion matrices and plots are displayed in the notebook.

## Prediction

Set `PREDICTION_FILE` to a CSV of weather conditions, including the same raw weather columns and units as the training data. This can be an observed-weather CSV or your already-predicted weather CSV. Leave it `None` to inspect a heldout example.

The prediction section outputs `matched_weather_profile` and the three class probabilities. A following cell shows the closest historical **training** weather examples for the first query row. Town/prefecture/date columns, when present, are retained as display metadata only; prediction does not require them.

These outputs indicate resemblance to weather around confirmed landfalls. This event-only dataset has no ordinary non-typhoon examples, so the class probabilities do not establish the probability of a typhoon occurring. There is no weather forecasting runtime in this notebook.

## Saved outputs

Use a new `EXPERIMENT_ID` for each run. With the default `weather_only_v1`:

- Models and prediction/training/test CSVs: `training/trained_models/risk/weather_only_v1/`.
- Settings, metrics and plots: `training/experiment_logs/risk/weather_only_v1/`.

Outputs are protected unless `OVERWRITE_OUTPUTS=True`. Paths work when the notebook kernel starts in `training`, `Typhoon`, or the repository root. `requirements.txt` contains only dependencies needed by this weather-only classifier notebook.

# Typhoon training

```text
Typhoon/
├── data/
│   ├── raw/
│   ├── processed/
│   └── gis/jp_shp/
├── preprocessing/
├── testing/
├── training/
│   ├── experiment_logs/
│   ├── trained_models/
│   ├── train_typhoon_models.ipynb
│   └── typhoon_nearest_prefecture_models.ipynb
└── README.md
```

Keep your existing data, preprocessing scripts, and testing scripts in their original folders. The download contains folder placeholders and revised notebooks; it does not replace those existing files. Data and generated models/logs are excluded from Git.

Run the notebooks from `Typhoon/training`, opening them in Jupyter or VS Code. Paths also work when the notebook kernel starts from `Typhoon` or the repository root.

1. **`typhoon_nearest_prefecture_models.ipynb`** trains a weather-only LSTM to reconstruct latitude/longitude, converts predicted positions to Japan's nearest prefecture, and saves the merged weather/prefecture CSV. It displays training loss, heldout coordinates and location errors.
2. **`train_typhoon_models.ipynb`** reads that CSV and trains Random Forest, CART, Gradient Boosting and SVM. It displays Low/Medium/High classification scores, confusion matrices and sample predictions.

Edit `EXPERIMENT_ID`, dates, model settings and file paths in the first settings cell. Both notebooks should use the same `EXPERIMENT_ID`. Outputs are protected unless `OVERWRITE_OUTPUTS` is enabled.

Required input files:

```text
data/processed/jma_day3_to_landfall_with_weather_prefecture.csv
data/gis/jp_shp/jp.shp
data/gis/jp_shp/jp.dbf
data/gis/jp_shp/jp.shx
data/gis/jp_shp/jp.prj
```

The default period is January 2021–January 2025. The supplied dataset contains 75 event rows across 42 dates, not a full daily series. Coordinate inputs exclude latitude/longitude; early unavailable weather steps are masked. Whole storms are held out and reconstructed training locations are generated out of fold. The classifier keeps the same split and displays the original `Moderate` class as `Medium`.

Models and result CSVs go under `training/trained_models/nearest_prefecture/<EXPERIMENT_ID>/` and `training/trained_models/risk/<EXPERIMENT_ID>/`. Configuration, plots and metric logs go under the corresponding folders in `training/experiment_logs/`.

These event labels describe days before confirmed landfall and do not establish ordinary daily occurrence risk. There is no future-weather pipeline in these notebooks.

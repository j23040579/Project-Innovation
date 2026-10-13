# Typhoon training notebooks

Open the notebooks in Jupyter or VS Code and run their cells from top to bottom:

1. **`typhoon_nearest_prefecture_models.ipynb`** — train a weather-only LSTM to reconstruct daily latitude/longitude, convert predicted positions to Japan's nearest prefecture, view location errors and plots, and save the reconstructed weather/prefecture CSV.
2. **`train_typhoon_models.ipynb`** — read that CSV and train Random Forest, CART, Gradient Boosting, and SVM. View Low/Medium/High classification scores, confusion matrices, and individual predictions.

Edit paths and training parameters in the first settings cell of each notebook. The Python pipeline and its CLI tests have been replaced by these two self-contained notebooks.

## Data and installation

Place your existing `Typhoon` folder next to the notebooks, or change `DATA_PATH` and `SHAPEFILE` in notebook 1:

```text
Project-Innovation/
  typhoon_nearest_prefecture_models.ipynb
  train_typhoon_models.ipynb
  Typhoon/
    data/processed/jma_day3_to_landfall_with_weather_prefecture.csv
    data/gis/jp_shp/jp.shp
    data/gis/jp_shp/jp.dbf
    data/gis/jp_shp/jp.shx
    data/gis/jp_shp/jp.prj
```

Use Python 3.12 and install `requirements.txt` in the notebook's environment. Data files are supplied separately; the notebooks do not download them. Select this Python environment as the notebook kernel.

## Outputs

Notebook 1 saves its model, preprocessing, metrics, and `reconstructed_weather_prefecture.csv` under `trained_models/nearest_prefecture/baseline_v1/`. Notebook 2 reads that file and saves four classifiers, comparison scores, and predictions under `trained_models/risk/baseline_v1/`. Use a new output directory for a new experiment; existing outputs are protected unless `OVERWRITE_OUTPUTS` is enabled.

The default historical period is January 2021–January 2025. The provided data contains 75 storm-event rows over 42 dates in this period, not a full daily series. Early sequence steps are masked when preceding weather is unavailable; every labelled day is retained. Whole storms are held out, and reconstructed training locations are generated out of fold. Both notebooks use the same train/test split.

Latitude/longitude are targets, never coordinate-model inputs. The original `Moderate` label is displayed as `Medium`. Risk labels describe days before confirmed landfall; these data do not establish risk on ordinary nonstorm dates. There is no future-weather runtime pipeline in this revision.

Both notebooks were executed with the supplied data and the displayed 30-epoch settings. Saved outputs are representative results, not a claim of validated daily warning accuracy.

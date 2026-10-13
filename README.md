# Project Innovation — Weather and Typhoon notebooks

- [`Future Weather Attributes/Future_Weather_Attributes_LSTM.ipynb`](Future%20Weather%20Attributes/Future_Weather_Attributes_LSTM.ipynb): predict one year of daily weather, save the future CSV in `Forecast`, and inspect historical error metrics and plots in `experiment_logs`. See its [folder and usage guide](Future%20Weather%20Attributes/README.md).
- [`Typhoon/training/train_typhoon_models.ipynb`](Typhoon/training/train_typhoon_models.ipynb): train weather-only event profiles using every supplied historical event row, then classify weather conditions as Low/Medium/High. See the [Typhoon guide](Typhoon/README.md).

Keep your existing datasets/model files in their project folders. Install `requirements.txt` in a Python 3.12 notebook environment. In the Typhoon notebook, set `PREDICTION_FILE` to the future-weather CSV to classify its weather profiles.

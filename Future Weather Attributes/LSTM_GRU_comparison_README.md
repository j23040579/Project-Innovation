# LSTM and GRU weather comparison

Open `Future_Weather_Attributes_LSTM_GRU.ipynb`, set `DATA_PATH` in section 1, then **Restart Kernel and Run All**. The `.py` file contains the same implementation for script/editor use. The uploaded original notebook is preserved.

The main switch is:

```python
MODEL_CHOICE = "BOTH"  # "LSTM", "GRU", or "BOTH"
```

Every experiment retains all **37 original physical weather attributes**, encoded as **38 weather channels** because wind direction uses sine and cosine. Historical calendar and station metadata are included for both models. The target-day calendar is supplied separately. Additional CSV identifiers and derived columns are handled as in the original notebook; they are not an attribute-removal experiment.

The default setting matches the original architecture widths and budget: 30 history days, recurrent layers of 96 and 64 units, a 96-unit dense layer, Adam learning rate 0.001, batch size 128, up to 30 epochs and 160,000 training windows. The recurrent layers become GRU layers in GRU runs. Equal widths give different parameter counts; both are logged.

To compare training settings without dropping attributes, add another entry:

```python
EXPERIMENT_SETTINGS = [
    {"name": "baseline", "lookback": 30, "recurrent_units": [96, 64],
     "dense_units": 96, "learning_rate": 0.001, "dropout": 0.0},
    {"name": "longer_history", "lookback": 60, "recurrent_units": [96, 64],
     "dense_units": 96, "learning_rate": 0.001, "dropout": 0.0},
]
SEEDS = [42]  # optionally [42, 52, 62] to measure initialization variability
```

This example trains four models with one seed: two architectures times two settings. More seeds multiply that count. Use a new `EXPERIMENT_ID` when changing a comparison's models, settings, seeds, dates, budget or evaluation/forecast options. Identical requests reuse their saved models unless `FORCE_RETRAIN=True`. A mismatched or reduced-attribute checkpoint is rejected.

## How to compare accuracy

Weather prediction is regression. Lower **MAE** and **RMSE** indicate smaller errors; **bias** shows average overprediction or underprediction. Errors are reported separately for every physical attribute in its original units. Wind-direction error wraps around 0/360 degrees.

The overall comparison uses mean normalized RMSE so large-unit fields do not dominate. Each field's RMSE is divided by its observed training standard deviation; direction uses 180 degrees. A constant training field uses one physical unit and remains included. All 37 fields must have observed evaluation values for a complete ranking.

Models share training-derived repairs, scalers, sampled windows, target dates and per-seed shuffle streams. If lookbacks differ, the largest lookback determines common target eligibility. Training and early-stopping validation exclude incomplete target rows. Physical validation/test metrics retain partial observations and score each attribute only where actually observed. Missing ground truth is never filled for scoring.

The notebook compares:

- **One-step predictions:** refresh observed history every day.
- **Recursive predictions:** start once at the beginning of a validation/test period, then feed predictions back.
- **Forecast horizons:** cumulative errors over days 1 through h, for configured horizons covered by the period.
- **Persistence:** use the preceding observation for one-step predictions or repeat the last pre-origin observation recursively.

The default periods are training through September 2025, validation October–December 2025 and test January 2026. Validation uses a **continuous 92-day recursive forecast**, rather than the three 31-day origins in the earlier attribute experiments, so its aggregate score should not be compared directly with those earlier scores. Checkpoint selection uses one-step scaled validation MSE. Architecture/setting selection uses mean recursive validation normalized RMSE, averaged over seeds, and is frozen before test evaluation. Future output uses the first listed seed of the selected configuration. Test results do not change that selection.

## Exported files

Within `experiment_logs/<EXPERIMENT_ID>/`:

| File or folder | Content |
| --- | --- |
| `experiment_request.json`, `comparison_plan.json` | Requested settings, data identity, full feature contract, windows and software versions |
| `shared/` | Data quality, repair counts, complete-target counts, shared scalers, sampled indices and score normalizers |
| `lstm/<setting>/seed_<n>/`, `gru/<setting>/seed_<n>/` | Separate epoch logs, loss plots, run configurations, validation/test attribute metrics, predictions and horizons |
| `validation_comparison.csv` | Every run's validation scores and training statistics |
| `validation_configuration_ranking.csv` | Architecture/setting ranking averaged over configured seeds |
| `selection_frozen.json` | Selection recorded before new test metrics |
| `test_comparison.csv` | Both models' heldout one-step and recursive aggregate errors |
| `test_all_attribute_metrics.csv` | Per-attribute test MAE, RMSE, bias, baseline errors, skill and observed counts |
| `model_comparison_report.md` | A training/evaluation/selected-model report based on this execution's actual logs |
| `completion.json` | Completion status for the current execution |

`experiment_exports/<EXPERIMENT_ID>_comparison_logs.zip` contains all logs, predictions, reports and plots. Trained Keras models, scalers and configurations are stored separately in `trained_models/future_weather/<EXPERIMENT_ID>/`.

The validation-selected model writes `Forecast/<EXPERIMENT_ID>/future_daily_weather_attributes.csv` with every physical weather attribute. Use `RUN_FORECAST=False` before starting a new comparison if you only want historical testing. To forecast using one specific architecture, run that model alone with the switch and a new experiment ID.

## Verification and limits

The update was checked with actual small TensorFlow training runs for both architectures, multiple settings and seeds, Keras serialization/reuse, independent metric recomputation and the real weather dataset's preparation. Tiny synthetic checks verify execution and arithmetic; **they are not weather accuracy results**. A new full-dataset GRU comparison has not been trained as part of this notebook-editing task.

One seed cannot establish a statistically reliable difference. Repeated seeds measure initialization variability, not independent weather years. January test data was already used in earlier project work; it is not a newly blind benchmark. Historical performance over the configured periods does not establish accuracy of an unobserved annual forecast.

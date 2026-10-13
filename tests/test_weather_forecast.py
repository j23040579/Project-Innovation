"""Weather runtime contracts using deterministic fake artifacts, without TensorFlow."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "weather_forecast.py"
spec = importlib.util.spec_from_file_location("weather_forecast", MODULE)
weather = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weather)


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values)

    def inverse_transform(self, values):
        return np.asarray(values)


class NextDayModel:
    def __init__(self):
        self.calls = []

    def predict(self, inputs, verbose=0):
        past, calendar = inputs
        self.calls.append((past.copy(), calendar.copy()))
        # Features are temperature, pressure and circular wind direction.
        result = past[:, -1, :4].copy()
        result[:, 0] += 1
        result[:, 1] += 2
        return result


def history(town="Town A", end="2025-01-03", latitude=35.0, longitude=139.0):
    dates = pd.date_range(end=end, periods=3)
    return pd.DataFrame({
        "prefecture": "tokyo", "town": town, "date": dates,
        "latitude": latitude, "longitude": longitude, "elevation": 10.0,
        "temperature_2m_mean": [10., 11., 12.],
        "surface_pressure_mean": [1000., 1001., 1002.],
        "wind_direction_10m_dominant": 90.0,
    })


def artifacts(model):
    return ({"features": ["temperature_2m_mean", "surface_pressure_mean", "wind_direction_sin", "wind_direction_cos"], "lookback": 3},
            {"scaler": IdentityScaler(), "loc_scaler": IdentityScaler()}, model)


class WeatherForecastTests(unittest.TestCase):
    def test_inclusive_endpoint_recursive_state_and_first_pressure_change(self):
        model = NextDayModel()
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(model)):
            result = weather.ensure_weather_through(history(), "artifacts", "2025-01-06")
        future = result.loc[result.weather_source.eq("forecast")]
        self.assertEqual(list(future.date.dt.strftime("%Y-%m-%d")), ["2025-01-04", "2025-01-05", "2025-01-06"])
        self.assertEqual(future.temperature_2m_mean.tolist(), [13., 14., 15.])
        self.assertEqual(future.pressure_change.tolist(), [2., 2., 2.])
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(model.calls[0][0].shape, (1, 3, 9))
        self.assertEqual(len(result), 6)

    def test_batches_all_towns_and_preserves_station_locations(self):
        initial = pd.concat([history(), history("Town B", latitude=36.0, longitude=140.0)], ignore_index=True)
        model = NextDayModel()
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(model)):
            result = weather.ensure_weather_through(initial, "artifacts", "2025-01-05")
        self.assertEqual([call[0].shape[0] for call in model.calls], [2, 2])
        self.assertTrue(result.loc[result.town.eq("Town A"), "latitude"].eq(35.0).all())
        self.assertTrue(result.loc[result.town.eq("Town B"), "longitude"].eq(140.0).all())
        future = result.loc[result.weather_source.eq("forecast")]
        self.assertTrue(np.allclose(future.wind_direction_10m_dominant, 90.0))

    def test_different_latest_dates_are_not_skipped(self):
        initial = pd.concat([history(), history("Town B", end="2025-01-04")], ignore_index=True)
        model = NextDayModel()
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(model)):
            result = weather.ensure_weather_through(initial, "artifacts", "2025-01-06")
        self.assertEqual([call[0].shape[0] for call in model.calls], [1, 2, 2])
        self.assertTrue(result.groupby("town").date.max().eq(pd.Timestamp("2025-01-06")).all())

    def test_gap_is_explicit_even_without_new_forecast(self):
        with self.assertRaisesRegex(ValueError, "missing daily dates"):
            weather.ensure_weather_through(history().drop(index=1), None, "2025-01-03")

    def test_insufficient_lookback_is_explicit(self):
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(NextDayModel())):
            with self.assertRaisesRegex(ValueError, "need 3 consecutive"):
                weather.ensure_weather_through(history().tail(2), "artifacts", "2025-01-04")

    def test_nonfinite_feature_is_explicit(self):
        initial = history()
        initial.loc[1, "temperature_2m_mean"] = np.nan
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(NextDayModel())):
            with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
                weather.ensure_weather_through(initial, "artifacts", "2025-01-04")

    def test_zero_forecast_needs_no_model_and_does_not_mutate_input(self):
        initial = history()
        original = initial.copy()
        with patch.object(weather, "_load_weather_artifacts", side_effect=AssertionError("Should not load")):
            result = weather.ensure_weather_through(initial, None, "2025-01-02")
        pd.testing.assert_frame_equal(initial, original)
        self.assertEqual(len(result), len(initial))
        self.assertTrue(result.weather_source.eq("provided").all())

    def test_existing_source_is_preserved_and_unannotated_input_is_provided(self):
        initial = history()
        initial["weather_source"] = ["observed", "forecast", None]
        result = weather.ensure_weather_through(initial, None, "2025-01-03")
        self.assertEqual(result.weather_source.tolist(), ["observed", "forecast", "provided"])

    def test_251_towns_cover_inclusive_endpoint_in_daily_batches(self):
        initial = pd.concat([history(f"Town {index:03d}") for index in range(251)], ignore_index=True)
        model = NextDayModel()
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(model)):
            result = weather.ensure_weather_through(initial, "artifacts", "2025-01-05")
        self.assertEqual([call[0].shape[0] for call in model.calls], [251, 251])
        forecast = result.loc[result.weather_source.eq("forecast")]
        self.assertEqual(len(forecast), 502)
        self.assertEqual(forecast.groupby("date").size().tolist(), [251, 251])
        self.assertTrue(forecast.groupby("town").date.max().eq(pd.Timestamp("2025-01-05")).all())

    def test_date_before_history_is_not_silently_treated_as_cached(self):
        with self.assertRaisesRegex(ValueError, "precedes the first"):
            weather.ensure_weather_through(history(), None, "2024-12-31")

    def test_filters_and_missing_station(self):
        initial = pd.concat([history(), history("Town B")], ignore_index=True)
        result = weather.ensure_weather_through(initial, None, "2025-01-03", town="Town A")
        self.assertEqual(result.town.unique().tolist(), ["Town A"])
        with self.assertRaisesRegex(ValueError, "No weather stations match"):
            weather.ensure_weather_through(initial, None, "2025-01-03", prefecture="missing")

    def test_elevation_fallback(self):
        initial = history().rename(columns={"elevation": "openmeteo_elevation"})
        model = NextDayModel()
        with patch.object(weather, "_load_weather_artifacts", return_value=artifacts(model)):
            result = weather.ensure_weather_through(initial, "artifacts", "2025-01-04")
        self.assertEqual(result.iloc[-1].elevation, 10.0)
        self.assertEqual(model.calls[0][0][0, -1, -1], 10.0)

    def test_missing_artifacts_explain_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "original scalers"):
                weather.ensure_weather_through(history(), directory, "2025-01-04")

    def test_read_daily_normalizes_dates_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weather.csv"
            initial = history()
            initial["date"] = initial.date.dt.strftime("%Y-%m-%d")
            initial.to_csv(path, index=False)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(weather.read_daily(path).date))
            pd.concat([initial, initial.iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                weather.read_daily(path)

    def test_cli_cannot_overwrite_input_or_model_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            history().to_csv(path, index=False)
            with self.assertRaises(SystemExit):
                weather.main(["--input", str(path), "--out", str(path), "--end-date", "2025-01-03"])
            with self.assertRaises(SystemExit):
                weather.main(["--input", str(path), "--out", str(Path(directory) / "models/output.csv"), "--model-dir", str(Path(directory) / "models"), "--end-date", "2025-01-03"])


if __name__ == "__main__":
    unittest.main()

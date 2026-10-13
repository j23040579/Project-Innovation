"""Contract tests for causality, series joins, and grouped coordinate stacking."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import typhoon_pipeline as pipeline


def frame(dates, values, **columns):
    result = pd.DataFrame({'date': pd.to_datetime(dates)})
    for name in pipeline.WEATHER:
        result[name] = values
    for name, value in columns.items():
        result[name] = value
    return result


class FakeMapper:
    def __init__(self, *_):
        pass

    def transform(self, coords):
        return ['Tokyo'] * len(coords), [1.0] * len(coords)


class FakeClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, x):
        return np.repeat(self.classes_[0], len(x))


class WindowTests(unittest.TestCase):
    def test_unsorted_noncontiguous_index_uses_chronological_positional_rows(self):
        weather = frame(['2024-01-03', '2024-01-01', '2024-01-02'], [3, 1, 2])
        weather.index = [120, 7, 900]
        windows, rows = pipeline.make_windows(weather, 2, [])
        np.testing.assert_array_equal(windows[:, :, 0], [[1, 2], [2, 3]])
        np.testing.assert_array_equal(rows, [2, 0])
        # Returned indices address the caller's iloc; input indices are preserved.
        self.assertEqual(list(weather.iloc[rows].date.dt.day), [2, 3])
        self.assertEqual(list(weather.index), [120, 7, 900])

    def test_no_future_day_or_coordinate_input(self):
        weather = frame(pd.date_range('2024-01-01', periods=4), [1, 2, 3, 4],
                        latitude=[70, 71, 72, 73], longitude=[140, 141, 142, 143])
        windows, rows = pipeline.make_windows(weather, 2, [])
        altered = weather.copy()
        altered['latitude'] = -1
        altered['longitude'] = -2
        altered.loc[3, pipeline.WEATHER] = 99
        new_windows, new_rows = pipeline.make_windows(altered, 2, [])
        np.testing.assert_array_equal(rows, new_rows)
        np.testing.assert_array_equal(windows[:2], new_windows[:2])
        self.assertFalse(np.array_equal(windows[-1], new_windows[-1]))
        self.assertEqual(windows.shape[-1], len(pipeline.LOCATION_FEATURES))

    def test_events_and_calendar_gaps_cannot_be_crossed(self):
        weather = frame(['2024-01-01', '2024-01-02', '2024-01-04',
                         '2024-01-03', '2024-01-04'], [1, 2, 4, 103, 104],
                        landfall_event_id=['A', 'A', 'A', 'B', 'B'])
        windows, rows = pipeline.make_windows(weather, 2, ['landfall_event_id'])
        np.testing.assert_array_equal(windows[:, :, 0], [[1, 2], [103, 104]])
        np.testing.assert_array_equal(rows, [1, 4])
        with self.assertRaisesRegex(ValueError, 'No consecutive'):
            pipeline.make_windows(weather, 3, ['landfall_event_id'])

    def test_ambiguous_date_keys_are_rejected(self):
        weather = frame(['2024-01-01', '2024-01-01'], [1, 2], town=['A', 'B'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pipeline.series_columns(weather)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pipeline.make_windows(weather, 1, [])
        self.assertEqual(pipeline.series_columns(weather, 'town'), ['town'])


class JoinTests(unittest.TestCase):
    def setUp(self):
        self.weather = frame(['2024-01-02', '2024-01-01', '2024-01-01'], [20, 10, 30],
                             prefecture=['tokyo'] * 3, town=['A', 'A', 'B'],
                             latitude=[35.0] * 3, longitude=[139.0] * 3,
                             elevation=[12.0] * 3)

    def test_series_keyed_join_preserves_rows_weather_and_station_coordinates(self):
        predictions = pipeline.location_frame(
            self.weather, np.array([2, 0, 1]),
            np.array([[30., 130.], [32., 132.], [31., 131.]]), FakeMapper())
        joined = pipeline.join_locations(self.weather, predictions, ['prefecture', 'town'])
        self.assertEqual(len(joined), len(self.weather))
        self.assertEqual(joined.town.tolist(), ['A', 'A', 'B'])
        self.assertEqual(joined.temperature_2m_mean.tolist(), [20, 10, 30])
        self.assertEqual(joined.latitude.tolist(), [32., 31., 30.])
        self.assertEqual(joined.input_latitude.tolist(), [35.] * 3)
        self.assertEqual(joined.input_longitude.tolist(), [139.] * 3)
        self.assertEqual(joined.openmeteo_elevation.tolist(), [12.] * 3)
        self.assertEqual(self.weather.latitude.tolist(), [35.] * 3)

    def test_date_only_join_of_multiple_towns_is_rejected(self):
        predictions = pipeline.location_frame(
            self.weather, np.arange(3), np.array([[30., 130.]] * 3), FakeMapper())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pipeline.join_locations(self.weather, predictions, [])

    def test_missing_warmup_target_is_not_silently_dropped(self):
        weather = frame(pd.date_range('2024-01-01', periods=3), [1, 2, 3])
        _, rows = pipeline.make_windows(weather, 3, [])
        predictions = pipeline.location_frame(weather, rows, np.array([[30., 130.]]), FakeMapper())
        with self.assertRaisesRegex(ValueError, 'preceding days'):
            pipeline.join_locations(weather.iloc[[0]], predictions, [])

    def test_location_frame_preserves_observed_coordinates(self):
        result = pipeline.location_frame(self.weather, [1], np.array([[29., 128.]]), FakeMapper())
        self.assertEqual(result.loc[0, 'latitude'], 29.)
        self.assertEqual(result.loc[0, 'input_latitude'], 35.)
        self.assertEqual(result.loc[0, 'location_source'], 'weather_only_lstm')


class PreparationTests(unittest.TestCase):
    def test_requested_date_requires_its_own_lookback_history(self):
        weather = frame(pd.date_range('2024-01-01', periods=3), [1, 2, 3])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'weather.csv'
            weather.to_csv(source, index=False)
            args = pipeline.parser().parse_args([
                'prepare', '--weather', str(source), '--location-model', 'unused',
                '--gis', 'unused', '--out', str(folder / 'output.csv'),
                '--request-date', '2024-01-02'])
            with mock.patch.object(pipeline, 'load_location', return_value=(object(), {}, {'lookback': 3})), \
                 mock.patch.object(pipeline, 'predict_location', return_value=np.array([[30., 130.]])), \
                 mock.patch.object(pipeline, 'PrefectureMapper', FakeMapper):
                with self.assertRaisesRegex(ValueError, 'earlier history|preceding days'):
                    pipeline.prepare(args)
            self.assertFalse((folder / 'output.csv').exists())


class GeographyTests(unittest.TestCase):
    def test_prefecture_names_repair_mojibake_and_preserve_valid_unicode(self):
        import geopandas as gpd
        from pyproj import Geod
        from shapely.geometry import box
        mapper = pipeline.PrefectureMapper.__new__(pipeline.PrefectureMapper)
        mapper.name_col = 'name'
        mapper.geod = Geod(ellps='WGS84')
        mapper.frame = gpd.GeoDataFrame(
            {'name': ['HyÅ\x8dgo', 'Å\x8csaka', 'Kōchi', 'Tokyo']},
            geometry=[box(130 + i * 2, 30, 131 + i * 2, 31) for i in range(4)],
            crs='EPSG:4326')
        names, distances = mapper.transform(np.array([[30.5, 130.5 + i * 2] for i in range(4)]))
        self.assertEqual(names, ['Hyōgo', 'Ōsaka', 'Kōchi', 'Tokyo'])
        self.assertEqual(distances, [0.] * 4)


class TrainingTests(unittest.TestCase):
    def test_risk_default_lookback_preserves_all_four_day_event_classes(self):
        args = pipeline.parser().parse_args(['train-risk', '--labels', 'unused', '--gis', 'unused', '--out', 'unused'])
        self.assertEqual(args.lookback, 1)

    def test_labelled_events_take_precedence_over_station_metadata(self):
        labelled = frame(list(pd.date_range('2024-01-01', periods=3)) * 2,
                         [1, 2, 3, 101, 102, 103],
                         landfall_event_id=['A'] * 3 + ['B'] * 3,
                         intl_id=['stormA'] * 3 + ['stormB'] * 3,
                         prefecture=['tokyo'] * 6, town=['same_station'] * 6,
                         latitude=[30.] * 6, longitude=[130.] * 6)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'labels.csv'
            labelled.to_csv(source, index=False)
            args = pipeline.parser().parse_args([
                'train-location', '--labels', str(source), '--out', 'unused',
                '--start-date', '2024-01-01', '--end-date', '2024-01-03', '--lookback', '2'])
            _, windows, _, samples, _ = pipeline.labelled_samples(args)
        np.testing.assert_array_equal(windows[:, :, 0], [[1, 2], [2, 3], [101, 102], [102, 103]])
        self.assertEqual(samples.landfall_event_id.tolist(), ['A', 'A', 'B', 'B'])

    def test_coordinate_stacking_keeps_all_events_from_each_storm_together(self):
        parts = []
        for storm in range(6):
            for event in range(2):
                dates = pd.date_range(f'2024-0{event + 1}-01', periods=4)
                parts.append(frame(dates, [storm + 10] * 4,
                                   intl_id=[f'storm_{storm}'] * 4,
                                   landfall_event_id=[f'storm_{storm}_event_{event}'] * 4,
                                   latitude=[30 + storm] * 4, longitude=[130 + storm] * 4,
                                   openmeteo_elevation=[1.] * 4,
                                   risk=['Low', 'Moderate', 'High', 'High']))
        labelled = pd.concat(parts, ignore_index=True)
        observed_splits = []

        def fit_location(x, y, lookback, epochs, seed=42):
            return set(x[:, :, 0].ravel()), {}

        def predict_location(x, trained_storms, prep):
            heldout_storms = set(x[:, :, 0].ravel())
            self.assertFalse(trained_storms & heldout_storms)
            observed_splits.append((trained_storms, heldout_storms))
            return np.column_stack((30 + x[:, -1, 0] / 10, 130 + x[:, -1, 0] / 10))

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            labels = folder / 'labels.csv'
            labelled.to_csv(labels, index=False)
            out = folder / 'models'
            args = pipeline.parser().parse_args([
                'train-risk', '--labels', str(labels), '--gis', 'unused', '--out', str(out),
                '--start-date', '2024-01-01', '--end-date', '2024-12-31', '--epochs', '1', '--folds', '3'])
            with mock.patch.object(pipeline, 'fit_location', side_effect=fit_location), \
                 mock.patch.object(pipeline, 'predict_location', side_effect=predict_location), \
                 mock.patch.object(pipeline, 'save_location'), \
                 mock.patch.object(pipeline, 'PrefectureMapper', FakeMapper), \
                 mock.patch.object(pipeline, 'risk_estimator', side_effect=lambda name: FakeClassifier()):
                pipeline.train_risk(args)
            config = json.loads((out / 'config.json').read_text())
            self.assertFalse(set(config['location_train_groups']) & set(config['test_groups']))
            self.assertEqual(len(observed_splits), 4)  # Three OOF fits and one outer holdout.
            train = pd.read_csv(out / 'training_reconstructed.csv')
            test = pd.read_csv(out / 'testing_reconstructed.csv')
            self.assertFalse(set(train.intl_id) & set(test.intl_id))
            self.assertEqual(set(train.risk), {'Low', 'Moderate', 'High'})
            self.assertEqual(train.location_source.unique().tolist(), ['weather_only_lstm'])


if __name__ == '__main__':
    unittest.main()

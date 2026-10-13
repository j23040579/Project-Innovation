#!/usr/bin/env python3
"""Weather-only location reconstruction, daily joins, and labelled risk classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WEATHER = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'relative_humidity_2m_mean', 'relative_humidity_2m_max', 'relative_humidity_2m_min',
    'dew_point_2m_mean', 'dew_point_2m_max', 'dew_point_2m_min',
    'precipitation_sum', 'rain_sum', 'precipitation_hours',
    'cloud_cover_mean', 'cloud_cover_max', 'cloud_cover_min',
    'pressure_msl_mean', 'pressure_msl_max', 'pressure_msl_min',
    'surface_pressure_mean', 'surface_pressure_max', 'surface_pressure_min',
    'wind_speed_10m_mean', 'wind_speed_10m_max', 'wind_speed_10m_min',
    'wind_gusts_10m_mean', 'wind_gusts_10m_max', 'wind_gusts_10m_min',
    'wind_direction_10m_dominant', 'shortwave_radiation_sum',
]
RISK_FEATURES = ['nearest_prefecture', 'year', 'month', 'day', 'openmeteo_elevation'] + WEATHER
LOCATION_FEATURES = [c for c in WEATHER if c != 'wind_direction_10m_dominant'] + [
    'wind_direction_sin', 'wind_direction_cos', 'calendar_sin', 'calendar_cos']
LABELS = {'LOW': 'Low', 'MEDIUM': 'Moderate', 'MODERATE': 'Moderate', 'HIGH': 'High'}


def keras_module():
    from tensorflow import keras
    return keras


def read_frame(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if 'date' not in df:
        raise ValueError('Input requires an ISO date column.')
    df['date'] = pd.to_datetime(df['date'], errors='raise').dt.normalize()
    if df['date'].isna().any():
        raise ValueError('Date contains missing values.')
    return df


def require_columns(df, columns, context):
    missing = set(columns) - set(df)
    if missing:
        raise ValueError(f'{context}: missing columns {sorted(missing)}')


def series_columns(df, explicit=None):
    if explicit is not None:
        cols = [c.strip() for c in explicit.split(',') if c.strip()]
    elif {'prefecture', 'town'} <= set(df):
        cols = ['prefecture', 'town']
    elif 'landfall_event_id' in df:
        cols = ['landfall_event_id']
    else:
        cols = []
    if 'date' in cols:
        raise ValueError('date cannot be a series column.')
    require_columns(df, cols, 'Series key')
    if df[cols].isna().any().any():
        raise ValueError('Series identifiers cannot be missing.')
    if df.duplicated(cols + ['date']).any():
        raise ValueError(f'Duplicate {cols + ["date"]} keys: dates alone cannot identify multiple towns/storms.')
    return cols


def encoded_weather(df):
    require_columns(df, WEATHER, 'Weather-only location model')
    x = df[[c for c in WEATHER if c != 'wind_direction_10m_dominant']].apply(pd.to_numeric, errors='raise')
    direction = np.deg2rad(pd.to_numeric(df['wind_direction_10m_dominant'], errors='raise'))
    x['wind_direction_sin'] = np.sin(direction)
    x['wind_direction_cos'] = np.cos(direction)
    angle = 2 * np.pi * (df['date'].dt.dayofyear - 1) / 365.25
    x['calendar_sin'], x['calendar_cos'] = np.sin(angle), np.cos(angle)
    values = x[LOCATION_FEATURES].to_numpy(dtype='float32')
    if np.isinf(values).any():
        raise ValueError('Weather includes infinite values.')
    return values


def make_windows(df, lookback, keys):
    """Include weather on the target date; never use future days or coordinate inputs."""
    if lookback < 1:
        raise ValueError('lookback must be positive.')
    df = df.reset_index(drop=True)
    series_columns(df, ','.join(keys))
    raw = encoded_weather(df)
    windows, rows = [], []
    groups = df.groupby(keys, sort=False, dropna=False) if keys else [(None, df)]
    for _, part in groups:
        part = part.sort_values('date')
        idx = part.index.to_numpy()
        days = part['date'].to_numpy(dtype='datetime64[D]').astype('int64')
        for end in range(lookback - 1, len(part)):
            start = end + 1 - lookback
            if np.all(np.diff(days[start:end + 1]) == 1):
                windows.append(raw[idx[start:end + 1]])
                rows.append(int(idx[end]))
    if not windows:
        raise ValueError(f'No consecutive {lookback}-day weather windows. Supply earlier history and valid series keys.')
    return np.asarray(windows), np.asarray(rows, dtype=int)


def fit_location(x, y, lookback, epochs, seed=42):
    keras = keras_module()
    keras.utils.set_random_seed(seed)
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x.reshape(-1, x.shape[-1]))
    xs = StandardScaler().fit(imputer.transform(x.reshape(-1, x.shape[-1])))
    ys = StandardScaler().fit(y)
    xx = xs.transform(imputer.transform(x.reshape(-1, x.shape[-1]))).reshape(x.shape).astype('float32')
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, len(LOCATION_FEATURES))),
        keras.layers.LSTM(32), keras.layers.Dense(16, activation='relu'), keras.layers.Dense(2),
    ])
    model.compile(optimizer='adam', loss='mse')
    # train_on_batch avoids large background tf.data thread pools in small cloud machines.
    rng = np.random.default_rng(seed)
    yy = ys.transform(y).astype('float32')
    for epoch in range(epochs):
        order = rng.permutation(len(xx))
        for offset in range(0, len(order), 32):
            batch = order[offset:offset + 32]
            model.train_on_batch(xx[batch], yy[batch])
    return model, {'imputer': imputer, 'x_scaler': xs, 'y_scaler': ys}


def predict_location(x, model, prep):
    values = prep['x_scaler'].transform(prep['imputer'].transform(x.reshape(-1, x.shape[-1])))
    xx = values.reshape(x.shape).astype('float32')
    batches = [model(xx[i:i + 512], training=False).numpy() for i in range(0, len(xx), 512)]
    result = prep['y_scaler'].inverse_transform(np.concatenate(batches))
    if not np.isfinite(result).all() or (np.abs(result[:, 0]) > 90).any() or (np.abs(result[:, 1]) > 180).any():
        raise ValueError('Location model produced nonfinite/out-of-range coordinates; evaluate or retrain the model.')
    return result


def save_location(folder, model, prep, lookback, train_df):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.save(folder / 'model.keras')
    joblib.dump(prep, folder / 'preprocessing.joblib')
    config = {'contract': 'weather_only_location_v1', 'features': LOCATION_FEATURES, 'lookback': lookback,
              'train_start': str(train_df.date.min().date()), 'train_end': str(train_df.date.max().date()),
              'target': ['latitude', 'longitude'], 'inputs_include_target_day_weather': True}
    (folder / 'config.json').write_text(json.dumps(config, indent=2))


def load_location(folder):
    folder = Path(folder)
    if not (folder / 'config.json').is_file():
        raise ValueError('Use a weather-only location model from train-location/train-risk. '
                         'The uploaded trajectory_lstm.keras needs previous coordinates and is incompatible.')
    cfg = json.loads((folder / 'config.json').read_text())
    if cfg.get('contract') != 'weather_only_location_v1' or cfg.get('features') != LOCATION_FEATURES:
        raise ValueError('Unsupported location feature contract; do not substitute the uploaded trajectory model.')
    return keras_module().models.load_model(folder / 'model.keras', compile=False), joblib.load(folder / 'preprocessing.joblib'), cfg


class PrefectureMapper:
    def __init__(self, path):
        import geopandas as gpd
        self.frame = gpd.read_file(path)
        if self.frame.crs is None:
            raise ValueError('Japan shapefile requires a coordinate reference system.')
        self.frame = self.frame.to_crs('EPSG:4326')
        if len(self.frame) != 47:
            raise ValueError(f'Expected 47 Japanese prefectures, found {len(self.frame)}. Use jp.shp, not country polygons.')
        candidates = ['name', 'NAME_1', 'NAME', 'name_1', 'prefecture', 'nam']
        self.name_col = next((c for c in candidates if c in self.frame), None)
        if self.name_col is None:
            raise ValueError(f'Cannot find prefecture name field among {list(self.frame)}.')
        if self.frame.geometry.isna().any() or not self.frame.geometry.is_valid.all():
            raise ValueError('Invalid/missing prefecture geometries.')
        from pyproj import Geod
        self.geod = Geod(ellps='WGS84')

    def transform(self, coords):
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        names, distances = [], []
        for latitude, longitude in coords:
            point = Point(float(longitude), float(latitude))
            dd = []
            for geom in self.frame.geometry:
                if geom.covers(point):
                    dd.append(0.0)
                else:
                    target = nearest_points(point, geom)[1]
                    dd.append(abs(self.geod.inv(point.x, point.y, target.x, target.y)[2]) / 1000)
            best = int(np.argmin(dd))
            name = str(self.frame.iloc[best][self.name_col])
            try:
                name = name.encode('latin-1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            names.append(name); distances.append(dd[best])
        return names, distances


def location_frame(df, rows, coords, mapper):
    result = df.iloc[rows].copy().reset_index(drop=True)
    # Preserve station/observed coordinates separately; coordinates in the output are reconstructed cyclone positions.
    for col in ['latitude', 'longitude']:
        if col in result:
            result['input_' + col] = result[col]
    result['latitude'], result['longitude'] = coords[:, 0], coords[:, 1]
    result['nearest_prefecture'], result['distance_to_prefecture_km'] = mapper.transform(coords)
    result['location_source'] = 'weather_only_lstm'
    result['year'], result['month'], result['day'] = result.date.dt.year, result.date.dt.month, result.date.dt.day
    if 'openmeteo_elevation' not in result and 'elevation' in result:
        result['openmeteo_elevation'] = result['elevation']
    return result


def join_locations(weather, locations, keys):
    """Join explicit daily/series keys; reject ambiguous joins instead of multiplying rows."""
    join = keys + ['date']
    series_columns(weather, ','.join(keys))
    series_columns(locations, ','.join(keys))
    cols = ['latitude', 'longitude', 'nearest_prefecture', 'distance_to_prefecture_km', 'location_source']
    require_columns(locations, join + cols, 'Reconstructed locations')
    base = weather.copy()
    for col in ['latitude', 'longitude']:
        if col in base:
            base['input_' + col] = base[col]
    base = base.drop(columns=[c for c in cols if c in base])
    merged = base.merge(locations[join + cols], on=join, how='left', validate='one_to_one')
    if merged['nearest_prefecture'].isna().any():
        raise ValueError('Some requested dates lack consecutive location-model history; supply preceding days.')
    merged['year'], merged['month'], merged['day'] = merged.date.dt.year, merged.date.dt.month, merged.date.dt.day
    if 'openmeteo_elevation' not in merged and 'elevation' in merged:
        merged['openmeteo_elevation'] = merged['elevation']
    return merged


def date_subset(df, start=None, end=None):
    if start:
        df = df[df.date >= pd.Timestamp(start)]
    if end:
        df = df[df.date <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError('No records in requested date range; event datasets do not cover every calendar day.')
    return df.copy().reset_index(drop=True)


def write_csv(df, path):
    path = Path(path)
    if path.exists():
        raise ValueError(f'Output already exists: {path}; choose a new output to preserve inputs/artifacts.')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    print(f'Saved {len(df)} rows to {path}')


def labelled_samples(args):
    df = date_subset(read_frame(args.labels), args.start_date, args.end_date)
    require_columns(df, ['latitude', 'longitude', args.group_column], 'Labelled coordinates')
    if df[args.group_column].isna().any():
        raise ValueError('Training group IDs cannot be missing.')
    selected_keys = args.series_columns
    if selected_keys is None and 'landfall_event_id' in df:
        selected_keys = 'landfall_event_id'
    keys = series_columns(df, selected_keys)
    x, rows = make_windows(df, args.lookback, keys)
    samples = df.iloc[rows].reset_index(drop=True)
    y = samples[['latitude', 'longitude']].apply(pd.to_numeric, errors='raise').to_numpy(dtype='float32')
    if not np.isfinite(y).all() or (np.abs(y[:, 0]) > 90).any() or (np.abs(y[:, 1]) > 180).any():
        raise ValueError('Historical target coordinates must be finite and valid.')
    return df, x, rows, samples, y


def train_location(args):
    if Path(args.out).exists():
        raise ValueError('Model output directory exists; choose a new one.')
    df, x, rows, samples, y = labelled_samples(args)
    groups = samples[args.group_column].astype(str)
    if groups.nunique() < 3:
        raise ValueError('Need at least 3 independent storms/series groups to train and evaluate.')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42).split(x, groups=groups))
    model, prep = fit_location(x[train], y[train], args.lookback, args.epochs)
    predicted = predict_location(x[test], model, prep)
    metrics = {'test_samples': len(test), 'heldout_groups': int(groups.iloc[test].nunique()),
               'latitude_mae_degrees': float(np.abs(predicted[:, 0] - y[test, 0]).mean()),
               'longitude_mae_degrees': float(np.abs(predicted[:, 1] - y[test, 1]).mean()),
               'note': 'Separate grouped holdout. Deployment model below is refit on all labelled samples.'}
    model, prep = fit_location(x, y, args.lookback, args.epochs)
    save_location(args.out, model, prep, args.lookback, df)
    (Path(args.out) / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print('No coordinates are inputs. This model requires weather on each target date.')


def risk_estimator(name):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    return {
        'random_forest': lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=2),
        'cart': lambda: DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'gradient_boosting': lambda: GradientBoostingClassifier(random_state=42),
        'svm': lambda: SVC(class_weight='balanced', probability=True, random_state=42),
    }[name]()


def train_risk(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError('Training output directory exists; choose a new one.')
    df, x, rows, samples, y = labelled_samples(args)
    require_columns(samples, ['risk', 'openmeteo_elevation'], 'Risk training needs labelled events, not just weather')
    targets = samples.risk.astype(str).str.upper().map(LABELS)
    if targets.isna().any():
        raise ValueError('Risk labels must be Low/Medium/Moderate/High.')
    groups = samples[args.group_column].astype(str)
    if groups.nunique() < 4:
        raise ValueError('Need at least 4 independent groups for heldout and out-of-fold location predictions.')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42).split(x, groups=groups))
    folds = min(args.folds, groups.iloc[train].nunique())
    if folds < 2:
        raise ValueError('Need at least 2 folds in training groups.')
    # Generate training features without giving each location model its own target coordinates.
    coords_train = np.empty((len(train), 2), dtype='float32')
    for fold, (fit, val) in enumerate(GroupKFold(n_splits=folds).split(x[train], groups=groups.iloc[train])):
        print(f'Location stacking fold {fold + 1}/{folds}', flush=True)
        model, prep = fit_location(x[train[fit]], y[train[fit]], args.lookback, args.epochs, seed=42 + fold)
        coords_train[val] = predict_location(x[train[val]], model, prep)
    # This final location model NEVER sees coordinate labels from risk test groups.
    model, prep = fit_location(x[train], y[train], args.lookback, args.epochs)
    coords_test = predict_location(x[test], model, prep)
    mapper = PrefectureMapper(args.gis)
    a = location_frame(samples, train, coords_train, mapper)
    b = location_frame(samples, test, coords_test, mapper)
    require_columns(a, RISK_FEATURES, 'Classifier inputs')
    if set(targets.iloc[train]) != {'Low', 'Moderate', 'High'}:
        raise ValueError('Training split lacks one or more risk classes; supply more labelled storms.')
    out.mkdir(parents=True)
    save_location(out / 'location', model, prep, args.lookback, samples.iloc[train])
    a['risk'], b['risk'] = targets.iloc[train].to_numpy(), targets.iloc[test].to_numpy()
    write_csv(a, out / 'training_reconstructed.csv'); write_csv(b, out / 'testing_reconstructed.csv')
    metrics = []
    for name in ['random_forest', 'cart', 'gradient_boosting', 'svm']:
        pre = ColumnTransformer([
            ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                                  ('scale', StandardScaler())]), RISK_FEATURES[1:]),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['nearest_prefecture']),
        ])
        classifier = Pipeline([('preprocessor', pre), ('model', risk_estimator(name))])
        classifier.fit(a[RISK_FEATURES], a.risk)
        pred = classifier.predict(b[RISK_FEATURES])
        joblib.dump(classifier, out / (name + '_model.joblib'))
        metrics.append({'model': name, 'accuracy': float(accuracy_score(b.risk, pred)),
                        'balanced_accuracy': float(balanced_accuracy_score(b.risk, pred)), 'test_rows': len(b)})
        (out / (name + '_report.json')).write_text(json.dumps(classification_report(b.risk, pred, output_dict=True, zero_division=0), indent=2))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (out / 'config.json').write_text(json.dumps({'features': RISK_FEATURES, 'group_column': args.group_column,
          'location_train_groups': sorted(groups.iloc[train].unique().tolist()),
          'test_groups': sorted(groups.iloc[test].unique().tolist()),
          'training_features': 'grouped out-of-fold weather-only reconstructed locations',
          'label_semantics': 'Days before confirmed landfall, not general daily occurrence risk.'}, indent=2))
    print(json.dumps(metrics, indent=2))


def prepare(args):
    from weather_forecast import ensure_weather_through
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output exists; choose a new output.')
    weather = read_frame(args.weather)
    if args.request_date and pd.Timestamp(args.request_date) < weather.date.min():
        raise ValueError('Requested date predates input weather; supply historical observations. Future models cannot reconstruct missing history.')
    require_columns(weather, WEATHER, 'Daily weather')
    if args.request_date:
        end = args.request_date
    else:
        end = args.end_date or str(weather.date.max().date())
    if args.prefecture or args.town or pd.Timestamp(end) > weather.date.max():
        weather = ensure_weather_through(weather, args.weather_model, end, args.prefecture, args.town)
    else:
        weather = weather[weather.date <= pd.Timestamp(end)].copy()
    keys = series_columns(weather, args.series_columns)
    # For all requested towns, extension must be considered even when a different town has a later date.
    if keys == ['prefecture', 'town'] and not (args.prefecture or args.town):
        weather = ensure_weather_through(weather, args.weather_model, end)
    model, prep, cfg = load_location(args.location_model)
    x, rows = make_windows(weather, int(cfg['lookback']), keys)
    reconstructed = location_frame(weather.reset_index(drop=True), rows, predict_location(x, model, prep), PrefectureMapper(args.gis))
    start = args.request_date or args.start_date
    wanted = date_subset(weather, start, end)
    if args.request_date:
        if not (wanted.date == pd.Timestamp(args.request_date)).all():
            raise ValueError('Requested date is absent from historical weather; supply model artifacts to forecast it.')
    merged = join_locations(wanted, reconstructed, keys)
    if args.risk_model:
        classifier = joblib.load(args.risk_model)
        expected = list(classifier.feature_names_in_)
        if set(expected) - set(RISK_FEATURES):
            raise ValueError('Classifier requires storm-only features. Use the weather-only classifier from train-risk.')
        require_columns(merged, expected, 'Risk model')
        if merged[expected].isna().any().any():
            raise ValueError('Risk model input includes missing weather/elevation; supply measured or model-produced values.')
        predicted = classifier.predict(merged[expected])
        merged['model_risk_label'] = predicted
        merged['risk_prediction'] = pd.Series(predicted).str.upper().replace({'MODERATE': 'MEDIUM'}).to_numpy()
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(merged[expected])
            for i, label in enumerate(classifier.classes_):
                merged['probability_' + ('MEDIUM' if label == 'Moderate' else str(label).upper())] = proba[:, i]
    write_csv(merged, out)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    for name, func in [('train-location', train_location), ('train-risk', train_risk)]:
        q = commands.add_parser(name)
        q.add_argument('--labels', required=True, help='Historical weather + true storm coordinates; risk labels for train-risk.')
        q.add_argument('--out', required=True, help='New output directory.')
        q.add_argument('--start-date', default='2021-01-01')
        q.add_argument('--end-date', default='2025-01-31')
        q.add_argument('--lookback', type=int, default=1, help='Consecutive days including target day. Default 1 preserves all labels in four-day event data.')
        q.add_argument('--epochs', type=int, default=30)
        q.add_argument('--test-size', type=float, default=0.2)
        q.add_argument('--series-columns', help='Comma-separated series key; defaults to event ID for storm labels.')
        q.add_argument('--group-column', default='intl_id', help='Hold out whole storms to avoid event leakage.')
        if name == 'train-risk':
            q.add_argument('--gis', required=True, help='Japan 47-prefecture jp.shp.')
            q.add_argument('--folds', type=int, default=3)
        q.set_defaults(func=func)
    q = commands.add_parser('prepare')
    q.add_argument('--weather', required=True)
    q.add_argument('--location-model', required=True, help='Directory produced by train-location or train-risk/location.')
    q.add_argument('--weather-model', help='Notebook weather directory with model.keras, scalers.joblib, config.json; needed beyond cache.')
    q.add_argument('--gis', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--start-date'); q.add_argument('--end-date')
    q.add_argument('--request-date', help='Return exactly this date; extends weather recursively if necessary.')
    q.add_argument('--prefecture'); q.add_argument('--town')
    q.add_argument('--series-columns', help='Comma-separated key, default prefecture,town or landfall_event_id.')
    q.add_argument('--risk-model', help='Optional weather-only classification pipeline joblib.')
    q.set_defaults(func=prepare)
    return p


def main():
    p = parser(); args = p.parse_args()
    if hasattr(args, 'epochs') and (args.epochs < 1 or not 0 < args.test_size < 1):
        p.error('epochs must be positive and 0 < test-size < 1.')
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        p.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

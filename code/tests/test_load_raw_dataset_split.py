"""load_raw_dataset(split=...) no dataset sintético de 3 noites (Requisitos 1.3–1.6)."""
import logging

import numpy as np
import pytest

from epb_detection_system_v5 import load_raw_dataset, extract_night_groups


def _load(ds, **kw):
    return load_raw_dataset(str(ds.images_dir), str(ds.labels_csv), **kw)


def _nights(df):
    return set(extract_night_groups(df))


def test_no_test_file_dev_equals_all(synthetic_dataset):
    ds = synthetic_dataset
    _, y_dev, df_dev = _load(ds, include_cloudy=True)
    _, y_all, df_all = _load(ds, include_cloudy=True, split='all')
    assert list(df_dev['filename']) == list(df_all['filename'])
    assert len(df_all) == 9 and np.array_equal(y_dev, y_all)


def test_dev_test_all_with_cloudy_test_night(synthetic_dataset):
    ds = synthetic_dataset
    ds.test_nights_path.write_text("# comentário\n\nCA_20260624\nBJL_20250722\n")
    X_dev, y_dev, df_dev = _load(ds, include_cloudy=True)
    X_test, y_test, df_test = _load(ds, include_cloudy=True, split='test')
    assert _nights(df_dev) == {'CA_20251022'}
    assert _nights(df_test) == {'CA_20260624', 'BJL_20250722'}
    assert X_dev.shape == (3, 512, 512) and set(y_dev) == {1}
    assert set(y_test) == {0, -1}
    # sem nublados: noite nublada some do teste
    _, y_t, df_t = _load(ds, split='test')
    assert _nights(df_t) == {'BJL_20250722'} and set(y_t) == {0}


def test_explicit_test_nights_path(synthetic_dataset, tmp_path):
    ds = synthetic_dataset
    other = tmp_path / 'outra_lista.txt'
    other.write_text("CA_20251022\n")
    _, _, df = _load(ds, include_cloudy=True, test_nights_path=str(other))
    assert _nights(df) == {'BJL_20250722', 'CA_20260624'}


def test_empty_test_split_returns_empty(synthetic_dataset):
    X, y, df = _load(synthetic_dataset, split='test')
    assert len(X) == len(y) == len(df) == 0


def test_warns_for_unknown_test_night(synthetic_dataset, caplog):
    ds = synthetic_dataset
    ds.test_nights_path.write_text("CA_20990101\n")
    with caplog.at_level(logging.INFO, logger='epb_detection_system_v5'):
        _, _, df = _load(ds, include_cloudy=True)
    assert len(df) == 9
    assert any('CA_20990101' in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_logs_excluded_frames_and_nights(synthetic_dataset, caplog):
    ds = synthetic_dataset
    ds.test_nights_path.write_text("CA_20251022\n")
    with caplog.at_level(logging.INFO, logger='epb_detection_system_v5'):
        _load(ds, include_cloudy=True)
    assert any('3 quadros de 1 noites excluídos' in r.getMessage() for r in caplog.records)


def test_rejects_unknown_split(synthetic_dataset):
    with pytest.raises(ValueError, match='split'):
        _load(synthetic_dataset, split='train')

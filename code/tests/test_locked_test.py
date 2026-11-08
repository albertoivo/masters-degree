"""load_test_nights e evaluate_locked_test (Requisitos 1.1, 1.2, 1.6, 1.8, 1.9).

Usa um dataset local de 5 noites (2 EPB, 2 sem EPB, 1 nublada) em tmp_path, porque o
`synthetic_dataset` do conftest tem só uma noite de cada classe e não treina com uma
noite reservada. O log é sempre gravado em tmp_path, nunca em data/test_evaluations.log.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from conftest import LABEL_COLUMNS, frame_name, write_png
from epb_detection_v5 import (EPBRecognitionSystemV5, evaluate_locked_test,
                              load_test_nights)

# night -> (site, início, has_epb, cloudy)
LOCAL_NIGHTS = {
    'CA_20251022': ('CA', datetime(2025, 10, 22, 22, 0), 1, 0),
    'CA_20251023': ('CA', datetime(2025, 10, 23, 22, 0), 1, 0),
    'BJL_20250722': ('BJL', datetime(2025, 7, 22, 22, 0), 0, 0),
    'BJL_20250723': ('BJL', datetime(2025, 7, 23, 22, 0), 0, 0),
    'CA_20260624': ('CA', datetime(2026, 6, 24, 23, 0), None, 1),
}
OFFSETS_MIN = [0, 60, 150, 200]   # 200 min após 22:00 -> quadro depois da meia-noite


@pytest.fixture
def four_nights(tmp_path):
    data_dir = tmp_path / 'data'
    images = data_dir / 'all-images'
    rows, seed = [], 0
    for night, (site, start, has_epb, cloudy) in LOCAL_NIGHTS.items():
        for off in OFFSETS_MIN:
            t = start + timedelta(minutes=off)
            name = frame_name(site, t)
            seed += 1
            write_png(images / name, seed)
            rows.append({'filename': name, 'has_epb': '' if has_epb is None else has_epb,
                         'cloudy': cloudy, 'site': site,
                         'datetime': f"{t:%Y-%m-%d %H:%M:%S}", 'filter': '630.0nm',
                         'kp': 1.3, 'dst': 8, 'moon_illumination': 10.0, 'moon_alt': -5.0,
                         'bz_gsm': 0.1, 'f107': 130.0})
    labels = data_dir / 'epb_labels.csv'
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(labels, index=False)
    return {'images': str(images), 'labels': str(labels),
            'nights_file': data_dir / 'test_nights.txt',
            'log': data_dir / 'test_evaluations.log'}


def _make_model():
    return EPBRecognitionSystemV5(n_components=2)


def _run(ds, note='teste', **kw):
    return evaluate_locked_test(_make_model, note, data_folder=ds['images'],
                                label_file=ds['labels'], **kw)


# ---------------------------------------------------------------- load_test_nights

def test_load_test_nights_ignores_comments_and_blank_lines(tmp_path):
    p = tmp_path / 'test_nights.txt'
    p.write_text("# instruções\n\n  CA_20251022  \n#BJL_20250722\n   \nBJL_20250723\n# fim\n")
    assert load_test_nights(str(p)) == {'CA_20251022', 'BJL_20250723'}


def test_load_test_nights_missing_file_is_empty(tmp_path):
    assert load_test_nights(str(tmp_path / 'nao_existe.txt')) == set()


def test_load_test_nights_only_comments_is_empty(tmp_path):
    p = tmp_path / 'test_nights.txt'
    p.write_text("# Uma noite por linha\n# Ex.: CA_20251022\n\n")
    assert load_test_nights(str(p)) == set()


# ---------------------------------------------------------------- evaluate_locked_test

def test_empty_test_set_raises_and_writes_nothing(four_nights):
    ds = four_nights
    ds['nights_file'].write_text("# vazio\n\n")
    with pytest.raises(ValueError, match='vazio'):
        _run(ds)
    assert not ds['log'].exists()


def test_missing_test_nights_file_raises(four_nights):
    ds = four_nights
    assert not ds['nights_file'].exists()
    with pytest.raises(ValueError, match='vazio'):
        _run(ds)
    assert not ds['log'].exists()


def test_unknown_test_night_warns_and_raises(four_nights, caplog):
    ds = four_nights
    ds['nights_file'].write_text("CA_20251022\nCA_20990101\n")
    with caplog.at_level(logging.INFO, logger='epb_detection_v5'):
        with pytest.raises(ValueError, match='CA_20990101'):
            _run(ds)
    assert any('CA_20990101' in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)
    assert not ds['log'].exists()


def test_cloudy_only_test_night_raises(four_nights):
    ds = four_nights
    ds['nights_file'].write_text("CA_20251022\nCA_20260624\n")
    with pytest.raises(ValueError, match='CA_20260624'):
        _run(ds)
    assert not ds['log'].exists()


def test_single_class_dev_raises(four_nights):
    ds = four_nights
    ds['nights_file'].write_text("BJL_20250722\nBJL_20250723\n")
    with pytest.raises(ValueError, match='duas classes'):
        _run(ds)
    assert not ds['log'].exists()


def test_appends_one_log_line_per_call(four_nights):
    ds = four_nights
    ds['nights_file'].write_text("# teste\nCA_20251022\nBJL_20250722\n")

    res = _run(ds, note='primeira')
    assert res['test_nights'] == ['BJL_20250722', 'CA_20251022']
    assert list(res['per_night']['night']) == ['BJL_20250722', 'CA_20251022']
    assert list(res['per_night']['n']) == [4, 4]
    assert res['summary']['n_test_frames'] == 8
    assert res['summary']['n_dev_frames'] == 8
    assert res['summary']['n_dev_nights'] == 2

    lines = ds['log'].read_text(encoding='utf-8').splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry) == {'timestamp', 'git_commit', 'test_nights', 'note', 'metrics', 'per_night'}
    assert entry['note'] == 'primeira'
    assert entry['test_nights'] == ['BJL_20250722', 'CA_20251022']
    assert isinstance(entry['git_commit'], str) and entry['git_commit']
    datetime.fromisoformat(entry['timestamp'])
    assert entry == res['log_entry']
    assert [r['night'] for r in entry['per_night']] == ['BJL_20250722', 'CA_20251022']

    _run(ds, note='segunda')
    lines = ds['log'].read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2
    assert [json.loads(l)['note'] for l in lines] == ['primeira', 'segunda']


def test_explicit_paths_override_defaults(four_nights, tmp_path):
    ds = four_nights
    other_nights = tmp_path / 'outra' / 'nights.txt'
    other_nights.parent.mkdir()
    other_nights.write_text("CA_20251023\nBJL_20250723\n")
    other_log = tmp_path / 'logs' / 'eval.log'
    res = _run(ds, test_nights_path=str(other_nights), log_path=str(other_log))
    assert res['test_nights'] == ['BJL_20250723', 'CA_20251023']
    assert len(other_log.read_text(encoding='utf-8').splitlines()) == 1
    assert not ds['log'].exists()

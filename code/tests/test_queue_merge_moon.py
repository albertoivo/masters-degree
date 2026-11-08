"""Testes unitários da fila de rotulagem, do merge e do script da Lua.
Requisitos 4.3, 4.8, 5.2, 5.6, 5.8. Tudo roda sobre o `synthetic_dataset` em tmp_path.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import label_queue as lq
import merge_labels as ml
import moon_illumination as moon


# --- fila -----------------------------------------------------------------

def test_even_spacing_picks_first_and_last():
    t0 = datetime(2025, 10, 22, 22, 0, 0)
    times = [t0 + timedelta(minutes=m) for m in range(0, 400, 10)]  # 40 quadros
    idx = lq.select_evenly_spaced(times, 5)
    assert len(idx) == 5
    assert idx[0] == 0 and idx[-1] == len(times) - 1
    assert idx == sorted(set(idx))
    # alvos 0, 97.5, 195, 292.5, 390 min -> quadros mais próximos
    assert [(times[i] - t0).seconds // 60 for i in idx] == [0, 100, 190, 290, 390]


def test_even_spacing_small_nights():
    t0 = datetime(2025, 1, 1, 22)
    times = [t0 + timedelta(minutes=m) for m in (0, 5, 50)]
    assert lq.select_evenly_spaced(times, 20) == [0, 1, 2]
    assert lq.select_evenly_spaced(times, 2) == [0, 2]
    assert lq.select_evenly_spaced([], 5) == []


def _run(ds, **kw):
    return lq.run(ds.util_dir, ds.labels_csv, ds.data_dir / 'label_queue.csv',
                  ds.data_dir / 'sheets', sheets=False, **kw)


def test_str_excluded_by_default(synthetic_dataset):
    ds = synthetic_dataset
    summary = _run(ds)
    queue = pd.read_csv(ds.data_dir / 'label_queue.csv', dtype=str)
    assert 'STR' not in set(queue['site'])
    assert 'STR_20250801' not in set(summary['night'])
    # cada noite rotulada: todos os não rotulados (<= 20) estão na fila, nada rotulado
    expected = {fn for n, fns in ds.unlabeled.items() if not n.startswith('STR') for fn in fns}
    assert set(queue['filename']) == expected

    _run(ds, exclude_sites=[], force=True)
    queue = pd.read_csv(ds.data_dir / 'label_queue.csv', dtype=str)
    assert set(ds.unlabeled['STR_20250801']) <= set(queue['filename'])


def test_force_required_to_overwrite_filled_queue(synthetic_dataset, capsys):
    ds = synthetic_dataset
    queue_csv = ds.data_dir / 'label_queue.csv'
    args = ['--util-dir', str(ds.util_dir), '--labels-csv', str(ds.labels_csv),
            '--queue-csv', str(queue_csv), '--no-sheets']
    assert lq.main(args) == 0

    q = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    q.loc[0, 'notes'] = 'visto'
    q.to_csv(queue_csv, index=False)
    filled = queue_csv.read_bytes()

    with pytest.raises(FileExistsError):
        _run(ds)
    assert lq.main(args) == 1
    assert '--force' in capsys.readouterr().err
    assert queue_csv.read_bytes() == filled          # fila preenchida intacta

    assert lq.main(args + ['--force']) == 0
    q = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    assert (q[['has_epb', 'cloudy', 'notes']] == '').all().all()


# --- merge ----------------------------------------------------------------

def _queue_rows(ds, night, labels):
    """Linhas de fila para os primeiros quadros não rotulados de `night`."""
    site = night.split('_')[0]
    return [{'filename': fn, 'night': night, 'site': site,
             'path': str(ds.processed_dir(night) / fn),
             'has_epb': h, 'cloudy': c, 'notes': ''}
            for fn, (h, c) in zip(ds.unlabeled[night], labels)]


def _write_queue(ds, rows):
    path = ds.data_dir / 'label_queue.csv'
    pd.DataFrame(rows, columns=lq.QUEUE_COLUMNS).to_csv(path, index=False)
    return path


def test_duplicate_filename_rejected(synthetic_dataset):
    ds = synthetic_dataset
    night = 'CA_20251022'
    rows = _queue_rows(ds, night, [('1', '0')])
    dup = ds.labeled[night][0]                       # já está no CSV
    rows.append({'filename': dup, 'night': night, 'site': 'CA',
                 'path': str(ds.processed_dir(night) / dup),
                 'has_epb': '0', 'cloudy': '0', 'notes': ''})
    queue = _write_queue(ds, rows)
    before = ds.labels_csv.read_bytes()
    images_before = sorted(p.name for p in ds.images_dir.iterdir())

    with pytest.raises(ml.MergeError) as exc:
        ml.merge(queue, ds.labels_csv, ds.images_dir, backup_dir=ds.root / 'bak')
    assert [fn for _, fn, _ in exc.value.rejected] == [dup]
    assert ds.labels_csv.read_bytes() == before
    assert sorted(p.name for p in ds.images_dir.iterdir()) == images_before
    assert not (ds.root / 'bak').exists()


def test_duplicate_within_queue_rejected(synthetic_dataset):
    ds = synthetic_dataset
    rows = _queue_rows(ds, 'BJL_20250722', [('0', '0')])
    rows.append(dict(rows[0]))
    with pytest.raises(ml.MergeError) as exc:
        ml.merge(_write_queue(ds, rows), ds.labels_csv, ds.images_dir)
    assert len(exc.value.rejected) == 1


def test_existing_image_not_overwritten(synthetic_dataset):
    ds = synthetic_dataset
    night = 'CA_20251022'
    rows = _queue_rows(ds, night, [('1', '0'), ('', '1')])
    existing = ds.images_dir / rows[0]['filename']
    existing.write_bytes(b'conteudo original')
    before = ds.labels_csv.read_bytes()

    result = ml.merge(_write_queue(ds, rows), ds.labels_csv, ds.images_dir,
                      backup_dir=ds.root / 'bak', now=datetime(2026, 1, 2, 3, 4, 5))
    assert existing.read_bytes() == b'conteudo original'
    assert result.kept_existing == [rows[0]['filename']]
    assert result.copied == [rows[1]['filename']]
    assert (ds.images_dir / rows[1]['filename']).read_bytes() == \
        (ds.processed_dir(night) / rows[1]['filename']).read_bytes()
    assert ds.labels_csv.read_bytes().startswith(before)
    assert result.backup.read_bytes() == before
    df = ds.labels()
    assert set(rows[i]['filename'] for i in range(2)) <= set(df['filename'])


# --- script da Lua ----------------------------------------------------------

def test_unknown_station_raises_naming_it():
    assert moon.get_site_coordinates('CA') == ('-7.38', '-36.52')
    with pytest.raises(ValueError, match='STR'):
        moon.get_site_coordinates('STR')

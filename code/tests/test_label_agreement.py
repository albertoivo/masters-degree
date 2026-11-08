"""Testes unitários de util/label_agreement.py (Requisitos 3.2, 3.3, 3.5).

Usa o dataset sintético de conftest.py: 3 noites com 3 quadros rotulados cada
(CA_20251022 = EPB, BJL_20250722 = Sem EPB, CA_20260624 = Indeterminado).
Nada é escrito em data/ real.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

import label_agreement as la


def read_csv(path) -> pd.DataFrame:
    # mesma leitura da CLI
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# --------------------------------------------------------------------------- sample (3.2)

def test_sample_has_blank_labels_and_min_two_per_night(synthetic_dataset):
    labels = read_csv(synthetic_dataset.labels_csv)
    sample = la.sample_frames(labels, n=6, seed=42)

    assert list(sample.columns) == la.SAMPLE_COLUMNS
    assert len(sample) == 6
    for col in ('has_epb', 'cloudy', 'notes'):
        assert (sample[col] == '').all()
    counts = sample.groupby('night').size().to_dict()
    assert set(counts) == set(synthetic_dataset.labeled)
    assert all(k >= 2 for k in counts.values())
    # só quadros rotulados, sem repetição, e noite consistente com o nome
    all_labeled = {f for fs in synthetic_dataset.labeled.values() for f in fs}
    assert set(sample['filename']) <= all_labeled
    assert sample['filename'].is_unique
    assert all(la.night_of(f) == n for f, n in zip(sample['filename'], sample['night']))


def test_sample_small_n_still_gives_two_per_night(synthetic_dataset):
    labels = read_csv(synthetic_dataset.labels_csv)
    sample = la.sample_frames(labels, n=1, seed=0)
    assert sample.groupby('night').size().min() >= 2


def test_sample_is_deterministic_for_seed(synthetic_dataset):
    labels = read_csv(synthetic_dataset.labels_csv)
    a = la.sample_frames(labels, n=7, seed=3)
    b = la.sample_frames(labels, n=7, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_cli_sample_writes_to_out(synthetic_dataset):
    out = synthetic_dataset.data_dir / 'relabel_sample.csv'
    rc = la.main(['sample', '--n', '6', '--seed', '1',
                  '--labels', str(synthetic_dataset.labels_csv), '--out', str(out)])
    assert rc == 0
    sample = read_csv(out)
    assert len(sample) == 6
    assert (sample[['has_epb', 'cloudy', 'notes']] == '').all().all()


# --------------------------------------------------------------------------- kappa (3.3)

def test_kappa_matches_hand_computed_table():
    # tabela (linhas = a, colunas = b):
    #            EPB  Sem  Ind
    #   EPB       2    1    0
    #   Sem       0    2    0
    #   Ind       0    0    1
    # po = 5/6; pe = (3*2 + 2*3 + 1*1)/36 = 13/36; kappa = (30-13)/(36-13) = 17/23
    a = ['EPB', 'EPB', 'EPB', 'Sem EPB', 'Sem EPB', 'Indeterminado']
    b = ['EPB', 'EPB', 'Sem EPB', 'Sem EPB', 'Sem EPB', 'Indeterminado']
    assert la.raw_agreement(a, b) == pytest.approx(5 / 6)
    assert la.cohen_kappa(a, b) == pytest.approx(17 / 23)
    conf = la.confusion_matrix(a, b, labels=la.VERDICTS)
    assert conf.loc['EPB', 'EPB'] == 2
    assert conf.loc['EPB', 'Sem EPB'] == 1
    assert conf.loc['Sem EPB', 'Sem EPB'] == 2
    assert conf.loc['Indeterminado', 'Indeterminado'] == 1
    assert conf.values.sum() == 6


def test_kappa_undefined_for_single_class_and_empty():
    assert math.isnan(la.cohen_kappa(['EPB'] * 4, ['EPB'] * 4))
    assert math.isnan(la.cohen_kappa([], []))


def test_verdict_rules():
    assert la.verdict('1', '0') == 'EPB'
    assert la.verdict('0', '0') == 'Sem EPB'
    assert la.verdict('', '1') == 'Indeterminado'
    assert la.verdict('', '') is None
    assert la.verdict('1', '', lenient_cloudy=True) == 'EPB'
    with pytest.raises(ValueError):
        la.verdict('1', '')


# --------------------------------------------------------------------------- score (3.3, 3.5)

def _relabel_frame(ds) -> pd.DataFrame:
    """Todos os 9 quadros; 1 EPB re-rotulado como Sem EPB, 1 Indeterminado deixado vazio.
    Um Sem EPB vem com cloudy vazio (leitura leniente -> cloudy=0)."""
    epb, sem, ind = (ds.labeled['CA_20251022'], ds.labeled['BJL_20250722'],
                     ds.labeled['CA_20260624'])
    rows = []
    rows += [{'filename': epb[0], 'has_epb': '0', 'cloudy': '0', 'notes': 'fraca'}]
    rows += [{'filename': f, 'has_epb': '1', 'cloudy': '0', 'notes': ''} for f in epb[1:]]
    rows += [{'filename': sem[0], 'has_epb': '0', 'cloudy': '', 'notes': ''}]
    rows += [{'filename': f, 'has_epb': '0', 'cloudy': '0', 'notes': ''} for f in sem[1:]]
    rows += [{'filename': ind[0], 'has_epb': '', 'cloudy': '', 'notes': ''}]   # ignorado
    rows += [{'filename': f, 'has_epb': '', 'cloudy': '1', 'notes': ''} for f in ind[1:]]
    return pd.DataFrame(rows, columns=['filename', 'has_epb', 'cloudy', 'notes'])


def test_score_ignores_blank_rows_and_matches_hand_kappa(synthetic_dataset):
    ds = synthetic_dataset
    labels = read_csv(ds.labels_csv)
    res = la.score_relabels(_relabel_frame(ds), labels)

    assert res['n_ignored'] == 1
    assert res['n_scored'] == 8
    # original: EPB 3, Sem 3, Ind 2; re-rótulo: EPB 2, Sem 4, Ind 2
    # po = 7/8; pe = (3*2 + 3*4 + 2*2)/64 = 22/64; kappa = (56-22)/(64-22) = 17/21
    assert res['agreement'] == pytest.approx(7 / 8)
    assert res['kappa'] == pytest.approx(17 / 21)
    conf = res['confusion']
    assert conf.loc['EPB', 'Sem EPB'] == 1
    assert conf.loc['EPB', 'EPB'] == 2
    assert conf.loc['Sem EPB', 'Sem EPB'] == 3
    assert conf.loc['Indeterminado', 'Indeterminado'] == 2
    assert conf.values.sum() == 8

    pn = res['per_night'].set_index('night')
    assert pn.loc['CA_20251022', 'n'] == 3
    assert pn.loc['CA_20251022', 'agreement'] == pytest.approx(2 / 3)
    assert pn.loc['BJL_20250722', 'agreement'] == pytest.approx(1.0)
    assert pn.loc['CA_20260624', 'n'] == 2   # linha vazia não conta

    dis = res['disagreements']
    assert list(dis['filename']) == [ds.labeled['CA_20251022'][0]]
    assert dis.loc[0, 'original'] == 'EPB' and dis.loc[0, 'relabel'] == 'Sem EPB'
    assert dis.loc[0, 'notes'] == 'fraca'


def test_score_all_blank_scores_nothing(synthetic_dataset):
    labels = read_csv(synthetic_dataset.labels_csv)
    sample = la.sample_frames(labels, n=6, seed=42)
    res = la.score_relabels(sample, labels)
    assert res['n_ignored'] == len(sample)
    assert res['n_scored'] == 0
    assert math.isnan(res['kappa'])


def test_score_rejects_frame_missing_from_labels(synthetic_dataset):
    labels = read_csv(synthetic_dataset.labels_csv)
    bad = pd.DataFrame([{'filename': 'O6_CA_20990101_220000.png',
                         'has_epb': '1', 'cloudy': '0', 'notes': ''}])
    with pytest.raises(ValueError):
        la.score_relabels(bad, labels)


def test_cli_score_writes_disagreements(synthetic_dataset, capsys):
    ds = synthetic_dataset
    relabel_csv = ds.data_dir / 'relabel_sample.csv'
    out = ds.data_dir / 'relabel_disagreements.csv'
    _relabel_frame(ds).to_csv(relabel_csv, index=False)
    rc = la.main(['score', str(relabel_csv), '--labels', str(ds.labels_csv), '--out', str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert 'Linhas ignoradas (rótulos vazios): 1' in text
    assert f"{17 / 21:.3f}" in text
    dis = read_csv(out)
    assert list(dis['filename']) == [ds.labeled['CA_20251022'][0]]

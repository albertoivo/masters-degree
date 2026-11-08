"""Property 4: a fila nunca contém quadros já rotulados.

**Validates: Requirements 4.1, 4.2**
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from hypothesis import given, settings, strategies as st

import label_queue as lq

SITES = ['CA', 'BJL', 'CP', 'STR']
BASE = datetime(2025, 7, 1, 18, 0, 0)


@st.composite
def frame_sets(draw):
    """Quadros únicos (site, instante) espalhados em algumas noites, com a hora em
    qualquer ponto do dia para exercitar a regra das 12:00 UT."""
    keys = draw(st.lists(
        st.tuples(st.sampled_from(SITES),
                  st.integers(0, 4),                 # dia
                  st.integers(0, 24 * 60 - 1)),      # minuto do dia
        min_size=0, max_size=60, unique=True))
    frames = []
    for site, day, minute in keys:
        t = BASE.replace(hour=0) + timedelta(days=day, minutes=minute)
        name = f"O6_{site}_{t:%Y%m%d_%H%M%S}.png"
        frames.append(lq.Frame(name, Path('/nonexistent') / name, site,
                               lq.night_id(site, t), t))
    return frames


@st.composite
def queue_inputs(draw):
    frames = draw(frame_sets())
    names = [f.filename for f in frames]
    labeled = set(draw(st.lists(st.sampled_from(names), unique=True))) if names else set()
    # nomes rotulados que não estão em disco não devem afetar nada
    labeled |= set(draw(st.lists(st.just('O6_CA_20990101_000000.png'), max_size=1)))
    exclude = draw(st.one_of(st.just(['STR']),
                             st.lists(st.sampled_from(SITES + ['str']), unique=True)))
    per_night = draw(st.one_of(st.none(), st.integers(1, 25)))
    return frames, labeled, exclude, per_night


@settings(max_examples=300, deadline=None)
@given(queue_inputs())
def test_queue_never_contains_labeled_or_excluded_frames(inputs):
    frames, labeled, exclude, per_night = inputs
    queue, summary = lq.build_queue(frames, labeled, exclude_sites=exclude,
                                    per_night=per_night)
    excluded = {s.upper() for s in exclude}
    input_set = set(frames)

    for f in queue:
        assert f.filename not in labeled                 # Req 4.1
        assert f.site.upper() not in excluded            # Req 4.2
        assert f in input_set                            # fila ⊆ quadros em disco
    assert len({f.filename for f in queue}) == len(queue)

    if per_night is None:                                # --all: todo não rotulado entra
        expected = {f.filename for f in frames
                    if f.filename not in labeled and f.site.upper() not in excluded}
        assert {f.filename for f in queue} == expected

    assert set(summary['site'].str.upper()).isdisjoint(excluded)
    assert int(summary['queued'].sum()) == len(queue)


def test_end_to_end_queue_csv_excludes_labeled_and_str(synthetic_dataset, tmp_path):
    ds = synthetic_dataset
    queue_csv = tmp_path / 'out' / 'label_queue.csv'
    rc = lq.main(['--util-dir', str(ds.util_dir), '--labels-csv', str(ds.labels_csv),
                  '--queue-csv', str(queue_csv), '--sheets-dir', str(tmp_path / 'sheets'),
                  '--no-sheets', '--all'])
    assert rc == 0
    q = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    labeled = set(ds.labels()['filename'])
    assert set(q['filename']).isdisjoint(labeled)
    assert 'STR' not in set(q['site'])
    assert all(Path(p).exists() for p in q['path'])
    expected = {n for night, names in ds.unlabeled.items()
                if not night.startswith('STR') for n in names}
    assert set(q['filename']) == expected

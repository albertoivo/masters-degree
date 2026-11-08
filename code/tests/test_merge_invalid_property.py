"""Property 3: entrada inválida deixa o CSV intacto.

Para qualquer fila com pelo menos uma linha inválida, `merge` levanta MergeError,
`epb_labels.csv` fica byte-idêntico, nenhum backup é criado, nenhuma PNG é copiada
e toda linha inválida aparece em `.rejected`.

**Validates: Requirements 5.2, 5.3**
"""
from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import merge_labels as ml

QUEUE_FIELDS = ['filename', 'night', 'site', 'path', 'has_epb', 'cloudy', 'notes']
LABEL_HEADER = ['filename', 'has_epb', 'cloudy', 'site', 'datetime', 'filter',
                'kp', 'dst', 'moon_illumination']
EXISTING = ['O6_CA_20250101_000000.png', 'O6_BJL_20250101_001000.png']
SITES = ['CA', 'BJL', 'CP']
BASE = datetime(2025, 10, 22, 22, 0, 0)

INVALID_KINDS = ['bad_has_epb', 'bad_cloudy', 'cloudy1_with_has_epb',
                 'cloudy0_empty_has_epb', 'has_epb_without_cloudy', 'already_in_csv',
                 'repeated_in_queue', 'bad_name', 'empty_path', 'missing_png']
VALID_VERDICTS = [('1', '0'), ('0', '0'), ('', '1')]
BAD_VALUES = ['2', '-1', 'x', 'yes', '1.0', '01', 'nan']
BAD_NAMES = ['foo.png', 'O6_CA_2025_01.png', 'XX_CA_20250101_000000.png',
             'O6_CA_20251399_000000.png', 'O6_CA_20250101_000000.jpg', 'O6__20250101_000000.png']

# Cada item: ('valid' | 'unlabeled' | tipo inválido, veredito, site)
items = st.one_of(
    st.tuples(st.just('valid'), st.sampled_from(VALID_VERDICTS), st.sampled_from(SITES)),
    st.tuples(st.just('unlabeled'), st.just(('', '')), st.sampled_from(SITES)),
    st.tuples(st.sampled_from(INVALID_KINDS), st.sampled_from(VALID_VERDICTS),
              st.sampled_from(SITES)),
)


@st.composite
def queues(draw):
    plan = draw(st.lists(items, min_size=0, max_size=12))
    forced = (draw(st.sampled_from(INVALID_KINDS)), draw(st.sampled_from(VALID_VERDICTS)),
              draw(st.sampled_from(SITES)))
    plan.insert(draw(st.integers(0, len(plan))), forced)
    bad_value = draw(st.sampled_from(BAD_VALUES))
    bad_name = draw(st.sampled_from(BAD_NAMES))
    return plan, bad_value, bad_name


def _build_rows(plan, bad_value, bad_name, src_dir: Path):
    """Linhas da fila e índices (0-based) das linhas que devem ser rejeitadas."""
    rows, invalid_idx = [], []
    counter = 0

    def fresh(site):
        nonlocal counter
        t = BASE + timedelta(minutes=counter)
        counter += 1
        fn = f"O6_{site}_{t:%Y%m%d_%H%M%S}.png"
        p = src_dir / fn
        p.write_bytes(b'\x89PNG fake ' + fn.encode())
        return fn, str(p)

    def row(fn, path, has_epb, cloudy, site):
        return {'filename': fn, 'night': f'{site}_20251022', 'site': site, 'path': path,
                'has_epb': has_epb, 'cloudy': cloudy, 'notes': ''}

    for kind, (h, c), site in plan:
        fn, path = fresh(site)
        if kind == 'valid':
            rows.append(row(fn, path, h, c, site))
            continue
        if kind == 'unlabeled':
            rows.append(row(fn, path, '', '', site))
            continue
        if kind == 'bad_has_epb':
            h, c = bad_value, '0'
        elif kind == 'bad_cloudy':
            h, c = '0', bad_value
        elif kind == 'cloudy1_with_has_epb':
            h, c = ('1' if h == '1' else '0'), '1'
        elif kind == 'cloudy0_empty_has_epb':
            h, c = '', '0'
        elif kind == 'has_epb_without_cloudy':
            h, c = ('1' if h == '1' else '0'), ''
        elif kind == 'already_in_csv':
            fn = EXISTING[counter % len(EXISTING)]
        elif kind == 'repeated_in_queue':
            rows.append(row(fn, path, h, c, site))  # original válido antes da repetição
        elif kind == 'bad_name':
            fn = bad_name
        elif kind == 'empty_path':
            path = ''
        elif kind == 'missing_png':
            path = str(src_dir / 'nao_existe' / fn)
        invalid_idx.append(len(rows))
        rows.append(row(fn, path, h, c, site))
    return rows, invalid_idx


def _snapshot(d: Path):
    return {p.relative_to(d): p.read_bytes() for p in sorted(d.rglob('*')) if p.is_file()}


@settings(max_examples=200, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(queues())
def test_invalid_queue_leaves_everything_untouched(q):
    plan, bad_value, bad_name = q
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir, src_dir, images_dir, backup_dir = (
            tmp / 'data', tmp / 'src', tmp / 'imgs', tmp / 'bak')
        for d in (data_dir, src_dir, images_dir, backup_dir):
            d.mkdir()
        (images_dir / 'O6_CA_20250101_000000.png').write_bytes(b'existing')

        labels_csv = data_dir / 'epb_labels.csv'
        with open(labels_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(LABEL_HEADER)
            w.writerow([EXISTING[0], '1', '0', 'CA', '2025-01-01 00:00:00', '630.0nm',
                        '3', '-20', '0.5'])
            w.writerow([EXISTING[1], '', '1', 'BJL', '2025-01-01 00:10:00', '630.0nm',
                        '', '', ''])

        rows, invalid_idx = _build_rows(plan, bad_value, bad_name, src_dir)
        queue_csv = tmp / 'label_queue.csv'
        with open(queue_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
            w.writeheader()
            w.writerows(rows)

        labels_before = labels_csv.read_bytes()
        data_before = _snapshot(data_dir)
        images_before = _snapshot(images_dir)

        # backup_dir explícito e padrão (pasta do CSV) são verificados
        for bdir in (backup_dir, None):
            with pytest.raises(ml.MergeError) as exc:
                ml.merge(queue_csv, labels_csv, images_dir, backup_dir=bdir,
                         now=datetime(2026, 1, 1, 12, 0, 0))

            assert labels_csv.read_bytes() == labels_before
            assert _snapshot(data_dir) == data_before          # nenhum backup ao lado do CSV
            assert list(backup_dir.iterdir()) == []             # nenhum backup na pasta dada
            assert _snapshot(images_dir) == images_before      # nenhuma PNG copiada

            rejected_lines = {n for n, _, _ in exc.value.rejected}
            expected_lines = {i + 2 for i in invalid_idx}       # linha 1 = cabeçalho
            assert rejected_lines == expected_lines             # só e todas as inválidas

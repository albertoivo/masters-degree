"""Property 2: Merging labels never alters existing rows.

**Validates: Requirements 5.5**

Para qualquer fila válida, depois de `merge`:
  * os bytes antigos de epb_labels.csv são prefixo exato do arquivo novo
    (exceto o terminador de linha acrescentado quando faltava no final);
  * as linhas existentes, lidas com csv.reader, continuam iguais;
  * o backup é idêntico ao arquivo antigo;
  * o número de linhas novas é o número de linhas rotuladas da fila.
Cada exemplo usa seu próprio TemporaryDirectory; data/ real nunca é tocado.
"""
from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

import merge_labels

LABEL_COLUMNS = ['filename', 'has_epb', 'cloudy', 'site', 'datetime', 'filter',
                 'kp', 'dst', 'moon_illumination', 'moon_alt', 'bz_gsm', 'f107']
QUEUE_COLUMNS = ['filename', 'night', 'site', 'path', 'has_epb', 'cloudy', 'notes']
LABELED_VERDICTS = [('1', '0'), ('0', '0'), ('', '1')]
UNLABELED = ('', '')
NOW = datetime(2026, 1, 2, 3, 4, 5)
BASE = datetime(2025, 7, 22, 20, 0, 0)

# Texto "estranho" para as células existentes: vírgulas, aspas, quebras de linha
# (sempre entre aspas pelo csv.writer), acentos e espaços. Sem '\r' isolado, que
# não seria citado com lineterminator='\n' e não é algo que o CSV real contenha.
cell_text = st.text(alphabet=st.sampled_from(list('abcXYZ019 .-_,"\nçãé')), max_size=12)


@st.composite
def existing_csv(draw):
    """(bytes do CSV existente, nomes existentes)."""
    n_rows = draw(st.integers(0, 6))
    rows = []
    for i in range(n_rows):
        # Nome existente nunca colide com os quadros novos (prefixo OLD_).
        name = f"OLD_{i}_" + draw(cell_text)
        rows.append([name] + [draw(cell_text) for _ in LABEL_COLUMNS[1:]])
    eol = draw(st.sampled_from(['\n', '\r\n']))
    quoting = draw(st.sampled_from([csv.QUOTE_MINIMAL, csv.QUOTE_ALL]))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=eol, quoting=quoting)
    w.writerow(LABEL_COLUMNS)
    w.writerows(rows)
    text = buf.getvalue()
    if draw(st.booleans()):  # sem terminador de linha no final
        text = text[:-len(eol)]
    return text.encode('utf-8'), [r[0] for r in rows]


@st.composite
def queue_rows(draw):
    """Lista de (filename, site, has_epb, cloudy) com nomes únicos e válidos."""
    offsets = draw(st.lists(st.integers(0, 12 * 3600), min_size=0, max_size=8, unique=True))
    out = []
    for off in offsets:
        site = draw(st.sampled_from(['CA', 'BJL', 'CP']))
        t = BASE + timedelta(seconds=off)
        verdict = draw(st.sampled_from(LABELED_VERDICTS + [UNLABELED]))
        out.append((f"O6_{site}_{t:%Y%m%d_%H%M%S}.png", site) + verdict)
    return out


def _parse(raw: bytes):
    return list(csv.reader(io.StringIO(raw.decode('utf-8'))))


@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
@given(existing=existing_csv(), queue=queue_rows())
def test_merge_never_alters_existing_rows(existing, queue):
    old_bytes, _ = existing
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        labels_csv = tmp / 'data' / 'epb_labels.csv'
        images_dir = tmp / 'data' / 'all-images'
        backup_dir = tmp / 'backups'
        src_dir = tmp / 'processed'
        labels_csv.parent.mkdir(parents=True)
        src_dir.mkdir()
        labels_csv.write_bytes(old_bytes)

        queue_csv = tmp / 'label_queue.csv'
        with open(queue_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(QUEUE_COLUMNS)
            for fn, site, has_epb, cloudy in queue:
                png = src_dir / fn
                Image.new('L', (2, 2), 7).save(png)
                w.writerow([fn, f"{site}_night", site, str(png), has_epb, cloudy, ''])

        old_rows = _parse(old_bytes)
        result = merge_labels.merge(queue_csv, labels_csv, images_dir,
                                    backup_dir=backup_dir, now=NOW)
        new_bytes = labels_csv.read_bytes()

        labeled = [q for q in queue if (q[2], q[3]) != UNLABELED]
        assert len(result.appended) == len(labeled)
        assert result.skipped_unlabeled == len(queue) - len(labeled)

        if not labeled:
            assert new_bytes == old_bytes
            assert result.backup is None
            return

        # Bytes antigos preservados: prefixo exato (mais o terminador que faltava).
        assert new_bytes.startswith(old_bytes)
        if not old_bytes.endswith(b'\n'):
            eol = b'\r\n' if b'\r\n' in old_bytes.split(b'\n', 1)[0] + b'\n' else b'\n'
            assert new_bytes[len(old_bytes):].startswith(eol)

        # Linhas existentes iguais; novas linhas = rotuladas, na ordem da fila.
        new_rows = _parse(new_bytes)
        assert new_rows[:len(old_rows)] == old_rows
        appended = new_rows[len(old_rows):]
        assert len(appended) == len(labeled)
        for row, (fn, site, has_epb, cloudy) in zip(appended, labeled):
            rec = dict(zip(LABEL_COLUMNS, row))
            assert (rec['filename'], rec['has_epb'], rec['cloudy'], rec['site']) == \
                (fn, has_epb, cloudy, site)
            assert rec['filter'] == '630.0nm'
            assert all(rec[c] == '' for c in LABEL_COLUMNS[6:])

        # Backup idêntico ao arquivo antigo.
        assert result.backup is not None and result.backup.parent == backup_dir
        assert result.backup.read_bytes() == old_bytes

"""Incorpora a fila de rotulagem preenchida em data/epb_labels.csv.

Fluxo (tudo ou nada):
    1. Lê data/label_queue.csv (filename, night, site, path, has_epb, cloudy, notes).
    2. Ignora linhas com has_epb e cloudy ambos vazios (ainda não rotuladas).
    3. Valida todas as linhas rotuladas; se alguma for inválida, lista todas e sai
       sem modificar nenhum arquivo.
    4. Grava o backup data/epb_labels.YYYYMMDD-HHMMSS.bak.csv.
    5. Acrescenta as linhas com filename, has_epb, cloudy, site, datetime, filter
       (demais colunas vazias). As linhas existentes não são reescritas.
    6. Copia cada PNG para data/all-images/ sem sobrescrever arquivo existente.
    7. Mostra os comandos para preencher os metadados restantes.

Vereditos válidos (data/LABELING_PROTOCOL.md):
    EPB            has_epb=1  cloudy=0
    Sem EPB        has_epb=0  cloudy=0
    Indeterminado  has_epb=   cloudy=1

Uso (a partir de code/):
    python util/merge_labels.py data/label_queue.csv
    python util/merge_labels.py fila.csv --labels-csv outro.csv --images-dir imgs/
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_CSV = CODE_ROOT / 'data' / 'epb_labels.csv'
DEFAULT_IMAGES_DIR = CODE_ROOT / 'data' / 'all-images'
DEFAULT_QUEUE_CSV = CODE_ROOT / 'data' / 'label_queue.csv'

QUEUE_REQUIRED = ['filename', 'path', 'has_epb', 'cloudy']
APPENDED_FIELDS = ['filename', 'has_epb', 'cloudy', 'site', 'datetime', 'filter']
# Prefixo do nome do arquivo -> valor da coluna `filter` (igual ao já usado no CSV).
FILTERS = {'O6': '630.0nm'}
FRAME_RE = re.compile(
    r'^(?P<filter>[A-Za-z0-9]+)_(?P<site>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})\.png$')
VALID_VERDICTS = {('1', '0'), ('0', '0'), ('', '1')}

NEXT_COMMANDS = [
    'python util/indice_kp_dst.py      # kp, dst, bz_gsm, f107 (dados OMNI)',
    'python util/moon_illumination.py  # moon_illumination, moon_alt',
]


class MergeError(Exception):
    """Fila inválida; `rejected` lista (linha, filename, motivo) de cada linha rejeitada."""

    def __init__(self, rejected: List[tuple]):
        self.rejected = rejected
        lines = [f"  linha {n}: {fn or '(sem nome)'}: {why}" for n, fn, why in rejected]
        super().__init__(f"{len(rejected)} linha(s) rejeitada(s); nada foi modificado:\n"
                         + '\n'.join(lines))


@dataclass
class MergeResult:
    appended: List[Dict[str, str]] = field(default_factory=list)  # linhas acrescentadas ao CSV
    skipped_unlabeled: int = 0                                    # linhas da fila sem rótulo
    backup: Optional[Path] = None
    copied: List[str] = field(default_factory=list)               # PNGs copiadas
    kept_existing: List[str] = field(default_factory=list)        # PNGs já presentes (não sobrescritas)


# ---------------------------------------------------------------------------
# Funções puras
# ---------------------------------------------------------------------------

def parse_frame_name(filename: str) -> Optional[Dict[str, str]]:
    """'O6_CA_20251023_001620.png' -> {'site': 'CA', 'datetime': '2025-10-23 00:16:20',
    'filter': '630.0nm'}. None se o nome estiver fora do padrão ou o filtro for desconhecido."""
    m = FRAME_RE.match(filename)
    if not m or m['filter'] not in FILTERS:
        return None
    try:
        t = datetime.strptime(m['date'] + m['time'], '%Y%m%d%H%M%S')
    except ValueError:
        return None
    return {'site': m['site'], 'datetime': f"{t:%Y-%m-%d %H:%M:%S}",
            'filter': FILTERS[m['filter']]}


def is_unlabeled(row: Dict[str, str]) -> bool:
    return _cell(row, 'has_epb') == '' and _cell(row, 'cloudy') == ''


def row_problems(row: Dict[str, str], known: set) -> List[str]:
    """Motivos de rejeição de uma linha rotulada (lista vazia = válida).
    `known` = nomes já no CSV ou já vistos antes nesta fila."""
    probs = []
    has_epb, cloudy = _cell(row, 'has_epb'), _cell(row, 'cloudy')
    fn = _cell(row, 'filename')
    if has_epb not in ('0', '1', ''):
        probs.append(f"has_epb={has_epb!r} fora de {{0, 1, vazio}}")
    if cloudy not in ('0', '1', ''):
        probs.append(f"cloudy={cloudy!r} fora de {{0, 1, vazio}}")
    if not probs and (has_epb, cloudy) not in VALID_VERDICTS:
        if cloudy == '1':
            probs.append('cloudy=1 com has_epb preenchido (Indeterminado exige has_epb vazio)')
        elif cloudy == '0':
            probs.append('cloudy=0 com has_epb vazio')
        else:
            probs.append('has_epb preenchido com cloudy vazio')
    if not fn:
        probs.append('filename vazio')
    elif parse_frame_name(fn) is None:
        probs.append('nome fora do padrão O6_SITE_YYYYMMDD_HHMMSS.png')
    if fn and fn in known:
        probs.append('filename já está no CSV de rótulos ou repetido na fila')
    path = _cell(row, 'path')
    if not path:
        probs.append('path vazio')
    elif not Path(path).is_file():
        probs.append(f"PNG não encontrada: {path}")
    return probs


def _cell(row: Dict[str, str], key: str) -> str:
    v = row.get(key)
    return '' if v is None else str(v).strip()


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------

def _read_queue(queue_csv: Path) -> List[Dict[str, str]]:
    with open(queue_csv, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = [c for c in QUEUE_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{queue_csv}: colunas ausentes: {', '.join(missing)}")
        return list(reader)


def _read_labels(labels_csv: Path):
    """(texto bruto, cabeçalho, nomes existentes, terminador de linha)."""
    raw = Path(labels_csv).read_bytes().decode('utf-8')
    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise ValueError(f"{labels_csv}: arquivo vazio (sem cabeçalho)")
    header = rows[0]
    missing = [c for c in APPENDED_FIELDS if c not in header]
    if missing:
        raise ValueError(f"{labels_csv}: colunas ausentes: {', '.join(missing)}")
    i = header.index('filename')
    existing = {r[i] for r in rows[1:] if len(r) > i}
    eol = '\r\n' if '\r\n' in raw.split('\n', 1)[0] + '\n' else '\n'
    return raw, header, existing, eol


def _backup_path(labels_csv: Path, backup_dir: Optional[Path], now: datetime) -> Path:
    d = Path(backup_dir) if backup_dir is not None else labels_csv.parent
    stem = f"{labels_csv.stem}.{now:%Y%m%d-%H%M%S}"
    p, k = d / f"{stem}.bak.csv", 1
    while p.exists():  # duas execuções no mesmo segundo
        p, k = d / f"{stem}-{k}.bak.csv", k + 1
    return p


def merge(queue_csv, labels_csv=DEFAULT_LABELS_CSV, images_dir=DEFAULT_IMAGES_DIR,
          backup_dir=None, now: Optional[datetime] = None) -> MergeResult:
    """Incorpora a fila em `labels_csv`. Levanta MergeError (sem tocar em nada) se
    alguma linha rotulada for inválida. Sem linhas rotuladas, também não altera nada."""
    queue_csv, labels_csv, images_dir = Path(queue_csv), Path(labels_csv), Path(images_dir)
    queue = _read_queue(queue_csv)
    raw, header, existing, eol = _read_labels(labels_csv)

    result = MergeResult()
    labeled, rejected, known = [], [], set(existing)
    for n, row in enumerate(queue, start=2):  # linha 1 = cabeçalho
        if is_unlabeled(row):
            result.skipped_unlabeled += 1
            continue
        probs = row_problems(row, known)
        fn = _cell(row, 'filename')
        if fn:
            known.add(fn)
        if probs:
            rejected.append((n, fn, '; '.join(probs)))
        else:
            labeled.append(row)
    if rejected:
        raise MergeError(rejected)
    if not labeled:
        return result

    images_dir.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now()
    result.backup = _backup_path(labels_csv, backup_dir, now)
    result.backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_csv, result.backup)

    new_rows = []
    for row in labeled:
        fn = _cell(row, 'filename')
        rec = {'filename': fn, 'has_epb': _cell(row, 'has_epb'),
               'cloudy': _cell(row, 'cloudy'), **parse_frame_name(fn)}
        new_rows.append(rec)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, restval='', lineterminator=eol)
    w.writerows(new_rows)
    prefix = '' if raw == '' or raw.endswith('\n') else eol
    with open(labels_csv, 'a', newline='', encoding='utf-8') as f:
        f.write(prefix + buf.getvalue())
    result.appended = new_rows

    for row in labeled:
        fn = _cell(row, 'filename')
        dst = images_dir / fn
        if dst.exists():
            result.kept_existing.append(fn)
        else:
            shutil.copy2(_cell(row, 'path'), dst)
            result.copied.append(fn)
    return result


def print_report(result: MergeResult, labels_csv: Path, images_dir: Path) -> None:
    print(f"Linhas sem rótulo ignoradas: {result.skipped_unlabeled}")
    if not result.appended:
        print("Nenhuma linha rotulada na fila; nada foi modificado.")
        return
    print(f"Backup: {result.backup}")
    print(f"{len(result.appended)} linha(s) acrescentada(s) em {labels_csv}")
    print(f"{len(result.copied)} PNG(s) copiada(s) para {images_dir}")
    if result.kept_existing:
        print(f"{len(result.kept_existing)} PNG(s) já existiam e não foram sobrescritas:")
        for fn in result.kept_existing:
            print(f"  {fn}")
    print("\nPreencha os metadados restantes (a partir de code/):")
    for cmd in NEXT_COMMANDS:
        print(f"  {cmd}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('queue_csv', type=Path, nargs='?', default=DEFAULT_QUEUE_CSV,
                   help='fila preenchida (padrão: data/label_queue.csv)')
    p.add_argument('--labels-csv', type=Path, default=DEFAULT_LABELS_CSV)
    p.add_argument('--images-dir', type=Path, default=DEFAULT_IMAGES_DIR)
    p.add_argument('--backup-dir', type=Path, default=None,
                   help='pasta do backup (padrão: a mesma do CSV de rótulos)')
    args = p.parse_args(argv)
    try:
        result = merge(args.queue_csv, args.labels_csv, args.images_dir, args.backup_dir)
    except MergeError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    print_report(result, args.labels_csv, args.images_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

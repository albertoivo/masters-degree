"""Fila de rotulagem a partir das noites já baixadas em util/*/processed/.

Lista os quadros que ainda não estão em data/epb_labels.csv, escolhe por noite
quadros igualmente espaçados no tempo (primeiro e último sempre incluídos) e grava:

    data/label_queue.csv               filename, night, site, path, has_epb, cloudy, notes
    data/label_queue_sheets/NIGHT.png  prancha com os quadros da fila em ordem temporal

Preencha has_epb / cloudy / notes seguindo data/LABELING_PROTOCOL.md e depois rode
`python util/merge_labels.py data/label_queue.csv`.

Uso (a partir de code/):
    python util/label_queue.py                    # 20 quadros por noite, STR excluída
    python util/label_queue.py --per-night 15
    python util/label_queue.py --all              # todos os quadros não rotulados
    python util/label_queue.py --exclude-sites STR CP
    python util/label_queue.py --force            # sobrescreve fila já preenchida
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UTIL_DIR = CODE_ROOT / 'util'
DEFAULT_LABELS_CSV = CODE_ROOT / 'data' / 'epb_labels.csv'
DEFAULT_QUEUE_CSV = CODE_ROOT / 'data' / 'label_queue.csv'
DEFAULT_SHEETS_DIR = CODE_ROOT / 'data' / 'label_queue_sheets'

QUEUE_COLUMNS = ['filename', 'night', 'site', 'path', 'has_epb', 'cloudy', 'notes']
LABEL_FIELDS = ['has_epb', 'cloudy', 'notes']
DEFAULT_EXCLUDE_SITES = ['STR']
DEFAULT_PER_NIGHT = 20

FRAME_RE = re.compile(r'^O6_(?P<site>[A-Za-z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})\.png$')


@dataclass(frozen=True)
class Frame:
    filename: str
    path: Path
    site: str
    night: str
    time: datetime


# ---------------------------------------------------------------------------
# Funções puras
# ---------------------------------------------------------------------------

def parse_frame_name(filename: str) -> Optional[tuple]:
    """'O6_CA_20251023_001620.png' -> ('CA', datetime). None se fora do padrão."""
    m = FRAME_RE.match(filename)
    if not m:
        return None
    try:
        t = datetime.strptime(m['date'] + m['time'], '%Y%m%d%H%M%S')
    except ValueError:
        return None
    return m['site'], t


def night_id(site: str, t: datetime) -> str:
    """Noite = SITE_YYYYMMDD da data de início; antes de 12:00 UT é a noite anterior."""
    start = t - timedelta(days=1) if t.hour < 12 else t
    return f"{site}_{start:%Y%m%d}"


def select_evenly_spaced(times: Sequence[datetime], k: int) -> List[int]:
    """Índices (ordenados) de até k quadros igualmente espaçados no tempo.

    `times` deve estar em ordem crescente. Com k >= 2 e len(times) >= 2, o primeiro e o
    último quadro estão sempre incluídos. Para cada instante-alvo interior escolhe-se o
    quadro ainda não escolhido mais próximo, então o resultado tem exatamente
    min(k, len(times)) índices distintos.
    """
    n = len(times)
    if k <= 0 or n == 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    secs = [(t - times[0]).total_seconds() for t in times]
    span = secs[-1]
    chosen = {0, n - 1}
    for j in range(1, k - 1):
        target = span * j / (k - 1)
        best = min((i for i in range(n) if i not in chosen),
                   key=lambda i: (abs(secs[i] - target), i))
        chosen.add(best)
    return sorted(chosen)


def build_queue(frames: Iterable[Frame], labeled: Iterable[str],
                exclude_sites: Iterable[str] = DEFAULT_EXCLUDE_SITES,
                per_night: Optional[int] = DEFAULT_PER_NIGHT) -> tuple:
    """Monta a fila. per_night=None equivale a --all.

    Retorna (queue: List[Frame] ordenada por noite e tempo,
             summary: DataFrame night, site, on_disk, labeled, queued).
    Noites de estações excluídas não aparecem em nenhum dos dois.
    """
    labeled = set(labeled)
    excluded = {s.upper() for s in exclude_sites}
    by_night: Dict[str, List[Frame]] = {}
    for f in frames:
        if f.site.upper() in excluded:
            continue
        by_night.setdefault(f.night, []).append(f)

    queue: List[Frame] = []
    rows = []
    for night in sorted(by_night):
        night_frames = sorted({f.filename: f for f in by_night[night]}.values(),
                              key=lambda f: (f.time, f.filename))
        unlabeled = [f for f in night_frames if f.filename not in labeled]
        if per_night is None:
            picked = unlabeled
        else:
            picked = [unlabeled[i] for i in
                      select_evenly_spaced([f.time for f in unlabeled], per_night)]
        queue.extend(picked)
        rows.append({'night': night, 'site': night_frames[0].site,
                     'on_disk': len(night_frames),
                     'labeled': len(night_frames) - len(unlabeled),
                     'queued': len(picked)})
    summary = pd.DataFrame(rows, columns=['night', 'site', 'on_disk', 'labeled', 'queued'])
    return queue, summary


def queue_dataframe(queue: Sequence[Frame]) -> pd.DataFrame:
    return pd.DataFrame([{'filename': f.filename, 'night': f.night, 'site': f.site,
                          'path': str(f.path), 'has_epb': '', 'cloudy': '', 'notes': ''}
                         for f in queue], columns=QUEUE_COLUMNS)


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------

def scan_frames(util_dir: Path) -> List[Frame]:
    """Lê util_dir/*/processed/*.png. A noite vem do nome do arquivo (regra das 12:00 UT);
    arquivos fora do padrão EMBRACE são ignorados."""
    frames = []
    for path in sorted(Path(util_dir).glob('*/processed/*.png')):
        parsed = parse_frame_name(path.name)
        if parsed is None:
            continue
        site, t = parsed
        frames.append(Frame(path.name, path.resolve(), site, night_id(site, t), t))
    return frames


def load_labeled_filenames(labels_csv: Path) -> set:
    if not Path(labels_csv).exists():
        return set()
    df = pd.read_csv(labels_csv, usecols=['filename'], dtype=str)
    return set(df['filename'].dropna().str.strip())


def queue_has_filled_labels(queue_csv: Path) -> bool:
    """True se a fila existe e alguma célula de has_epb, cloudy ou notes está preenchida."""
    if not Path(queue_csv).exists():
        return False
    df = pd.read_csv(queue_csv, dtype=str, keep_default_na=False)
    for col in LABEL_FIELDS:
        if col in df.columns and (df[col].str.strip() != '').any():
            return True
    return False


def write_contact_sheet(frames: Sequence[Frame], out_path: Path,
                        thumb: int = 200, cols: int = 5) -> None:
    """Prancha com miniaturas em ordem temporal e o horário (UT) sob cada uma."""
    from PIL import Image, ImageDraw

    frames = sorted(frames, key=lambda f: f.time)
    if not frames:
        return
    label_h = 16
    rows = (len(frames) + cols - 1) // cols
    ncols = min(cols, len(frames))
    sheet = Image.new('RGB', (ncols * thumb, rows * (thumb + label_h)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
        with Image.open(f.path) as im:
            im = im.convert('RGB')
            im.thumbnail((thumb, thumb))
            sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw.text((x + 4, y + thumb + 2), f"{f.time:%Y-%m-%d %H:%M:%S} UT", fill='black')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def run(util_dir: Path = DEFAULT_UTIL_DIR, labels_csv: Path = DEFAULT_LABELS_CSV,
        queue_csv: Path = DEFAULT_QUEUE_CSV, sheets_dir: Path = DEFAULT_SHEETS_DIR,
        exclude_sites: Iterable[str] = DEFAULT_EXCLUDE_SITES,
        per_night: Optional[int] = DEFAULT_PER_NIGHT, force: bool = False,
        sheets: bool = True) -> pd.DataFrame:
    """Gera fila e pranchas; retorna o resumo por noite.
    Levanta FileExistsError se a fila já tem rótulos e force=False."""
    queue_csv, sheets_dir = Path(queue_csv), Path(sheets_dir)
    if not force and queue_has_filled_labels(queue_csv):
        raise FileExistsError(
            f"{queue_csv} já contém rótulos preenchidos. Rode util/merge_labels.py "
            f"antes, ou use --force para sobrescrever (os rótulos da fila serão perdidos).")

    queue, summary = build_queue(scan_frames(util_dir), load_labeled_filenames(labels_csv),
                                 exclude_sites, per_night)
    queue_csv.parent.mkdir(parents=True, exist_ok=True)
    queue_dataframe(queue).to_csv(queue_csv, index=False)

    if sheets:
        by_night: Dict[str, List[Frame]] = {}
        for f in queue:
            by_night.setdefault(f.night, []).append(f)
        for night, frames in by_night.items():
            write_contact_sheet(frames, sheets_dir / f"{night}.png")
    return summary


def print_summary(summary: pd.DataFrame, queue_csv: Path) -> None:
    if summary.empty:
        print("Nenhum quadro encontrado.")
    else:
        total = summary[['on_disk', 'labeled', 'queued']].sum()
        table = pd.concat([summary.drop(columns='site'),
                           pd.DataFrame([{'night': 'TOTAL', **total.to_dict()}])],
                          ignore_index=True)
        table.columns = ['noite', 'no disco', 'rotulados', 'na fila']
        print(table.to_string(index=False))
    print(f"\nFila gravada em {queue_csv}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument('--per-night', type=int, default=DEFAULT_PER_NIGHT,
                   help=f'quadros por noite, igualmente espaçados (padrão {DEFAULT_PER_NIGHT})')
    g.add_argument('--all', action='store_true', help='enfileira todo quadro não rotulado')
    p.add_argument('--exclude-sites', nargs='*', default=DEFAULT_EXCLUDE_SITES,
                   help='estações excluídas (padrão: STR); sem valores = nenhuma')
    p.add_argument('--force', action='store_true', help='sobrescreve fila já preenchida')
    p.add_argument('--no-sheets', action='store_true', help='não gera as pranchas')
    p.add_argument('--util-dir', type=Path, default=DEFAULT_UTIL_DIR)
    p.add_argument('--labels-csv', type=Path, default=DEFAULT_LABELS_CSV)
    p.add_argument('--queue-csv', type=Path, default=DEFAULT_QUEUE_CSV)
    p.add_argument('--sheets-dir', type=Path, default=DEFAULT_SHEETS_DIR)
    args = p.parse_args(argv)
    if not args.all and args.per_night < 1:
        p.error('--per-night deve ser >= 1')

    try:
        summary = run(args.util_dir, args.labels_csv, args.queue_csv, args.sheets_dir,
                      args.exclude_sites, None if args.all else args.per_night,
                      args.force, sheets=not args.no_sheets)
    except FileExistsError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    print_summary(summary, args.queue_csv)
    return 0


if __name__ == '__main__':
    sys.exit(main())

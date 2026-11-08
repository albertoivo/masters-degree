#!/usr/bin/env python
"""Re-rotulagem cega e concordância intra-avaliador (kappa de Cohen).

Uso (a partir de code/):
    python util/label_agreement.py sample --n 50 --seed 42
        -> data/relabel_sample.csv  (filename, night, has_epb, cloudy, notes; rótulos vazios)
    python util/label_agreement.py score data/relabel_sample.csv
        -> tabela de confusão, concordância bruta e kappa (geral e por noite)
        -> data/relabel_disagreements.csv

Veredicto (ver data/LABELING_PROTOCOL.md):
    cloudy=1, has_epb vazio -> Indeterminado
    cloudy=0, has_epb=1     -> EPB
    cloudy=0, has_epb=0     -> Sem EPB
    ambos vazios            -> ainda não rotulado (ignorado no score)
No arquivo re-rotulado, `cloudy` vazio com `has_epb` preenchido é lido como cloudy=0.

Noite: SITE_YYYYMMDD da data de início; quadros antes de 12:00 UT pertencem à noite
que começou no dia anterior (mesma regra de extract_night_groups).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = CODE_DIR / 'data' / 'epb_labels.csv'
DEFAULT_SAMPLE = CODE_DIR / 'data' / 'relabel_sample.csv'
DEFAULT_DISAGREEMENTS = CODE_DIR / 'data' / 'relabel_disagreements.csv'

VERDICTS = ('EPB', 'Sem EPB', 'Indeterminado')
SAMPLE_COLUMNS = ['filename', 'night', 'has_epb', 'cloudy', 'notes']
MIN_PER_NIGHT = 2

_FNAME_RE = re.compile(r'^O6_([A-Za-z0-9]+)_(\d{8})_(\d{2})(\d{2})(\d{2})')


# --------------------------------------------------------------------------- noites
def night_of(filename: str) -> str:
    """'O6_CA_20251023_011500.png' -> 'CA_20251022' (antes de 12:00 UT = noite anterior)."""
    m = _FNAME_RE.match(Path(str(filename)).name)
    if not m:
        raise ValueError(f"Nome de arquivo fora do padrão O6_SITE_YYYYMMDD_HHMMSS: '{filename}'")
    site, day, hour = m.group(1), datetime.strptime(m.group(2), '%Y%m%d'), int(m.group(3))
    if hour < 12:
        day -= timedelta(days=1)
    return f"{site}_{day:%Y%m%d}"


# --------------------------------------------------------------------------- veredicto
def _cell(value) -> Optional[str]:
    """Normaliza uma célula de rótulo: None se vazia, senão '0'/'1' (ou o texto original)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    if s == '' or s.lower() == 'nan':
        return None
    try:
        f = float(s)
        if f in (0.0, 1.0):
            return str(int(f))
    except ValueError:
        pass
    return s


def verdict(has_epb, cloudy, lenient_cloudy: bool = False) -> Optional[str]:
    """Veredicto de três classes a partir de (has_epb, cloudy); None se não rotulado.

    lenient_cloudy=True aceita cloudy vazio com has_epb preenchido (lido como cloudy=0).
    Levanta ValueError para combinações inválidas.
    """
    h, c = _cell(has_epb), _cell(cloudy)
    if h is None and c is None:
        return None
    if c is None and lenient_cloudy:
        c = '0'
    if c == '1' and h is None:
        return 'Indeterminado'
    if c == '0' and h == '1':
        return 'EPB'
    if c == '0' and h == '0':
        return 'Sem EPB'
    raise ValueError(f"combinação inválida has_epb={has_epb!r}, cloudy={cloudy!r}")


# --------------------------------------------------------------------------- kappa
def confusion_matrix(a: Sequence[Hashable], b: Sequence[Hashable],
                     labels: Optional[Sequence[Hashable]] = None) -> pd.DataFrame:
    """Contagens: linhas = a (original), colunas = b (re-rótulo)."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("vetores de rótulos com tamanhos diferentes")
    if labels is None:
        labels = sorted(set(a) | set(b), key=str)
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return pd.DataFrame(m, index=labels, columns=labels)


def raw_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("vetores de rótulos com tamanhos diferentes")
    if not a:
        return float('nan')
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Kappa de Cohen entre dois vetores de rótulos (função pura).

    kappa = (po - pe) / (1 - pe). Retorna NaN se os vetores forem vazios ou se
    pe = 1 (uma única classe nos dois vetores: kappa indefinido).
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("vetores de rótulos com tamanhos diferentes")
    n = len(a)
    if n == 0:
        return float('nan')
    po = sum(x == y for x, y in zip(a, b)) / n
    ca: Dict[Hashable, int] = {}
    cb: Dict[Hashable, int] = {}
    for x in a:
        ca[x] = ca.get(x, 0) + 1
    for y in b:
        cb[y] = cb.get(y, 0) + 1
    # soma em ordem canônica para o resultado não depender da ordem dos argumentos
    keys = sorted(set(ca) | set(cb), key=repr)
    pe = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys) / (n * n)
    if math.isclose(pe, 1.0):
        return float('nan')
    return (po - pe) / (1.0 - pe)


# --------------------------------------------------------------------------- sample
def _allocate(sizes: Dict[str, int], n: int) -> Dict[str, int]:
    """Quotas por noite: proporcional ao tamanho, mínimo 2 (ou o que houver), soma = n
    quando possível. Se n < 2 * noites, cada noite recebe o mínimo (total > n)."""
    quota = {k: min(MIN_PER_NIGHT, s) for k, s in sizes.items()}
    total_avail = sum(sizes.values())
    target = min(n, total_avail)
    remaining = target - sum(quota.values())
    while remaining > 0:
        room = {k: sizes[k] - quota[k] for k in sizes if sizes[k] > quota[k]}
        if not room:
            break
        weight = sum(sizes[k] for k in room)
        ideal = {k: remaining * sizes[k] / weight for k in room}
        add = {k: min(room[k], int(math.floor(ideal[k]))) for k in room}
        left = remaining - sum(add.values())
        # maiores restos (empate: noite maior, depois nome) recebem as unidades que sobram
        order = sorted(room, key=lambda k: (-(ideal[k] - math.floor(ideal[k])), -sizes[k], k))
        for k in order:
            if left <= 0:
                break
            if add[k] < room[k]:
                add[k] += 1
                left -= 1
        if sum(add.values()) == 0:
            break
        for k, v in add.items():
            quota[k] += v
        remaining = target - sum(quota.values())
    return quota


def sample_frames(labels: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sorteia quadros rotulados estratificados por noite; rótulos em branco, ordem embaralhada."""
    if n <= 0:
        raise ValueError("--n deve ser positivo")
    df = labels.copy()
    df['_verdict'] = [verdict(h, c) for h, c in zip(df['has_epb'], df['cloudy'])]
    df = df[df['_verdict'].notna()].drop_duplicates('filename')
    if df.empty:
        raise ValueError("nenhum quadro rotulado no CSV")
    df['night'] = [night_of(f) for f in df['filename']]
    sizes = df.groupby('night').size().to_dict()
    quota = _allocate(sizes, n)
    if sum(quota.values()) > n:
        print(f"Aviso: n={n} é menor que {MIN_PER_NIGHT} x {len(sizes)} noites; "
              f"sorteando {sum(quota.values())} quadros (mínimo por noite).", file=sys.stderr)
    rng = np.random.default_rng(seed)
    picked: List[pd.DataFrame] = []
    for night in sorted(quota):
        g = df[df['night'] == night].sort_values('filename')
        idx = rng.choice(len(g), size=quota[night], replace=False)
        picked.append(g.iloc[np.sort(idx)])
    out = pd.concat(picked, ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    return pd.DataFrame({'filename': out['filename'], 'night': out['night'],
                         'has_epb': '', 'cloudy': '', 'notes': ''}, columns=SAMPLE_COLUMNS)


def _has_filled_labels(path: Path) -> bool:
    try:
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return False
    cols = [c for c in ('has_epb', 'cloudy') if c in old.columns]
    return any(old[c].str.strip().ne('').any() for c in cols)


# --------------------------------------------------------------------------- score
def score_relabels(relabels: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Compara re-rótulos com o CSV de rótulos. Retorna dict com
    n_ignored, n_scored, confusion, agreement, kappa, per_night (DataFrame), disagreements."""
    orig = {}
    for f, h, c in zip(labels['filename'], labels['has_epb'], labels['cloudy']):
        orig[str(f)] = verdict(h, c)

    problems, rows, n_ignored = [], [], 0
    notes_col = relabels['notes'] if 'notes' in relabels.columns else [''] * len(relabels)
    for i, (f, h, c, note) in enumerate(zip(relabels['filename'], relabels['has_epb'],
                                            relabels['cloudy'], notes_col), start=2):
        f = str(f).strip()
        try:
            new = verdict(h, c, lenient_cloudy=True)
        except ValueError as e:
            problems.append(f"linha {i} ({f}): {e}")
            continue
        if new is None:
            n_ignored += 1
            continue
        if orig.get(f) is None:
            problems.append(f"linha {i} ({f}): sem rótulo em epb_labels.csv")
            continue
        rows.append({'filename': f, 'night': night_of(f), 'original': orig[f],
                     'relabel': new, 'notes': '' if _cell(note) is None else str(note)})
    if problems:
        raise ValueError("arquivo de re-rotulagem inválido:\n  " + "\n  ".join(problems))

    scored = pd.DataFrame(rows, columns=['filename', 'night', 'original', 'relabel', 'notes'])
    a, b = list(scored['original']), list(scored['relabel'])
    per_night = []
    for night, g in scored.groupby('night', sort=True):
        per_night.append({'night': night, 'n': len(g),
                          'agreement': raw_agreement(g['original'], g['relabel']),
                          'kappa': cohen_kappa(list(g['original']), list(g['relabel']))})
    return {
        'n_ignored': n_ignored,
        'n_scored': len(scored),
        'confusion': confusion_matrix(a, b, labels=VERDICTS),
        'agreement': raw_agreement(a, b),
        'kappa': cohen_kappa(a, b),
        'per_night': pd.DataFrame(per_night, columns=['night', 'n', 'agreement', 'kappa']),
        'disagreements': scored[scored['original'] != scored['relabel']].reset_index(drop=True),
    }


def _fmt(x: float) -> str:
    return 'n/d' if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


# --------------------------------------------------------------------------- CLI
def cmd_sample(args) -> int:
    out = Path(args.out)
    if out.exists() and _has_filled_labels(out) and not args.force:
        print(f"Erro: {out} já tem rótulos preenchidos; use --force para sobrescrever.",
              file=sys.stderr)
        return 1
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    sample = sample_frames(labels, args.n, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    counts = sample.groupby('night').size()
    print(f"{len(sample)} quadros de {len(counts)} noites -> {out}")
    for night, k in counts.items():
        print(f"  {night}: {k}")
    print("Preencha has_epb / cloudy / notes sem consultar epb_labels.csv e rode:")
    print(f"  python util/label_agreement.py score {out}")
    return 0


def cmd_score(args) -> int:
    relabels = pd.read_csv(args.file, dtype=str, keep_default_na=False)
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    try:
        res = score_relabels(relabels, labels)
    except ValueError as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1
    print(f"Linhas ignoradas (rótulos vazios): {res['n_ignored']}")
    print(f"Quadros comparados: {res['n_scored']}")
    if res['n_scored'] == 0:
        print("Nada a comparar.")
        return 0
    conf = res['confusion'].copy()
    conf.index.name, conf.columns.name = 'original', 're-rótulo'
    print("\nTabela de confusão (linhas = epb_labels.csv, colunas = re-rótulo):")
    print(conf.to_string())
    print(f"\nConcordância bruta: {_fmt(res['agreement'])}")
    print(f"Kappa de Cohen:     {_fmt(res['kappa'])}")
    print("\nPor noite:")
    pn = res['per_night'].copy()
    pn['agreement'] = pn['agreement'].map(_fmt)
    pn['kappa'] = pn['kappa'].map(_fmt)
    print(pn.to_string(index=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res['disagreements'].to_csv(out, index=False)
    print(f"\n{len(res['disagreements'])} discordâncias -> {out}")
    if not math.isnan(res['kappa']) and res['kappa'] < 0.6:
        print("Kappa < 0.6: revise as definições do protocolo antes de rotular mais.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample', help='sorteia quadros para re-rotulagem cega')
    s.add_argument('--n', type=int, default=50)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--labels', default=str(DEFAULT_LABELS))
    s.add_argument('--out', default=str(DEFAULT_SAMPLE))
    s.add_argument('--force', action='store_true',
                   help='sobrescreve uma amostra que já tem rótulos preenchidos')
    s.set_defaults(func=cmd_sample)
    c = sub.add_parser('score', help='compara re-rótulos com epb_labels.csv')
    c.add_argument('file')
    c.add_argument('--labels', default=str(DEFAULT_LABELS))
    c.add_argument('--out', default=str(DEFAULT_DISAGREEMENTS))
    c.set_defaults(func=cmd_score)
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

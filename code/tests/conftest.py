"""Fixtures compartilhadas: dataset sintético em tmp_path (nunca toca data/ ou util/ reais).

Estrutura criada por `synthetic_dataset`:

    tmp_path/
      data/epb_labels.csv          3 noites rotuladas (EPB, sem EPB, nublada)
      data/all-images/*.png        uma PNG 512x512 'L' para cada linha do CSV
      util/NIGHT/processed/*.png   quadros rotulados + não rotulados de cada noite
      util/STR_20250801/processed  noite STR (só não rotulados; excluída por padrão)

Regra de noite: quadros antes de 12:00 UT pertencem à noite que começou no dia anterior,
por isso cada noite tem quadros depois da meia-noite com data do dia seguinte.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from PIL import Image

LABEL_COLUMNS = ['filename', 'has_epb', 'cloudy', 'site', 'datetime', 'filter',
                 'kp', 'dst', 'moon_illumination', 'moon_alt', 'bz_gsm', 'f107']
IMG_SIZE = 512

# night_id -> (site, start, labeled (has_epb, cloudy), labeled offsets min, unlabeled offsets min)
# offsets em minutos a partir de `start`; has_epb=None significa célula vazia.
NIGHTS = {
    'CA_20251022': ('CA', datetime(2025, 10, 22, 22, 0, 0), (1, 0),
                    [0, 60, 180], [10, 20, 30, 120, 240, 300]),
    'BJL_20250722': ('BJL', datetime(2025, 7, 22, 22, 0, 0), (0, 0),
                     [0, 60, 150], [15, 45, 90, 200, 260]),
    'CA_20260624': ('CA', datetime(2026, 6, 24, 23, 0, 0), (None, 1),
                    [0, 180, 240], [30, 90, 300]),
}
STR_NIGHT = ('STR_20250801', 'STR', datetime(2025, 8, 1, 23, 0, 0), [0, 30, 60])


def frame_name(site: str, t: datetime) -> str:
    """Nome no padrão EMBRACE: O6_SITE_YYYYMMDD_HHMMSS.png"""
    return f"O6_{site}_{t:%Y%m%d_%H%M%S}.png"


def write_png(path: Path, seed: int) -> None:
    """PNG 512x512 em tons de cinza, determinística: disco brilhante + ruído leve."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    r = np.hypot(yy - IMG_SIZE / 2, xx - IMG_SIZE / 2)
    img = np.where(r < 0.44 * IMG_SIZE, 120, 10).astype(np.int16)
    img += rng.integers(-8, 9, size=img.shape, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='L').save(path)


@dataclass
class SyntheticDataset:
    root: Path
    data_dir: Path
    images_dir: Path
    labels_csv: Path
    util_dir: Path
    test_nights_path: Path                      # data/test_nights.txt (não criado)
    labeled: Dict[str, List[str]] = field(default_factory=dict)    # night -> filenames no CSV
    unlabeled: Dict[str, List[str]] = field(default_factory=dict)  # night -> filenames só em processed/
    verdict: Dict[str, str] = field(default_factory=dict)          # night -> EPB / Sem EPB / Indeterminado

    def labels(self) -> pd.DataFrame:
        return pd.read_csv(self.labels_csv)

    def processed_dir(self, night: str) -> Path:
        return self.util_dir / night / 'processed'


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> SyntheticDataset:
    data_dir = tmp_path / 'data'
    images_dir = data_dir / 'all-images'
    util_dir = tmp_path / 'util'
    images_dir.mkdir(parents=True)
    util_dir.mkdir()
    ds = SyntheticDataset(root=tmp_path, data_dir=data_dir, images_dir=images_dir,
                          labels_csv=data_dir / 'epb_labels.csv', util_dir=util_dir,
                          test_nights_path=data_dir / 'test_nights.txt')

    rows, seed = [], 0
    for night, (site, start, (has_epb, cloudy), lab_off, unlab_off) in NIGHTS.items():
        ds.verdict[night] = 'Indeterminado' if cloudy == 1 else ('EPB' if has_epb == 1 else 'Sem EPB')
        ds.labeled[night], ds.unlabeled[night] = [], []
        for off in sorted(lab_off + unlab_off):
            t = start + timedelta(minutes=off)
            name = frame_name(site, t)
            seed += 1
            write_png(ds.processed_dir(night) / name, seed)
            if off in lab_off:
                write_png(images_dir / name, seed)
                ds.labeled[night].append(name)
                rows.append({'filename': name,
                             'has_epb': '' if has_epb is None else has_epb,
                             'cloudy': cloudy, 'site': site,
                             'datetime': f"{t:%Y-%m-%d %H:%M:%S}", 'filter': '630.0nm',
                             'kp': 1.3, 'dst': 8, 'moon_illumination': 10.0, 'moon_alt': -5.0,
                             'bz_gsm': 0.1, 'f107': 130.0})
            else:
                ds.unlabeled[night].append(name)

    str_night, str_site, str_start, str_off = STR_NIGHT
    ds.unlabeled[str_night] = []
    for off in str_off:
        name = frame_name(str_site, str_start + timedelta(minutes=off))
        seed += 1
        write_png(ds.processed_dir(str_night) / name, seed)
        ds.unlabeled[str_night].append(name)

    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(ds.labels_csv, index=False)
    return ds

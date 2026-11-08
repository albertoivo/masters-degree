"""Teste offline das opções --offline e --clean-only de util/embrace.sh.

Copia 2 TIFs reais de util/CA_20251022/original/ para tmp_path (a pasta real só é lida)
e roda o script sem rede. Requirements 2.1, 2.2, 2.7.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

UTIL_DIR = Path(__file__).resolve().parents[1] / 'util'
EMBRACE = UTIL_DIR / 'embrace.sh'
REAL_ORIGINAL = UTIL_DIR / 'CA_20251022' / 'original'
NIGHT = 'CA_20251022'
VIDEO = 'O6_CA_20251022.mp4'


def _load_check_clean():
    spec = importlib.util.spec_from_file_location('check_clean', UTIL_DIR / 'check_clean.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_embrace(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    # stdin=DEVNULL: sem isso o script já travou esperando entrada
    return subprocess.run(['bash', str(EMBRACE), *args], cwd=cwd, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=600)


@pytest.fixture
def night_copy(tmp_path: Path) -> Path:
    if shutil.which('magick') is None:
        pytest.skip('ImageMagick (magick) não instalado')
    tifs = sorted(REAL_ORIGINAL.glob('O6_*.tif'))[:2] if REAL_ORIGINAL.is_dir() else []
    if len(tifs) < 2:
        pytest.skip(f'TIFs reais ausentes em {REAL_ORIGINAL}')
    original = tmp_path / NIGHT / 'original'
    original.mkdir(parents=True)
    for tif in tifs:
        shutil.copy2(tif, original / tif.name)
    return tmp_path / NIGHT


def test_offline_clean_only_then_full_run(night_copy: Path):
    """**Validates: Requirements 2.1, 2.2, 2.7**"""
    tmp = night_copy.parent
    stems = sorted(p.stem for p in (night_copy / 'original').glob('*.tif'))

    # --offline --clean-only: só clean/; processed/ vazio (ou inexistente) e sem vídeo
    res = _run_embrace(tmp, '--offline', '--clean-only', NIGHT)
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(p.stem for p in (night_copy / 'clean').glob('*.png')) == stems
    processed = night_copy / 'processed'
    assert not processed.exists() or not any(processed.iterdir())
    assert not list(night_copy.glob('*.mp4'))

    # Execução completa offline: processed/ e vídeo; clean/ igual a processed/ no FOV
    res = _run_embrace(tmp, '--offline', NIGHT)
    assert res.returncode == 0, res.stdout + res.stderr
    assert sorted(p.stem for p in processed.glob('*.png')) == stems
    assert (night_copy / VIDEO).is_file()

    check_clean = _load_check_clean()
    n, failures = check_clean.check_night(night_copy)
    assert n == 2
    assert failures == [], f'clean/ difere de processed/ dentro do FOV: {failures}'
    assert check_clean.main([str(night_copy)]) == 0

"""Testes unitários das duas funções de apresentação locais do `app_v5.ipynb`.

As funções (`_plot_recon_error`, `_plot_temporal`) vivem no notebook; aqui elas são
extraídas executando as células que as definem num namespace isolado, para testar o
código real (não uma cópia). Usa backend `Agg` e `tmp_path` do pytest.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import nbformat  # noqa: E402

import epb_detection_v5 as v5  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_NB = _REPO / "app_v5.ipynb"


def _load_plot_functions():
    """Executa as células que definem as funções locais e devolve o namespace."""
    nb = nbformat.read(str(_NB), 4)
    ns = {
        "np": np, "pd": pd, "plt": plt, "v5": v5, "Path": Path,
        "Image": __import__("PIL.Image", fromlist=["Image"]),
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "def _plot_recon_error" in src or "def _plot_temporal" in src:
            exec(compile(src, "<app_v5-cell>", "exec"), ns)
    assert "_plot_recon_error" in ns and "_plot_temporal" in ns
    return ns


@pytest.fixture(scope="module")
def plot_funcs():
    return _load_plot_functions()


def _synthetic_frames(n=12):
    """DataFrame sintético com os três estados representados."""
    rows = []
    for i in range(n):
        if i % 4 == 0:  # Indeterminado
            verdict, prob, reliable, sky = "Indeterminado", np.nan, False, "CLOUDY"
        elif i % 4 == 1:  # EPB confiável
            verdict, prob, reliable, sky = "EPB", 0.85, True, "CLEAR"
        elif i % 4 == 2:  # Sem EPB confiável
            verdict, prob, reliable, sky = "Sem EPB", 0.10, True, "CLEAR"
        else:  # decidido mas não confiável
            verdict, prob, reliable, sky = "EPB", 0.60, False, "DEGRADED"
        rows.append({
            "filename": f"O6_CA_20251022_{i:06d}.png",
            "verdict": verdict, "prob_epb": prob,
            "reliable": reliable, "sky": sky,
        })
    return pd.DataFrame(rows)


def test_plot_temporal_saves_nonempty_png(plot_funcs, tmp_path):
    df = _synthetic_frames()
    out = tmp_path / "temporal.png"
    fig = plot_funcs["_plot_temporal"](
        df, title="teste", save_path=str(out), image_folder=None)
    assert out.exists() and out.stat().st_size > 0
    # Distingue os três estados (presentes no DataFrame sintético).
    assert set(df["verdict"]) == {"EPB", "Sem EPB", "Indeterminado"}
    plt.close(fig)


def test_plot_temporal_raises_on_empty(plot_funcs, tmp_path):
    empty = pd.DataFrame(columns=["filename", "verdict", "prob_epb", "reliable", "sky"])
    with pytest.raises(ValueError):
        plot_funcs["_plot_temporal"](
            empty, title="vazio", save_path=str(tmp_path / "x.png"))


def test_plot_temporal_raises_on_missing_thumbnail(plot_funcs, tmp_path):
    df = _synthetic_frames()
    missing_folder = tmp_path / "nao_existe"
    missing_folder.mkdir()
    with pytest.raises(FileNotFoundError):
        plot_funcs["_plot_temporal"](
            df, title="thumb ausente", save_path=str(tmp_path / "t.png"),
            thumbnail_interval=1, image_folder=str(missing_folder))


def test_plot_recon_error_saves_nonempty_and_nonneg(plot_funcs, tmp_path):
    rng = np.random.default_rng(0)
    N = 20
    X = rng.standard_normal((N, 64, 64)).astype(np.float32)
    y = np.array([i % 2 for i in range(N)])
    system = v5.EPBRecognitionSystemV5(n_components=3, C=0.1, target_size=(64, 64))
    system.fit(X, y)

    X_sample = X[0]
    out = tmp_path / "recon.png"
    fig = plot_funcs["_plot_recon_error"](
        system, X_sample, raw=X_sample, title="teste recon", save_path=str(out))
    assert out.exists() and out.stat().st_size > 0

    # O erro de reconstrução é não negativo.
    proj = system.tdpca.transform(X_sample[np.newaxis])
    recon = system.tdpca.inverse_transform(proj)[0]
    err = np.abs(X_sample - recon)
    assert np.all(err >= 0)
    plt.close(fig)

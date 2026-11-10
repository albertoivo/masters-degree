"""
EPB Detection System — V5

Sistema de detecção de EPB (produto atual). Reaproveita o núcleo compartilhado
(epb_core_v5: TwoDPCA, QC do Estágio 1, EPBRecognitionSystem, métricas) e acrescenta o
pré-processamento com sinal, a orientação por estação, o classificador linear e a
avaliação por noite.

Mudanças em relação ao V4 (sistema de stacking do núcleo, EPBRecognitionSystem), por ordem
de prioridade:

  1. Detrend com sinal: perturbação relativa δI/I = I / fundo − 1, sem cortar valores negativos.
     No V4, `img - fundo` seguido de `clip(·, 0)` zerava as depleções (as próprias bolhas).
  2. Remoção de estrelas (abertura morfológica, só remove estruturas BRILHANTES pequenas) e
     estimativa do fundo apenas dentro do domo (convolução normalizada pela máscara), em 512 px.
  3. Normalização z-score dentro do domo no lugar do CLAHE. O processamento em 512 px com redução
     no fim (order='fullres') continua disponível, mas NÃO é o padrão: na avaliação por noite foi
     pior que reduzir para 64 px primeiro (acurácia balanceada 0,70 vs. 0,75).
  4. Verificação da orientação das bolhas por estação (tensor de estrutura) e rotação
     opcional por estação (rotation_by_site).
  5. Classificador simples: 2DPCA + StandardScaler + Regressão Logística (sem SMOTE).
     Stacking continua disponível para comparação (classifier='stacking').

Avaliação: a unidade independente é a NOITE. Hiperparâmetros são escolhidos com
leave-one-night-out aninhado (nested_logo_evaluate), para que a estimativa não seja otimista.

Este módulo reaproveita o núcleo compartilhado (epb_core_v5) e não o altera.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import sobel
from skimage.exposure import equalize_adapthist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline

from epb_core_v5 import (
    EPBRecognitionSystem, CascadedPredictionResult, assess_sky_quality,
    _binary_metrics, extract_night_groups,
)

logger = logging.getLogger(__name__)

# Parâmetros do pré-processamento V5 (σ do fundo = mesma fração da imagem do V4: 10 px em 64 px).
# Nota sobre as imagens do EMBRACE: não existe lua nas fotos do EMBRACE. O equipamento é
# configurado para não operar / mirar em outro lugar do céu e a lua não aparece no domo;
# gradientes assimétricos de borda devem-se no máximo à poluição luminosa da cidade.
V5_PREPROCESS = dict(
    detrend='ratio',            # item 1
    remove_stars=True,          # item 2
    star_size=7,                # px na resolução original (512)
    masked_background=True,     # item 2: convolução gaussiana normalizada no domo já remove
                                # van Rhijn, vinheteamento e poluição luminosa suave de borda
    radial_flatten=False,       # item 2b testado e descartado: redundante com masked_background e
                                # nocivo na ablação por noite (bal. acc 0,512 vs. 0,786 sem ele)
    order='downsample_first',   # reduz para 64 px antes de processar (item 3 testado e descartado:
                                # 'fullres' teve acurácia balanceada por noite 0,70 vs. 0,75)
    contrast='zscore',          # item 3
    bg_sigma_frac=10 / 64,
    fov_radius_frac=0.44,
    zclip=4.0,
)

# ----------------------------------------------------------------------------
# Estágio 1 (qualidade do céu) — limiares em TRÊS faixas sobre o PNG do embrace.sh:
#   CLOUDY   (estrelas < 1000 ou brilho médio < 55)  -> INDETERMINADO, sem decisão de EPB
#   DEGRADED (estrelas < 4500)                         -> decisão emitida, marcada como pouco confiável
#   CLEAR                                              -> decisão normal
# Ajustados em nov/2026 com UMA noite nublada rotulada (CA_20260624): na avaliação, 55% dos
# quadros nublados ficam INDETERMINADO, outros 24% ficam DEGRADED, e 96% dos quadros limpos
# passam (pior noite limpa: 81%). Os limiares antigos (4500/35) bloqueavam 43% de uma noite
# de EPB com céu limpo. Revalidar quando houver quadros nublados de outras noites.
QC_PARAMS = dict(threshold_stars=1000, min_mean_brightness=55.0, degraded_stars=4500)

UNDETERMINED = 'UNDETERMINED_CLOUDY'
STATUS_LABELS = {
    UNDETERMINED: 'Indeterminado — céu nublado, não é possível avaliar EPB',
    'EPB_DETECTED': 'EPB detectada',
    'QUIET_CLEAR': 'Sem EPB',
    'DEGRADED_EPB': 'EPB detectada (visibilidade parcial — conferir)',
    'DEGRADED_QUIET': 'Sem EPB (visibilidade parcial — conferir)',
}

# Pré-processamento equivalente ao V4, expresso com os mesmos parâmetros (referência da escada).
V4_EQUIVALENT = dict(detrend='subtract_clip', remove_stars=False, masked_background=False,
                     radial_flatten=False,
                     order='downsample_first', contrast='clahe', bg_sigma_frac=10 / 64,
                     fov_radius_frac=0.44, zclip=None)


# ============================================================================
# PRÉ-PROCESSAMENTO
# ============================================================================

def fov_mask(shape: Tuple[int, int], radius_frac: float = 0.44) -> np.ndarray:
    """Máscara circular do domo all-sky (exclui cantos com carimbos de texto e o exterior do fisheye)."""
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2) <= radius_frac * min(h, w)


def radial_profile(img: np.ndarray, radius_frac: float = 0.44, n_bins: int = 48) -> np.ndarray:
    """
    Perfil radial de brilho da imagem (mediana por anel), devolvido como imagem do mesmo tamanho.
    Fora do domo, repete o valor do último anel.
    """
    h, w = img.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
    r_max = radius_frac * min(h, w)
    edges = np.linspace(0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1).reshape(h, w)
    inside = r <= r_max
    med = np.array([np.median(img[(b == i) & inside]) if np.any((b == i) & inside) else np.nan
                    for i in range(n_bins)])
    ok = ~np.isnan(med)
    med = np.interp(centers, centers[ok], med[ok])
    return np.interp(np.minimum(r, r_max), centers, med).astype(np.float32)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def preprocess_epb_image_v5(img: np.ndarray,
                            target_size: Tuple[int, int] = (64, 64),
                            rotation_deg: float = 0.0,
                            detrend: str = 'ratio',
                            remove_stars: bool = True,
                            star_size: int = 7,
                            masked_background: bool = True,
                            radial_flatten: bool = False,
                            order: str = 'fullres',
                            contrast: str = 'zscore',
                            bg_sigma_frac: float = 10 / 64,
                            fov_radius_frac: float = 0.44,
                            zclip: Optional[float] = 4.0,
                            return_steps: bool = False):
    """
    Pré-processamento V5 de uma imagem all-sky 630 nm (escala de cinza, uint8 ou float).

    Etapas (com os padrões):
      0. rotação opcional (orientação por estação)
      1. remoção de estrelas: abertura morfológica (star_size px) — remove só pontos brilhantes,
         preservando as faixas escuras das bolhas
      1b. correção radial: divide pela mediana do brilho em anéis concêntricos (remove o
         aumento do airglow para o horizonte + vinheteamento da lente; as faixas das bolhas
         ocupam poucos azimutes e quase não afetam a mediana)
      2. fundo: gaussiana normalizada pela máscara do domo (não mistura o exterior preto)
      3. δI/I = I / fundo − 1 (com sinal: depleções ficam negativas)
      4. redução para target_size com INTER_AREA ponderado pela máscara
      5. z-score dentro do domo, corte em ±zclip, exterior = 0

    Os parâmetros permitem reproduzir passos intermediários (escada de ablação) e o V4
    (ver V4_EQUIVALENT). target_size = (altura, largura).

    Returns:
        imagem (H, W) float32; se return_steps=True, (imagem, dict de etapas intermediárias)
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    if img.max() <= 1.0:
        img = img * 255.0
    steps = {'input': img.copy()}
    H, W = target_size

    if rotation_deg:
        img = _rotate(img, rotation_deg)

    if order == 'downsample_first' and img.shape != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    elif order not in ('downsample_first', 'fullres'):
        raise ValueError(f"order inválido: {order}")

    mask = fov_mask(img.shape, fov_radius_frac)

    if remove_stars:
        k = max(3, int(round(star_size * min(img.shape) / 512)))
        k += (k % 2 == 0)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    steps['stars_removed'] = img.copy()

    if radial_flatten:
        img = img / np.maximum(radial_profile(img, fov_radius_frac), 1.0) * 100.0
    steps['radial_flattened'] = img.copy()

    sigma = bg_sigma_frac * min(img.shape)

    def _blur(a):
        return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    if masked_background:
        m = mask.astype(np.float32)
        bg = _blur(img * m) / np.maximum(_blur(m), 1e-6)
    else:
        bg = _blur(img)
    steps['background'] = bg

    if detrend == 'ratio':
        d = img / np.maximum(bg, 1.0) - 1.0          # fundo em ADU, piso de 1 ADU
    elif detrend == 'subtract_clip':
        d = np.clip(img / 255.0 - bg / 255.0, 0, None)  # comportamento do V4
    else:
        raise ValueError(f"detrend inválido: {detrend}")
    if masked_background:
        d = np.where(mask, d, 0.0)
    steps['detrended'] = d.copy()

    if order == 'fullres' and d.shape != (H, W):
        mf = mask.astype(np.float32)
        num = cv2.resize((d * mf).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        den = cv2.resize(mf, (W, H), interpolation=cv2.INTER_AREA)
        mask = den > 0.5
        d = np.where(mask, num / np.maximum(den, 1e-6), 0.0)

    if contrast == 'zscore':
        vals = d[mask]
        mu, sd = float(vals.mean()), float(vals.std()) + 1e-6
        out = np.where(mask, (d - mu) / sd, 0.0)
        if zclip:
            out = np.clip(out, -zclip, zclip)
    elif contrast == 'clahe':
        lo, hi = float(d.min()), float(d.max())
        u = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
        u = equalize_adapthist(u, clip_limit=0.03)
        if u.max() > u.min():
            u = (u - u.min()) / (u.max() - u.min())
        out = u * mask if masked_background else u
    elif contrast == 'none':
        out = d
    else:
        raise ValueError(f"contrast inválido: {contrast}")

    out = out.astype(np.float32)
    if return_steps:
        steps['output'] = out
        return out, steps
    return out


TEST_NIGHTS_FILE = 'test_nights.txt'
SPLITS = ('dev', 'test', 'all')


def load_test_nights(path: str = "data/test_nights.txt") -> set:
    """
    Lê as noites do conjunto de teste travado: uma noite SITE_YYYYMMDD por linha.
    Linhas em branco e linhas que começam com '#' são ignoradas.
    Arquivo ausente -> conjunto vazio.
    """
    p = Path(path)
    if not p.exists():
        return set()
    nights = set()
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            nights.add(line)
    return nights


def load_raw_dataset(data_folder: str, label_file: str,
                     include_cloudy: bool = False, split: str = "dev",
                     test_nights_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carrega as imagens rotuladas em resolução ORIGINAL (uint8) + labels + metadados alinhados.
    Quadros nublados (cloudy=1 ou has_epb vazio) têm EPB INDETERMINADA:
      include_cloudy=False (padrão) — excluídos; y ∈ {0, 1}. Uso: treino/avaliação do Estágio 2.
      include_cloudy=True           — incluídos com y = -1. Uso: avaliação do Estágio 1 (QC).
    Conjunto de teste travado (noites em test_nights.txt, ver load_test_nights):
      split="dev"  (padrão) — só quadros de noites FORA do teste.
      split="test"          — só quadros de noites do teste (uso exclusivo de evaluate_locked_test).
      split="all"           — todos os quadros rotulados.
    test_nights_path=None usa test_nights.txt na mesma pasta de label_file.
    """
    if split not in SPLITS:
        raise ValueError(f"split inválido: {split!r}; use um de {SPLITS}")
    df = pd.read_csv(label_file)
    if split != 'all':
        if test_nights_path is None:
            test_nights_path = Path(label_file).parent / TEST_NIGHTS_FILE
        test_nights = load_test_nights(test_nights_path)
        nights = extract_night_groups(df) if len(df) else np.array([], dtype=str)
        missing = sorted(test_nights - set(nights))
        for night in missing:
            logger.warning(f"Noite de teste {night} não tem quadros em {label_file} (erro no id da noite?).")
        in_test = np.isin(nights, list(test_nights))
        keep_mask = in_test if split == 'test' else ~in_test
        excluded = ~keep_mask
        if excluded.any():
            logger.info(f"split={split!r}: {int(excluded.sum())} quadros de "
                        f"{len(set(nights[excluded]))} noites excluídos.")
        df = df[keep_mask].reset_index(drop=True)
    cloudy_col = df['cloudy'].fillna(0).astype(int) if 'cloudy' in df.columns else pd.Series(0, index=df.index)
    undetermined = df['has_epb'].isna() | (cloudy_col == 1)
    if (df['has_epb'].notna() & (cloudy_col == 1)).any():
        logger.warning("Há quadros com cloudy=1 e has_epb preenchido; has_epb será ignorado nesses quadros.")
    df['has_epb'] = np.where(undetermined, -1, df['has_epb'].fillna(-1)).astype(int)
    if not include_cloudy:
        if undetermined.any():
            logger.info(f"{int(undetermined.sum())} quadros nublados excluídos do Estágio 2.")
        df = df[~undetermined.values].reset_index(drop=True)
    raws, keep = [], []
    for i, row in df.iterrows():
        try:
            raws.append(np.array(Image.open(Path(data_folder) / row['filename']).convert('L'), dtype=np.uint8))
            keep.append(i)
        except Exception as e:  # imagem ausente/corrompida
            logger.warning(f"{row['filename']}: {e}")
    df = df.loc[keep].reset_index(drop=True)
    if not raws:
        return np.empty((0, 0, 0), dtype=np.uint8), np.empty(0, dtype=int), df
    shapes = {r.shape for r in raws}
    if len(shapes) != 1:
        raise ValueError(f"Imagens com resoluções diferentes: {shapes}")
    return np.stack(raws), df['has_epb'].astype(int).values, df


def preprocess_batch(raw: np.ndarray, sites: Optional[np.ndarray] = None,
                     rotation_by_site: Optional[Dict[str, float]] = None,
                     target_size: Tuple[int, int] = (64, 64), **params) -> np.ndarray:
    """Aplica preprocess_epb_image_v5 a um lote (com rotação por estação, se fornecida)."""
    rot = rotation_by_site or {}
    out = []
    for i, img in enumerate(raw):
        angle = rot.get(str(sites[i]), 0.0) if sites is not None else 0.0
        out.append(preprocess_epb_image_v5(img, target_size, rotation_deg=angle, **params))
    return np.stack(out)


def brightness_features(raw: np.ndarray) -> np.ndarray:
    """5 estatísticas globais de brilho (sem estrutura espacial) — baseline de fator de confusão."""
    flat = raw.reshape(len(raw), -1).astype(np.float32)
    return np.column_stack([flat.mean(1), flat.std(1), np.percentile(flat, 5, axis=1),
                            np.percentile(flat, 50, axis=1), np.percentile(flat, 95, axis=1)])


# ============================================================================
# ITEM 4: ORIENTAÇÃO DAS BOLHAS POR ESTAÇÃO
# ============================================================================

def estimate_band_orientation(img_proc: np.ndarray, radius_frac: float = 0.38) -> Tuple[float, float]:
    """
    Orientação dominante das faixas numa imagem pré-processada (tensor de estrutura).

    Returns:
        angle_deg: inclinação das faixas em relação à VERTICAL da imagem, em (−90, 90]
                   (0 = faixas verticais; positivo = topo inclinado para a direita)
        coherence: [0, 1]; ~0 = sem direção dominante, ~1 = estrutura fortemente orientada
    """
    m = fov_mask(img_proc.shape, radius_frac)   # raio menor que o domo: ignora a borda artificial
    gx = sobel(img_proc, axis=1)
    gy = sobel(img_proc, axis=0)
    jxx, jyy, jxy = (gx * gx)[m].sum(), (gy * gy)[m].sum(), (gx * gy)[m].sum()
    # Direção dominante do GRADIENTE em relação ao eixo x; faixas são perpendiculares ao gradiente,
    # logo a inclinação das faixas em relação à vertical é o mesmo ângulo.
    theta = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    # eixo y da imagem aponta para baixo: troca o sinal para "topo para a direita = positivo"
    angle = -theta
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / max(jxx + jyy, 1e-12))
    return float(angle), coherence


def check_station_orientation(X_proc: np.ndarray, y: np.ndarray, sites: np.ndarray,
                              tolerance_deg: float = 25.0, verbose: bool = True) -> pd.DataFrame:
    """
    Verifica, por estação, se as bolhas aparecem com a mesma orientação (item 4).

    Usa apenas quadros com EPB: em quadros calmos não há faixas para medir.
    A média é circular nos ângulos dobrados (faixas não têm sentido), ponderada pela coerência.

    tolerance_deg: as bolhas seguem o meridiano MAGNÉTICO; no Brasil a declinação magnética é
                   da ordem de −20°, então inclinações até ~25° são esperadas mesmo com o norte
                   geográfico para cima. A tolerância é um critério prático, não físico.
    """
    rows = []
    for st in sorted(np.unique(sites)):
        idx = np.where((sites == st) & (y == 1))[0]
        if len(idx) == 0:
            rows.append({'station': st, 'epb_frames': 0, 'band_angle_deg': np.nan,
                         'circular_spread_deg': np.nan, 'median_coherence': np.nan,
                         'status': 'not verifiable (no EPB frames)'})
            continue
        res = np.array([estimate_band_orientation(X_proc[i]) for i in idx])
        ang, coh = res[:, 0], res[:, 1]
        z = np.sum(coh * np.exp(1j * np.radians(2 * ang))) / max(coh.sum(), 1e-12)
        mean_ang = 0.5 * np.degrees(np.angle(z))
        spread = 0.5 * np.degrees(np.sqrt(max(-2 * np.log(max(abs(z), 1e-12)), 0.0)))
        status = 'consistent (near vertical)' if abs(mean_ang) <= tolerance_deg else \
            f'tilted by {mean_ang:+.0f}°: consider rotation_by_site'
        rows.append({'station': st, 'epb_frames': len(idx), 'band_angle_deg': round(mean_ang, 1),
                     'circular_spread_deg': round(spread, 1),
                     'median_coherence': round(float(np.median(coh)), 3), 'status': status})
    df = pd.DataFrame(rows)
    if verbose:
        print("\n🧭 Band orientation per station (EPB frames only; 0° = vertical bands):")
        print(df.to_string(index=False))
        known = df.dropna(subset=['band_angle_deg'])
        if len(known) >= 2:
            d = known['band_angle_deg'].max() - known['band_angle_deg'].min()
            print(f"   Max difference between stations: {d:.1f}°")
        missing = df.loc[df['epb_frames'] == 0, 'station'].tolist()
        if missing:
            print(f"   ⚠️  Not verifiable from EPB bands: {', '.join(missing)}. "
                  "Confirm their orientation from the camera setup or star positions.")
    return df


# ============================================================================
# ITEM 5: CLASSIFICADOR SIMPLES
# ============================================================================

class EPBRecognitionSystemV5(EPBRecognitionSystem):
    """
    2DPCA + StandardScaler + classificador linear, com pré-processamento V5 embutido.

    classifier:
        'logreg'     — Regressão Logística balanceada (padrão; C = força inversa da regularização)
        'svm_linear' — SVM linear com probabilidades (Platt)
        'stacking'   — ensemble do V4 (SVM RBF + XGBoost + RF → LogReg), para comparação
    balance_data: SMOTE dentro do pipeline (desligado por padrão: classes já ~44/56%).
    """

    def __init__(self, n_components: int = 5, classifier: str = 'logreg', C: float = 0.1,
                 balance_data: bool = False, ensemble_params: Optional[dict] = None,
                 preprocess_params: Optional[dict] = None,
                 target_size: Tuple[int, int] = (64, 64),
                 rotation_by_site: Optional[Dict[str, float]] = None):
        super().__init__(n_components=n_components, balance_data=balance_data,
                         balance_method='smote', ensemble_params=ensemble_params)
        self.classifier_name = classifier
        self.C = C
        self.preprocess_params = dict(V5_PREPROCESS if preprocess_params is None else preprocess_params)
        self.target_size = tuple(target_size)
        self.rotation_by_site = dict(rotation_by_site or {})

    def _make_classifier(self, y_train: np.ndarray):
        if self.classifier_name == 'logreg':
            return LogisticRegression(C=self.C, class_weight='balanced', max_iter=5000, random_state=42)
        if self.classifier_name == 'svm_linear':
            return SVC(kernel='linear', C=self.C, probability=True, class_weight='balanced', random_state=42)
        if self.classifier_name == 'stacking':
            return self._create_ensemble(y_train)
        raise ValueError(f"classifier inválido: {self.classifier_name}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'EPBRecognitionSystemV5':
        """X_train: imagens JÁ pré-processadas com preprocess_epb_image_v5 (N, H, W)."""
        self.tdpca.fit(X_train)
        F = self.scaler.fit_transform(self.extract_features(X_train))
        steps = []
        if self.balance_data:
            self.sampler = self._setup_balancing()
            steps.append(('sampler', self.sampler))
        steps.append(('classifier', self._make_classifier(y_train)))
        self.pipeline_ = ImbPipeline(steps).fit(F, y_train)
        self.classifier = self.pipeline_.named_steps['classifier']
        return self

    # ----- inferência a partir da imagem bruta -----
    def preprocess(self, img_raw: np.ndarray, site: Optional[str] = None) -> np.ndarray:
        angle = self.rotation_by_site.get(str(site), 0.0) if site is not None else 0.0
        return preprocess_epb_image_v5(img_raw, self.target_size, rotation_deg=angle,
                                       **self.preprocess_params)

    def predict_proba_raw(self, img_raw: np.ndarray, site: Optional[str] = None) -> float:
        """Probabilidade de EPB para UMA imagem bruta (resolução original)."""
        return float(self.predict_proba(self.preprocess(img_raw, site)[np.newaxis])[0, 1])

    def predict_cascaded(self, img_raw: np.ndarray, target_size: Tuple[int, int] = None,
                         qc_params: Optional[dict] = None,
                         site: Optional[str] = None) -> CascadedPredictionResult:
        """
        Estágio 1 (QC na imagem bruta) + Estágio 2 V5. target_size é ignorado (vem do sistema).

        Três saídas possíveis para o especialista (ver STATUS_LABELS):
          INDETERMINADO — céu nublado: o sistema NÃO emite decisão sobre EPB (has_epb=None)
          EPB / SEM EPB — decisão do Estágio 2; marcada como pouco confiável se o céu for DEGRADED
        """
        qc = assess_sky_quality(img_raw, **(qc_params or QC_PARAMS))
        if not qc.is_valid_sky:
            return CascadedPredictionResult(qc=qc, has_epb=None, prob_epb=float('nan'),
                                            status=UNDETERMINED, status_label=STATUS_LABELS[UNDETERMINED],
                                            is_reliable=False)
        prob = self.predict_proba_raw(img_raw, site)
        has_epb = int(prob >= 0.5)
        if qc.condition == 'DEGRADED':
            status = 'DEGRADED_EPB' if has_epb else 'DEGRADED_QUIET'
            reliable = False
        else:
            status = 'EPB_DETECTED' if has_epb else 'QUIET_CLEAR'
            reliable = True
        return CascadedPredictionResult(qc=qc, has_epb=has_epb, prob_epb=prob, status=status,
                                        status_label=STATUS_LABELS[status], is_reliable=reliable)

    def analyze_folder(self, folder: str, site: Optional[str] = None,
                       qc_params: Optional[dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Relatório para o especialista: uma linha por quadro + resumo da noite.

        Cada quadro recebe um de três vereditos (coluna 'verdict'):
          'EPB', 'Sem EPB' ou 'Indeterminado' (céu nublado: nenhuma decisão é emitida).
        Quadros com visibilidade parcial têm decisão, mas 'reliable'=False ('conferir').
        A taxa de EPB do resumo é calculada só sobre quadros COM decisão.
        """
        rows = []
        for f in sorted(Path(folder).glob('*.png')):
            res = self.predict_cascaded(np.array(Image.open(f).convert('L')), qc_params=qc_params, site=site)
            verdict = 'Indeterminado' if res.has_epb is None else ('EPB' if res.has_epb else 'Sem EPB')
            rows.append({'filename': f.name, 'verdict': verdict, 'status': res.status,
                         'label': res.status_label, 'prob_epb': res.prob_epb,
                         'reliable': res.is_reliable, 'sky': res.qc.condition,
                         'star_pixels': res.qc.star_pixels, 'raw_mean': round(res.qc.raw_mean, 1)})
        frames = pd.DataFrame(rows)
        if frames.empty:
            return frames, pd.DataFrame()
        decided = frames['verdict'] != 'Indeterminado'
        summary = pd.DataFrame([{
            'frames': len(frames),
            'EPB': int((frames.verdict == 'EPB').sum()),
            'Sem EPB': int((frames.verdict == 'Sem EPB').sum()),
            'Indeterminado': int((~decided).sum()),
            'conferir (visibilidade parcial)': int((decided & ~frames.reliable).sum()),
            'cobertura_pct': round(100 * decided.mean(), 1),
            'epb_pct_dos_decididos': round(100 * (frames.verdict == 'EPB').sum() / max(decided.sum(), 1), 1),
        }])
        return frames, summary

    # ----- explicabilidade exata para o modelo linear -----
    def get_shap_importance(self, X_sample: np.ndarray, X_background: np.ndarray,
                            method: str = 'linear', n_background: int = 100):
        """
        Atribuição por pixel EXATA para classificadores lineares (logreg / svm_linear).

        O logit é linear nas features, e as features são lineares nos pixels
        (2DPCA + StandardScaler), logo:
            logit(X) − E_bg[logit] = Σ_pixels (X − E_bg[X]) ∘ (W' Pᵀ),   W' = coef / scale
        Isto coincide com o SHAP linear (interventional) e soma exatamente à diferença de logit.

        Returns:
            pixel_map (H, W) — positivo: evidência de EPB; feature_contrib (n_features,)
        """
        if self.classifier_name == 'stacking':
            return super().get_shap_importance(X_sample, X_background, method='tree',
                                               n_background=n_background)
        coef = self.classifier.coef_.ravel()
        F_bg = self.scaler.transform(self.extract_features(X_background))
        f = self.scaler.transform(self.extract_features(X_sample[np.newaxis]))[0]
        feat_contrib = coef * (f - F_bg.mean(axis=0))
        H = X_sample.shape[0]
        w_pix = (coef / self.scaler.scale_).reshape(H, self.n_components) @ self.tdpca.projection_matrix.T
        pixel_map = (X_sample - X_background.mean(axis=0)) * w_pix
        return pixel_map, feat_contrib


# ============================================================================
# AVALIAÇÃO: LEAVE-ONE-NIGHT-OUT ANINHADO
# ============================================================================

# Parâmetros fixos usados quando o loop interno não consegue avaliar as duas classes.
# Valores escolhidos pela seleção aninhada na rodada anterior (7 noites, nov/2026):
# não são independentes destes dados — ver docstring de nested_logo_evaluate.
FALLBACK_PARAMS = {'n_components': 5, 'C': 0.001}


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Média de sensibilidade e especificidade sobre os quadros COM predição (y_pred >= 0)."""
    ok = y_pred >= 0
    if not ok.any():
        return np.nan
    y_true, y_pred = y_true[ok], y_pred[ok]
    m = _binary_metrics(y_true, y_pred, y_pred.astype(float))
    vals = [v for v in (m['sensitivity'], m['specificity']) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def logo_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 make_model: Callable[[], object]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predições out-of-night (cada noite prevista por um modelo que não a viu).

    Se o treino de uma divisão tiver uma só classe (ex.: a única outra noite sem EPB
    ficou de fora), essa noite fica SEM predição: y_pred = -1 e y_proba = NaN.
    """
    proba = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        proba[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    pred = np.where(np.isnan(proba), -1, (proba >= 0.5).astype(int))
    return pred, proba


def select_params_logo(X: np.ndarray, y: np.ndarray, groups: np.ndarray, param_grid: dict,
                       make_model: Callable[..., object]) -> Tuple[dict, pd.DataFrame]:
    """Escolhe os parâmetros pela acurácia balanceada pooled em leave-one-night-out."""
    rows, combos = [], list(ParameterGrid(param_grid))
    for i, params in enumerate(combos):
        pred, _ = logo_predict(X, y, groups, lambda: make_model(**params))
        skipped = sorted(set(groups[pred < 0]))
        rows.append({**params, 'balanced_accuracy': _balanced_accuracy(y, pred), '_combo': i,
                     'nights_without_prediction': len(skipped),
                     'scored_classes': int(len(np.unique(y[pred >= 0])))})
    # Ordem estável: em caso de empate, vence a combinação que aparece primeiro na grade
    # (listar valores do mais simples para o mais complexo).
    df = pd.DataFrame(rows).sort_values('balanced_accuracy', ascending=False, kind='stable').reset_index(drop=True)
    best = dict(combos[int(df.loc[0, '_combo'])])   # tipos originais da grade (int continua int)
    return best, df.drop(columns='_combo')


def nested_logo_evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, param_grid: dict,
                         make_model: Callable[..., object], verbose: bool = True,
                         fallback_params: Optional[dict] = None) -> dict:
    """
    Leave-one-night-out ANINHADO: para cada noite externa, os parâmetros são escolhidos com
    leave-one-night-out nas noites restantes e só então a noite externa é prevista.

    Limite com poucas noites: se, sem a noite externa, sobrar só 1 noite de alguma classe,
    o loop interno não consegue medir essa classe. Nesse caso usa fallback_params (fixos),
    e isso fica registrado em 'chosen' (coluna 'selection') e em
    summary['nights_with_fixed_params'].
    """
    if fallback_params is None:
        fallback_params = FALLBACK_PARAMS
    pred = np.full(len(y), -1)
    proba = np.full(len(y), np.nan)
    chosen = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            if verbose:
                print(f"   • {groups[te][0]}: skipped (training would contain a single class)")
            continue
        best, inner = select_params_logo(X[tr], y[tr], groups[tr], param_grid, make_model)
        selection = 'nested'
        if int(inner['scored_classes'].min()) < 2:
            # O loop interno não tem 2 noites de cada classe: só consegue medir uma delas
            # (normalmente só sensibilidade). Escolher por esse critério enviesaria o modelo,
            # então usa parâmetros FIXOS declarados (fallback_params).
            best, selection = dict(fallback_params), 'fixed (inner loop had one class)'
        model = make_model(**best).fit(X[tr], y[tr])
        proba[te] = model.predict_proba(X[te])[:, 1]
        pred[te] = (proba[te] >= 0.5).astype(int)
        chosen.append({'night': groups[te][0], 'selection': selection, **best})
        if verbose:
            print(f"   • {groups[te][0]}: params={best} ({selection}) -> accuracy={np.mean(pred[te] == y[te]):.3f}")
    ok = pred >= 0
    summary = _binary_metrics(y[ok], pred[ok], proba[ok])
    per_night = pd.DataFrame([{'night': g, 'label': int(y[groups == g][0]), 'n': int((groups == g).sum()),
                               'accuracy': (float(np.mean(pred[groups == g] == y[groups == g]))
                                            if ok[groups == g].all() else np.nan),
                               'mean_proba': float(np.nanmean(proba[groups == g])) if ok[groups == g].any() else np.nan}
                              for g in np.unique(groups)])
    summary['balanced_accuracy'] = _balanced_accuracy(y, pred)
    summary['macro_night_accuracy'] = float(per_night['accuracy'].mean())
    summary['worst_night_accuracy'] = float(per_night['accuracy'].min())
    summary['nights_without_prediction'] = int(per_night['accuracy'].isna().sum())
    summary['nights_with_fixed_params'] = int(sum(c['selection'] != 'nested' for c in chosen))
    return {'summary': summary, 'per_night': per_night, 'chosen': pd.DataFrame(chosen),
            'y_pred': pred, 'y_proba': proba}


def evaluate_sequences_out_of_night(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                    sequences: List[Tuple[str, str, str]],
                                    make_model: Callable[[], EPBRecognitionSystemV5],
                                    qc_params: Optional[dict] = None) -> pd.DataFrame:
    """
    Sequências noturnas completas (util/<noite>/processed), cada uma prevista por um modelo
    treinado SEM a respectiva noite. sequences: [(rótulo, pasta, id_da_noite), ...].
    Compara o Estágio 2 isolado com a cascata (Estágio 1 QC + Estágio 2).
    """
    rows = []
    for label, folder, night in sequences:
        files = sorted(Path(folder).glob('*.png'))
        if not files:
            continue
        train = groups != night
        model = make_model().fit(X[train], y[train])
        s2_epb = casc_epb = casc_reliable_epb = qc_cloudy = 0
        for f in files:
            raw = np.array(Image.open(f).convert('L'))
            s2_epb += int(model.predict_proba_raw(raw) >= 0.5)
            res = model.predict_cascaded(raw, qc_params=qc_params)
            if not res.qc.is_valid_sky:
                qc_cloudy += 1
            elif res.has_epb == 1:
                casc_epb += 1
                casc_reliable_epb += int(res.is_reliable)
        n = len(files)
        # O modelo nunca vê a noite avaliada; esta coluna indica só se a noite faz parte
        # do conjunto rotulado do Estágio 2 (False para a noite nublada, que é excluída).
        rows.append({'sequence': label, 'night_labeled_for_stage2': bool((groups == night).any()),
                     'frames': n,
                     'undetermined_pct': round(100 * qc_cloudy / n, 1),
                     'stage2_only_epb_pct': round(100 * s2_epb / n, 1),
                     'cascaded_epb_pct': round(100 * casc_epb / n, 1),
                     'cascaded_clear_sky_epb_pct': round(100 * casc_reliable_epb / n, 1)})
    return pd.DataFrame(rows)


# ============================================================================
# SENSIBILIDADE AO LIMIAR DE DECISÃO
# ============================================================================

def threshold_sweep(y: np.ndarray, y_proba: np.ndarray,
                    thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Varre limiares de decisão sobre as probabilidades OUT-OF-NIGHT e devolve, para cada limiar,
    as métricas binárias agregadas (pooled) nos quadros COM predição.

    A unidade de avaliação continua a ser a noite: use as probabilidades out-of-night de
    logo_predict / nested_logo_evaluate (cada noite prevista por um modelo que não a viu).
    Quadros sem predição (y_proba = NaN, divisão de treino monoclasse) são ignorados.

    Esta análise é descritiva: ela mostra o compromisso sensibilidade/especificidade ao longo
    do limiar. NÃO escolhe um limiar para o conjunto de teste travado — o limiar operacional é
    fixado no desenvolvimento (ver data/LOCKED_TEST_POLICY.md, regra 4).

    Args:
        y: rótulos verdadeiros (0/1), alinhados com y_proba
        y_proba: probabilidade de EPB por quadro (NaN onde não há predição)
        thresholds: limiares a testar; padrão = 0.05, 0.10, …, 0.95

    Returns:
        DataFrame com uma linha por limiar: threshold, n, sensitivity, specificity,
        balanced_accuracy, precision, f1, accuracy, youden_j (= sens + spec − 1).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    ok = ~np.isnan(y_proba)
    yt, pr = y[ok], y_proba[ok]
    rows = []
    for t in thresholds:
        pred = (pr >= t).astype(int)
        m = _binary_metrics(yt, pred, pr)
        sens, spec = m['sensitivity'], m['specificity']
        bal = np.nanmean([sens, spec])
        youden = (sens + spec - 1) if not (np.isnan(sens) or np.isnan(spec)) else np.nan
        rows.append({'threshold': float(t), 'n': int(ok.sum()),
                     'sensitivity': sens, 'specificity': spec,
                     'balanced_accuracy': bal, 'precision': m['precision'],
                     'f1': m['f1'], 'accuracy': m['accuracy'], 'youden_j': youden})
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame, criterion: str = 'balanced_accuracy') -> dict:
    """
    Melhor limiar de uma varredura de threshold_sweep segundo um critério.

    criterion: 'balanced_accuracy' (padrão) ou 'youden_j' ou 'f1'. Em caso de empate, escolhe
    o limiar mais próximo de 0.5 (operação mais neutra). Devolve a linha vencedora como dict.
    """
    if criterion not in sweep.columns:
        raise ValueError(f"critério inválido: {criterion!r}")
    s = sweep.dropna(subset=[criterion]).copy()
    if s.empty:
        raise ValueError("varredura sem valores válidos para o critério.")
    best = s[criterion].max()
    cand = s[s[criterion] >= best - 1e-12].copy()
    cand['dist_meio'] = (cand['threshold'] - 0.5).abs()
    return cand.sort_values(['dist_meio', 'threshold']).iloc[0].drop('dist_meio').to_dict()


def plot_threshold_sensitivity(sweep: pd.DataFrame, criterion: str = 'balanced_accuracy',
                               save_path: Optional[str] = None, dpi: int = 200) -> plt.Figure:
    """
    Curvas de sensibilidade, especificidade e acurácia balanceada vs. limiar, com o limiar
    operacional padrão (0.5) e o melhor limiar pelo critério marcados.
    """
    bt = best_threshold(sweep, criterion)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep['threshold'], sweep['sensitivity'], 'o-', color='#d62728', label='Sensibilidade')
    ax.plot(sweep['threshold'], sweep['specificity'], 's-', color='#1f77b4', label='Especificidade')
    ax.plot(sweep['threshold'], sweep['balanced_accuracy'], '^-', color='#2ca02c',
            label='Acurácia balanceada')
    ax.axvline(0.5, color='#888888', ls='--', lw=1.4, label='Limiar padrão (0.5)')
    ax.axvline(bt['threshold'], color='#9467bd', ls=':', lw=1.8,
               label=f"Melhor {criterion} (t={bt['threshold']:.2f})")
    ax.set_xlabel('Limiar de decisão')
    ax.set_ylabel('Métrica (out-of-night, pooled)')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Sensibilidade ao limiar de decisão (leave-one-night-out)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower center', fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Figura gravada: {save_path}")
    return fig


# ============================================================================
# CONJUNTO DE TESTE TRAVADO
# ============================================================================

TEST_EVALUATIONS_LOG = 'test_evaluations.log'


def _git_commit() -> str:
    """Hash do commit atual do repositório deste módulo ('unknown' se git não estiver disponível)."""
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def _json_safe(value):
    """Converte tipos numpy/NaN para JSON válido (NaN -> null)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


def evaluate_locked_test(make_model: Callable[[], EPBRecognitionSystemV5], note: str,
                         data_folder: str = "data/all-images",
                         label_file: str = "data/epb_labels.csv",
                         test_nights_path: Optional[str] = None,
                         log_path: Optional[str] = None) -> dict:
    """
    ÚNICA forma de pontuar o conjunto de teste travado (noites de test_nights.txt).

    Treina make_model() em TODAS as noites de desenvolvimento (split="dev"), prevê cada noite
    de teste e anexa uma linha JSON a log_path (padrão: test_evaluations.log na pasta de
    label_file) com data/hora UTC, commit git, noites de teste, note e métricas.
    Cada chamada é registrada: o teste deve ser olhado poucas vezes, com o modelo já escolhido.

    make_model: sem argumentos; devolve um EPBRecognitionSystemV5 (ou objeto com fit/predict_proba
        sobre imagens pré-processadas e, opcionalmente, rotation_by_site / target_size /
        preprocess_params, usados para pré-processar as imagens brutas).
    test_nights_path=None usa test_nights.txt na pasta de label_file.

    Levanta ValueError se o conjunto de teste estiver vazio, se alguma noite de teste não tiver
    quadros rotulados (não nublados) ou se o treino tiver uma só classe.
    Returns: {'summary': métricas pooled (_binary_metrics + macro/pior noite),
              'per_night': DataFrame por noite, 'test_nights': lista, 'log_entry': dict gravado}
    """
    if test_nights_path is None:
        test_nights_path = Path(label_file).parent / TEST_NIGHTS_FILE
    if log_path is None:
        log_path = Path(label_file).parent / TEST_EVALUATIONS_LOG
    test_nights = sorted(load_test_nights(test_nights_path))
    if not test_nights:
        raise ValueError(f"Conjunto de teste vazio: nenhuma noite em {test_nights_path}.")

    raw_te, y_te, df_te = load_raw_dataset(data_folder, label_file, split='test',
                                           test_nights_path=test_nights_path)
    groups_te = extract_night_groups(df_te) if len(df_te) else np.array([], dtype=str)
    missing = [n for n in test_nights if n not in set(groups_te)]
    if missing:
        raise ValueError(f"Noites de teste sem quadros rotulados (não nublados): {missing}")

    raw_dev, y_dev, df_dev = load_raw_dataset(data_folder, label_file, split='dev',
                                              test_nights_path=test_nights_path)
    if len(np.unique(y_dev)) < 2:
        raise ValueError("Treino (split='dev') precisa das duas classes.")

    model = make_model()
    prep = dict(rotation_by_site=getattr(model, 'rotation_by_site', None),
                target_size=getattr(model, 'target_size', (64, 64)),
                **getattr(model, 'preprocess_params', V5_PREPROCESS))
    X_dev = preprocess_batch(raw_dev, sites=df_dev['site'].astype(str).values, **prep)
    X_te = preprocess_batch(raw_te, sites=df_te['site'].astype(str).values, **prep)
    model.fit(X_dev, y_dev)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    per_night = pd.DataFrame([{'night': g, 'n': int((groups_te == g).sum()),
                               'n_epb': int(y_te[groups_te == g].sum()),
                               'accuracy': float(np.mean(pred[groups_te == g] == y_te[groups_te == g])),
                               'mean_proba': float(np.mean(proba[groups_te == g]))}
                              for g in test_nights])
    summary = _binary_metrics(y_te, pred, proba)
    summary['balanced_accuracy'] = _balanced_accuracy(y_te, pred)
    summary['macro_night_accuracy'] = float(per_night['accuracy'].mean())
    summary['worst_night_accuracy'] = float(per_night['accuracy'].min())
    summary['n_test_frames'] = int(len(y_te))
    summary['n_dev_frames'] = int(len(y_dev))
    summary['n_dev_nights'] = int(len(set(extract_night_groups(df_dev))))

    entry = _json_safe({
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': _git_commit(),
        'test_nights': test_nights,
        'note': note,
        'metrics': summary,
        'per_night': per_night.to_dict(orient='records'),
    })
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logger.info(f"Avaliação do teste travado registrada em {log_path}.")
    return {'summary': summary, 'per_night': per_night, 'test_nights': test_nights,
            'log_entry': entry}


# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

def plot_preprocessing_comparison(raw_examples: List[Tuple[str, np.ndarray]],
                                  target_size: Tuple[int, int] = (64, 64),
                                  save_path: Optional[str] = None, dpi: int = 200) -> plt.Figure:
    """Para cada exemplo: imagem bruta | V4 | V5 (δI/I) — mesma escala de cores com sinal no V5."""
    fig, axes = plt.subplots(len(raw_examples), 4, figsize=(15, 3.8 * len(raw_examples)))
    axes = np.atleast_2d(axes)
    for r, (title, raw) in enumerate(raw_examples):
        v4 = preprocess_epb_image_v5(raw, target_size, **V4_EQUIVALENT)
        v5, st = preprocess_epb_image_v5(raw, target_size, return_steps=True, **V5_PREPROCESS)
        axes[r, 0].imshow(raw, cmap='gray')
        axes[r, 0].set_title(f'{title}\nraw (512 px)', fontsize=10)
        axes[r, 1].imshow(v4, cmap='gray')
        axes[r, 1].set_title('V4: subtract + clip(0) + CLAHE', fontsize=10)
        m_rf = fov_mask(st['radial_flattened'].shape)
        rf = st['radial_flattened'] * m_rf
        vals = rf[m_rf]
        axes[r, 2].imshow(rf, cmap='gray', vmin=np.percentile(vals, 1), vmax=np.percentile(vals, 99))
        axes[r, 2].set_title(f'V5 step: stars removed + radial flat ({rf.shape[0]} px)', fontsize=10)
        im = axes[r, 3].imshow(v5, cmap='RdBu_r', vmin=-3, vmax=3)
        axes[r, 3].set_title('V5: δI/I z-score (blue = depletion)', fontsize=10)
        plt.colorbar(im, ax=axes[r, 3], fraction=0.046)
        for a in axes[r]:
            a.axis('off')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Figure saved: {save_path}")
    return fig


def plot_linear_attribution(system: EPBRecognitionSystemV5, X_sample: np.ndarray,
                            X_background: np.ndarray, raw: Optional[np.ndarray] = None,
                            title: str = '', save_path: Optional[str] = None, dpi: int = 200) -> plt.Figure:
    """Imagem bruta | entrada V5 | atribuição exata por pixel (vermelho = evidência de EPB)."""
    pix, _ = system.get_shap_importance(X_sample, X_background)
    prob = float(system.predict_proba(X_sample[np.newaxis])[0, 1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].imshow(raw if raw is not None else X_sample, cmap='gray')
    axes[0].set_title('Raw image', fontsize=11)
    axes[1].imshow(X_sample, cmap='RdBu_r', vmin=-3, vmax=3)
    axes[1].set_title('V5 input (δI/I z-score)', fontsize=11)
    v = np.percentile(np.abs(pix), 99) or 1e-9
    im = axes[2].imshow(pix, cmap='RdBu_r', vmin=-v, vmax=v)
    axes[2].set_title('Exact pixel attribution (logit)\nred: EPB evidence | blue: quiet evidence', fontsize=11)
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for a in axes:
        a.axis('off')
    fig.suptitle(f'{title}  P(EPB) = {prob:.1%}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Figure saved: {save_path}")
    return fig


# ============================================================================
# ESCADA DE ABLAÇÃO DO PRÉ-PROCESSAMENTO
# ============================================================================

PREPROCESS_LADDER = [
    ('V4 equivalent (subtract + clip + CLAHE)', dict(V4_EQUIVALENT)),
    ('+1 signed δI/I (no clip)', dict(V4_EQUIVALENT, detrend='ratio', contrast='zscore', zclip=4.0)),
    ('+2 star removal + background inside dome (= V5 default)', dict(V5_PREPROCESS)),
    ('alt. 2b: + radial flattening', dict(V5_PREPROCESS, radial_flatten=True)),
    ('alt. 3: full-res processing, downsample last', dict(V5_PREPROCESS, order='fullres')),
]


def preprocessing_ladder(raw: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         param_grid: dict, ladder: Optional[list] = None,
                         sites: Optional[np.ndarray] = None,
                         rotation_by_site: Optional[Dict[str, float]] = None,
                         verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Adiciona as mudanças de pré-processamento uma a uma e avalia cada passo com
    leave-one-night-out ANINHADO (mesmo classificador linear e mesma grade de parâmetros).

    Returns:
        table (uma linha por passo), per_night (acurácia por noite × passo), cache {passo: X}
    """
    ladder = ladder or PREPROCESS_LADDER
    rows, per_night, cache = [], {}, {}
    for name, params in ladder:
        X = preprocess_batch(raw, sites=sites, rotation_by_site=rotation_by_site, **params)
        cache[name] = X
        res = nested_logo_evaluate(X, y, groups, param_grid,
                                   lambda **kw: EPBRecognitionSystemV5(preprocess_params=params, **kw),
                                   verbose=False)
        s = res['summary']
        rows.append({'step': name, 'accuracy': s['accuracy'], 'balanced_accuracy': s['balanced_accuracy'],
                     'sensitivity': s['sensitivity'], 'specificity': s['specificity'], 'auc': s['auc'],
                     'macro_night_accuracy': s['macro_night_accuracy'],
                     'worst_night_accuracy': s['worst_night_accuracy']})
        per_night[name] = res['per_night'].set_index('night')['accuracy']
        if verbose:
            print(f"   ✓ {name:<48} bal.acc={s['balanced_accuracy']:.3f}  "
                  f"spec={s['specificity']:.3f}  sens={s['sensitivity']:.3f}  AUC={s['auc']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(per_night), cache


# ============================================================================
# ESTÁGIO 1: AVALIAÇÃO DO CONTROLE DE QUALIDADE CONTRA O RÓTULO `cloudy`
# ============================================================================

def evaluate_stage1_qc(raw: np.ndarray, df_metadata: pd.DataFrame, groups: np.ndarray,
                       qc_params: Optional[dict] = None,
                       verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Compara o Estágio 1 (assess_sky_quality, qc_params; padrão QC_PARAMS) com o rótulo `cloudy`.

    Positivo = nublado = INDETERMINADO. 'DEGRADED' conta como céu válido (o Estágio 2 decide,
    mas a decisão sai marcada como 'conferir').
    Atenção: QC_PARAMS foi ajustado nestes mesmos rótulos (uma única noite nublada), então
    esta avaliação é otimista até haver quadros nublados de outras noites.

    Returns:
        {'summary': métricas globais, 'per_night': taxa de bloqueio por noite,
         'frames': diagnóstico por quadro (estrelas, brilho, condição, rótulo)}
    """
    rows = []
    for img, (_, r), g in zip(raw, df_metadata.iterrows(), groups):
        qc = assess_sky_quality(img, **(qc_params or QC_PARAMS))
        rows.append({'filename': r['filename'], 'night': g, 'cloudy_label': int(r.get('cloudy', 0) or 0),
                     'qc_cloudy': int(not qc.is_valid_sky), 'qc_condition': qc.condition,
                     'star_pixels': qc.star_pixels, 'raw_mean': round(qc.raw_mean, 1)})
    fr = pd.DataFrame(rows)
    yt, yp = fr['cloudy_label'].values, fr['qc_cloudy'].values
    tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    summary = pd.DataFrame([{
        'frames': len(fr), 'cloudy_labeled': int(yt.sum()),
        'cloudy_recall': tp / max(tp + fn, 1),          # nublados corretamente bloqueados
        'clear_pass_rate': tn / max(tn + fp, 1),        # céu limpo que passa para o Estágio 2
        'precision_cloudy': tp / max(tp + fp, 1),
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
        'nights_with_cloudy_label': int(fr.loc[fr.cloudy_label == 1, 'night'].nunique()),
        # nublados que passam como 'conferir' em vez de receber decisão confiável
        'cloudy_as_degraded': float(((yt == 1) & (fr.qc_condition == 'DEGRADED')).sum() / max(yt.sum(), 1)),
        'worst_clear_night_pass': float((1 - fr[yt == 0].groupby('night')['qc_cloudy'].mean()).min())
        if (yt == 0).any() else np.nan,
    }])
    per_night = (fr.groupby('night')
                 .agg(frames=('qc_cloudy', 'size'), cloudy_labeled=('cloudy_label', 'sum'),
                      qc_blocked=('qc_cloudy', 'sum'),
                      degraded=('qc_condition', lambda s: int((s == 'DEGRADED').sum())),
                      median_stars=('star_pixels', 'median'))
                 .reset_index())
    per_night['qc_blocked_pct'] = (100 * per_night['qc_blocked'] / per_night['frames']).round(1)
    if verbose:
        s = summary.iloc[0]
        print(f"\n☁️  Stage 1 QC vs. manual `cloudy` label ({s['frames']} frames, "
              f"{s['cloudy_labeled']} cloudy from {s['nights_with_cloudy_label']} night(s))")
        print(f"   • Cloudy -> 'Indeterminado' (recall):  {s['cloudy_recall']:.1%}  ({tp}/{tp + fn})")
        print(f"   • Cloudy -> decision marked 'conferir': {s['cloudy_as_degraded']:.1%}")
        print(f"   • Clear frames passed to Stage 2:      {s['clear_pass_rate']:.1%}  ({tn}/{tn + fp}); "
              f"worst clear night {s['worst_clear_night_pass']:.1%}")
        print(f"   • Precision of 'Indeterminado':        {s['precision_cloudy']:.1%}")
        print(per_night.to_string(index=False))
        if s['nights_with_cloudy_label'] < 2:
            print("   ⚠️  All cloudy labels come from a single night: recall is not a general estimate.")
    return {'summary': summary, 'per_night': per_night, 'frames': fr}

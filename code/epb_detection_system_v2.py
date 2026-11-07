"""
Sistema Completo de Reconhecimento de EPBs usando 2DPCA
Autor: Sistema de ML para Detecção de Bolhas de Plasma Equatorial
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

# Sklearn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Processamento de imagem
from scipy.ndimage import gaussian_filter, rotate, shift
from scipy.stats import spearmanr
from skimage.exposure import equalize_adapthist

import re
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import joblib


@dataclass
class SkyQualityResult:
    """Resultado da análise de qualidade de céu e detecção de nuvens (Estágio 1 - QC)."""
    condition: str          # 'CLEAR', 'CLOUDY', 'DEGRADED'
    is_valid_sky: bool      # True se apto para análise ionosférica
    star_pixels: int        # Picos de alta frequência correspondentes a estrelas
    raw_mean: float         # Intensidade média da imagem bruta
    raw_std: float          # Desvio padrão bruto da imagem
    confidence_qc: float    # Confiança da classificação de céu [0, 1]
    reason: str             # Explicação diagnóstica do estado


@dataclass
class CascadedPredictionResult:
    """Resultado da predição em cascata (Estágio 1 QC + Estágio 2 2DPCA)."""
    qc: SkyQualityResult
    has_epb: Optional[int]       # 1 (EPB), 0 (No EPB), None (se Céu Nublado)
    prob_epb: float              # Probabilidade de EPB (0.0 se nublado)
    status: str                  # 'EPB_DETECTED', 'QUIET_CLEAR', 'CLOUDY_OBSCURED', 'DEGRADED_EPB', 'DEGRADED_QUIET'
    status_label: str            # Texto amigável em português
    is_reliable: bool            # True se a classificação de EPB é confiável


@dataclass
class EPBModelData:
    """Estrutura validada para persistência do modelo treinado."""
    version: str
    system: object  # EPBRecognitionSystem
    n_components: int
    target_size: Tuple[int, int]
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'version': self.version,
            'system': self.system,
            'n_components': self.n_components,
            'target_size': self.target_size,
            'test_metrics': self.test_metrics,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EPBModelData':
        required = {'system', 'n_components', 'target_size'}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Modelo inválido — campos obrigatórios em falta: {missing}")
        return cls(
            version=data.get('version', 'unknown'),
            system=data['system'],
            n_components=data['n_components'],
            target_size=data['target_size'],
            test_metrics=data.get('test_metrics', {}),
        )


logger = logging.getLogger(__name__)


def save_model(model_data: EPBModelData, filepath: str) -> str:
    """Salva o modelo com joblib e retorna o hash SHA-256 do ficheiro."""
    joblib.dump(model_data.to_dict(), filepath, compress=3)
    file_hash = _compute_file_hash(filepath)
    
    # Salva hash ao lado do modelo
    hash_path = filepath + '.sha256'
    with open(hash_path, 'w') as f:
        f.write(file_hash)
    
    logger.info(f"Modelo salvo: {filepath} (SHA-256: {file_hash[:16]}...)")
    return file_hash


def load_model(filepath: str, verify_hash: bool = True) -> EPBModelData:
    """Carrega o modelo com joblib e verifica integridade via SHA-256."""
    if verify_hash:
        hash_path = filepath + '.sha256'
        try:
            with open(hash_path, 'r') as f:
                expected_hash = f.read().strip()
            actual_hash = _compute_file_hash(filepath)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integridade do modelo comprometida!\n"
                    f"  Esperado: {expected_hash[:16]}...\n"
                    f"  Obtido:   {actual_hash[:16]}..."
                )
            logger.info(f"Integridade verificada (SHA-256: {actual_hash[:16]}...)")
        except FileNotFoundError:
            logger.warning(f"Ficheiro de hash não encontrado ({hash_path}). Verificacão ignorada.")
    
    data = joblib.load(filepath)
    return EPBModelData.from_dict(data)


def _compute_file_hash(filepath: str) -> str:
    """Calcula o SHA-256 de um ficheiro."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def setup_logging(level: int = logging.INFO) -> None:
    """Configura logging para uso em notebooks."""
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
    )
    logger.setLevel(level)


# ============================================================================
# PARTE 1: PRÉ-PROCESSAMENTO E CARREGAMENTO DE DADOS
# ============================================================================

def load_epb_dataset(data_folder: str, label_file: str, target_size: Tuple[int, int] = (64, 64), apply_preprocessing: bool = True, apply_circular_mask: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carrega dataset de imagens de EPB
    
    Estrutura esperada:
        data_folder/
            ├── image_001.png
            ├── image_002.png
            └── ...
        
        label_file.csv:
            filename,has_epb
            image_001.png,1
            image_002.png,0
            ...
    
    Args:
        data_folder: Pasta com as imagens
        label_file: CSV com colunas [filename, has_epb]
        target_size: Tamanho para redimensionar imagens
        apply_preprocessing: Aplica pré-processamento especializado
        apply_circular_mask: Se True, aplica máscara circular do domo all-sky (remove textos de borda)
    
    Returns:
        images: array (n_samples, height, width)
        labels: array (n_samples,)
        filenames: lista de nomes dos arquivos
    """
    logger.info("Carregando dataset de EPBs...")
    
    # Lê arquivo de labels
    df = pd.read_csv(label_file)
    
    required_cols = {'filename', 'has_epb'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV faltando colunas obrigatórias: {missing}. Colunas encontradas: {list(df.columns)}")
    
    logger.info(f"Labels carregados: {len(df)} registros")
    
    images = []
    labels = []
    filenames = []
    errors = []
    
    for idx, row in df.iterrows():
        img_path = Path(data_folder) / row['filename']
        
        try:
            # Carrega imagem
            img = Image.open(img_path).convert('L')  # Grayscale
            img = np.array(img.resize(target_size))
            
            # Aplica pré-processamento
            if apply_preprocessing:
                img = preprocess_epb_image(img, apply_circular_mask=apply_circular_mask)
            else:
                img = img.astype(np.float32) / 255.0  # Apenas normaliza
                if apply_circular_mask:
                    h_m, w_m = img.shape
                    yg, xg = np.ogrid[:h_m, :w_m]
                    fov_m = np.sqrt((xg - (w_m - 1) / 2.0)**2 + (yg - (h_m - 1) / 2.0)**2) <= 0.44 * min(h_m, w_m)
                    img = img * fov_m
            
            images.append(img)
            labels.append(int(row['has_epb']))
            filenames.append(row['filename'])
            
        except Exception as e:
            errors.append((row['filename'], str(e)))
    
    if errors:
        logger.warning(f"{len(errors)} imagens com erro:")
        for fname, error in errors[:5]:  # Mostra primeiros 5
            logger.warning(f"  {fname}: {error}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    logger.info(f"{len(images)} imagens carregadas com sucesso")
    logger.info(f"Dimensão das imagens: {images.shape[1]}x{images.shape[2]}")
    
    # Estatísticas do dataset
    n_epb = np.sum(labels)
    n_no_epb = len(labels) - n_epb
    logger.info("Distribuição das classes:")
    logger.info(f"  EPBs detectados: {n_epb} ({n_epb/len(labels)*100:.1f}%)")
    logger.info(f"  No EPB: {n_no_epb} ({n_no_epb/len(labels)*100:.1f}%)")
    
    if n_epb / len(labels) < 0.3 or n_epb / len(labels) > 0.7:
        logger.warning("Dataset desbalanceado! Será aplicado balanceamento.")
    
    return images, labels, filenames


def temporal_train_test_split(
    images: np.ndarray, labels: np.ndarray, filenames: List[str],
    test_size: float = 0.25, date_pattern: str = r'(\d{8})',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Divisão treino/teste baseada em noites de observação (temporal split).

    Agrupa imagens por noite (extraída do nome do ficheiro), ordena
    cronologicamente, e aloca as noites mais recentes para teste —
    separadamente para cada classe — garantindo representação de ambas
    em treino e teste e respeitando a proporção test_size.

    Args:
        images: array (n_samples, height, width)
        labels: array (n_samples,)
        filenames: lista de nomes de ficheiros
        test_size: Fracção desejada de imagens para teste (aprox.)
        date_pattern: Regex para extrair a data (YYYYMMDD) do filename

    Returns:
        X_train, X_test, y_train, y_test, files_train, files_test
    """
    # 1. Extrai data de cada ficheiro
    dates = []
    for fn in filenames:
        match = re.search(date_pattern, fn)
        if match:
            dates.append(match.group(1))
        else:
            dates.append('00000000')
    dates = np.array(dates)

    # 2. Agrupa noites e ordena cronologicamente
    unique_nights = sorted(set(dates))
    night_info = []
    for night in unique_nights:
        mask = dates == night
        night_labels = labels[mask]
        night_info.append({
            'night': night,
            'count': int(mask.sum()),
            'class': 'epb' if night_labels.sum() > 0 else 'no_epb',
        })

    logger.info(f"Temporal split: {len(unique_nights)} noites encontradas")
    for ni in night_info:
        logger.info(f"  {ni['night']}: {ni['count']} imgs "
                     f"({'EPB' if ni['class'] == 'epb' else 'No EPB'})")

    # 3. Separa noites por classe (mantendo ordem cronológica)
    epb_nights = [ni for ni in night_info if ni['class'] == 'epb']
    no_epb_nights = [ni for ni in night_info if ni['class'] == 'no_epb']

    # 4. Para cada classe, aloca as noites mais recentes para teste (~test_size)
    def _select_test_nights(nights: List[dict], target_frac: float) -> set:
        """Selecciona noites mais recentes para teste, respeitando a fracção."""
        total_cls = sum(n['count'] for n in nights)
        target = int(total_cls * target_frac)
        selected = set()
        count = 0
        # Pelo menos 1 noite no treino
        for ni in reversed(nights[:-1] if len(nights) > 1 else []):
            # Para com tolerância de 30%
            if count + ni['count'] > target * 1.5 and count > 0:
                break
            selected.add(ni['night'])
            count += ni['count']
            if count >= target:
                break
        # Se não seleccionou nenhuma e há ≥2 noites, pega a última
        if not selected and len(nights) >= 2:
            selected.add(nights[-1]['night'])
        return selected

    test_nights = set()
    if len(epb_nights) >= 2:
        test_nights |= _select_test_nights(epb_nights, test_size)
    elif len(epb_nights) == 1:
        logger.warning("Apenas 1 noite com EPB — ficará no treino")
    else:
        logger.warning("Sem noites com EPB!")

    if len(no_epb_nights) >= 2:
        test_nights |= _select_test_nights(no_epb_nights, test_size)
    elif len(no_epb_nights) == 1:
        logger.warning("Apenas 1 noite sem EPB — ficará no treino")
    else:
        logger.warning("Sem noites sem EPB!")

    # 5. Monta máscaras
    test_mask = np.isin(dates, list(test_nights))
    train_mask = ~test_mask

    X_train = images[train_mask]
    X_test = images[test_mask]
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    files_train = [f for f, m in zip(filenames, train_mask) if m]
    files_test = [f for f, m in zip(filenames, test_mask) if m]

    # 6. Log final
    train_nights_sorted = sorted({dates[i] for i in range(len(dates)) if train_mask[i]})
    test_nights_sorted = sorted({dates[i] for i in range(len(dates)) if test_mask[i]})

    logger.info(f"\n📅 Temporal Split (estratificado por classe):")
    logger.info(f"  Train: {len(X_train)} imgs de {len(train_nights_sorted)} noites "
                f"({train_nights_sorted[0]}–{train_nights_sorted[-1]})")
    logger.info(f"  Test:  {len(X_test)} imgs de {len(test_nights_sorted)} noites "
                f"({test_nights_sorted[0]}–{test_nights_sorted[-1]})")

    n_epb_train = int(y_train.sum())
    n_epb_test = int(y_test.sum())
    logger.info(f"  Train — EPB: {n_epb_train}, No EPB: {len(y_train) - n_epb_train}")
    logger.info(f"  Test  — EPB: {n_epb_test}, No EPB: {len(y_test) - n_epb_test}")

    actual_test_ratio = len(X_test) / len(images)
    logger.info(f"  Ratio teste: {actual_test_ratio:.1%} (objectivo: {test_size:.0%})")

    if n_epb_test == 0 or n_epb_test == len(y_test):
        logger.warning("⚠️  ATENÇÃO: Test sem representação de ambas as classes! "
                       "Considere adicionar mais noites de observação.")

    return X_train, X_test, y_train, y_test, files_train, files_test


def detect_stars(
    img_raw: np.ndarray,
    sigma: float = 2.0,
    threshold_diff: float = 8.0,
    min_intensity: float = 30.0,
    mask_fov: bool = True
) -> Tuple[int, np.ndarray]:
    """
    Detecta fontes pontuais brilhantes de alta frequência (estrelas) no campo de visão.
    
    Estrelas no céu noturno operam como testemunhas da transparência troposférica:
    - Céu límpido: centenas a milhares de picos estelares pontuais são detectados.
    - Céu nublado/obscurecido: as nuvens atenuam/bloqueiam as estrelas, derrubando a contagem.
    
    Args:
        img_raw: Imagem em escala de cinza (uint8 ou float [0, 255])
        sigma: Desvio padrão do filtro gaussiano para modelar fundo difuso
        threshold_diff: Diferença mínima de intensidade para considerar pico estelar
        min_intensity: Brilho absoluto mínimo para evitar ruído térmico em áreas pretas
        mask_fov: Restringe a contagem ao domo circular útil (exclui anotações de texto nos cantos)
        
    Returns:
        star_pixels: Quantidade de pixels pertencentes a fontes estelares
        star_mask: Máscara binária booleana com as estrelas identificadas
    """
    img_f = img_raw.astype(np.float32)
    if img_f.max() <= 1.0:
        img_f = img_f * 255.0
        
    smooth = gaussian_filter(img_f, sigma=sigma)
    high_freq = img_f - smooth
    
    star_mask = (high_freq > threshold_diff) & (img_f > min_intensity)
    if mask_fov:
        h_s, w_s = img_f.shape
        yg, xg = np.ogrid[:h_s, :w_s]
        fov_circle = np.sqrt((xg - (w_s - 1) / 2.0)**2 + (yg - (h_s - 1) / 2.0)**2) <= 0.44 * min(h_s, w_s)
        star_mask = star_mask & fov_circle
    star_pixels = int(np.sum(star_mask))
    
    return star_pixels, star_mask


def assess_sky_quality(
    img_raw: np.ndarray,
    threshold_stars: int = 4500,
    min_mean_brightness: float = 35.0,
    degraded_stars: int = 6500
) -> SkyQualityResult:
    """
    Avalia a condição do céu e a presença de nuvens (Estágio 1 - Quality Control).
    
    Analisa a transparência óptica antes da detecção ionosférica, evitando que imagens
    obscurecidas por nuvens gerem falsos positivos causados pelo CLAHE.
    
    Args:
        img_raw: Imagem bruta em escala de cinza (uint8 ou float)
        threshold_stars: Limiar mínimo de pixels de estrelas para validar céu limpo
        min_mean_brightness: Intensidade média mínima para descartar céu totalmente encoberto
        degraded_stars: Limiar para alerta de visibilidade intermediária / degradada
        
    Returns:
        SkyQualityResult com condição ('CLEAR', 'CLOUDY', 'DEGRADED'), métricas e validação.
    """
    img_f = img_raw.astype(np.float32)
    if img_f.max() <= 1.0:
        img_f = img_f * 255.0
        
    star_pixels, _ = detect_stars(img_f, mask_fov=True)
    raw_mean = float(np.mean(img_f))
    raw_std = float(np.std(img_f))
    
    if raw_mean < min_mean_brightness or star_pixels < threshold_stars:
        condition = 'CLOUDY'
        is_valid_sky = False
        confidence_qc = max(0.0, 1.0 - (star_pixels / max(threshold_stars, 1)))
        reason = f"Cloudy / Obscured Sky (Stars: {star_pixels}, Mean brightness: {raw_mean:.1f})"
    elif star_pixels < degraded_stars:
        condition = 'DEGRADED'
        is_valid_sky = True
        confidence_qc = 0.5 + 0.5 * ((star_pixels - threshold_stars) / max(degraded_stars - threshold_stars, 1))
        reason = f"Partially Clear / Degraded Visibility (Stars: {star_pixels}, Mean brightness: {raw_mean:.1f})"
    else:
        condition = 'CLEAR'
        is_valid_sky = True
        confidence_qc = min(1.0, 0.8 + 0.2 * (star_pixels / max(degraded_stars * 2, 1)))
        reason = f"Clear Sky with High Visibility (Stars: {star_pixels}, Mean brightness: {raw_mean:.1f})"
        
    return SkyQualityResult(
        condition=condition,
        is_valid_sky=is_valid_sky,
        star_pixels=star_pixels,
        raw_mean=raw_mean,
        raw_std=raw_std,
        confidence_qc=confidence_qc,
        reason=reason
    )


def preprocess_epb_image(img: np.ndarray, apply_background_removal: bool = True, enhance_contrast: bool = True, apply_circular_mask: bool = False, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Pré-processamento específico para realçar estruturas de EPB
    
    Args:
        img: Imagem em escala de cinza [0, 255] ou [0, 1]
        apply_background_removal: Remove gradientes de fundo
        enhance_contrast: Aplica CLAHE para realçar contraste local
        apply_circular_mask: Zera pixels fora do domo circular útil (remove carimbos de texto)
        target_size: Redimensiona opcionalmente para (H, W) antes do pré-processamento
    
    Returns:
        Imagem pré-processada normalizada [0, 1]
    """
    if target_size is not None and img.shape != target_size:
        img_uint8 = img.astype(np.uint8) if img.max() > 1.0 else (img * 255.0).astype(np.uint8)
        img = np.array(Image.fromarray(img_uint8).resize((target_size[1], target_size[0])))

    # Garante [0, 1]
    if img.max() > 1.0:
        img = img / 255.0
    
    img = img.astype(np.float32)
    
    # 1. Remove gradientes de fundo (iluminação não-uniforme)
    if apply_background_removal:
        # Estima background com filtro gaussiano
        background = gaussian_filter(img, sigma=10)
        img = img - background
        # Garante valores positivos
        img = np.clip(img, 0, None)
    
    # 2. Realce de contraste (similar a CLAHE)
    if enhance_contrast:
        # Equalização adaptativa por blocos
        img = equalize_adapthist(img, clip_limit=0.03)
    
    # 3. Normalização final
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    # 4. Máscara circular opcional do campo de visão (FOV)
    if apply_circular_mask:
        h_m, w_m = img.shape
        yg, xg = np.ogrid[:h_m, :w_m]
        fov_m = np.sqrt((xg - (w_m - 1) / 2.0)**2 + (yg - (h_m - 1) / 2.0)**2) <= 0.44 * min(h_m, w_m)
        img = (img * fov_m).astype(np.float32)
    
    return img


def augment_epb_data(images: np.ndarray, labels: np.ndarray, augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data Augmentation estratificado para aumentar dataset.
    
    Aplica mais augmentation na classe minoritária para ajudar no balanceamento
    antes do SMOTE, preservando características de EPB.
    
    Args:
        images: array (n_samples, height, width)
        labels: array (n_samples,)
        augment_factor: Fator base de multiplicação (a minoritária recebe o dobro)
    
    Returns:
        images_aug: array aumentado
        labels_aug: array de labels aumentado
    """
    logger.info(f"Aplicando Data Augmentation estratificado (fator base={augment_factor})...")
    
    # Calcula fator diferenciado por classe
    n_class0 = np.sum(labels == 0)
    n_class1 = np.sum(labels == 1)
    minority_label = 1 if n_class1 < n_class0 else 0
    
    factor_majority = max(augment_factor - 1, 1)
    factor_minority = factor_majority * 2  # Dobro para a classe minoritária
    
    logger.info(f"  Classe 0: {n_class0} imgs (fator={'x' + str(factor_minority + 1) if 0 == minority_label else 'x' + str(factor_majority + 1)})")
    logger.info(f"  Classe 1: {n_class1} imgs (fator={'x' + str(factor_minority + 1) if 1 == minority_label else 'x' + str(factor_majority + 1)})")
    
    augmented_images = list(images)
    augmented_labels = list(labels)
    
    for img, label in zip(images, labels):
        n_augments = factor_minority if label == minority_label else factor_majority
        for _ in range(n_augments):
            aug_img = _apply_random_augmentation(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    n_new0 = np.sum(augmented_labels == 0)
    n_new1 = np.sum(augmented_labels == 1)
    logger.info(f"Dataset expandido: {len(images)} → {len(augmented_images)} imagens "
                f"(classe 0: {n_class0}→{n_new0}, classe 1: {n_class1}→{n_new1})")
    
    return augmented_images, augmented_labels


def _apply_random_augmentation(img: np.ndarray) -> np.ndarray:
    """Aplica uma transformação aleatória preservando características de EPB."""
    aug_type = np.random.choice(['flip', 'rotate', 'shift', 'noise'])
    
    if aug_type == 'flip':
        return np.fliplr(img)
    elif aug_type == 'rotate':
        angle = np.random.uniform(-10, 10)
        return rotate(img, angle, reshape=False, mode='constant')
    elif aug_type == 'shift':
        shift_x = np.random.randint(-5, 5)
        shift_y = np.random.randint(-5, 5)
        return shift(img, [shift_y, shift_x], mode='constant')
    else:  # noise
        noise = np.random.normal(0, 0.02, img.shape)
        return np.clip(img + noise, 0, 1)


# ============================================================================
# PARTE 2: 2DPCA E EXTRAÇÃO DE FEATURES
# ============================================================================

class TwoDPCA:
    """2DPCA para extração de features de imagens de EPB"""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.projection_matrix: Optional[np.ndarray] = None
        self.mean_image: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        self.explained_variance_ratio: Optional[np.ndarray] = None
        self.all_eigenvalues: Optional[np.ndarray] = None
        self.all_explained_variance_ratio: Optional[np.ndarray] = None
    
    def fit(self, images: np.ndarray) -> 'TwoDPCA':
        """Treina o 2DPCA"""
        n_samples, height, width = images.shape
        
        self.mean_image = np.mean(images, axis=0)
        centered_images = images - self.mean_image
        
        # Matriz de covariância
        covariance_matrix = np.zeros((width, width))
        for img in centered_images:
            covariance_matrix += img.T @ img
        covariance_matrix /= n_samples
        
        # Autovalores e autovetores
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Ordena em ordem decrescente
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = np.maximum(eigenvalues[idx], 0.0)
        eigenvectors = eigenvectors[:, idx]
        
        # Armazena espectro completo para análise e gráficos
        total_variance = np.sum(eigenvalues)
        self.all_eigenvalues = eigenvalues
        self.all_explained_variance_ratio = eigenvalues / total_variance if total_variance > 0 else eigenvalues
        
        # Seleciona top n_components
        self.projection_matrix = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.all_explained_variance_ratio[:self.n_components]
        
        return self
    
    def transform(self, images: np.ndarray) -> np.ndarray:
        """Projeta imagens no espaço reduzido"""
        centered_images = images - self.mean_image
        projected = np.array([img @ self.projection_matrix 
                             for img in centered_images])
        return projected
    
    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)
    
    def inverse_transform(self, projected_images: np.ndarray) -> np.ndarray:
        """Reconstrói imagens (útil para visualização)"""
        reconstructed = np.array([proj @ self.projection_matrix.T 
                                 for proj in projected_images])
        return reconstructed + self.mean_image


class TwoDPCATransformer(BaseEstimator, TransformerMixin):
    """Wrapper sklearn-compatible para TwoDPCA.

    Aceita dados achatados (n_samples, h*w) na interface sklearn,
    reconstrói as imagens 2D internamente, aplica 2DPCA e devolve
    features achatadas (n_samples, h * n_components).
    """

    def __init__(self, n_components: int = 15,
                 image_shape: Tuple[int, int] = (64, 64)):
        self.n_components = n_components
        self.image_shape = image_shape

    def fit(self, X: np.ndarray, y=None) -> 'TwoDPCATransformer':
        images = X.reshape(-1, *self.image_shape)
        self.tdpca_ = TwoDPCA(n_components=self.n_components)
        self.tdpca_.fit(images)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        images = X.reshape(-1, *self.image_shape)
        projected = self.tdpca_.transform(images)
        return projected.reshape(len(X), -1)


def optimize_n_components(X_train: np.ndarray, y_train: np.ndarray,
                          max_components: int = 30, cv_folds: int = 5,
                          use_augmentation: bool = False,
                          augment_factor: int = 3) -> Tuple[int, List[dict]]:
    """
    Otimiza número de componentes 2DPCA via validação cruzada.

    O augmentation (se activado) é aplicado **dentro** de cada fold para evitar
    data leakage — imagens aumentadas nunca aparecem no fold de validação.

    Args:
        X_train: Imagens originais (sem augmentation)
        y_train: Labels originais
        max_components: Número máximo de componentes a testar
        cv_folds: Número de folds para validação cruzada
        use_augmentation: Aplicar augmentation dentro de cada fold
        augment_factor: Fator de augmentation (usado se use_augmentation=True)
    """
    logger.info(f"Otimizando número de componentes (máx={max_components})...")
    if use_augmentation:
        logger.info(f"  Augmentation dentro dos folds: fator={augment_factor}")
    
    results = []
    candidate_k = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    test_range = [c for c in candidate_k if c <= min(max_components, X_train.shape[2])]
    if not test_range:
        test_range = list(range(5, min(max_components + 1, X_train.shape[2]), 5))
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for n_comp in test_range:
        fold_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Augmentation apenas no treino do fold
            if use_augmentation:
                X_fold_train, y_fold_train = augment_epb_data(
                    X_fold_train, y_fold_train, augment_factor=augment_factor
                )
            
            # 2DPCA
            tdpca = TwoDPCA(n_components=n_comp)
            tdpca.fit(X_fold_train)
            
            features_train = tdpca.transform(X_fold_train).reshape(len(X_fold_train), -1)
            features_val = tdpca.transform(X_fold_val).reshape(len(X_fold_val), -1)
            
            # Normaliza
            scaler = StandardScaler()
            features_train = scaler.fit_transform(features_train)
            features_val = scaler.transform(features_val)
            
            # Treina e avalia
            clf = SVC(kernel='rbf', gamma='scale', class_weight='balanced', random_state=42)
            clf.fit(features_train, y_fold_train)
            y_pred_fold = clf.predict(features_val)
            fold_f1 = f1_score(y_fold_val, y_pred_fold, zero_division=0)
            fold_scores.append(fold_f1)
        
        scores = np.array(fold_scores)
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Calcula variância explicada (no dataset completo)
        tdpca_full = TwoDPCA(n_components=n_comp)
        tdpca_full.fit(X_train)
        var_explained = np.sum(tdpca_full.explained_variance_ratio)
        
        results.append({
            'n_components': n_comp,
            'f1_score': mean_score,
            'f1_std': std_score,
            'variance_explained': var_explained
        })
        
        logger.info(f"  {n_comp:2d} comp: F1={mean_score:.4f}±{std_score:.4f} | "
                    f"Var={var_explained:.2%}")
    
    # Encontra melhor configuração
    best_result = max(results, key=lambda x: x['f1_score'])
    best_n = best_result['n_components']
    
    logger.info(f"Melhor configuração: {best_n} componentes")
    logger.info(f"  F1-Score: {best_result['f1_score']:.4f}")
    logger.info(f"  Variância explicada: {best_result['variance_explained']:.2%}")
    
    return best_n, results


def optimize_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray,
    image_shape: Tuple[int, int] = (64, 64),
    n_iter: int = 30,
    cv_folds: int = 5,
    scoring: str = 'f1',
    random_state: int = 42,
) -> Tuple[dict, pd.DataFrame, plt.Figure]:
    """
    Otimização conjunta de hiperparâmetros via RandomizedSearchCV.

    Constrói um pipeline completo (2DPCA → Scaler → SMOTE → Ensemble)
    e busca os melhores hiperparâmetros por busca aleatória com CV
    estratificada.

    Parâmetros otimizados:
      - n_components (2DPCA)
      - C, gamma (SVM RBF)
      - n_estimators, max_depth, learning_rate (XGBoost)
      - n_estimators, max_depth (Random Forest)
      - C (meta-classificador LogisticRegression do StackingClassifier)

    Args:
        X_train: Imagens de treino (n, h, w) — sem augmentation
        y_train: Labels
        image_shape: Forma das imagens (height, width)
        n_iter: Combinações aleatórias a testar
        cv_folds: Folds de validação cruzada
        scoring: Metric de otimização ('f1', 'accuracy', 'roc_auc')
        random_state: Seed para reprodutibilidade

    Returns:
        best_params: Dict com os melhores hiperparâmetros (nomes limpos)
        df_results: DataFrame com resultados de todas as combinações
        fig: Figura com visualização dos resultados
    """
    print(f"\n{'='*70}")
    print(f"🔧 OTIMIZAÇÃO CONJUNTA DE HIPERPARÂMETROS")
    print(f"{'='*70}")
    print(f"   Combinações a testar: {n_iter}")
    print(f"   Folds CV: {cv_folds}")
    print(f"   Metric: {scoring}")
    print(f"   Amostras: {len(X_train)}\n")

    # Flatten images para interface sklearn
    X_flat = X_train.reshape(len(X_train), -1)

    # Ratio de classes para XGBoost
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    # Pipeline completo
    pipe = ImbPipeline([
        ('tdpca', TwoDPCATransformer(image_shape=image_shape)),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=random_state)),
        ('clf', StackingClassifier(
            estimators=[
                ('svm_rbf', SVC(kernel='rbf', probability=True,
                                class_weight='balanced',
                                random_state=random_state)),
                ('xgboost', XGBClassifier(eval_metric='logloss',
                                          scale_pos_weight=spw,
                                          random_state=random_state)),
                ('rf', RandomForestClassifier(class_weight='balanced',
                                              min_samples_split=5,
                                              random_state=random_state)),
            ],
            final_estimator=LogisticRegression(
                max_iter=1000, random_state=random_state),
            cv=5,
            passthrough=False,
        )),
    ])

    # Espaço de busca
    param_distributions = {
        'tdpca__n_components': [5, 10, 15, 20, 25],
        'clf__svm_rbf__C': [0.5, 1.0, 5.0, 10.0, 20.0],
        'clf__svm_rbf__gamma': ['scale', 'auto'],
        'clf__xgboost__n_estimators': [100, 200, 300],
        'clf__xgboost__max_depth': [3, 6, 9],
        'clf__xgboost__learning_rate': [0.05, 0.1, 0.2],
        'clf__rf__n_estimators': [100, 200, 300],
        'clf__rf__max_depth': [10, 15, 20, None],
        'clf__final_estimator__C': [0.1, 0.5, 1.0, 5.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_state)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
        error_score='raise',
    )

    search.fit(X_flat, y_train)

    # Monta dict com nomes limpos
    best = search.best_params_
    best_params = {
        'n_components': best.get('tdpca__n_components', 15),
        'svm_C': best.get('clf__svm_rbf__C', 5.0),
        'svm_gamma': best.get('clf__svm_rbf__gamma', 'scale'),
        'xgb_n_estimators': best.get('clf__xgboost__n_estimators', 200),
        'xgb_max_depth': best.get('clf__xgboost__max_depth', 6),
        'xgb_learning_rate': best.get('clf__xgboost__learning_rate', 0.1),
        'rf_n_estimators': best.get('clf__rf__n_estimators', 200),
        'rf_max_depth': best.get('clf__rf__max_depth', 15),
        'meta_C': best.get('clf__final_estimator__C', 1.0),
    }
    best_score = search.best_score_

    print(f"\n{'='*70}")
    print(f"🏆 MELHORES HIPERPARÂMETROS ({scoring}={best_score:.4f})")
    print(f"{'='*70}")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # Resultados
    df_results = pd.DataFrame(search.cv_results_).sort_values('rank_test_score')

    print(f"\n📊 Top 5 combinações:")
    param_cols = [c for c in df_results.columns if c.startswith('param_')]
    for _, row in df_results.head(5).iterrows():
        print(f"   #{int(row['rank_test_score'])} | "
              f"Test: {row['mean_test_score']:.4f}±{row['std_test_score']:.4f} | "
              f"Train: {row['mean_train_score']:.4f}")
        for pc in param_cols:
            short = pc.replace('param_clf__', '').replace('param_tdpca__', '')
            short = short.replace('param_', '')
            print(f"       {short}: {row[pc]}")

    # --- Visualização ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Score vs n_components
    ax = axes[0]
    for nc in sorted(df_results['param_tdpca__n_components'].dropna().unique()):
        mask = df_results['param_tdpca__n_components'] == nc
        scores = df_results.loc[mask, 'mean_test_score']
        ax.scatter([nc] * len(scores), scores, alpha=0.5, s=30)
    ax.set_xlabel('n_components (2DPCA)', fontsize=11)
    ax.set_ylabel(f'{scoring} (teste)', fontsize=11)
    ax.set_title('Score vs. Components 2DPCA', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # 2. Distribuição dos scores
    ax = axes[1]
    ax.hist(df_results['mean_test_score'], bins=15,
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(best_score, color='red', linestyle='--', linewidth=2,
               label=f'Melhor: {best_score:.4f}')
    ax.set_xlabel(f'{scoring} (teste)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribuição dos Scores', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Train vs Test (diagnóstico de overfitting)
    ax = axes[2]
    ax.scatter(df_results['mean_train_score'],
               df_results['mean_test_score'],
               alpha=0.5, s=30, c='steelblue')
    lims = [min(df_results['mean_test_score'].min(),
                df_results['mean_train_score'].min()) - 0.02, 1.02]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Train = Test')
    ax.set_xlabel(f'{scoring} (treino)', fontsize=11)
    ax.set_ylabel(f'{scoring} (teste)', fontsize=11)
    ax.set_title('Train vs. Test (Overfitting?)', fontsize=12,
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Otimização de Hiperparâmetros — Melhor {scoring}: '
                 f'{best_score:.4f}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return best_params, df_results, fig


# ============================================================================
# PARTE 3: SISTEMA COMPLETO COM ENSEMBLE E BALANCEAMENTO
# ============================================================================

class EPBRecognitionSystem:
    """
    Sistema completo de reconhecimento de EPBs
    - 2DPCA para extração de features
    - Balanceamento de classes (SMOTE)
    - Ensemble de classificadores
    """
    
    def __init__(self, n_components: int = 15, balance_data: bool = True,
                 balance_method: str = 'smote',
                 ensemble_params: Optional[dict] = None):
        """
        Args:
            n_components: Número de componentes 2DPCA
            balance_data: Aplicar balanceamento de classes
            balance_method: 'smote', 'undersample', ou 'combined'
            ensemble_params: Dict com hiperparâmetros do ensemble
                             (produzido por optimize_hyperparameters).
                             Se None, usa valores padrão.
        """
        self.n_components = n_components
        self.balance_data = balance_data
        self.balance_method = balance_method
        self.ensemble_params = ensemble_params or {}
        
        self.tdpca = TwoDPCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.sampler = None
        
        # Pipeline imblearn (sampler + classificador)
        self.pipeline_ = None
        self.classifier = None
    
    def _create_ensemble(self, y_train: np.ndarray) -> StackingClassifier:
        """Ensemble otimizado para detecção de EPBs via stacking.

        Usa StackingClassifier com LogisticRegression como meta-classificador,
        que aprende automaticamente a melhor combinação dos base learners.

        Se self.ensemble_params estiver preenchido (de optimize_hyperparameters),
        usa os hiperparâmetros otimizados; caso contrário, usa defaults.
        """
        p = getattr(self, 'ensemble_params', {}) or {}

        # Calcula ratio de classes para XGBoost
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if p:
            logger.info("Usando hiperparâmetros otimizados do ensemble:")
            for k, v in p.items():
                logger.info(f"  {k}: {v}")

        estimators = [
            ('svm_rbf', SVC(
                kernel='rbf',
                probability=True,
                gamma=p.get('svm_gamma', 'scale'),
                C=p.get('svm_C', 5.0),
                class_weight='balanced',
                random_state=42
            )),
            ('xgboost', XGBClassifier(
                n_estimators=p.get('xgb_n_estimators', 200),
                max_depth=p.get('xgb_max_depth', 6),
                learning_rate=p.get('xgb_learning_rate', 0.1),
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=42,
                eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=p.get('rf_n_estimators', 200),
                max_depth=p.get('rf_max_depth', 15),
                min_samples_split=5,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ))
        ]

        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=p.get('meta_C', 1.0),
                max_iter=1000,
                random_state=42,
            ),
            cv=5,
            passthrough=False,
        )
    
    def _setup_balancing(self):
        """Configura método de balanceamento"""
        if not self.balance_data:
            return None
        
        if self.balance_method == 'smote':
            # SMOTE: sobreamostragem sintética
            return SMOTE(random_state=42)
        
        elif self.balance_method == 'undersample':
            # Subamostragem da classe majoritária
            return RandomUnderSampler(random_state=42)
        
        elif self.balance_method == 'combined':
            # Combina ambos
            from imblearn.combine import SMOTEENN
            return SMOTEENN(random_state=42)
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """Extrai features usando 2DPCA"""
        projected = self.tdpca.transform(images)
        features = projected.reshape(projected.shape[0], -1)
        return features
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'EPBRecognitionSystem':
        """Treina o sistema completo"""
        logger.info("Iniciando treinamento do sistema...")
        
        # 1. Treina 2DPCA
        logger.info("Treinando 2DPCA para extração de features...")
        self.tdpca.fit(X_train)
        var_explained = np.sum(self.tdpca.explained_variance_ratio)
        logger.info(f"Variância explicada: {var_explained:.2%}")
        
        # 2. Extrai features
        logger.info("Extraindo features...")
        features_train = self.extract_features(X_train)
        logger.info(f"Shape das features: {features_train.shape}")
        
        # 3. Normaliza
        logger.info("Normalizando features...")
        features_train = self.scaler.fit_transform(features_train)
        
        # 4. Cria ensemble
        logger.info("Criando classificador ensemble...")
        ensemble = self._create_ensemble(y_train)
        
        # 5. Monta pipeline imblearn (sampler + classificador)
        #    Garante que o SMOTE só é aplicado no fit, nunca no predict,
        #    e durante CV interna o SMOTE atua apenas nos folds de treino.
        if self.balance_data:
            logger.info(f"Montando pipeline com balanceamento ({self.balance_method})...")
            original_dist = np.bincount(y_train)
            
            self.sampler = self._setup_balancing()
            
            # Log distribuição esperada após SMOTE
            _, resampled_y = self.sampler.fit_resample(features_train, y_train)
            new_dist = np.bincount(resampled_y)
            logger.info(f"Classe 0: {original_dist[0]} → {new_dist[0]}")
            logger.info(f"Classe 1: {original_dist[1]} → {new_dist[1]}")
            
            # Pipeline imblearn: sampler + classificador
            # O sampler é re-criado para garantir estado limpo no pipeline
            self.sampler = self._setup_balancing()
            self.pipeline_ = ImbPipeline([
                ('sampler', self.sampler),
                ('classifier', ensemble),
            ])
            self.pipeline_.fit(features_train, y_train)
        else:
            self.pipeline_ = ImbPipeline([
                ('classifier', ensemble),
            ])
            self.pipeline_.fit(features_train, y_train)
        
        # Referência directa ao classificador (retrocompatibilidade)
        self.classifier = self.pipeline_.named_steps['classifier']
        logger.info("Treinamento concluído!")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prediz se há EPB"""
        features_test = self.extract_features(X_test)
        features_test = self.scaler.transform(features_test)
        return self.pipeline_.predict(features_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Retorna probabilidades"""
        features_test = self.extract_features(X_test)
        features_test = self.scaler.transform(features_test)
        return self.pipeline_.predict_proba(features_test)

    def predict_single_image(self, img_proc: np.ndarray) -> dict:
        """Prediz classe e probabilidade para uma única imagem já pré-processada (2D array)."""
        batch = img_proc[np.newaxis, ...] if img_proc.ndim == 2 else img_proc
        prob = self.predict_proba(batch)[0]
        pred = int(prob[1] >= 0.5)
        return {
            'prediction': pred,
            'label': 'EPB' if pred == 1 else 'No EPB',
            'probability': float(prob[1]),
            'prob_no_epb': float(prob[0]),
        }
    
    def predict_cascaded(self, img_raw: np.ndarray, target_size: Tuple[int, int] = (64, 64),
                         threshold_stars: int = 4500, min_mean_brightness: float = 35.0) -> CascadedPredictionResult:
        """
        Predição em cascata com controle de qualidade de céu (QC + 2DPCA).
        
        Args:
            img_raw: Imagem bruta (2D array, resolução original ou redimensionada)
            target_size: Dimensão esperada pelo modelo 2DPCA
            threshold_stars: Limiar de estrelas para validação de céu limpo
            min_mean_brightness: Brilho médio mínimo para céu aberto
            
        Returns:
            CascadedPredictionResult com status, probabilidades e diagnóstico QC.
        """
        qc = assess_sky_quality(img_raw, threshold_stars=threshold_stars,
                                min_mean_brightness=min_mean_brightness)
        
        # Se céu estiver nublado/obstruído, evita inferência errônea
        if not qc.is_valid_sky:
            return CascadedPredictionResult(
                qc=qc,
                has_epb=None,
                prob_epb=0.0,
                status='CLOUDY_OBSCURED',
                status_label='Céu Nublado / Obstruído',
                is_reliable=False
            )
            
        # Céu válido: pré-processa e infere com o ensemble 2DPCA
        if img_raw.shape != target_size:
            img_resized = np.array(Image.fromarray(img_raw).resize(target_size))
        else:
            img_resized = img_raw
            
        img_prep = preprocess_epb_image(img_resized)
        prob = float(self.predict_proba(img_prep[np.newaxis, ...])[0, 1])
        has_epb = int(prob >= 0.5)
        
        if qc.condition == 'DEGRADED':
            status = 'DEGRADED_EPB' if has_epb == 1 else 'DEGRADED_QUIET'
            status_label = "EPB Detectada (Visibilidade Parcial)" if has_epb == 1 else "Céu Quieto (Visibilidade Parcial)"
            is_reliable = False
        else:
            status = 'EPB_DETECTED' if has_epb == 1 else 'QUIET_CLEAR'
            status_label = "Bolha de Plasma (EPB) Detectada" if has_epb == 1 else "Céu Limpo Sem EPB (Quieto)"
            is_reliable = True
            
        return CascadedPredictionResult(
            qc=qc,
            has_epb=has_epb,
            prob_epb=prob,
            status=status,
            status_label=status_label,
            is_reliable=is_reliable
        )

    def predict_proba_cascaded(self, img_raw: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> Tuple[float, SkyQualityResult]:
        """Retorna (prob_epb, SkyQualityResult) via inferência em cascata."""
        res = self.predict_cascaded(img_raw, target_size=target_size)
        return res.prob_epb, res.qc
    
    def get_feature_importance(self, X_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analisa quais regiões da imagem são mais importantes
        
        Usa gradiente das features em relação à imagem original
        """
        # Extrai features
        features = self.extract_features(X_sample.reshape(1, *X_sample.shape))
        
        # Reconstrói imagem projetada
        projected = self.tdpca.transform(X_sample.reshape(1, *X_sample.shape))
        reconstructed = self.tdpca.inverse_transform(projected)[0]
        
        # Diferença mostra importância
        importance_map = np.abs(X_sample - reconstructed)
        
        return importance_map, reconstructed

    def get_shap_importance(self, X_sample: np.ndarray,
                            X_background: np.ndarray,
                            method: str = 'tree',
                            n_background: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mapa de importância baseado em SHAP (contribuição real para a classificação).

        Calcula SHAP values no espaço de features do classificador e retro-projeta
        para o espaço da imagem via componentes do 2DPCA:
            shap_map = shap_features_2d @ projection_matrix.T

        Isto mostra quais regiões da imagem realmente influenciam a decisão,
        ao contrário do erro de reconstrução (get_feature_importance).

        Args:
            X_sample: Imagem pré-processada (H, W)
            X_background: Imagens de referência para SHAP (N, H, W)
            method: 'tree' (rápido, usa XGBoost) ou 'kernel' (preciso, usa ensemble)
            n_background: Máximo de amostras de background (usado em 'kernel')

        Returns:
            shap_map: SHAP Attribution no espaço da imagem (H, W) — com sinal:
                      positivo → evidência a favor de EPB,
                      negativo → evidência contra EPB
            shap_features: Valuees SHAP no espaço de features (n_features,)
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "O pacote 'shap' é necessário para esta funcionalidade. "
                "Instale com: pip install shap"
            )

        # Extrai e normaliza features da amostra
        features_sample = self.scaler.transform(
            self.extract_features(X_sample.reshape(1, *X_sample.shape))
        )

        if method == 'tree':
            # TreeExplainer num modelo tree-based do ensemble.
            # Tenta XGBoost primeiro (peso mais alto); se falhar por
            # incompatibilidade de versão, usa RandomForest.
            tree_models = [
                ('xgboost', self.classifier.named_estimators_['xgboost']),
                ('rf', self.classifier.named_estimators_['rf']),
            ]
            for model_name, tree_model in tree_models:
                try:
                    explainer = shap.TreeExplainer(tree_model)
                    shap_values = explainer.shap_values(features_sample)
                    break
                except (ValueError, TypeError):
                    logger.warning(
                        f"TreeExplainer falhou com {model_name}, "
                        f"tentando próximo modelo..."
                    )
                    continue
            else:
                raise RuntimeError(
                    "TreeExplainer falhou com todos os modelos do ensemble. "
                    "Use method='kernel' como alternativa."
                )
            # shap_values: lista [class0, class1] ou ndarray (várias formas)
            n_feats = features_sample.shape[1]
            if isinstance(shap_values, list):
                sv = np.asarray(shap_values[1]).flatten()[:n_feats]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    # (n_classes, n_samples, n_features) ou (n_samples, n_features, n_classes)
                    if shap_values.shape[0] == 2:
                        sv = shap_values[1][0]
                    elif shap_values.shape[-1] == 2:
                        sv = shap_values[0, :, 1]
                    else:
                        sv = shap_values[0].flatten()[:n_feats]
                else:
                    sv = shap_values.flatten()[:n_feats]
            else:
                sv = np.asarray(shap_values).flatten()[:n_feats]

        elif method == 'kernel':
            # KernelExplainer no ensemble completo (mais preciso, mais lento)
            if X_background is None:
                raise ValueError("X_background é obrigatório para method='kernel'")
            features_bg = self.scaler.transform(self.extract_features(X_background))
            if len(features_bg) > n_background:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(features_bg), n_background, replace=False)
                features_bg = features_bg[idx]
            bg_summary = shap.kmeans(features_bg, min(50, len(features_bg)))
            explainer = shap.KernelExplainer(
                self.classifier.predict_proba, bg_summary
            )
            shap_values = explainer.shap_values(features_sample, nsamples='auto')
            sv = shap_values[1][0]  # classe 1 (EPB)

        else:
            raise ValueError(f"Método SHAP inválido: '{method}'. Use 'tree' ou 'kernel'.")

        # Retro-projeção para espaço da imagem (matematicamente correta para 2DPCA + Scaler)
        # O modelo opera em Y_scaled = (Y - E[Y]) / sigma = (X @ P - E[X @ P]) / sigma
        # Distribuímos a atribuição SHAP (sv) proporcionalmente à variação de cada pixel
        H = X_sample.shape[0]
        
        if X_background is not None:
            E_X = np.mean(X_background, axis=0)
            features_bg = self.extract_features(X_background)
            features_bg_scaled = self.scaler.transform(features_bg)
            E_Y_scaled = np.mean(features_bg_scaled, axis=0)
        else:
            # Fallback caso X_background seja None
            E_X = np.zeros_like(X_sample)
            E_Y_scaled = np.zeros_like(sv)

        delta_Y_scaled = features_sample[0] - E_Y_scaled
        
        # Evita divisão por zero
        epsilon = 1e-8
        weight = sv / (delta_Y_scaled + epsilon * np.sign(delta_Y_scaled + 1e-15))
        
        # Desfaz o flatten e o scaler
        weight_2d = weight.reshape(H, self.n_components)
        scale_2d = self.scaler.scale_.reshape(H, self.n_components)
        weight_scaled_2d = weight_2d / scale_2d
        
        # Projeta de volta pelo PCA e multiplica pelo desvio do pixel na imagem real
        delta_X = X_sample - E_X
        shap_map = delta_X * (weight_scaled_2d @ self.tdpca.projection_matrix.T)

        return shap_map, sv


# ============================================================================
# PARTE 4: VISUALIZAÇÃO E INTERPRETAÇÃO
# ============================================================================

def plot_classification_examples(X_test: np.ndarray, y_test: np.ndarray,
                                 y_pred: np.ndarray, y_proba: np.ndarray,
                                 filenames_test: Optional[List[str]] = None, 
                                 n_examples: int = 6, axes=None) -> Optional[plt.Figure]:
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, n_examples, figsize=(20, 3.2))
        if n_examples == 1:
            axes = [axes]
    
    for i in range(min(n_examples, len(X_test))):
        ax = axes[i]
        ax.imshow(X_test[i], cmap='plasma', aspect='auto')
        
        true_label = "EPB" if y_test[i] == 1 else "No EPB"
        pred_label = "EPB" if y_pred[i] == 1 else "No EPB"
        prob = (y_proba[i, 1] if y_proba.ndim == 2 else y_proba[i]) * 100
        
        color = 'green' if y_test[i] == y_pred[i] else 'red'
        title = f'Real: {true_label}\nPred: {pred_label} ({prob:.0f}%)'
        if filenames_test:
            title = f"{filenames_test[i][:15]}...\n" + title
        
        ax.set_title(title, fontsize=8, color=color)
        ax.axis('off')
        
    if fig:
        plt.tight_layout()
        return fig
    return None

def plot_2dpca_variance(system: 'EPBRecognitionSystem', axes=None, save_path: Optional[str] = None, dpi: int = 300, max_display_components: int = 30, cv_component_results: Optional[List[dict]] = None) -> Optional[plt.Figure]:
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        
    all_var = getattr(system.tdpca, 'all_explained_variance_ratio', None)
    if all_var is None or len(all_var) == 0:
        all_var = system.tdpca.explained_variance_ratio
    n_show = min(max_display_components, len(all_var))
    var_show = all_var[:n_show]
    comp_indices = np.arange(1, n_show + 1)
    k_sel = system.n_components
    
    # Variância por componente (destacando os k selecionados)
    ax = axes[0]
    bar_colors = ['#1d4ed8' if i <= k_sel else '#94a3b8' for i in comp_indices]
    ax.bar(comp_indices, var_show, color=bar_colors, alpha=0.85, edgecolor='none')
    ax.axvline(x=k_sel + 0.5, color='#dc2626', linestyle='--', linewidth=1.8,
               label=f'Selected k={k_sel} ({np.sum(all_var[:k_sel])*100:.1f}% var)')
    ax.set_xlabel('Number of Components', fontsize=10)
    ax.set_ylabel('Explained Variance Ratio', fontsize=10)
    ax.set_title('2DPCA Eigenvalue Spectrum (Selected vs. Residual)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # Variância cumulativa (e F1-Score de CV opcional no eixo secundário)
    ax = axes[1]
    cumulative = np.cumsum(var_show)
    ax.plot(comp_indices, cumulative, 'o-', color='#1d4ed8', linewidth=2, markersize=4,
            label='Cumulative Explained Variance')
    ax.axvline(x=k_sel, color='#dc2626', linestyle='--', linewidth=1.8,
               label=f'Selected k={k_sel} ({cumulative[min(k_sel-1, len(cumulative)-1)]*100:.1f}%)')
    ax.axhline(y=0.90, color='#16a34a', linestyle=':', linewidth=1.5, alpha=0.85, label='90% Variance Threshold')
    ax.axhline(y=0.95, color='#d97706', linestyle=':', linewidth=1.5, alpha=0.85, label='95% Variance Threshold')
    
    if cv_component_results:
        ax_f1 = ax.twinx()
        c_nums = [r['n_components'] for r in cv_component_results if r['n_components'] <= n_show]
        c_f1s = [r['f1_score'] for r in cv_component_results if r['n_components'] <= n_show]
        ax_f1.plot(c_nums, c_f1s, 's-', color='#7c3aed', linewidth=2, markersize=6, label='5-Fold CV F1-Score')
        ax_f1.set_ylabel('Cross-Validated F1-Score', fontsize=10, color='#7c3aed')
        ax_f1.tick_params(axis='y', labelcolor='#7c3aed')
        ax_f1.set_ylim(0.80, 1.02)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_f1.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8.5)
    else:
        ax.legend(loc='lower right', fontsize=8.5)
        
    ax.set_xlabel('Number of Components', fontsize=10)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=10)
    ax.set_ylim(0.25, 1.03)
    ax.set_title('Cumulative Variance & Discriminative Subspace', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    if fig:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Figure saved: {save_path} ({dpi} DPI)")
        return fig
    return None

def plot_confusion_matrix_analysis(y_test: np.ndarray, y_pred: np.ndarray, ax=None, save_path: Optional[str] = None, dpi: int = 300) -> Optional[plt.Figure]:
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No EPB', 'EPB'],
                yticklabels=['No EPB', 'EPB'], ax=ax, 
                cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_xlabel('Predicted Class', fontsize=10)
    
    if fig:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path} ({dpi} DPI)")
        return fig
    return None

def plot_performance_metrics(y_test: np.ndarray, y_proba: np.ndarray, axes=None, save_path: Optional[str] = None, dpi: int = 300) -> Optional[plt.Figure]:
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
    y_p1 = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    
    # ROC Curve
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_test, y_p1)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('ROC Curve', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    # Precision-Recall Curve
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_test, y_p1)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color='purple', lw=2,
            label=f'PR (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Precision-Recall Curve', fontsize=11, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    
    # Distribuição de probabilidades
    ax = axes[2]
    epb_probs = y_p1[y_test == 1]
    no_epb_probs = y_p1[y_test == 0]
    
    ax.hist(no_epb_probs, bins=20, alpha=0.6, label='No EPB (real)', 
            color='blue', density=True)
    ax.hist(epb_probs, bins=20, alpha=0.6, label='EPB (real)', 
            color='red', density=True)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
               label='Threshold (0.5)')
    ax.set_xlabel('EPB Probability', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Predicted Probabilities Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if fig:
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Gráfico salvo: {save_path} ({dpi} DPI)")
        return fig
    return None

def plot_feature_interpretation(system: 'EPBRecognitionSystem', X_test: np.ndarray, y_test: np.ndarray, axes=None) -> Optional[plt.Figure]:
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
        
    # Pega exemplos de cada classe
    epb_idx = np.where(y_test == 1)[0][0]
    no_epb_idx = np.where(y_test == 0)[0][0]
    
    for i, (idx, title_prefix) in enumerate([(epb_idx, 'EPB'), (no_epb_idx, 'No EPB')]):
        # Imagem original
        ax = axes[i*3]
        ax.imshow(X_test[idx], cmap='plasma')
        ax.set_title(f'{title_prefix}: Original', fontsize=10)
        ax.axis('off')
        
        # Mapa de importância
        importance_map, reconstructed = system.get_feature_importance(X_test[idx])
        
        ax = axes[i*3+1]
        im = ax.imshow(importance_map, cmap='hot')
        ax.set_title(f'{title_prefix}: Importance', fontsize=10)
        ax.axis('off')
        
        # Reconstrução
        ax = axes[i*3+2]
        ax.imshow(reconstructed, cmap='plasma')
        ax.set_title(f'{title_prefix}: Reconstructed', fontsize=10)
        ax.axis('off')
        
    if fig:
        plt.tight_layout()
        return fig
    return None

def plot_pca_components_visual(system: 'EPBRecognitionSystem', X_test: np.ndarray, n_components: int = 6, axes=None) -> Optional[plt.Figure]:
    n_comp_to_show = min(n_components, system.n_components)
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, n_comp_to_show, figsize=(3 * n_comp_to_show, 4))
        if n_comp_to_show == 1:
            axes = [axes]
            
    variance_ratio = system.tdpca.explained_variance_ratio
    for i in range(n_comp_to_show):
        ax = axes[i]
        component = system.tdpca.projection_matrix[:, i].reshape(-1, 1)
        # Expande para visualização
        component_img = np.tile(component, (1, X_test.shape[1]))
        ax.imshow(component_img.T, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Comp {i+1}\n({variance_ratio[i]:.1%})', fontsize=9)
        ax.axis('off')
        
    if fig:
        plt.tight_layout()
        return fig
    return None

def plot_comprehensive_analysis(system: 'EPBRecognitionSystem', X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                y_pred: np.ndarray, y_proba: np.ndarray,
                                filenames_test: Optional[List[str]] = None) -> plt.Figure:
    """Análise visual completa do sistema (Refatorada)"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 6, hspace=1, wspace=0.4)
    
    # ROW 1: EXEMPLOS DE CLASSIFICAÇÃO
    axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(6)]
    plot_classification_examples(X_test, y_test, y_pred, y_proba, filenames_test, axes=axes_row1)
    
    # ROW 2: ANÁLISE DE FEATURES E MATRIZ DE CONFUSÃO
    ax_var1 = fig.add_subplot(gs[1, 0:2])
    ax_var2 = fig.add_subplot(gs[1, 2:4])
    plot_2dpca_variance(system, axes=[ax_var1, ax_var2])
    
    ax_cm = fig.add_subplot(gs[1, 4:6])
    plot_confusion_matrix_analysis(y_test, y_pred, ax=ax_cm)
    
    # ROW 3: MÉTRICAS DE PERFORMANCE
    axes_row3 = [fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[2, 2:4]), fig.add_subplot(gs[2, 4:6])]
    plot_performance_metrics(y_test, y_proba, axes=axes_row3)
    
    # ROW 4: FEATURE INTERPRETATION
    axes_row4 = [fig.add_subplot(gs[3, i]) for i in range(6)]
    plot_feature_interpretation(system, X_test, y_test, axes=axes_row4)
    
    # ROW 5: COMPONENTES PRINCIPAIS
    n_comp = min(6, system.n_components)
    axes_row5 = [fig.add_subplot(gs[4, i]) for i in range(n_comp)]
    plot_pca_components_visual(system, X_test, n_components=n_comp, axes=axes_row5)
    
    plt.suptitle('🌌 Comprehensive Analysis: EPB Recognition System', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def explain_feature_importance(system: 'EPBRecognitionSystem', X_sample: np.ndarray, y_sample: int,
                               filename: Optional[str] = None,
                               mode: str = 'full', image_folder: str = 'img-teste',
                               save_path: Optional[str] = None, dpi: int = 300,
                               show_filename: bool = False) -> np.ndarray:
    """
    Explica quais regiões da imagem contribuem para a reconstrução 2DPCA e resíduo espacial.
    """
    importance_map, reconstructed = system.get_feature_importance(X_sample)
    
    # Predição (uma só chamada)
    prob = system.predict_proba(X_sample.reshape(1, *X_sample.shape))[0]
    pred = np.argmax(prob)
    
    suptitle = (f'2DPCA Subspace Analysis: {"EPB" if y_sample == 1 else "No EPB"} '
                f'(Predicted: {"EPB" if pred == 1 else "No EPB"}, {prob[1]*100:.1f}%)'
                f'{" — " + filename if (filename and show_filename) else ""}')
    
    # --- Modo FULL: análise textual + visualização completa ---
    if mode == 'full':
        print("\n" + "="*70)
        print("🔍 FEATURE INTERPRETATION")
        print("="*70)
        
        print(f"\n📸 Imagem: {filename if filename else 'N/A'}")
        print(f"   • True class: {'EPB' if y_sample == 1 else 'No EPB'}")
        print(f"   • Prediction: {'EPB' if pred == 1 else 'No EPB'}")
        print(f"   • Probabilidade EPB: {prob[1]*100:.2f}%")
        print(f"   • Confidence: {'Alta' if max(prob) > 0.8 else 'Média' if max(prob) > 0.6 else 'Baixa'}")
        
        threshold = np.percentile(importance_map, 90)
        important_pixels = importance_map > threshold
        
        h, w = importance_map.shape
        quadrants = {
            'Upper Left': important_pixels[:h//2, :w//2].sum(),
            'Upper Right': important_pixels[:h//2, w//2:].sum(),
            'Lower Left': important_pixels[h//2:, :w//2].sum(),
            'Lower Right': important_pixels[h//2:, w//2:].sum()
        }
        
        print(f"\n📊 Most important regions (critical pixels):")
        total = sum(quadrants.values())
        for region, count in sorted(quadrants.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"   • {region}: {count} pixels ({percentage:.1f}%)")
        
        print(f"\n💡 Interpretation:")
        if pred == 1:
            print("   ✓ The model identified characteristic EPB patterns:")
            print("     - Vertical/irregular structures")
            print("     - Localized plasma depletions")
            print("     - Typical spatial distribution of bubbles")
        else:
            print("   ✓ The model did NOT detect EPB patterns:")
            print("     - Uniform plasma distribution")
            print("     - Absence of irregular structures")
            print("     - Quiet ionosphere characteristics")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(X_sample, cmap='plasma')
        axes[0].set_title('Pre-Processed Image (X)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(importance_map, cmap='hot')
        axes[1].set_title('Reconstruction Residual |X - X̂|\n(High-frequency anomalies)',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, label='Residual Magnitude')
        
        axes[2].imshow(reconstructed, cmap='plasma')
        axes[2].set_title(f'2DPCA Reconstruction (X̂, k={system.n_components})\n(Low-rank principal subspace)',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Figure saved: {save_path} ({dpi} DPI)")
        plt.show()
    
    # --- Modo SIMPLE: imagem do disco + mapa de importância ---
    elif mode == 'simple':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        img = _load_disk_image(filename, image_folder)
        if img is None:
            img = X_sample
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Airglow Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(importance_map, cmap='hot')
        axes[1].set_title('2DPCA Reconstruction Residual |X - X̂|',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, label='Residual Magnitude')
        
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Figure saved: {save_path} ({dpi} DPI)")
        plt.show()
    
    # --- Modo COMPARISON: disco + processada + mapa ---
    elif mode == 'comparison':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        img_disk = _load_disk_image(filename, image_folder)
        if img_disk is None:
            img_disk = X_sample
        
        axes[0].imshow(img_disk, cmap='gray')
        axes[0].set_title('Original Airglow Image\n(Raw 630.0 nm Observation)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(X_sample, cmap='plasma')
        axes[1].set_title('Physics-Informed Pre-Processed (X)\n(Gaussian Background Removal + CLAHE)',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        im = axes[2].imshow(importance_map, cmap='hot')
        axes[2].set_title('2DPCA Reconstruction Residual |X - X̂|\n(Unexplained Spatial Variance)',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, label='Residual Magnitude')
        
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"💾 Figure saved: {save_path} ({dpi} DPI)")
        plt.show()
    
    else:
        raise ValueError(f"Modo inválido: '{mode}'. Use 'full', 'simple' ou 'comparison'.")
    
    return importance_map


def _load_disk_image(filename: Optional[str], image_folder: str) -> Optional[np.ndarray]:
    """Carrega a imagem original do disco (sem pré-processamento)."""
    if not filename:
        return None
    candidate = Path(image_folder) / filename
    if candidate.exists():
        try:
            return np.array(Image.open(candidate).convert('L'))
        except Exception:
            return None
    return None


# Aliases para retrocompatibilidade
def explain_feature_importance_2(system: 'EPBRecognitionSystem', X_sample: np.ndarray, y_sample: int,
                                 filename: Optional[str] = None, image_folder: str = 'img-teste',
                                 save_path: Optional[str] = None, dpi: int = 300,
                                 show_filename: bool = False) -> np.ndarray:
    return explain_feature_importance(system, X_sample, y_sample, filename,
                                      mode='simple', image_folder=image_folder,
                                      save_path=save_path, dpi=dpi, show_filename=show_filename)

def explain_feature_importance_3(system: 'EPBRecognitionSystem', X_sample: np.ndarray, y_sample: int,
                                 filename: Optional[str] = None, image_folder: str = 'img-teste',
                                 save_path: Optional[str] = None, dpi: int = 300,
                                 show_filename: bool = False) -> np.ndarray:
    return explain_feature_importance(system, X_sample, y_sample, filename,
                                      mode='comparison', image_folder=image_folder,
                                      save_path=save_path, dpi=dpi, show_filename=show_filename)


# ============================================================================
# PARTE 5: ANÁLISE TEMPORAL DE SEQUÊNCIA DE IMAGES
# ============================================================================

def analyze_temporal_sequence(system: 'EPBRecognitionSystem', image_folder: str,
                              target_size: Tuple[int, int] = (64, 64),
                              show_thumbnails: bool = True, thumbnail_interval: int = 10,
                              use_qc: bool = True, threshold_stars: int = 4500,
                              min_mean_brightness: float = 35.0,
                              save_path: Optional[str] = None,
                              dpi: int = 300,
                              theme: str = 'light') -> Optional[pd.DataFrame]:
    """
    Analisa uma sequência temporal de imagens com suporte a controle de qualidade (QC) e detecção de EPB.
    
    Esta função é útil para:
    - Verificar se o modelo detecta EPBs antes de serem visíveis
    - Filtrar noites/períodos nublados (Estágio 1 - QC) para evitar falsos positivos
    - Analisar a evolução temporal de eventos de EPB
    - Identificar o momento de início/fim de perturbações ionosféricas
    
    Args:
        system: Sistema EPBRecognitionSystem treinado
        image_folder: Pasta com imagens ordenadas cronologicamente
        target_size: Tamanho para redimensionar imagens (deve ser igual ao treino)
        show_thumbnails: Se True, mostra thumbnails das imagens no gráfico
        thumbnail_interval: Intervalo entre thumbnails (ex: 10 = mostra a cada 10 imagens)
        use_qc: Se True, aplica Estágio 1 (QC) para descartar imagens nubladas
        threshold_stars: Limiar de estrelas para validação de céu limpo
        min_mean_brightness: Brilho médio mínimo para descartar céu totalmente encoberto
        save_path: Caminho opcional para salvar a figura
        dpi: Resolução DPI da figura
        theme: 'light' (fundo claro para publicações) ou 'dark' (fundo escuro)
    
    Returns:
        DataFrame com timestamps, probabilidades, predições e diagnósticos QC
    """
    print("="*70)
    print("🕐 TEMPORAL SEQUENCE ANALYSIS OF IMAGES (Cascaded QC + 2DPCA)")
    print("="*70)
    
    image_folder = Path(image_folder)
    
    # Lista e ordena imagens
    image_files = sorted(list(image_folder.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"⚠️  No image found in: {image_folder}")
        return None
    
    print(f"📂 Folder: {image_folder}")
    print(f"📸 Total images: {len(image_files)}")
    print(f"🛡️  Sky Quality Control (QC): {'Enabled' if use_qc else 'Disabled'}")
    
    # Processa cada imagem
    results = []
    
    for img_path in image_files:
        try:
            # Extrai timestamp do nome do arquivo (formato: O6_XX_YYYYMMDD_HHMMSS.png)
            match = re.search(r'(\d{8})_(\d{6})', img_path.name)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            else:
                timestamp = None
            
            # Carrega imagem bruta em resolução total para análise de estrelas
            img_pil = Image.open(img_path).convert('L')
            img_raw = np.array(img_pil)
            
            if use_qc:
                qc = assess_sky_quality(img_raw, threshold_stars=threshold_stars,
                                        min_mean_brightness=min_mean_brightness)
                
                if not qc.is_valid_sky:
                    prob = [1.0, 0.0]
                    pred = 0
                    pred_label = 'Cloudy Sky (QC)'
                    state_code = 2  # 0=Quiet, 1=EPB, 2=Cloudy
                else:
                    img_resized = np.array(img_pil.resize(target_size))
                    img_processed = preprocess_epb_image(img_resized)
                    prob = system.predict_proba(img_processed.reshape(1, *img_processed.shape))[0]
                    pred = int(prob[1] >= 0.5)
                    pred_label = 'EPB' if pred == 1 else 'No EPB'
                    state_code = 1 if pred == 1 else 0
            else:
                qc = SkyQualityResult('NOT_EVALUATED', True, 0, float(np.mean(img_raw)), float(np.std(img_raw)), 1.0, '')
                img_resized = np.array(img_pil.resize(target_size))
                img_processed = preprocess_epb_image(img_resized)
                prob = system.predict_proba(img_processed.reshape(1, *img_processed.shape))[0]
                pred = int(prob[1] >= 0.5)
                pred_label = 'EPB' if pred == 1 else 'No EPB'
                state_code = 1 if pred == 1 else 0
            
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'timestamp': timestamp,
                'prob_no_epb': prob[0],
                'prob_epb': prob[1],
                'prediction': pred,
                'prediction_label': pred_label,
                'state_code': state_code,
                'sky_condition': qc.condition,
                'is_valid_sky': qc.is_valid_sky,
                'star_pixels': qc.star_pixels,
                'raw_mean': qc.raw_mean,
            })
            
        except Exception as e:
            logger.warning(f"Erro em {img_path.name}: {str(e)}")
    
    # Cria DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        logger.error("Nenhuma imagem processada com sucesso.")
        return None
    
    # Estatísticas
    n_epb = ((df['prediction'] == 1) & (df['state_code'] == 1)).sum()
    n_no_epb = ((df['prediction'] == 0) & (df['state_code'] == 0)).sum()
    n_cloudy = (df['state_code'] == 2).sum()
    
    print(f"\n📊 Prediction summary:")
    print(f"   • Clear Sky with EPB:    {n_epb} images ({n_epb/len(df)*100:.1f}%)")
    print(f"   • Clear Sky without EPB: {n_no_epb} images ({n_no_epb/len(df)*100:.1f}%)")
    if use_qc:
        print(f"   • Cloudy / Obscured Sky: {n_cloudy} images ({n_cloudy/len(df)*100:.1f}%)")
    print(f"   • Average EPB probability (Clear Sky): {df[df['state_code'] != 2]['prob_epb'].mean()*100:.1f}%" if (len(df) - n_cloudy) > 0 else "   • No clear-sky images")
    
    # Identifica transições (mudanças de estado)
    df['state_change'] = df['state_code'].diff().fillna(0).abs()
    transitions = df[df['state_change'] > 0]
    
    if len(transitions) > 0:
        print(f"\n🔄 Detected transitions ({len(transitions)}):")
        for _, row in transitions.iterrows():
            time_str = row['timestamp'].strftime('%H:%M:%S') if row['timestamp'] else 'N/A'
            print(f"   • {time_str} - Changed to: {row['prediction_label']} ({row['prob_epb']*100:.1f}%)")
    
    # =========================================================================
    # VISUALIZAÇÃO ELEGANTE - Suporte a tema claro (publicação) e escuro
    # =========================================================================
    
    _rc_backup = plt.rcParams.copy()
    plt.style.use('default')
    
    if theme == 'dark':
        COLORS = {
            'background': '#0d1117',
            'card_bg': '#161b22',
            'text': '#e6edf3',
            'text_secondary': '#8b949e',
            'line_main': '#58a6ff',
            'fill_gradient_top': '#58a6ff',
            'fill_gradient_bottom': '#1f6feb',
            'threshold': '#3fb950',
            'confidence_medium': '#d29922',
            'confidence_high': '#f85149',
            'transition': '#a371f7',
            'grid': '#30363d',
            'epb_positive': '#f85149',
            'epb_negative': '#3fb950',
            'cloudy': '#8b949e',
        }
    else:
        COLORS = {
            'background': '#ffffff',
            'card_bg': '#f8fafc',
            'text': '#0f172a',
            'text_secondary': '#334155',
            'line_main': '#0284c7',
            'fill_gradient_top': '#38bdf8',
            'fill_gradient_bottom': '#0284c7',
            'threshold': '#15803d',
            'confidence_medium': '#b45309',
            'confidence_high': '#b91c1c',
            'transition': '#7c3aed',
            'grid': '#cbd5e1',
            'epb_positive': '#dc2626',
            'epb_negative': '#16a34a',
            'cloudy': '#64748b',
        }
    
    fig = plt.figure(figsize=(18, 11), facecolor=COLORS['background'])
    
    if show_thumbnails:
        gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.8, 1], hspace=0.85,
                              left=0.06, right=0.94, top=0.88, bottom=0.08)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 0.8], hspace=0.85,
                              left=0.06, right=0.94, top=0.88, bottom=0.08)
    
    # -------------------------------------------------------------------------
    # GRÁFICO PRINCIPAL: Probabilidade ao longo do tempo
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0], facecolor=COLORS['card_bg'])
    
    if df['timestamp'].notna().all():
        x_values = df['timestamp']
        x_label = 'Time (UT)'
    else:
        x_values = range(len(df))
        x_label = 'Image Index'
    
    prob_values = df['prob_epb'] * 100
    
    # Preenchimento
    ax1.fill_between(x_values, 0, prob_values, alpha=0.18, color=COLORS['fill_gradient_top'])
    ax1.fill_between(x_values, 0, prob_values * 0.7, alpha=0.15, color=COLORS['fill_gradient_bottom'])
    
    # Linha principal
    ax1.plot(x_values, prob_values, color=COLORS['line_main'], linewidth=3.5, alpha=0.25)
    ax1.plot(x_values, prob_values, color=COLORS['line_main'], linewidth=2.2, 
             label='EPB Probability (Clear Sky)', zorder=5)
    
    # Pontos coloridos por estado
    point_colors = []
    for sc, p in zip(df['state_code'], prob_values):
        if sc == 2:
            point_colors.append(COLORS['cloudy'])
        elif p >= 50:
            point_colors.append(COLORS['epb_positive'])
        else:
            point_colors.append(COLORS['epb_negative'])
            
    ax1.scatter(x_values, prob_values, c=point_colors, s=16, alpha=0.85, zorder=6, edgecolors='none')
    
    # Linhas de referência
    ax1.axhline(y=50, color=COLORS['threshold'], linestyle='--', linewidth=2, 
                label='Decision Threshold (50%)', alpha=0.9)
    ax1.axhline(y=70, color=COLORS['confidence_medium'], linestyle=':', linewidth=1.8, 
                label='Medium Confidence (70%)', alpha=0.85)
    ax1.axhline(y=90, color=COLORS['confidence_high'], linestyle=':', linewidth=1.8, 
                label='High Confidence (90%)', alpha=0.85)
    
    # Marca transições
    for _, row in transitions.iterrows():
        if df['timestamp'].notna().all():
            x_pos = row['timestamp']
        else:
            x_pos = df[df['filename'] == row['filename']].index[0]
        ax1.axvline(x=x_pos, color=COLORS['transition'], linestyle='-', alpha=0.45, linewidth=1.5)
    
    ax1.set_xlabel(x_label, fontsize=12, color=COLORS['text'], fontweight='medium', labelpad=10)
    ax1.set_ylabel('EPB Probability (%)', fontsize=12, color=COLORS['text'], 
                   fontweight='medium', labelpad=10)
    ax1.set_ylim(-2, 105)
    ax1.set_xlim(x_values.iloc[0] if hasattr(x_values, 'iloc') else x_values[0], 
                 x_values.iloc[-1] if hasattr(x_values, 'iloc') else x_values[-1])
    
    ax1.grid(True, linestyle='-', alpha=0.45 if theme == 'light' else 0.2, color=COLORS['grid'], linewidth=0.7)
    ax1.set_axisbelow(True)
    
    ax1.tick_params(axis='both', colors=COLORS['text_secondary'], labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)
    
    if df['timestamp'].notna().all():
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    legend = ax1.legend(loc='upper left', fontsize=10, framealpha=0.95,
                        facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
                        labelcolor=COLORS['text'], bbox_to_anchor=(0.01, 0.99))
    legend.get_frame().set_linewidth(0.8)
    
    ax1.set_title('Temporal Evolution of EPB Probability (with Sky Quality Gatekeeper)', fontsize=14, 
                  color=COLORS['text'], fontweight='bold', pad=15, loc='left')
    
    # -------------------------------------------------------------------------
    # GRÁFICO SECUNDÁRIO: Timeline de classificação (3 Estados)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], facecolor=COLORS['card_bg'])
    
    bar_colors = []
    for sc in df['state_code']:
        if sc == 2:
            bar_colors.append(COLORS['cloudy'])
        elif sc == 1:
            bar_colors.append(COLORS['epb_positive'])
        else:
            bar_colors.append(COLORS['epb_negative'])
            
    bars = ax2.bar(range(len(df)), [1] * len(df), color=bar_colors, alpha=0.85, width=1.0, 
                   edgecolor='none')
    
    for bar, sc, prob in zip(bars, df['state_code'], df['prob_epb']):
        if sc == 2:
            bar.set_alpha(0.65)
        else:
            bar.set_alpha(0.35 + 0.6 * abs(prob - 0.5) * 2)
    
    ax2.set_ylabel('Status', fontsize=11, color=COLORS['text'], fontweight='medium', labelpad=10)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels([''])
    ax2.set_xlabel('Image Sequence Index', fontsize=11, color=COLORS['text'], 
                   fontweight='medium', labelpad=10)
    ax2.set_xlim(-0.5, len(df) - 0.5)
    ax2.set_ylim(0, 1)
    
    ax2.tick_params(axis='both', colors=COLORS['text_secondary'], labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.8)
    
    ax2.set_title('Sky Condition & EPB Classification Timeline', fontsize=12, 
                  color=COLORS['text'], fontweight='bold', pad=10, loc='left')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['epb_negative'], alpha=0.85, label='Clear Sky - No EPB'),
        Patch(facecolor=COLORS['epb_positive'], alpha=0.85, label='Clear Sky - With EPB'),
        Patch(facecolor=COLORS['cloudy'], alpha=0.85, label='Cloudy / Obscured (QC)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95,
               facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    # -------------------------------------------------------------------------
    # THUMBNAILS (se habilitado)
    # -------------------------------------------------------------------------
    if show_thumbnails:
        ax3 = fig.add_subplot(gs[2], facecolor=COLORS['background'])
        ax3.axis('off')
        ax3.set_title('Representative All-Sky Image Samples', fontsize=12, 
                      color=COLORS['text'], fontweight='bold', pad=10, loc='left')
        
        indices = list(range(0, len(df), thumbnail_interval))
        if len(indices) > 14:
            step = len(indices) // 14
            indices = indices[::step][:14]
        
        n_thumbs = len(indices)
        thumb_width = 1.0 / n_thumbs
        
        for i, idx in enumerate(indices):
            row_data = df.iloc[idx]
            try:
                img = Image.open(row_data['filepath']).convert('L')
                img = np.array(img.resize((64, 64)))
                
                left = i * thumb_width + 0.005
                bottom = 0.15
                width = thumb_width - 0.01
                height = 0.65
                
                ax_thumb = ax3.inset_axes([left, bottom, width, height])
                ax_thumb.imshow(img, cmap='gray', aspect='equal')
                ax_thumb.axis('off')
                
                if row_data['state_code'] == 2:
                    border_color = COLORS['cloudy']
                elif row_data['prediction'] == 1:
                    border_color = COLORS['epb_positive']
                else:
                    border_color = COLORS['epb_negative']
                    
                for spine in ax_thumb.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)
                
                time_label = row_data['timestamp'].strftime('%H:%M') if row_data['timestamp'] else str(idx)
                if row_data['state_code'] == 2:
                    prob_label = "CLOUD"
                else:
                    prob_label = f"{row_data['prob_epb']*100:.0f}%"
                
                ax_thumb.text(0.5, -0.08, time_label, transform=ax_thumb.transAxes,
                             fontsize=8.5, color=COLORS['text'], ha='center', va='top',
                             fontweight='bold')
                ax_thumb.text(0.5, -0.28, prob_label, transform=ax_thumb.transAxes,
                             fontsize=7.5, color=COLORS['text_secondary'], ha='center', va='top',
                             fontweight='medium')
                
            except Exception:
                pass
    
    # -------------------------------------------------------------------------
    # TÍTULO GERAL
    # -------------------------------------------------------------------------
    if df['timestamp'].notna().all():
        date_str = df['timestamp'].iloc[0].strftime('%Y-%m-%d')
        time_range = f"{df['timestamp'].iloc[0].strftime('%H:%M')} – {df['timestamp'].iloc[-1].strftime('%H:%M')} UT"
        title_text = 'Nocturnal Temporal Analysis of Equatorial Plasma Bubbles'
        subtitle_text = f'{date_str}  •  {time_range}  •  {len(df)} All-Sky Frames'
    else:
        title_text = 'Nocturnal Temporal Analysis of Equatorial Plasma Bubbles'
        subtitle_text = f'{len(df)} analyzed images'
    
    fig.text(0.5, 0.96, title_text, ha='center', va='top', fontsize=17, 
             color=COLORS['text'], fontweight='bold')
    fig.text(0.5, 0.925, subtitle_text, ha='center', va='top', fontsize=11, 
             color=COLORS['text_secondary'], fontweight='normal')
    
    # Adiciona estatísticas no canto
    stats_text = f"EPB: {n_epb}/{len(df)}  |  Cloudy: {n_cloudy}/{len(df)}  |  Avg Prob: {df['prob_epb'].mean()*100:.1f}%"
    fig.text(0.94, 0.96, stats_text, ha='right', va='top', fontsize=10, 
             color=COLORS['text'], fontweight='medium', 
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['card_bg'], 
                      edgecolor=COLORS['grid'], linewidth=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"💾 High-resolution timeline saved: {save_path} ({dpi} DPI)")
    
    plt.show()
    plt.rcParams.update(_rc_backup)
    
    print("\n" + "="*70)
    print("✅ TEMPORAL ANALYSIS COMPLETED")
    print("="*70)
    
    return df


def analyze_critical_periods(df_temporal: pd.DataFrame) -> None:
    """
    Analisa períodos críticos de um DataFrame temporal retornado por analyze_temporal_sequence.
    
    Detecta transições, analisa o início de eventos EPB e gera estatísticas da noite.
    
    Args:
        df_temporal: DataFrame retornado por analyze_temporal_sequence (deve conter
                     colunas: timestamp, prob_epb, prediction, prediction_label, state_change)
    """
    from datetime import timedelta
    
    if df_temporal is None or df_temporal.empty:
        print("⚠️  Empty DataFrame. Run analyze_temporal_sequence first.")
        return
    
    print("="*70)
    print("🔍 CRITICAL PERIODS ANALYSIS")
    print("="*70)
    
    # Detecta a data da observação
    if df_temporal['timestamp'].notna().all():
        obs_date = df_temporal['timestamp'].iloc[0].strftime('%m/%d/%Y')
        time_start = df_temporal['timestamp'].iloc[0].strftime('%H:%M')
        time_end = df_temporal['timestamp'].iloc[-1].strftime('%H:%M')
        print(f"📅 Date: {obs_date} | Period: {time_start} - {time_end} UT\n")
    
    # -------------------------------------------------------------------------
    # 1. Detecta TRANSIÇÕES (mudanças de estado No EPB <-> Com EPB)
    # -------------------------------------------------------------------------
    transitions = df_temporal[df_temporal['state_change'] == 1].copy()
    
    if len(transitions) > 0:
        print("🔄 DETECTED TRANSITIONS")
        print("-"*70)
        print("   Moments where the model changed classification:\n")
        
        for i, (idx, row) in enumerate(transitions.iterrows()):
            time_str = row['timestamp'].strftime('%H:%M:%S') if row['timestamp'] else 'N/A'
            prob = row['prob_epb'] * 100
            
            if row['prediction'] == 1:
                direction = "No EPB → WITH EPB 🔴"
            else:
                direction = "WITH EPB → No EPB 🟢"
            
            print(f"   {i+1}. {time_str} UT | {direction} | Prob: {prob:.1f}%")
        
        print()
    
    # -------------------------------------------------------------------------
    # 2. Análise da PRIMEIRA TRANSIÇÃO para EPB (início do evento)
    # -------------------------------------------------------------------------
    first_epb_transition = transitions[transitions['prediction'] == 1]
    
    if len(first_epb_transition) > 0:
        first_epb = first_epb_transition.iloc[0]
        first_epb_time = first_epb['timestamp']
        
        print("⚡ EPB EVENT START ANALYSIS")
        print("-"*70)
        print(f"   First EPB detection: {first_epb_time.strftime('%H:%M:%S')} UT\n")
        
        window_before = timedelta(minutes=15)
        window_after = timedelta(minutes=15)
        
        start_window = first_epb_time - window_before
        end_window = first_epb_time + window_after
        
        mask = (df_temporal['timestamp'] >= start_window) & (df_temporal['timestamp'] <= end_window)
        df_window = df_temporal[mask].copy()
        
        print(f"   Analysis window: {start_window.strftime('%H:%M')} - {end_window.strftime('%H:%M')} UT")
        print(f"   Images in window: {len(df_window)}\n")
        
        print("   Details:")
        print("   " + "-"*60)
        
        for _, row in df_window.iterrows():
            time_str = row['timestamp'].strftime('%H:%M:%S')
            prob = row['prob_epb'] * 100
            pred = row['prediction_label']
            
            if prob > 90:
                emoji = "🔴"
            elif prob > 70:
                emoji = "🟠"
            elif prob > 50:
                emoji = "🟡"
            else:
                emoji = "🟢"
            
            marker = " ◄── TRANSITION" if row['timestamp'] == first_epb_time else ""
            
            print(f"   {time_str} UT | {emoji} {prob:5.1f}% | {pred}{marker}")
        
        print("   " + "-"*60)
        
        # Análise de detecção precoce
        pre_transition = df_temporal[df_temporal['timestamp'] < first_epb_time].tail(5)
        if len(pre_transition) > 0:
            avg_prob_before = pre_transition['prob_epb'].mean() * 100
            max_prob_before = pre_transition['prob_epb'].max() * 100
            
            print(f"\n   📊 Pre-transition statistics (last 5 images):")
            print(f"      • Average probability: {avg_prob_before:.1f}%")
            print(f"      • Maximum probability: {max_prob_before:.1f}%")
            
            if max_prob_before > 40:
                print(f"\n   💡 INSIGHT: The model already indicated {max_prob_before:.1f}% probability")
                print(f"      BEFORE the classification changed to EPB!")
                print(f"      This suggests early detection of ionospheric disturbances.")
        
        print()
    
    # -------------------------------------------------------------------------
    # 3. Estatísticas gerais da noite
    # -------------------------------------------------------------------------
    print("📈 OVERALL NIGHT STATISTICS")
    print("-"*70)
    
    n_total = len(df_temporal)
    n_epb = (df_temporal['prediction'] == 1).sum()
    n_no_epb = (df_temporal['prediction'] == 0).sum()
    
    print(f"   • Total images: {n_total}")
    print(f"   • With EPB detected: {n_epb} ({n_epb/n_total*100:.1f}%)")
    print(f"   • No EPB detected: {n_no_epb} ({n_no_epb/n_total*100:.1f}%)")
    print(f"   • Number of transitions: {len(transitions)}")
    print(f"\n   • Average EPB probability: {df_temporal['prob_epb'].mean()*100:.1f}%")
    print(f"   • Maximum probability: {df_temporal['prob_epb'].max()*100:.1f}%")
    print(f"   • Minimum probability: {df_temporal['prob_epb'].min()*100:.1f}%")
    
    idx_max = df_temporal['prob_epb'].idxmax()
    peak = df_temporal.loc[idx_max]
    if peak['timestamp']:
        print(f"\n   🎯 Activity peak: {peak['timestamp'].strftime('%H:%M:%S')} UT ({peak['prob_epb']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED")
    print("="*70)


# ============================================================================
# PARTE 7: ANÁLISE DE SENSIBILIDADE AO THRESHOLD
# ============================================================================

def analyze_threshold_sensitivity(y_true: np.ndarray, y_proba: np.ndarray,
                                   thresholds: Optional[np.ndarray] = None) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Analisa como sensitivity, specificity, precision, F1 e accuracy variam
    em função do threshold de decisão.

    Gera um gráfico de curvas threshold vs. métricas e retorna um DataFrame
    com os valores para cada threshold testado.

    Args:
        y_true: Labels reais (0 ou 1)
        y_proba: Probabilidades preditas para a classe positiva (shape (n,) ou (n,2))
        thresholds: Array de thresholds a testar. Se None, usa np.arange(0.05, 1.0, 0.01)

    Returns:
        fig: Figura matplotlib com o gráfico
        df_thresholds: DataFrame com colunas [threshold, sensitivity, specificity,
                        precision, f1, accuracy]
    """
    # Aceita tanto (n,) como (n, 2)
    if y_proba.ndim == 2:
        probs = y_proba[:, 1]
    else:
        probs = y_proba

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.01)

    records = []
    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)

        records.append({
            'threshold': round(float(t), 4),
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'precision': round(prec, 4),
            'f1': round(f1, 4),
            'accuracy': round(acc, 4),
        })

    df_thresholds = pd.DataFrame(records)

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Todas as métricas vs threshold
    ax = axes[0]
    ax.plot(df_thresholds['threshold'], df_thresholds['sensitivity'],
            label='Sensitivity (Recall)', linewidth=2, color='tab:red')
    ax.plot(df_thresholds['threshold'], df_thresholds['specificity'],
            label='Specificity', linewidth=2, color='tab:blue')
    ax.plot(df_thresholds['threshold'], df_thresholds['precision'],
            label='Precision (PPV)', linewidth=2, color='tab:green')
    ax.plot(df_thresholds['threshold'], df_thresholds['f1'],
            label='F1-Score', linewidth=2, color='tab:purple')
    ax.plot(df_thresholds['threshold'], df_thresholds['accuracy'],
            label='Accuracy', linewidth=2, color='tab:orange', linestyle='--')

    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Default threshold (0.5)')

    # Marca o threshold que maximiza F1
    best_f1_idx = df_thresholds['f1'].idxmax()
    best_t = df_thresholds.loc[best_f1_idx, 'threshold']
    best_f1 = df_thresholds.loc[best_f1_idx, 'f1']
    ax.axvline(x=best_t, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.scatter([best_t], [best_f1], color='purple', s=80, zorder=5,
              label=f'Best F1={best_f1:.3f} (t={best_t:.2f})')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Decision Threshold Sensitivity', fontsize=13, fontweight='bold')
    ax.legend(loc='center left', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    # Subplot 2: Sensitivity vs Specificity (trade-off direto)
    ax2 = axes[1]
    ax2.plot(df_thresholds['threshold'], df_thresholds['sensitivity'],
             label='Sensitivity', linewidth=2.5, color='tab:red')
    ax2.plot(df_thresholds['threshold'], df_thresholds['specificity'],
             label='Specificity', linewidth=2.5, color='tab:blue')

    # Ponto de cruzamento (equilíbrio)
    diff = np.abs(df_thresholds['sensitivity'].values - df_thresholds['specificity'].values)
    eq_idx = diff.argmin()
    eq_t = df_thresholds.loc[eq_idx, 'threshold']
    eq_val = (df_thresholds.loc[eq_idx, 'sensitivity'] + df_thresholds.loc[eq_idx, 'specificity']) / 2

    ax2.axvline(x=eq_t, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.scatter([eq_t], [eq_val], color='black', s=100, zorder=5, marker='X',
               label=f'Equilibrium (t={eq_t:.2f}, ≈{eq_val:.3f})')
    ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Metric', fontsize=12)
    ax2.set_title('Trade-off Sensitivity vs. Specificity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------- Resumo textual ----------
    row_default = df_thresholds.loc[(df_thresholds['threshold'] - 0.5).abs().idxmin()]
    row_best_f1 = df_thresholds.loc[best_f1_idx]
    row_eq = df_thresholds.loc[eq_idx]

    print("\n" + "="*70)
    print("📊 ANÁLISE DE SENSIBILIDADE AO THRESHOLD")
    print("="*70)
    print(f"\n   {'Metric':<20} {'t=0.50':>10} {'Best F1':>12} {'Equilibrium':>12}")
    print("   " + "-"*56)
    print(f"   {'Threshold':<20} {row_default['threshold']:>10.2f} {row_best_f1['threshold']:>12.2f} {row_eq['threshold']:>12.2f}")
    print(f"   {'Sensitivity':<20} {row_default['sensitivity']:>10.4f} {row_best_f1['sensitivity']:>12.4f} {row_eq['sensitivity']:>12.4f}")
    print(f"   {'Specificity':<20} {row_default['specificity']:>10.4f} {row_best_f1['specificity']:>12.4f} {row_eq['specificity']:>12.4f}")
    print(f"   {'Precision':<20} {row_default['precision']:>10.4f} {row_best_f1['precision']:>12.4f} {row_eq['precision']:>12.4f}")
    print(f"   {'F1-Score':<20} {row_default['f1']:>10.4f} {row_best_f1['f1']:>12.4f} {row_eq['f1']:>12.4f}")
    print(f"   {'Accuracy':<20} {row_default['accuracy']:>10.4f} {row_best_f1['accuracy']:>12.4f} {row_eq['accuracy']:>12.4f}")

    print(f"\n   💡 Recomendação:")
    if best_t < 0.5:
        print(f"      O melhor F1 ocorre em t={best_t:.2f} (< 0.50) — o threshold padrão")
        print(f"      é conservador. Baixar o threshold aumenta a detecção de EPBs.")
    elif best_t > 0.5:
        print(f"      O melhor F1 ocorre em t={best_t:.2f} (> 0.50) — o threshold padrão")
        print(f"      gera falsos positivos desnecessários. Subir o threshold melhora a precisão.")
    else:
        print(f"      O threshold padrão (0.50) já maximiza o F1-Score.")

    print("\n" + "="*70)

    return fig, df_thresholds


# ============================================================================
# PARTE 8: ANÁLISE DE CALIBRAÇÃO DE PROBABILIDADES
# ============================================================================

def plot_calibration_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                              n_bins: int = 10,
                              system: Optional['EPBRecognitionSystem'] = None,
                              X_train_features: Optional[np.ndarray] = None,
                              y_train: Optional[np.ndarray] = None,
                              X_test_features: Optional[np.ndarray] = None) -> Tuple[plt.Figure, dict]:
    """
    Gera um reliability diagram (calibration plot) e, opcionalmente,
    compara com probabilidades calibradas via CalibratedClassifierCV.

    Args:
        y_true: Labels reais do conjunto de teste (0 ou 1)
        y_proba: Probabilidades preditas (shape (n,) ou (n, 2))
        n_bins: Número de bins para o reliability diagram
        system: (Opcional) EPBRecognitionSystem treinado — se fornecido junto com
                X_train_features e y_train, calibra o classificador e compara
        X_train_features: Features de treino já normalizadas (para calibrar)
        y_train: Labels de treino (para calibrar)
        X_test_features: Features de teste já normalizadas (para gerar probs calibradas)

    Returns:
        fig: Figura matplotlib
        stats: Dict com ECE (Expected Calibration Error) antes e depois
    """
    # Aceita tanto (n,) como (n, 2)
    if y_proba.ndim == 2:
        probs = y_proba[:, 1]
    else:
        probs = y_proba

    # --- Calibration curve original ---
    fraction_pos, mean_predicted = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')

    # ECE (Expected Calibration Error)
    bin_counts = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
    total = len(probs)
    ece = 0.0
    for i in range(len(fraction_pos)):
        weight = bin_counts[i] / total if total > 0 else 0
        ece += weight * abs(fraction_pos[i] - mean_predicted[i])

    stats = {'ece_original': round(float(ece), 4)}

    # --- Calibração opcional ---
    has_calibration = (system is not None and X_train_features is not None
                       and y_train is not None and X_test_features is not None)
    cal_fraction_pos = cal_mean_predicted = cal_probs = None
    if has_calibration:
        cal_clf = CalibratedClassifierCV(system.classifier, cv=5, method='isotonic')
        cal_clf.fit(X_train_features, y_train)
        cal_proba = cal_clf.predict_proba(X_test_features)[:, 1]
        cal_fraction_pos, cal_mean_predicted = calibration_curve(
            y_true, cal_proba, n_bins=n_bins, strategy='uniform'
        )
        cal_probs = cal_proba

        cal_bin_counts = np.histogram(cal_proba, bins=n_bins, range=(0, 1))[0]
        ece_cal = 0.0
        for i in range(len(cal_fraction_pos)):
            w = cal_bin_counts[i] / total if total > 0 else 0
            ece_cal += w * abs(cal_fraction_pos[i] - cal_mean_predicted[i])
        stats['ece_calibrated'] = round(float(ece_cal), 4)

    # --- Plot ---
    n_cols = 2 if not has_calibration else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # Subplot 1: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly calibrated')
    ax.plot(mean_predicted, fraction_pos, 's-', color='tab:red', linewidth=2,
            label=f'Original model (ECE={ece:.3f})')
    if has_calibration:
        ax.plot(cal_mean_predicted, cal_fraction_pos, 'o-', color='tab:blue', linewidth=2,
                label=f'After calibration (ECE={stats["ece_calibrated"]:.3f})')
    ax.set_xlabel('Mean predicted probability', fontsize=11)
    ax.set_ylabel('Fraction of true positives', fontsize=11)
    ax.set_title('Reliability Diagram (Calibration Plot)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    # Subplot 2: Histograma de probabilidades
    ax2 = axes[1]
    ax2.hist(probs, bins=n_bins, range=(0, 1), alpha=0.7, color='tab:red',
             edgecolor='black', label='Original')
    if has_calibration:
        ax2.hist(cal_probs, bins=n_bins, range=(0, 1), alpha=0.5, color='tab:blue',
                 edgecolor='black', label='Calibrated')
    ax2.set_xlabel('Predicted probability', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Probability Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Subplot 3 (opcional): Comparação por classe
    if has_calibration:
        ax3 = axes[2]
        for label_val, label_name, color in [(0, 'No EPB', 'tab:blue'), (1, 'EPB', 'tab:red')]:
            mask = y_true == label_val
            ax3.hist(probs[mask], bins=n_bins, range=(0, 1), alpha=0.4,
                     color=color, label=f'{label_name} (original)', edgecolor='black')
            ax3.hist(cal_probs[mask], bins=n_bins, range=(0, 1), alpha=0.4,
                     color=color, linestyle='--', histtype='step', linewidth=2,
                     label=f'{label_name} (calibrado)')
        ax3.set_xlabel('Predicted probability', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Original vs. Calibrated por Classe', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- Resumo textual ---
    print("\n" + "="*70)
    print("📊 ANÁLISE DE CALIBRAÇÃO DE PROBABILIDADES")
    print("="*70)
    print(f"\n   ECE original (Expected Calibration Error): {ece:.4f}")
    if ece < 0.05:
        print("   ✓ Probabilidades bem calibradas")
    elif ece < 0.10:
        print("   ⚠️  Calibração razoável — considere CalibratedClassifierCV")
    else:
        print("   ❌ Probabilidades mal calibradas — calibração recomendada")

    if has_calibration:
        print(f"   ECE após calibração isotónica:              {stats['ece_calibrated']:.4f}")
        improvement = ece - stats['ece_calibrated']
        if improvement > 0.005:
            print(f"   ✓ Calibração melhorou o ECE em {improvement:.4f}")
        else:
            print(f"   ⚠️  Calibração não trouxe melhoria significativa (Δ={improvement:.4f})")

    print("\n   💡 Interpretação do reliability diagram:")
    print("      Se os pontos ficam ACIMA da diagonal: modelo é subconfiante")
    print("        (diz 60%, mas na verdade 80% são positivos)")
    print("      Se os pontos ficam ABAIXO da diagonal: modelo é sobreconfiante")
    print("        (diz 80%, mas na verdade só 60% são positivos)")
    print("\n" + "="*70)

    return fig, stats


# ============================================================================
# PARTE 9: AVALIAÇÃO COM K-FOLD CROSS-VALIDATION (ESTRATIFICADO E TEMPORAL)
# ============================================================================

def _extract_temporal_groups(df_metadata: pd.DataFrame, group_by: str = 'temporal_block') -> np.ndarray:
    """Extrai identificadores de grupo (blocos temporais de 30 min, data ou estação)."""
    import re
    groups = []
    for idx, row in df_metadata.iterrows():
        fname = str(row.get('filename', ''))
        site = str(row.get('site', 'UNK'))
        date_str = str(row.get('date', '00000000'))
        m = re.search(r'_(\d{8})_(\d{2})(\d{2})(\d{2})', fname)
        if m:
            d_str, hh, mm, ss = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            block_30m = f"{site}_{d_str}_{hh:02d}_{mm // 30}"
        else:
            block_30m = f"{site}_{date_str}"
        if group_by == 'temporal_block':
            groups.append(block_30m)
        elif group_by == 'date':
            groups.append(f"{site}_{date_str}")
        elif group_by == 'site':
            groups.append(site)
        else:
            groups.append(str(idx))
    return np.array(groups)


def evaluate_kfold_cv(X: np.ndarray, y: np.ndarray,
                      n_components: int = 5,
                      n_folds: int = 5,
                      balance_method: str = 'smote',
                      use_augmentation: bool = False,
                      augment_factor: int = 2,
                      ensemble_params: Optional[dict] = None,
                      n_splits: Optional[int] = None,
                      group_by: Optional[str] = None,
                      df_metadata: Optional[pd.DataFrame] = None,
                      return_oof: bool = False,
                      save_path: Optional[str] = None,
                      dpi: int = 300,
                      random_state: int = 42):
    """
    Avaliação com k-fold cross-validation (Estratificado ou Agrupado por Bloco Temporal)
    livre de vazamento de dados (leakage-free: pré-processamento estatístico, 2DPCA,
    augmentation e SMOTE executados estritamente dentro de cada fold de treino).
    """
    from sklearn.model_selection import StratifiedGroupKFold
    if n_splits is not None:
        n_folds = n_splits

    if group_by is not None and df_metadata is not None:
        groups = _extract_temporal_groups(df_metadata, group_by=group_by)
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y, groups=groups)
        cv_label = f"{n_folds}-Fold Group CV ({group_by}, {len(np.unique(groups))} groups)"
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
        cv_label = f"{n_folds}-Fold Stratified CV"

    fold_metrics = []
    oof_y_true = np.zeros(len(y), dtype=int)
    oof_y_pred = np.zeros(len(y), dtype=int)
    oof_y_proba = np.zeros(len(y), dtype=float)

    print(f"\n{'='*70}")
    print(f"📊 CROSS-VALIDATION EVALUATION: {cv_label}")
    print(f"{'='*70}")
    print(f"   Components: {n_components} | Balancing: {balance_method} (inside each fold)")
    print(f"   Augmentation: {'Yes (factor={})'.format(augment_factor) if use_augmentation else 'No'}")
    print(f"   Total samples: {len(X)} ({int(np.sum(y))} EPB, {int(len(y) - np.sum(y))} No EPB)\n")

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Augmentation estritamente no treino do fold
        if use_augmentation:
            X_fold_train, y_fold_train = augment_epb_data(
                X_fold_train, y_fold_train, augment_factor=augment_factor
            )

        # Treina sistema completo do zero dentro do fold
        sys_fold = EPBRecognitionSystem(
            n_components=n_components,
            balance_data=True,
            balance_method=balance_method,
            ensemble_params=ensemble_params
        )
        sys_fold.fit(X_fold_train, y_fold_train)

        # Avalia no fold de validação (apenas imagens reais não aumentadas)
        y_pred_fold = sys_fold.predict(X_fold_val)
        y_proba_fold = sys_fold.predict_proba(X_fold_val)[:, 1]

        oof_y_true[val_idx] = y_fold_val
        oof_y_pred[val_idx] = y_pred_fold
        oof_y_proba[val_idx] = y_proba_fold

        cm = confusion_matrix(y_fold_val, y_pred_fold, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)

        fpr, tpr, _ = roc_curve(y_fold_val, y_proba_fold)
        roc_auc_val = auc(fpr, tpr)
        prec_c, rec_c, _ = precision_recall_curve(y_fold_val, y_proba_fold)
        pr_auc_val = auc(rec_c, prec_c)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'n_val': len(val_idx),
            'accuracy': round(acc, 4),
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'precision': round(prec, 4),
            'f1': round(f1, 4),
            'auc': round(roc_auc_val, 4),
            'pr_auc': round(pr_auc_val, 4),
        })

        print(f"   Fold {fold_idx + 1}/{n_folds} (n={len(val_idx):3d}): "
              f"Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | "
              f"Prec={prec:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc_val:.4f} | PR-AUC={pr_auc_val:.4f}")

    df_results = pd.DataFrame(fold_metrics)
    df_results.attrs['cv_label'] = cv_label

    # --- Resumo estatístico ---
    metrics_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc', 'pr_auc']

    print(f"\n{'='*70}")
    print(f"📈 SUMMARY RESULTS: {cv_label}")
    print(f"{'='*70}")
    print(f"\n   {'Metric':<20} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}")
    print(f"   {'-'*60}")

    for col in metrics_cols:
        mean_val = df_results[col].mean()
        std_val = df_results[col].std()
        min_val = df_results[col].min()
        max_val = df_results[col].max()
        print(f"   {col.upper() if 'auc' in col else col.capitalize():<20} "
              f"{mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    print(f"{'='*70}")

    # --- Plot ---
    plot_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Boxplot de todas as métricas
    ax = axes[0]
    df_melt = df_results[plot_cols].melt(var_name='Metric', value_name='Score')
    sns.boxplot(data=df_melt, x='Metric', y='Score', hue='Metric', ax=ax, palette='Set2', legend=False)
    sns.stripplot(data=df_melt, x='Metric', y='Score', ax=ax,
                  color='black', size=5, alpha=0.6)
    ax.set_title(f'Metric Distribution ({cv_label})', fontsize=12, fontweight='bold')
    ax.set_ylim(0.75, 1.03)
    ax.set_xticks(range(len(plot_cols)))
    ax.set_xticklabels([c.upper() if c == 'auc' else c.capitalize() for c in plot_cols], rotation=25, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # Subplot 2: Metrics por fold
    ax2 = axes[1]
    for col in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']:
        ax2.plot(df_results['fold'], df_results[col], 'o-',
                 label=col.upper() if col == 'auc' else col.capitalize(), linewidth=2)
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title(f'Per-Fold Performance ({cv_label})', fontsize=12, fontweight='bold')
    ax2.set_xticks(df_results['fold'])
    ax2.set_ylim(0.75, 1.03)
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Cross-validation figure saved: {save_path} ({dpi} DPI)")
    plt.show()

    if return_oof:
        return fig, df_results, {
            'y_true': oof_y_true,
            'y_pred': oof_y_pred,
            'y_proba': oof_y_proba,
        }
    return fig, df_results


# ============================================================================
# PARTE 10: BENCHMARKING CONTRA BASELINES E REDE CONVOLUCIONAL 2D (2D-CNN)
# ============================================================================

class Conv2DCNNClassifier:
    """
    Rede Neural Convolucional 2D (2D-CNN) vetorizada em NumPy/SciPy (im2col via sliding_window_view + einsum).
    Arquitetura:
      Input (1, 64, 64) -> AvgPool(2x2) [32x32]
      -> Conv2D(8 filtros, 5x5) + ReLU + MaxPool(2x2) [14x14]
      -> Conv2D(16 filtros, 3x3) + ReLU + MaxPool(2x2) [6x6]
      -> Flatten (576) -> Dense(32) + ReLU -> Dense(1, Sigmoid)
    Treinada com mini-batch Adam, ponderação de classes e regularização L2.
    """
    def __init__(self, epochs: int = 25, batch_size: int = 32, lr: float = 0.004,
                 l2_reg: float = 1e-4, random_state: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2_reg = l2_reg
        self.random_state = random_state

    @staticmethod
    def _conv2d_forward(X, W, b):
        from numpy.lib.stride_tricks import sliding_window_view
        _, _, kH, kW = W.shape
        windows = sliding_window_view(X, (kH, kW), axis=(2, 3))
        out = np.einsum('nchwkl,fckl->nfhw', windows, W, optimize=True) + b[None, :, None, None]
        return out, windows

    @staticmethod
    def _conv2d_backward(dout, windows, X_shape, W):
        from numpy.lib.stride_tricks import sliding_window_view
        dW = np.einsum('nfhw,nchwkl->fckl', dout, windows, optimize=True)
        db = dout.sum(axis=(0, 2, 3))
        _, _, kH, kW = W.shape
        dout_pad = np.pad(dout, ((0, 0), (0, 0), (kH - 1, kH - 1), (kW - 1, kW - 1)))
        W_flip = W[:, :, ::-1, ::-1]
        win_dout = sliding_window_view(dout_pad, (kH, kW), axis=(2, 3))
        dX = np.einsum('nfhwkl,fckl->nchw', win_dout, W_flip, optimize=True)
        return dX, dW, db

    @staticmethod
    def _maxpool2x2_forward(X):
        N, C, H, W = X.shape
        Xr = X.reshape(N, C, H // 2, 2, W // 2, 2)
        out = Xr.max(axis=(3, 5))
        mask = (Xr == out[:, :, :, None, :, None])
        mask = mask / np.maximum(mask.sum(axis=(3, 5), keepdims=True), 1.0)
        return out, mask

    @staticmethod
    def _maxpool2x2_backward(dout, mask):
        N, C, H2, _, W2, _ = mask.shape
        return (dout[:, :, :, None, :, None] * mask).reshape(N, C, H2 * 2, W2 * 2)

    def _preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X[:, None, :, :]
        # Reduz 64x64 -> 32x32 via average pooling 2x2 e normaliza por imagem
        N, C, H, W = X.shape
        if H == 64 and W == 64:
            X = X.reshape(N, C, 32, 2, 32, 2).mean(axis=(3, 5))
        mean = X.mean(axis=(2, 3), keepdims=True)
        std = X.std(axis=(2, 3), keepdims=True) + 1e-6
        return (X - mean) / std

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        Xp = self._preprocess_batch(X)
        y = y.astype(np.float64)

        self.W1 = rng.randn(8, 1, 5, 5) * np.sqrt(2.0 / 25.0)
        self.b1 = np.zeros(8)
        self.W2 = rng.randn(16, 8, 3, 3) * np.sqrt(2.0 / (8 * 9.0))
        self.b2 = np.zeros(16)
        self.W3 = rng.randn(16 * 6 * 6, 32) * np.sqrt(2.0 / (16 * 36.0))
        self.b3 = np.zeros(32)
        self.W4 = rng.randn(32, 1) * np.sqrt(2.0 / 32.0)
        self.b4 = np.zeros(1)

        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        m_adam = [np.zeros_like(p) for p in params]
        v_adam = [np.zeros_like(p) for p in params]
        t_step = 0

        n_pos = max(np.sum(y == 1), 1)
        n_neg = max(np.sum(y == 0), 1)
        w_pos = len(y) / (2.0 * n_pos)
        w_neg = len(y) / (2.0 * n_neg)

        for _ in range(self.epochs):
            perm = rng.permutation(len(Xp))
            for i in range(0, len(Xp), self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = Xp[idx]
                yb = y[idx][:, None]
                wb = np.where(yb == 1, w_pos, w_neg)
                B = len(xb)

                z1, win1 = self._conv2d_forward(xb, self.W1, self.b1)
                a1 = np.maximum(0, z1)
                p1, mask1 = self._maxpool2x2_forward(a1)

                z2, win2 = self._conv2d_forward(p1, self.W2, self.b2)
                a2 = np.maximum(0, z2)
                p2, mask2 = self._maxpool2x2_forward(a2)

                flat = p2.reshape(B, -1)
                z3 = flat @ self.W3 + self.b3
                a3 = np.maximum(0, z3)
                logits = a3 @ self.W4 + self.b4
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

                dlogits = (probs - yb) * wb / B
                dW4 = a3.T @ dlogits + self.l2_reg * self.W4
                db4 = dlogits.sum(axis=0)
                da3 = dlogits @ self.W4.T
                dz3 = da3 * (z3 > 0)
                dW3 = flat.T @ dz3 + self.l2_reg * self.W3
                db3 = dz3.sum(axis=0)

                dp2 = (dz3 @ self.W3.T).reshape(p2.shape)
                da2 = self._maxpool2x2_backward(dp2, mask2)
                dz2 = da2 * (z2 > 0)
                dp1, dW2, db2 = self._conv2d_backward(dz2, win2, p1.shape, self.W2)
                dW2 += self.l2_reg * self.W2

                da1 = self._maxpool2x2_backward(dp1, mask1)
                dz1 = da1 * (z1 > 0)
                _, dW1, db1 = self._conv2d_backward(dz1, win1, xb.shape, self.W1)
                dW1 += self.l2_reg * self.W1

                grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]
                t_step += 1
                for j, (p, g) in enumerate(zip(params, grads)):
                    m_adam[j] = 0.9 * m_adam[j] + 0.1 * g
                    v_adam[j] = 0.999 * v_adam[j] + 0.001 * (g * g)
                    m_hat = m_adam[j] / (1.0 - 0.9 ** t_step)
                    v_hat = v_adam[j] / (1.0 - 0.999 ** t_step)
                    p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess_batch(X)
        z1, _ = self._conv2d_forward(Xp, self.W1, self.b1)
        p1, _ = self._maxpool2x2_forward(np.maximum(0, z1))
        z2, _ = self._conv2d_forward(p1, self.W2, self.b2)
        p2, _ = self._maxpool2x2_forward(np.maximum(0, z2))
        flat = p2.reshape(len(Xp), -1)
        a3 = np.maximum(0, flat @ self.W3 + self.b3)
        logits = (a3 @ self.W4 + self.b4).ravel()
        p_pos = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def benchmark_baselines(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        n_components: int = 5,
                        balance_method: str = 'smote',
                        ensemble_params: Optional[dict] = None,
                        X_train_raw: Optional[np.ndarray] = None,
                        X_test_raw: Optional[np.ndarray] = None,
                        X_train_masked: Optional[np.ndarray] = None,
                        X_test_masked: Optional[np.ndarray] = None,
                        y_train_orig: Optional[np.ndarray] = None,
                        X_all: Optional[np.ndarray] = None,
                        y_all: Optional[np.ndarray] = None,
                        X_all_raw: Optional[np.ndarray] = None,
                        X_all_masked: Optional[np.ndarray] = None,
                        df_metadata: Optional[pd.DataFrame] = None,
                        include_cnn: bool = True,
                        save_path: Optional[str] = None,
                        dpi: int = 300) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compara o sistema 2DPCA+Stacking Ensemble contra baselines clássicos,
    uma Rede Neural Convolucional 2D (2D-CNN) e estudos de ablação, incluindo
    avaliação de generalização zero-shot entre observatórios (Treino: CA+BJL -> Teste: CP).
    """
    import time
    from sklearn.base import clone
    print(f"\n{'='*96}")
    print(f"📊 COMPREHENSIVE BENCHMARK, ABLATION & CROSS-STATION GENERALIZATION STUDY")
    print(f"{'='*96}")
    print(f"   Train: {len(X_train)} | Holdout Test: {len(X_test)} | 2DPCA Components: {n_components}\n")

    n_samples_train, h, w = X_train.shape
    n_samples_test = X_test.shape[0]

    # Flatten
    X_train_flat = X_train.reshape(n_samples_train, -1)
    X_test_flat = X_test.reshape(n_samples_test, -1)

    # 2DPCA features
    t0_2dpca = time.perf_counter()
    tdpca = TwoDPCA(n_components=n_components)
    tdpca.fit(X_train)
    X_train_2dpca = tdpca.transform(X_train).reshape(n_samples_train, -1)
    t_2dpca_fit = time.perf_counter() - t0_2dpca
    t0_2dpca_inf = time.perf_counter()
    X_test_2dpca = tdpca.transform(X_test).reshape(n_samples_test, -1)
    t_2dpca_inf = (time.perf_counter() - t0_2dpca_inf) / max(n_samples_test, 1)

    # 1D-PCA features (mesmo nº de componentes principais k ou limitado a n_samples)
    n_pca_components = min(X_train_2dpca.shape[1], X_train_flat.shape[1], n_samples_train - 1)
    t0_pca = time.perf_counter()
    pca = PCA(n_components=n_pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    t_pca_fit = time.perf_counter() - t0_pca
    t0_pca_inf = time.perf_counter()
    X_test_pca = pca.transform(X_test_flat)
    t_pca_inf = (time.perf_counter() - t0_pca_inf) / max(n_samples_test, 1)

    # Prepara split Zero-Shot Cross-Station (Treino: CA+BJL -> Teste: CP) se disponível
    has_zs = (X_all is not None and y_all is not None and df_metadata is not None and 'site' in df_metadata.columns)
    if has_zs:
        mask_cp = (df_metadata['site'].values == 'CP')
        X_zs_tr, y_zs_tr = X_all[~mask_cp], y_all[~mask_cp]
        X_zs_te, y_zs_te = X_all[mask_cp], y_all[mask_cp]
        X_zs_tr_flat = X_zs_tr.reshape(len(X_zs_tr), -1)
        X_zs_te_flat = X_zs_te.reshape(len(X_zs_te), -1)
        n_pca_zs = min(h * n_components, X_zs_tr_flat.shape[1], len(X_zs_tr) - 1)
        pca_zs = PCA(n_components=n_pca_zs, random_state=42).fit(X_zs_tr_flat)
        X_zs_tr_pca = pca_zs.transform(X_zs_tr_flat)
        X_zs_te_pca = pca_zs.transform(X_zs_te_flat)
        tdpca_zs = TwoDPCA(n_components=n_components).fit(X_zs_tr)
        X_zs_tr_2d = tdpca_zs.transform(X_zs_tr).reshape(len(X_zs_tr), -1)
        X_zs_te_2d = tdpca_zs.transform(X_zs_te).reshape(len(X_zs_te), -1)

    def _eval_predictions(name, y_true, y_pred, y_proba, train_time_s, infer_ms, zs_cp_rec=np.nan):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)
        fpr_b, tpr_b, _ = roc_curve(y_true, y_proba)
        auc_val = auc(fpr_b, tpr_b)
        prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
        pr_auc_val = auc(rec_c, prec_c)
        zs_str = f" | ZS-CP={zs_cp_rec:.4f}" if not np.isnan(zs_cp_rec) else ""
        print(f"   {name:<34} Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | "
              f"F1={f1:.4f} | AUC={auc_val:.4f}{zs_str} | {infer_ms:.3f} ms/img")
        return {
            'model': name,
            'accuracy': round(acc, 4),
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'precision': round(prec, 4),
            'f1': round(f1, 4),
            'auc': round(auc_val, 4),
            'pr_auc': round(pr_auc_val, 4),
            'zs_cp_recall': round(float(zs_cp_rec), 4) if not np.isnan(zs_cp_rec) else np.nan,
            'train_time_s': round(train_time_s, 2),
            'infer_ms': round(infer_ms, 3),
        }

    baselines = [
        ('Flatten + LogReg', 'flat', X_train_flat, X_test_flat, 0.0, 0.0,
         LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('1D-PCA + LogReg', 'pca', X_train_pca, X_test_pca, t_pca_fit, t_pca_inf,
         LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('1D-PCA + SVM (RBF)', 'pca', X_train_pca, X_test_pca, t_pca_fit, t_pca_inf,
         SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
        ('Flatten + Random Forest', 'flat', X_train_flat, X_test_flat, 0.0, 0.0,
         RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('2DPCA + SVM (RBF)', '2dpca', X_train_2dpca, X_test_2dpca, t_2dpca_fit, t_2dpca_inf,
         SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)),
        ('2DPCA + Random Forest', '2dpca', X_train_2dpca, X_test_2dpca, t_2dpca_fit, t_2dpca_inf,
         RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('2DPCA + XGBoost', '2dpca', X_train_2dpca, X_test_2dpca, t_2dpca_fit, t_2dpca_inf,
         XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, eval_metric='logloss', n_jobs=-1)),
    ]

    results = []

    for name, ftype, X_tr, X_te, extra_fit_t, extra_inf_t, clf in baselines:
        t0 = time.perf_counter()
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', clf),
        ])
        pipe.fit(X_tr_s, y_train)
        t_fit = (time.perf_counter() - t0) + extra_fit_t

        t0_inf = time.perf_counter()
        X_te_s = scaler.transform(X_te)
        y_pred_b = pipe.predict(X_te_s)
        y_proba_b = pipe.predict_proba(X_te_s)[:, 1]
        t_inf_ms = ((time.perf_counter() - t0_inf) / max(n_samples_test, 1) + extra_inf_t) * 1000.0

        zs_rec = np.nan
        if has_zs:
            X_ztr = X_zs_tr_flat if ftype == 'flat' else (X_zs_tr_pca if ftype == 'pca' else X_zs_tr_2d)
            X_zte = X_zs_te_flat if ftype == 'flat' else (X_zs_te_pca if ftype == 'pca' else X_zs_te_2d)
            sc_z = StandardScaler()
            p_z = ImbPipeline([('smote', SMOTE(random_state=42)), ('clf', clone(clf))])
            p_z.fit(sc_z.fit_transform(X_ztr), y_zs_tr)
            zs_rec = float(np.mean(p_z.predict(sc_z.transform(X_zte)) == y_zs_te))

        results.append(_eval_predictions(name, y_test, y_pred_b, y_proba_b, t_fit, t_inf_ms, zs_rec))

    # 2D-CNN Baseline
    if include_cnn:
        t0_cnn = time.perf_counter()
        cnn = Conv2DCNNClassifier(epochs=25, batch_size=32, lr=0.004, random_state=42)
        cnn.fit(X_train, y_train)
        t_cnn_fit = time.perf_counter() - t0_cnn
        t0_cnn_inf = time.perf_counter()
        y_proba_cnn = cnn.predict_proba(X_test)[:, 1]
        y_pred_cnn = (y_proba_cnn >= 0.5).astype(int)
        t_cnn_inf_ms = ((time.perf_counter() - t0_cnn_inf) / max(n_samples_test, 1)) * 1000.0
        zs_cnn = np.nan
        if has_zs:
            cnn_z = Conv2DCNNClassifier(epochs=25, batch_size=32, lr=0.004, random_state=42).fit(X_zs_tr, y_zs_tr)
            zs_cnn = float(np.mean(cnn_z.predict(X_zs_te) == y_zs_te))
        results.append(_eval_predictions('2D-CNN (ConvNet Baseline)', y_test, y_pred_cnn, y_proba_cnn, t_cnn_fit, t_cnn_inf_ms, zs_cnn))

    # Ablation 1: 2DPCA + Stacking (No Physics Preprocessing)
    if X_train_raw is not None and X_test_raw is not None:
        t0_raw = time.perf_counter()
        y_tr_raw = y_train_orig if (y_train_orig is not None and len(X_train_raw) == len(y_train_orig)) else y_train
        sys_raw = EPBRecognitionSystem(
            n_components=n_components, balance_data=True, balance_method=balance_method,
            ensemble_params=ensemble_params
        )
        sys_raw.fit(X_train_raw, y_tr_raw)
        t_raw_fit = time.perf_counter() - t0_raw
        t0_raw_inf = time.perf_counter()
        y_pred_raw = sys_raw.predict(X_test_raw)
        y_proba_raw = sys_raw.predict_proba(X_test_raw)[:, 1]
        t_raw_inf_ms = ((time.perf_counter() - t0_raw_inf) / max(n_samples_test, 1)) * 1000.0
        zs_raw = np.nan
        if has_zs and X_all_raw is not None:
            sys_raw_z = EPBRecognitionSystem(
                n_components=n_components, balance_data=True, balance_method=balance_method,
                ensemble_params=ensemble_params
            ).fit(X_all_raw[~mask_cp], y_zs_tr)
            zs_raw = float(np.mean(sys_raw_z.predict(X_all_raw[mask_cp]) == y_zs_te))
        results.append(_eval_predictions('Ablation: No Physics Preproc', y_test, y_pred_raw, y_proba_raw, t_raw_fit, t_raw_inf_ms, zs_raw))

    # Ablation 2: 2DPCA + Stacking (No SMOTE)
    t0_nosmote = time.perf_counter()
    sys_nosmote = EPBRecognitionSystem(
        n_components=n_components, balance_data=False, balance_method=balance_method,
        ensemble_params=ensemble_params
    )
    sys_nosmote.fit(X_train, y_train)
    t_nosmote_fit = time.perf_counter() - t0_nosmote
    t0_ns_inf = time.perf_counter()
    y_pred_ns = sys_nosmote.predict(X_test)
    y_proba_ns = sys_nosmote.predict_proba(X_test)[:, 1]
    t_ns_inf_ms = ((time.perf_counter() - t0_ns_inf) / max(n_samples_test, 1)) * 1000.0
    zs_ns = np.nan
    if has_zs:
        sys_ns_z = EPBRecognitionSystem(
            n_components=n_components, balance_data=False, balance_method=balance_method,
            ensemble_params=ensemble_params
        ).fit(X_zs_tr, y_zs_tr)
        zs_ns = float(np.mean(sys_ns_z.predict(X_zs_te) == y_zs_te))
    results.append(_eval_predictions('Ablation: No SMOTE', y_test, y_pred_ns, y_proba_ns, t_nosmote_fit, t_ns_inf_ms, zs_ns))

    # Ablation 3: 2DPCA + Stacking (Circular FOV Mask - No Corner Text)
    if X_train_masked is not None and X_test_masked is not None:
        t0_msk = time.perf_counter()
        y_tr_msk = y_train_orig if (y_train_orig is not None and len(X_train_masked) == len(y_train_orig)) else y_train
        sys_msk = EPBRecognitionSystem(
            n_components=n_components, balance_data=True, balance_method=balance_method,
            ensemble_params=ensemble_params
        )
        sys_msk.fit(X_train_masked, y_tr_msk)
        t_msk_fit = time.perf_counter() - t0_msk
        t0_msk_inf = time.perf_counter()
        y_pred_msk = sys_msk.predict(X_test_masked)
        y_proba_msk = sys_msk.predict_proba(X_test_masked)[:, 1]
        t_msk_inf_ms = ((time.perf_counter() - t0_msk_inf) / max(n_samples_test, 1)) * 1000.0
        zs_msk = np.nan
        if has_zs and X_all_masked is not None:
            sys_msk_z = EPBRecognitionSystem(
                n_components=n_components, balance_data=True, balance_method=balance_method,
                ensemble_params=ensemble_params
            ).fit(X_all_masked[~mask_cp], y_zs_tr)
            zs_msk = float(np.mean(sys_msk_z.predict(X_all_masked[mask_cp]) == y_zs_te))
        results.append(_eval_predictions('Ablation: Circular FOV Mask', y_test, y_pred_msk, y_proba_msk, t_msk_fit, t_msk_inf_ms, zs_msk))

    # Sistema completo (2DPCA + Stacking Ensemble)
    t0_full = time.perf_counter()
    sys_full = EPBRecognitionSystem(
        n_components=n_components, balance_data=True, balance_method=balance_method,
        ensemble_params=ensemble_params
    )
    sys_full.fit(X_train, y_train)
    t_full_fit = time.perf_counter() - t0_full
    t0_full_inf = time.perf_counter()
    y_pred_full = sys_full.predict(X_test)
    y_proba_full = sys_full.predict_proba(X_test)[:, 1]
    t_full_inf_ms = ((time.perf_counter() - t0_full_inf) / max(n_samples_test, 1)) * 1000.0
    zs_full = np.nan
    if has_zs:
        sys_full_z = EPBRecognitionSystem(
            n_components=n_components, balance_data=True, balance_method=balance_method,
            ensemble_params=ensemble_params
        ).fit(X_zs_tr, y_zs_tr)
        zs_full = float(np.mean(sys_full_z.predict(X_zs_te) == y_zs_te))

    results.append(_eval_predictions('2DPCA + Stacking Ensemble ★', y_test, y_pred_full, y_proba_full, t_full_fit, t_full_inf_ms, zs_full))

    df_results = pd.DataFrame(results)

    # --- Tabela formatada ---
    print(f"\n{'='*102}")
    print(f"{'Model / Configuration':<34} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'Prec':>7} {'F1':>7} {'AUC':>7} {'ZS-CP':>7} {'ms/img':>8}")
    print(f"{'-'*102}")
    for _, row in df_results.iterrows():
        zs_v = f"{row['zs_cp_recall']:>7.4f}" if not np.isnan(row.get('zs_cp_recall', np.nan)) else "    N/A"
        print(f"{row['model']:<34} {row['accuracy']:>7.4f} {row['sensitivity']:>7.4f} "
              f"{row['specificity']:>7.4f} {row['precision']:>7.4f} {row['f1']:>7.4f} "
              f"{row['auc']:>7.4f} {zs_v} {row['infer_ms']:>8.3f}")
    print(f"{'='*102}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax = axes[0]
    x = np.arange(len(df_results))
    bar_w = 0.25

    ax.barh(x - bar_w, df_results['f1'], bar_w, label='Holdout F1-Score',
            color=['#27ae60' if '★' in m else '#3498db' for m in df_results['model']])
    ax.barh(x, df_results['auc'], bar_w, label='Holdout ROC-AUC',
            color=['#e67e22' if '★' in m else '#e74c3c' for m in df_results['model']])
    if has_zs and not df_results['zs_cp_recall'].isna().all():
        ax.barh(x + bar_w, df_results['zs_cp_recall'].fillna(0.0), bar_w, label='Zero-Shot CP Recall',
                color=['#8e44ad' if '★' in m else '#9b59b6' for m in df_results['model']])
    else:
        ax.barh(x + bar_w, df_results['sensitivity'], bar_w, label='Sensitivity',
                color=['#8e44ad' if '★' in m else '#9b59b6' for m in df_results['model']])

    ax.set_yticks(x)
    ax.set_yticklabels(df_results['model'], fontsize=9)
    ax.set_xlabel('Score', fontsize=11)
    ax.set_title('Model, Ablation & Cross-Station Generalization', fontsize=12, fontweight='bold')
    ax.set_xlim(0.0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    ax2 = axes[1]
    if has_zs and not df_results['zs_cp_recall'].isna().all():
        metrics_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc', 'zs_cp_recall']
        col_labels = ['Acc', 'Sens', 'Spec', 'Prec', 'F1', 'ROC-AUC', 'ZS-CP']
    else:
        metrics_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc', 'pr_auc']
        col_labels = ['Acc', 'Sens', 'Spec', 'Prec', 'F1', 'ROC-AUC', 'PR-AUC']
    heatmap_data = df_results.set_index('model')[metrics_cols]
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                ax=ax2, vmin=0.50, vmax=1.0, linewidths=0.5,
                cbar_kws={'label': 'Score'})
    ax2.set_title('Performance Metrics Across Architectures', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(col_labels, rotation=25, ha='right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 Benchmark figure saved: {save_path} ({dpi} DPI)")
    plt.show()

    return fig, df_results


# ============================================================================
# PARTE 10b: BENCHMARK PAREADO POR VALIDAÇÃO CRUZADA (MESMOS FOLDS PARA TODOS)
# ============================================================================

def extract_night_groups(df_metadata: pd.DataFrame) -> np.ndarray:
    """
    Agrupa cada imagem pela NOITE de observação (estação + noite local).

    Uma noite de observação atravessa a meia-noite UTC; por isso imagens tomadas
    antes das 12:00 UTC são atribuídas à noite que começou no dia anterior.
    Ex.: O6_CA_20251023_011500 -> 'CA_20251022'.
    """
    groups = []
    for _, row in df_metadata.iterrows():
        fname = str(row.get('filename', ''))
        site = str(row.get('site', 'UNK'))
        m = re.search(r'_(\d{8})_(\d{2})(\d{2})(\d{2})', fname)
        if m:
            d = datetime.strptime(m.group(1), '%Y%m%d')
            if int(m.group(2)) < 12:
                d = d - pd.Timedelta(days=1)
            groups.append(f"{site}_{d.strftime('%Y%m%d')}")
        else:
            groups.append(f"{site}_{row.get('date', 'UNK')}")
    return np.array(groups)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Métricas binárias robustas (AUC = NaN quando o conjunto possui uma única classe)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    if np.isnan(sens) or np.isnan(prec) or (prec + sens) == 0:
        f1 = np.nan if np.isnan(sens) else 0.0
    else:
        f1 = 2 * prec * sens / (prec + sens)
    acc = (tp + tn) / max(len(y_true), 1)
    if len(np.unique(y_true)) == 2:
        roc = roc_auc_score(y_true, y_proba)
        pr = average_precision_score(y_true, y_proba)
    else:
        roc, pr = np.nan, np.nan
    return {'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
            'precision': prec, 'f1': f1, 'auc': roc, 'pr_auc': pr,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


class _SklearnImageModel:
    """Adaptador: extrator de features (flatten / PCA / 2DPCA) + StandardScaler + SMOTE + classificador."""

    def __init__(self, feature: str, clf, n_components: int = 5, use_smote: bool = True,
                 random_state: int = 42):
        self.feature = feature
        self.clf = clf
        self.n_components = n_components
        self.use_smote = use_smote
        self.random_state = random_state

    def _features(self, X: np.ndarray, fit: bool) -> np.ndarray:
        n = len(X)
        if self.feature == 'flat':
            return X.reshape(n, -1)
        if self.feature == 'pca':
            flat = X.reshape(n, -1)
            if fit:
                k = min(X.shape[1] * self.n_components, n - 1, flat.shape[1])
                self.pca_ = PCA(n_components=k, random_state=self.random_state).fit(flat)
            return self.pca_.transform(flat)
        if self.feature == '2dpca':
            if fit:
                self.tdpca_ = TwoDPCA(n_components=self.n_components).fit(X)
            return self.tdpca_.transform(X).reshape(n, -1)
        raise ValueError(self.feature)

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.base import clone
        F = self._features(X, fit=True)
        steps = [('scaler', StandardScaler())]
        if self.use_smote:
            steps.append(('smote', SMOTE(random_state=self.random_state)))
        steps.append(('clf', clone(self.clf)))
        self.pipe_ = ImbPipeline(steps).fit(F, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe_.predict_proba(self._features(X, fit=False))


def build_benchmark_methods(n_components: int = 5,
                            ensemble_params: Optional[dict] = None,
                            include_cnn: bool = True,
                            include_ablations: bool = True) -> List[dict]:
    """
    Define os modelos comparados no benchmark pareado.

    Cada entrada: {'name', 'group', 'data' ('proc' | 'raw' | 'masked'), 'factory'}.
    Todos os modelos são re-treinados do zero em cada fold (sem vazamento).
    """
    rs = 42
    methods = [
        ('Flatten + LogReg', 'Baseline', 'proc',
         lambda: _SklearnImageModel('flat', LogisticRegression(max_iter=3000, random_state=rs))),
        ('Flatten + Random Forest', 'Baseline', 'proc',
         lambda: _SklearnImageModel('flat', RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1))),
        ('1D-PCA + LogReg', 'Baseline', 'proc',
         lambda: _SklearnImageModel('pca', LogisticRegression(max_iter=3000, random_state=rs), n_components)),
        ('1D-PCA + SVM (RBF)', 'Baseline', 'proc',
         lambda: _SklearnImageModel('pca', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=rs), n_components)),
        ('2DPCA + LogReg', 'Single learner', 'proc',
         lambda: _SklearnImageModel('2dpca', LogisticRegression(max_iter=3000, random_state=rs), n_components)),
        ('2DPCA + SVM (RBF)', 'Single learner', 'proc',
         lambda: _SklearnImageModel('2dpca', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=rs), n_components)),
        ('2DPCA + Random Forest', 'Single learner', 'proc',
         lambda: _SklearnImageModel('2dpca', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1), n_components)),
        ('2DPCA + XGBoost', 'Single learner', 'proc',
         lambda: _SklearnImageModel('2dpca', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                           eval_metric='logloss', random_state=rs, n_jobs=-1), n_components)),
    ]
    if include_cnn:
        methods.append(('2D-CNN', 'Baseline', 'proc',
                        lambda: Conv2DCNNClassifier(epochs=25, batch_size=32, lr=0.004, random_state=rs)))
    methods.append(('2DPCA + Stacking (proposed)', 'Proposed', 'proc',
                    lambda: EPBRecognitionSystem(n_components=n_components, balance_data=True,
                                                 balance_method='smote', ensemble_params=ensemble_params)))
    if include_ablations:
        methods += [
            ('Ablation: no physics preprocessing', 'Ablation', 'raw',
             lambda: EPBRecognitionSystem(n_components=n_components, balance_data=True,
                                          balance_method='smote', ensemble_params=ensemble_params)),
            ('Ablation: no SMOTE', 'Ablation', 'proc',
             lambda: EPBRecognitionSystem(n_components=n_components, balance_data=False,
                                          balance_method='smote', ensemble_params=ensemble_params)),
            ('Ablation: circular FOV mask', 'Ablation', 'masked',
             lambda: EPBRecognitionSystem(n_components=n_components, balance_data=True,
                                          balance_method='smote', ensemble_params=ensemble_params)),
        ]
    return [dict(name=n, group=g, data=d, factory=f) for n, g, d, f in methods]


def _corrected_resampled_ttest(diffs: np.ndarray, n_train: float, n_test: float) -> Tuple[float, float]:
    """
    Teste t corrigido para reamostragem (Nadeau & Bengio, 2003), apropriado para
    comparar dois classificadores avaliados nos MESMOS folds de r x k-fold CV.
    """
    from scipy.stats import t as t_dist
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    J = len(diffs)
    if J < 2:
        return np.nan, np.nan
    mean_d = diffs.mean()
    var_d = diffs.var(ddof=1)
    if var_d == 0:
        return (0.0, 1.0) if mean_d == 0 else (np.inf * np.sign(mean_d), 0.0)
    t_stat = mean_d / np.sqrt((1.0 / J + n_test / n_train) * var_d)
    p_val = 2 * t_dist.sf(np.abs(t_stat), df=J - 1)
    return float(t_stat), float(p_val)


def _mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[int, int, float]:
    """Teste exato de McNemar sobre predições pareadas (b = A certo/B errado, c = A errado/B certo)."""
    from scipy.stats import binomtest
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    p = 1.0 if (b + c) == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, float(p)


def benchmark_cv(X: np.ndarray, y: np.ndarray,
                 protocol: str = 'stratified',
                 groups: Optional[np.ndarray] = None,
                 X_raw: Optional[np.ndarray] = None,
                 X_masked: Optional[np.ndarray] = None,
                 n_components: int = 5,
                 ensemble_params: Optional[dict] = None,
                 n_folds: int = 5,
                 n_repeats: int = 3,
                 include_cnn: bool = True,
                 include_ablations: bool = True,
                 reference: str = '2DPCA + Stacking (proposed)',
                 random_state: int = 42,
                 verbose: bool = True) -> dict:
    """
    Benchmark PAREADO: todos os modelos são avaliados exatamente nos mesmos folds.

    Protocolos:
      - 'stratified'   : r x k-fold estratificado aleatório (quadros vizinhos podem cair em folds diferentes).
      - 'group'        : r x k-fold estratificado AGRUPADO (ex.: blocos temporais de 30 min).
      - 'logo'         : leave-one-group-out (ex.: deixa uma NOITE inteira de fora por vez).
                         Como cada noite é monoclasse, as métricas são calculadas sobre as
                         predições out-of-fold agregadas (pooled) e por noite (acurácia).

    Retorna dict com:
      'folds'   : métricas por (modelo, repetição, fold)   [stratified/group]
      'summary' : média ± desvio por modelo (ou métricas pooled no 'logo')
      'oof'     : {modelo: {'y_pred', 'y_proba'}} da 1ª repetição
      'stats'   : testes de significância contra o modelo de referência
      'per_group': acurácia por grupo/noite (apenas 'logo')
    """
    import time
    from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut

    methods = build_benchmark_methods(n_components, ensemble_params, include_cnn, include_ablations)
    data_map = {'proc': X, 'raw': X_raw, 'masked': X_masked}
    methods = [m for m in methods if data_map[m['data']] is not None]

    # ---- Splits ----
    splits = []  # (repeat, fold, train_idx, test_idx)
    if protocol == 'logo':
        if groups is None:
            raise ValueError("protocol='logo' requer groups")
        for f, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups)):
            splits.append((0, f, tr, te))
        n_repeats_eff = 1
    else:
        for r in range(n_repeats):
            if protocol == 'group':
                if groups is None:
                    raise ValueError("protocol='group' requer groups")
                sp = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state + r)
                it = sp.split(X, y, groups)
            else:
                sp = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state + r)
                it = sp.split(X, y)
            for f, (tr, te) in enumerate(it):
                splits.append((r, f, tr, te))
        n_repeats_eff = n_repeats

    label = {'stratified': f'{n_repeats}x{n_folds}-fold stratified CV',
             'group': f'{n_repeats}x{n_folds}-fold grouped CV ({len(np.unique(groups)) if groups is not None else 0} groups)',
             'logo': f'Leave-one-group-out ({len(np.unique(groups)) if groups is not None else 0} groups)'}[protocol]
    if verbose:
        print(f"\n{'='*88}\n📊 PAIRED BENCHMARK — {label} | {len(methods)} models | {len(splits)} train/test splits\n{'='*88}")

    fold_rows = []
    oof = {m['name']: {r: {'y_pred': np.full(len(y), -1), 'y_proba': np.full(len(y), np.nan)}
                       for r in range(n_repeats_eff)} for m in methods}
    infer_ms = {m['name']: [] for m in methods}

    for (r, f, tr, te) in splits:
        for m in methods:
            Xd = data_map[m['data']]
            model = m['factory']()
            model.fit(Xd[tr], y[tr])
            t0 = time.perf_counter()
            proba = model.predict_proba(Xd[te])[:, 1]
            infer_ms[m['name']].append((time.perf_counter() - t0) / max(len(te), 1) * 1000.0)
            pred = (proba >= 0.5).astype(int)
            oof[m['name']][r]['y_pred'][te] = pred
            oof[m['name']][r]['y_proba'][te] = proba
            if protocol != 'logo':
                row = _binary_metrics(y[te], pred, proba)
                row.update(model=m['name'], group=m['group'], repeat=r, fold=f, n_test=len(te), n_train=len(tr))
                fold_rows.append(row)
        if verbose:
            print(f"   ✓ split {r}.{f} done (train={len(tr)}, test={len(te)})")

    metric_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc', 'pr_auc']
    summary_rows, stats_rows, per_group = [], [], None
    df_folds = pd.DataFrame(fold_rows)

    if protocol != 'logo':
        for m in methods:
            d = df_folds[df_folds.model == m['name']]
            row = {'model': m['name'], 'group': m['group'], 'infer_ms': float(np.median(infer_ms[m['name']]))}
            for c in metric_cols:
                row[f'{c}_mean'] = float(np.nanmean(d[c]))
                row[f'{c}_std'] = float(np.nanstd(d[c], ddof=1))
            summary_rows.append(row)
        n_tr = float(df_folds['n_train'].mean())
        n_te = float(df_folds['n_test'].mean())
        ref = df_folds[df_folds.model == reference].sort_values(['repeat', 'fold'])
        for m in methods:
            if m['name'] == reference:
                continue
            d = df_folds[df_folds.model == m['name']].sort_values(['repeat', 'fold'])
            out = {'model': m['name']}
            for c in ['f1', 'accuracy', 'auc']:
                diffs = d[c].values - ref[c].values
                t_stat, p = _corrected_resampled_ttest(diffs, n_tr, n_te)
                out[f'delta_{c}'] = float(np.nanmean(diffs))
                out[f'p_{c}'] = p
            b, c_, p_mc = _mcnemar_exact(y, oof[reference][0]['y_pred'], oof[m['name']][0]['y_pred'])
            out.update(mcnemar_ref_only=b, mcnemar_other_only=c_, p_mcnemar=p_mc)
            stats_rows.append(out)
    else:
        grp_rows = []
        for m in methods:
            o = oof[m['name']][0]
            row = _binary_metrics(y, o['y_pred'], o['y_proba'])
            row.update(model=m['name'], group=m['group'], infer_ms=float(np.median(infer_ms[m['name']])))
            # Acurácia média por noite (macro) — cada noite pesa igualmente
            accs = []
            for g in np.unique(groups):
                mask = groups == g
                acc_g = float(np.mean(o['y_pred'][mask] == y[mask]))
                accs.append(acc_g)
                grp_rows.append({'model': m['name'], 'group_id': g, 'n': int(mask.sum()),
                                 'label': int(y[mask].max()), 'accuracy': acc_g,
                                 'mean_proba': float(np.mean(o['y_proba'][mask]))})
            row['macro_night_accuracy'] = float(np.mean(accs))
            row['worst_night_accuracy'] = float(np.min(accs))
            summary_rows.append(row)
        per_group = pd.DataFrame(grp_rows)
        for m in methods:
            if m['name'] == reference:
                continue
            b, c_, p_mc = _mcnemar_exact(y, oof[reference][0]['y_pred'], oof[m['name']][0]['y_pred'])
            stats_rows.append({'model': m['name'], 'mcnemar_ref_only': b,
                               'mcnemar_other_only': c_, 'p_mcnemar': p_mc})

    df_summary = pd.DataFrame(summary_rows)
    df_stats = pd.DataFrame(stats_rows)

    if verbose:
        print(f"\n{'-'*88}\n{label}\n{'-'*88}")
        if protocol != 'logo':
            print(f"{'Model':<38}{'Acc':>14}{'Sens':>14}{'Spec':>14}{'F1':>14}{'ROC-AUC':>16}")
            for _, s in df_summary.iterrows():
                print(f"{s['model']:<38}"
                      f"{s['accuracy_mean']:>8.4f}±{s['accuracy_std']:.3f}"
                      f"{s['sensitivity_mean']:>8.4f}±{s['sensitivity_std']:.3f}"
                      f"{s['specificity_mean']:>8.4f}±{s['specificity_std']:.3f}"
                      f"{s['f1_mean']:>8.4f}±{s['f1_std']:.3f}"
                      f"{s['auc_mean']:>9.4f}±{s['auc_std']:.4f}")
        else:
            print(f"{'Model':<38}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'F1':>8}{'AUC':>8}{'MacroNight':>12}{'Worst':>8}")
            for _, s in df_summary.iterrows():
                print(f"{s['model']:<38}{s['accuracy']:>8.4f}{s['sensitivity']:>8.4f}{s['specificity']:>8.4f}"
                      f"{s['f1']:>8.4f}{s['auc']:>8.4f}{s['macro_night_accuracy']:>12.4f}{s['worst_night_accuracy']:>8.4f}")
        print(f"\nSignificance vs. '{reference}':")
        print(df_stats.round(4).to_string(index=False))

    return {'protocol': protocol, 'label': label, 'folds': df_folds, 'summary': df_summary,
            'stats': df_stats, 'oof': {k: v[0] for k, v in oof.items()}, 'per_group': per_group}


def plot_benchmark_cv(results: List[dict], metric: str = 'f1',
                      save_path: Optional[str] = None, dpi: int = 300) -> plt.Figure:
    """
    Figura comparativa dos protocolos de benchmark pareado (um painel por protocolo).
    Barras = média (± desvio) do F1 nos folds (stratified/group) ou F1 pooled (logo).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    order = results[0]['summary']['model'].tolist()
    colors = {'Baseline': '#94a3b8', 'Single learner': '#60a5fa', 'Proposed': '#16a34a', 'Ablation': '#f59e0b'}
    for ax, res in zip(axes, results):
        s = res['summary'].set_index('model').loc[order].reset_index()
        if res['protocol'] == 'logo':
            vals = s[metric].values
            errs = None
        else:
            vals = s[f'{metric}_mean'].values
            errs = s[f'{metric}_std'].values
        ypos = np.arange(len(s))[::-1]
        ax.barh(ypos, vals, xerr=errs, color=[colors.get(g, '#cbd5e1') for g in s['group']],
                edgecolor='#334155', linewidth=0.6, capsize=2.5, error_kw=dict(lw=0.8))
        for yp, v in zip(ypos, vals):
            ax.text(min(v, 1.0) + 0.004, yp, f'{v:.3f}', va='center', fontsize=7.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(s['model'], fontsize=8.5)
        lo = np.nanmin(vals - (errs if errs is not None else 0))
        ax.set_xlim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.04)
        ax.set_xlabel('F1-score' if metric == 'f1' else metric)
        ax.set_title(res['label'], fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()],
               loc='lower center', ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"💾 Benchmark figure saved: {save_path} ({dpi} DPI)")
    plt.show()
    return fig



def evaluate_hard_negatives_and_generalization(
    X: np.ndarray,
    y: np.ndarray,
    df_metadata: pd.DataFrame,
    oof_y_pred: np.ndarray,
    oof_y_proba: np.ndarray,
    system: 'EPBRecognitionSystem',
    util_dir: str = 'util',
    threshold_stars: int = 4500,
) -> Dict[str, pd.DataFrame]:
    """
    Avalia o comportamento do sistema em cenários operacionais críticos ("hard negatives" e generalização):
      1. Estratificação por Iluminação Lunar (Moon >= 50% vs. Moon < 50%) nas predições out-of-fold.
      2. Desempenho por Observatório (CA, BJL, CP) nas predições out-of-fold.
      3. Generalização Zero-Shot Cross-Station: Treino em CA + BJL e teste na estação nunca vista CP.
      4. Redução de Falsos Positivos por Nuvens (Estágio 1 QC + Estágio 2 vs. Estágio 2 isolado)
         nas sequências noturnas contínuas (céu limpo BJL_20250722 vs. céu nublado CA_20260624).
    """
    from glob import glob
    print(f"\n{'='*80}")
    print(f"🛡️ HARD-NEGATIVE, LUNAR ILLUMINATION & CROSS-STATION GENERALIZATION ANALYSIS")
    print(f"{'='*80}")

    # 1. Lunar Illumination Stratification (Out-of-Fold)
    moon_rows = []
    if 'moon_illumination' in df_metadata.columns:
        moon_vals = df_metadata['moon_illumination'].values
        subsets = [
            ('Low/Moderate Moon (< 50%)', moon_vals < 50.0),
            ('Bright Moon (>= 50%)', moon_vals >= 50.0),
            ('All Labeled Images', np.ones(len(y), dtype=bool)),
        ]
        for label_str, mask in subsets:
            yt, yp, ypr = y[mask], oof_y_pred[mask], oof_y_proba[mask]
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            acc = (tp + tn) / max(len(yt), 1)
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * sens / max(prec + sens, 1e-8)
            moon_rows.append({
                'Regime': label_str,
                'N': int(mask.sum()),
                'EPB': int(yt.sum()),
                'No_EPB': int((yt == 0).sum()),
                'Accuracy': round(acc, 4),
                'Sensitivity': round(sens, 4),
                'Specificity': round(spec, 4),
                'F1': round(f1, 4),
            })
    df_moon = pd.DataFrame(moon_rows)
    print("\n1️⃣ Lunar Illumination Stratification (5-Fold Out-of-Fold Predictions):")
    print(df_moon.to_string(index=False))

    # 2. Per-Station Performance & Zero-Shot CP Holdout
    station_rows = []
    if 'site' in df_metadata.columns:
        sites = df_metadata['site'].values
        for st in ['CA', 'BJL', 'CP']:
            mask = (sites == st)
            if mask.sum() == 0:
                continue
            yt, yp = y[mask], oof_y_pred[mask]
            acc = np.mean(yt == yp)
            station_rows.append({
                'Station': st,
                'Evaluation_Mode': '5-Fold CV (Out-of-Fold)',
                'N': int(mask.sum()),
                'EPB': int(yt.sum()),
                'No_EPB': int((yt == 0).sum()),
                'Accuracy': round(float(acc), 4),
            })

        # Zero-Shot Holdout on CP (Trained strictly on CA + BJL)
        mask_cp = (sites == 'CP')
        if mask_cp.sum() > 0 and (~mask_cp).sum() > 0:
            sys_zs = EPBRecognitionSystem(
                n_components=system.n_components,
                balance_data=True,
                balance_method=system.balance_method,
                ensemble_params=system.ensemble_params,
            )
            sys_zs.fit(X[~mask_cp], y[~mask_cp])
            yp_cp = sys_zs.predict(X[mask_cp])
            ypr_cp = sys_zs.predict_proba(X[mask_cp])[:, 1]
            acc_cp = np.mean(y[mask_cp] == yp_cp)
            station_rows.append({
                'Station': 'CP (Unseen Station Holdout)',
                'Evaluation_Mode': f'Zero-Shot (Trained on CA+BJL, n={int((~mask_cp).sum())})',
                'N': int(mask_cp.sum()),
                'EPB': int(y[mask_cp].sum()),
                'No_EPB': int((y[mask_cp] == 0).sum()),
                'Accuracy': round(float(acc_cp), 4),
            })
            print(f"\n2️⃣ Zero-Shot Cross-Station Generalization (Train: CA+BJL -> Test: CP):")
            print(f"   • Detected {int(yp_cp.sum())}/{int(mask_cp.sum())} EPB frames ({acc_cp*100:.2f}% Recall, "
                  f"Mean Confidence: {ypr_cp.mean()*100:.2f}%)")
    df_station = pd.DataFrame(station_rows)
    print("\n   Station Breakdown:")
    print(df_station.to_string(index=False))

    # 3. Cloud Contamination & Stage 1 QC Effectiveness on Nocturnal Sequences
    # Train Stage 2 reference on clear-sky frames (cloudy == 0) so overcast night CA_20260624 is unseen
    if 'cloudy' in df_metadata.columns and (df_metadata['cloudy'].values == 0).sum() > 0:
        clear_mask = (df_metadata['cloudy'].values == 0)
        sys_seq = EPBRecognitionSystem(
            n_components=system.n_components,
            balance_data=True,
            balance_method=system.balance_method,
            ensemble_params=system.ensemble_params,
        )
        sys_seq.fit(X[clear_mask], y[clear_mask])
    else:
        sys_seq = system

    qc_rows = []
    seq_specs = [
        ('BJL_20250722 (Clear Quiet Sky)', os.path.join(util_dir, 'BJL_20250722', 'processed')),
        ('CA_20260624 (Cloud-Contaminated Sky)', os.path.join(util_dir, 'CA_20260624', 'processed')),
        ('CA_20251022 (Active EPB Night)', os.path.join(util_dir, 'CA_20251022', 'processed')),
    ]
    for seq_label, folder in seq_specs:
        if not os.path.isdir(folder):
            continue
        files = sorted(glob(os.path.join(folder, '*.png')))
        n_total = len(files)
        s2_only_epb = 0
        cascaded_epb = 0
        cascaded_clear_epb = 0
        qc_blocked = 0
        qc_degraded = 0
        for fpath in files:
            img_pil = Image.open(fpath).convert('L')
            img_raw = np.array(img_pil, dtype=np.float64)
            # Stage 2 alone (without Stage 1 QC)
            img_proc = preprocess_epb_image(img_raw, target_size=(64, 64))
            pred_s2 = int(sys_seq.predict(img_proc[np.newaxis, ...])[0])
            s2_only_epb += pred_s2
            # Two-stage cascaded pipeline (Stage 1 QC + Stage 2)
            res = sys_seq.predict_cascaded(img_raw, target_size=(64, 64), threshold_stars=threshold_stars)
            if not res.qc.is_valid_sky:
                qc_blocked += 1
            else:
                if res.qc.condition == 'DEGRADED':
                    qc_degraded += 1
                if res.has_epb == 1:
                    cascaded_epb += 1
                    if res.is_reliable:
                        cascaded_clear_epb += 1
        qc_rows.append({
            'Nocturnal_Sequence': seq_label,
            'Total_Frames': n_total,
            'QC_Flagged_Cloudy': f"{qc_blocked} ({qc_blocked/max(n_total,1)*100:.1f}%)",
            'QC_Degraded': f"{qc_degraded} ({qc_degraded/max(n_total,1)*100:.1f}%)",
            'Stage2_Only_EPB_Triggers': f"{s2_only_epb} ({s2_only_epb/max(n_total,1)*100:.1f}%)",
            'Cascaded_Valid_EPB_Detections': f"{cascaded_epb} ({cascaded_epb/max(n_total,1)*100:.1f}%)",
            'Cascaded_ClearSky_EPB': f"{cascaded_clear_epb} ({cascaded_clear_epb/max(n_total,1)*100:.1f}%)",
        })
    df_qc = pd.DataFrame(qc_rows)
    if len(df_qc) > 0:
        print("\n3️⃣ Stage 1 Sky Quality Control vs. Stage 2 Alone on Nocturnal Sequences:")
        print(df_qc.to_string(index=False))
    print(f"{'='*80}")

    return {
        'lunar': df_moon,
        'station': df_station,
        'qc_sequences': df_qc,
    }


def plot_pipeline_architecture(save_path: Optional[str] = None, dpi: int = 300) -> plt.Figure:
    """
    Gera um diagrama esquemático vetorial em alta resolução da arquitetura em cascata de 2 estágios
    para inclusão direta no artigo científico (waimlap2026/epb/figs/pipeline_architecture.png).

    Todo o texto é posicionado em coordenadas internas de cada card (título + separador + corpo),
    garantindo que nenhum rótulo extrapole as bordas das caixas.
    """
    from matplotlib.patches import FancyBboxPatch

    W_TOT, H_TOT = 17.7, 7.0
    fig, ax = plt.subplots(figsize=(W_TOT * 0.9, H_TOT * 0.9), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(0, W_TOT)
    ax.set_ylim(0, H_TOT)
    ax.set_aspect('equal')
    ax.axis('off')

    def _card(x, y, w, h, title, body, fc, ec, tc, bc, body_fs=9.0, title_fs=10.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.18",
                                    facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=2))
        n_title = title.count('\n') + 1
        title_h = 0.30 * n_title + 0.25
        ax.text(x + w / 2, y + h - 0.18, title, ha='center', va='top',
                fontsize=title_fs, fontweight='bold', color=tc, linespacing=1.15, zorder=3)
        sep_y = y + h - title_h - 0.12
        ax.plot([x + 0.18, x + w - 0.18], [sep_y, sep_y], color=ec, lw=0.9, alpha=0.6, zorder=3)
        if body:
            ax.text(x + w / 2, (y + sep_y) / 2, body, ha='center', va='center',
                    fontsize=body_fs, color=bc, linespacing=1.45, zorder=3)
        return sep_y

    def _arrow(x1, y1, x2, y2, color='#475569'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle='-|>,head_width=0.28,head_length=0.45',
                                    color=color, lw=1.8, shrinkA=0, shrinkB=0))

    def _label(x, y, text, color, ha='center', va='bottom'):
        ax.text(x, y, text, ha=ha, va=va, fontsize=8.5, fontstyle='italic',
                fontweight='bold', color=color, linespacing=1.1, zorder=4,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none'))

    # ---------------- Linha superior: fluxo principal ----------------
    bw, bh, gap = 2.7, 2.75, 0.9
    y_top = 3.85
    xs = [0.3 + i * (bw + gap) for i in range(5)]
    yc = y_top + bh / 2

    _card(xs[0], y_top, bw, bh, 'Raw All-Sky\nImage',
          'OI 630.0 nm airglow\n512 \u00d7 512 px\nCA \u00b7 BJL \u00b7 CP\nobservatories',
          '#f1f5f9', '#64748b', '#0f172a', '#334155')
    _card(xs[1], y_top, bw, bh, 'Stage 1\nSky Quality Control',
          'Circular FOV mask\nHigh-pass star detection\n'
          r'$N_{\mathrm{star}} \geq 4500$ px' '\n'
          r'and mean $\geq 35$ ADU',
          '#fef3c7', '#d97706', '#92400e', '#78350f')
    _card(xs[2], y_top, bw, bh, 'Stage 2A\nPhysics Preprocessing',
          'Resize to 64 \u00d7 64\nGaussian background\n'
          r'subtraction ($\sigma = 10$)' '\nCLAHE + min\u2013max norm.',
          '#e0f2fe', '#0284c7', '#075985', '#0c4a6e')
    _card(xs[3], y_top, bw, bh, 'Stage 2B\n2DPCA Projection',
          r'$G = \frac{1}{N}\sum (X_i-\bar{X})^{\top}(X_i-\bar{X})$' '\n'
          r'$Y = X\,W,\;\; k = 5$' '\n'
          '64 \u00d7 5 = 320 features',
          '#ede9fe', '#7c3aed', '#5b21b6', '#4c1d95')
    _card(xs[4], y_top, bw, bh, 'Stage 2C\nStacking Ensemble',
          'SVM-RBF \u00b7 RF \u00b7 XGBoost\nMeta-learner: LogReg\nIn-fold SMOTE\n'
          r'Output: $P(\mathrm{EPB}\mid X)$',
          '#dcfce7', '#16a34a', '#166534', '#14532d')

    _arrow(xs[0] + bw, yc, xs[1], yc)
    _arrow(xs[1] + bw, yc, xs[2], yc, color='#16a34a')
    _label(xs[1] + bw + gap / 2, yc + 0.12, 'clear\nsky', '#15803d')
    _arrow(xs[2] + bw, yc, xs[3], yc)
    _arrow(xs[3] + bw, yc, xs[4], yc)

    # ---------------- Linha inferior ----------------
    y_bot, bh2 = 0.3, 2.35

    # Ramo de rejeição (Stage 1 -> Inconclusive)
    cx1 = xs[1] + bw / 2
    _arrow(cx1, y_top, cx1, y_bot + bh2, color='#dc2626')
    _label(cx1 + 0.12, (y_top + y_bot + bh2) / 2, 'cloudy /\novercast', '#b91c1c',
           ha='left', va='center')
    _card(xs[1], y_bot, bw, bh2, 'Rejected:\nInconclusive',
          'Frame flagged as cloudy;\nno EPB decision issued\n(avoids cloud-induced\nfalse alarms)',
          '#fee2e2', '#dc2626', '#991b1b', '#7f1d1d')

    # Módulo de explicabilidade (abaixo de 2B e 2C)
    ex_x0 = xs[3] - 0.3
    ex_w = xs[4] + bw - ex_x0
    sep_y = _card(ex_x0, y_bot, ex_w, bh2,
                  'Dual-Level Spatial Explainability\n(pixel space, 64 \u00d7 64)', '',
                  '#f8fafc', '#475569', '#1e293b', '#334155')
    col_l = ex_x0 + ex_w * 0.25
    col_r = ex_x0 + ex_w * 0.75
    ax.plot([ex_x0 + ex_w / 2] * 2, [y_bot + 0.2, sep_y - 0.15], color='#cbd5e1', lw=0.9, zorder=3)
    mid = (y_bot + sep_y) / 2
    for cx, head, l1, eq in [
        (col_l, '(1) Unsupervised', 'Reconstruction error map', r'$E = |X - Y\,W^{\top}|$'),
        (col_r, '(2) Supervised', 'SHAP linear back-projection', r'$\Phi_{\mathrm{pixel}} = \Phi_{\mathrm{feat}}\,W^{\top}$'),
    ]:
        ax.text(cx, mid + 0.40, head, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color='#1e293b', zorder=3)
        ax.text(cx, mid + 0.02, l1, ha='center', va='center', fontsize=9, color='#334155', zorder=3)
        ax.text(cx, mid - 0.45, eq, ha='center', va='center', fontsize=10.5, color='#0f172a', zorder=3)

    cx3 = xs[3] + bw / 2
    cx4 = xs[4] + bw / 2
    _arrow(cx3, y_top, cx3, y_bot + bh2, color='#6d28d9')
    _label(cx3 + 0.12, (y_top + y_bot + bh2) / 2, r'$W$', '#6d28d9', ha='left', va='center')
    _arrow(cx4, y_top, cx4, y_bot + bh2, color='#15803d')
    _label(cx4 + 0.12, (y_top + y_bot + bh2) / 2, 'SHAP', '#15803d', ha='left', va='center')

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.08, facecolor='white')
        print(f"\U0001F4BE Pipeline architecture diagram saved: {save_path} ({dpi} DPI)")
    plt.show()
    return fig


# ============================================================================
# PARTE 11: ANÁLISE DE IMPORTÂNCIA BASEADA EM SHAP
# ============================================================================

def explain_shap_importance(system: 'EPBRecognitionSystem',
                            X_sample: np.ndarray, y_sample: int,
                            X_background: np.ndarray,
                            filename: Optional[str] = None,
                            method: str = 'tree',
                            image_folder: str = 'img-teste',
                            save_path: Optional[str] = None,
                            dpi: int = 300,
                            show_filename: bool = False) -> np.ndarray:
    """
    Explicação baseada em SHAP: quais regiões da imagem contribuem para a
    decisão de classificação via retroprojeção linear exata (Φ_pixel = Φ_feat · W^T).
    """
    # 1. Calcula SHAP
    shap_map, _ = system.get_shap_importance(
        X_sample, X_background, method=method
    )
    abs_shap_map = np.abs(shap_map)

    # 2. Importance antiga (erro de reconstrução)
    old_importance, _ = system.get_feature_importance(X_sample)

    # 3. Predição
    prob = system.predict_proba(X_sample.reshape(1, *X_sample.shape))[0]
    pred = np.argmax(prob)

    # 4. Análise textual
    print(f"\n{'='*70}")
    print(f"🔬 SHAP EXPLAINABILITY ANALYSIS — Supervised Pixel Attribution")
    print(f"{'='*70}")
    print(f"\n📸 Image: {filename or 'N/A'}")
    print(f"   • True class: {'EPB' if y_sample == 1 else 'No EPB'}")
    print(f"   • Prediction: {'EPB' if pred == 1 else 'No EPB'} ({prob[1]*100:.1f}%)")
    print(f"   • SHAP Method: {'TreeExplainer (XGBoost + RF)' if method == 'tree' else 'KernelExplainer (Full Ensemble)'}")

    # Análise por quadrante
    H, W = shap_map.shape
    quadrants = {
        'Upper Left': (slice(None, H // 2), slice(None, W // 2)),
        'Upper Right':  (slice(None, H // 2), slice(W // 2, None)),
        'Lower Left': (slice(H // 2, None), slice(None, W // 2)),
        'Lower Right':  (slice(H // 2, None), slice(W // 2, None)),
    }

    quadrant_values = {}
    for region, (rs, cs) in quadrants.items():
        quadrant_values[region] = (
            abs_shap_map[rs, cs].sum(),
            shap_map[rs, cs].mean(),
        )

    total_abs = abs_shap_map.sum()
    print(f"\n📊 SHAP contribution by quadrant:")
    for region in sorted(quadrant_values,
                         key=lambda r: quadrant_values[r][0], reverse=True):
        val, mean_signed = quadrant_values[region]
        pct = (val / total_abs * 100) if total_abs > 0 else 0
        dir_str = '↑ EPB' if mean_signed > 0 else '↓ No EPB'
        print(f"   • {region}: {pct:.1f}% ({dir_str})")

    # Correlação entre SHAP e erro de reconstrução
    corr, _ = spearmanr(abs_shap_map.ravel(), old_importance.ravel())
    print(f"\n📐 Spearman Correlation (|SHAP| vs. 2DPCA Reconstruction Error): {corr:.3f}")

    # 5. Visualização — 3 painéis (Original | Overlay SHAP | Mapa SHAP puro)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    img = _load_disk_image(filename, image_folder)
    if img is None:
        img = X_sample

    abs_vals = np.abs(shap_map)
    p99 = np.percentile(abs_vals[abs_vals > 0], 99) if np.any(abs_vals > 0) else 1e-10
    vmax = max(p99, 1e-10)

    # Painel 1: Imagem original
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Airglow Image', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Painel 2: Overlay SHAP sobre a imagem original
    from scipy.ndimage import zoom
    if img.shape != shap_map.shape:
        zoom_factors = (img.shape[0] / shap_map.shape[0],
                        img.shape[1] / shap_map.shape[1])
        shap_resized = zoom(shap_map, zoom_factors, order=1)
    else:
        shap_resized = shap_map

    axes[1].imshow(img, cmap='gray')
    im_overlay = axes[1].imshow(
        shap_resized, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        alpha=0.55, interpolation='bilinear'
    )
    axes[1].set_title('SHAP + Airglow Overlay\n(Red: EPB Evidence  |  Blue: Quiet Sky Evidence)',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im_overlay, ax=axes[1], fraction=0.046, label='SHAP value')

    # Painel 3: Mapa SHAP puro (quantitativo)
    im_pure = axes[2].imshow(shap_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             interpolation='bilinear')
    axes[2].set_title('Quantitative SHAP Attribution Map\n(Linear Subspace Back-Projection)',
                      fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im_pure, ax=axes[2], fraction=0.046, label='SHAP value')

    fname_suffix = f" — {filename}" if (filename and show_filename) else ""
    suptitle = (f'SHAP Attribution Analysis: True={"EPB" if y_sample == 1 else "No EPB"} | '
                f'Predicted={"EPB" if pred == 1 else "No EPB"} ({prob[1]*100:.1f}%){fname_suffix}')
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"💾 High-resolution SHAP map saved: {save_path} ({dpi} DPI)")
    plt.show()

    return shap_map


def predict_with_qc(
    system: 'EPBRecognitionSystem',
    img_raw: np.ndarray,
    target_size: Tuple[int, int] = (64, 64),
    threshold_stars: int = 4500,
    min_mean_brightness: float = 35.0
) -> CascadedPredictionResult:
    """
    Executa a inferência completa em 2 estágios para uma única imagem bruta.
    """
    return system.predict_cascaded(
        img_raw,
        target_size=target_size,
        threshold_stars=threshold_stars,
        min_mean_brightness=min_mean_brightness
    )


def analyze_temporal_sequence_with_qc(
    system: 'EPBRecognitionSystem',
    image_folder: str,
    target_size: Tuple[int, int] = (64, 64),
    show_thumbnails: bool = True,
    thumbnail_interval: int = 10,
    threshold_stars: int = 4500,
    min_mean_brightness: float = 35.0,
    save_path: Optional[str] = None,
    dpi: int = 300,
    theme: str = 'light'
) -> Optional[pd.DataFrame]:
    """Alias explícito para analyze_temporal_sequence com use_qc=True."""
    return analyze_temporal_sequence(
        system=system,
        image_folder=image_folder,
        target_size=target_size,
        show_thumbnails=show_thumbnails,
        thumbnail_interval=thumbnail_interval,
        use_qc=True,
        threshold_stars=threshold_stars,
        min_mean_brightness=min_mean_brightness,
        save_path=save_path,
        dpi=dpi,
        theme=theme
    )


"""
EPB Detection — núcleo compartilhado (V5)

Biblioteca enxuta reutilizada pelo sistema de detecção V5 (epb_detection_v5.py) e
pelos notebooks. Contém apenas o que os consumidores atuais usam:

  - 2DPCA (extração de features) ............................ TwoDPCA
  - Controle de qualidade do céu / Estágio 1 ............... SkyQualityResult,
                                                             detect_stars, assess_sky_quality
  - Resultado da predição em cascata ....................... CascadedPredictionResult
  - Agrupamento por noite e métricas binárias .............. extract_night_groups, _binary_metrics
  - Logging para notebooks ................................. setup_logging
  - Persistência do modelo (joblib + SHA-256) ............. EPBModelData, save_model
  - Sistema de reconhecimento 2DPCA + ensemble ............ EPBRecognitionSystem
  - Gráficos usados pelo notebook principal ............... plot_performance_metrics,
                                                             plot_confusion_matrix_analysis

Historicamente estes símbolos viviam no módulo "V4" (epb_detection_system_v2.py), que
misturava este núcleo com muito código de análise/benchmark já não utilizado. Este módulo
é o resultado de enxugar: só o conjunto alcançável a partir dos notebooks e dos testes.

Autor: Alberto Ivo da Costa Vieira
"""

import os
import re
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from scipy.ndimage import gaussian_filter

import joblib


logger = logging.getLogger(__name__)


# ============================================================================
# ESTRUTURAS DE RESULTADO
# ============================================================================

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


# ============================================================================
# PERSISTÊNCIA DO MODELO
# ============================================================================

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
# ESTÁGIO 1: QUALIDADE DO CÉU (DETECÇÃO DE ESTRELAS / NUVENS)
# ============================================================================

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


# ============================================================================
# 2DPCA — EXTRAÇÃO DE FEATURES
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


# ============================================================================
# SISTEMA COMPLETO: 2DPCA + ENSEMBLE + BALANCEAMENTO
# ============================================================================

class EPBRecognitionSystem:
    """
    Sistema completo de reconhecimento de EPBs
    - 2DPCA para extração de features
    - Balanceamento de classes (SMOTE)
    - Ensemble de classificadores

    Usado diretamente como linha de comparação "V4" (stacking + SMOTE) e como classe
    base de EPBRecognitionSystemV5 (que reutiliza extract_features / predict / predict_proba
    / _setup_balancing e sobrescreve fit).
    """

    def __init__(self, n_components: int = 15, balance_data: bool = True,
                 balance_method: str = 'smote',
                 ensemble_params: Optional[dict] = None):
        """
        Args:
            n_components: Número de componentes 2DPCA
            balance_data: Aplicar balanceamento de classes
            balance_method: 'smote', 'undersample', ou 'combined'
            ensemble_params: Dict com hiperparâmetros do ensemble. Se None, usa defaults.
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

        Se self.ensemble_params estiver preenchido, usa os hiperparâmetros otimizados;
        caso contrário, usa defaults.
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
        ao contrário do erro de reconstrução.

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
# AGRUPAMENTO POR NOITE E MÉTRICAS
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


# ============================================================================
# VISUALIZAÇÃO (usada pelo notebook principal)
# ============================================================================

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

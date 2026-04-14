"""
Sistema Completo de Reconhecimento de EPBs usando 2DPCA
Autor: Sistema de ML para Detecção de Bolhas de Plasma Equatorial
"""

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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

def load_epb_dataset(data_folder: str, label_file: str, target_size: Tuple[int, int] = (64, 64), apply_preprocessing: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
                img = preprocess_epb_image(img)
            else:
                img = img / 255.0  # Apenas normaliza
            
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
    logger.info(f"  Sem EPB: {n_no_epb} ({n_no_epb/len(labels)*100:.1f}%)")
    
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
                     f"({'EPB' if ni['class'] == 'epb' else 'Sem EPB'})")

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
    logger.info(f"  Treino: {len(X_train)} imgs de {len(train_nights_sorted)} noites "
                f"({train_nights_sorted[0]}–{train_nights_sorted[-1]})")
    logger.info(f"  Teste:  {len(X_test)} imgs de {len(test_nights_sorted)} noites "
                f"({test_nights_sorted[0]}–{test_nights_sorted[-1]})")

    n_epb_train = int(y_train.sum())
    n_epb_test = int(y_test.sum())
    logger.info(f"  Treino — EPB: {n_epb_train}, Sem EPB: {len(y_train) - n_epb_train}")
    logger.info(f"  Teste  — EPB: {n_epb_test}, Sem EPB: {len(y_test) - n_epb_test}")

    actual_test_ratio = len(X_test) / len(images)
    logger.info(f"  Ratio teste: {actual_test_ratio:.1%} (objectivo: {test_size:.0%})")

    if n_epb_test == 0 or n_epb_test == len(y_test):
        logger.warning("⚠️  ATENÇÃO: Teste sem representação de ambas as classes! "
                       "Considere adicionar mais noites de observação.")

    return X_train, X_test, y_train, y_test, files_train, files_test


def preprocess_epb_image(img: np.ndarray, apply_background_removal: bool = True, enhance_contrast: bool = True) -> np.ndarray:
    """
    Pré-processamento específico para realçar estruturas de EPB
    
    Args:
        img: Imagem em escala de cinza [0, 255] ou [0, 1]
        apply_background_removal: Remove gradientes de fundo
        enhance_contrast: Aplica CLAHE para realçar contraste local
    
    Returns:
        Imagem pré-processada normalizada [0, 1]
    """
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
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Seleciona top n_components
        self.projection_matrix = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance
        
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
    test_range = range(5, min(max_components + 1, X_train.shape[2]), 5)
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
      - weights (VotingClassifier)

    Args:
        X_train: Imagens de treino (n, h, w) — sem augmentation
        y_train: Labels
        image_shape: Forma das imagens (height, width)
        n_iter: Combinações aleatórias a testar
        cv_folds: Folds de validação cruzada
        scoring: Métrica de otimização ('f1', 'accuracy', 'roc_auc')
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
    print(f"   Métrica: {scoring}")
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
        ('clf', VotingClassifier(
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
            voting='soft',
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
        'clf__weights': [[1, 1, 1], [2, 3, 2], [1, 2, 1], [2, 2, 3], [1, 3, 1]],
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
        'voting_weights': list(best.get('clf__weights', [2, 3, 2])),
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
    ax.set_title('Score vs. Componentes 2DPCA', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)

    # 2. Distribuição dos scores
    ax = axes[1]
    ax.hist(df_results['mean_test_score'], bins=15,
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(best_score, color='red', linestyle='--', linewidth=2,
               label=f'Melhor: {best_score:.4f}')
    ax.set_xlabel(f'{scoring} (teste)', fontsize=11)
    ax.set_ylabel('Contagem', fontsize=11)
    ax.set_title('Distribuição dos Scores', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Treino vs Teste (diagnóstico de overfitting)
    ax = axes[2]
    ax.scatter(df_results['mean_train_score'],
               df_results['mean_test_score'],
               alpha=0.5, s=30, c='steelblue')
    lims = [min(df_results['mean_test_score'].min(),
                df_results['mean_train_score'].min()) - 0.02, 1.02]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Treino = Teste')
    ax.set_xlabel(f'{scoring} (treino)', fontsize=11)
    ax.set_ylabel(f'{scoring} (teste)', fontsize=11)
    ax.set_title('Treino vs. Teste (Overfitting?)', fontsize=12,
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
    
    def _create_ensemble(self, y_train: np.ndarray) -> VotingClassifier:
        """Ensemble otimizado para detecção de EPBs.

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
                random_state=42,
                eval_metric='logloss'
            )),
            ('rf', RandomForestClassifier(
                n_estimators=p.get('rf_n_estimators', 200),
                max_depth=p.get('rf_max_depth', 15),
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            ))
        ]

        return VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=p.get('voting_weights', [2, 3, 2])
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
            shap_map: Atribuição SHAP no espaço da imagem (H, W) — com sinal:
                      positivo → evidência a favor de EPB,
                      negativo → evidência contra EPB
            shap_features: Valores SHAP no espaço de features (n_features,)
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

        # Retro-projeção para espaço da imagem
        # features = image_row @ projection_matrix → (H, n_comp)
        # shap_map = shap_2d @ projection_matrix.T → (H, W)
        H = X_sample.shape[0]
        sv_2d = sv.reshape(H, self.n_components)
        shap_map = sv_2d @ self.tdpca.projection_matrix.T

        return shap_map, sv


# ============================================================================
# PARTE 4: VISUALIZAÇÃO E INTERPRETAÇÃO
# ============================================================================

def plot_comprehensive_analysis(system: 'EPBRecognitionSystem', X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                y_pred: np.ndarray, y_proba: np.ndarray,
                                filenames_test: Optional[List[str]] = None) -> plt.Figure:
    """Análise visual completa do sistema"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(5, 6, hspace=1, wspace=0.4)
    
    # ========== ROW 1: EXEMPLOS DE CLASSIFICAÇÃO ==========
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_test[i], cmap='plasma', aspect='auto')
        
        true_label = "EPB" if y_test[i] == 1 else "Sem EPB"
        pred_label = "EPB" if y_pred[i] == 1 else "Sem EPB"
        prob = y_proba[i, 1] * 100
        
        color = 'green' if y_test[i] == y_pred[i] else 'red'
        title = f'Real: {true_label}\nPred: {pred_label} ({prob:.0f}%)'
        if filenames_test:
            title = f"{filenames_test[i][:15]}...\n" + title
        
        ax.set_title(title, fontsize=8, color=color)
        ax.axis('off')
    
    # ========== ROW 2: ANÁLISE DE FEATURES ==========
    # Variância por componente
    ax = fig.add_subplot(gs[1, 0:2])
    variance_ratio = system.tdpca.explained_variance_ratio
    ax.bar(range(len(variance_ratio)), variance_ratio, color='steelblue', alpha=0.7)
    ax.set_xlabel('Componente', fontsize=10)
    ax.set_ylabel('Variância Explicada', fontsize=10)
    ax.set_title('Importância de Cada Componente 2DPCA', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Variância cumulativa
    ax = fig.add_subplot(gs[1, 2:4])
    cumulative = np.cumsum(variance_ratio)
    ax.plot(cumulative, 'o-', color='steelblue', linewidth=2)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
    ax.set_xlabel('Número de Componentes', fontsize=10)
    ax.set_ylabel('Variância Acumulada', fontsize=10)
    ax.set_title('Variância Cumulativa Explicada', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Matriz de confusão
    ax = fig.add_subplot(gs[1, 4:6])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sem EPB', 'EPB'],
                yticklabels=['Sem EPB', 'EPB'], ax=ax, 
                cbar_kws={'label': 'Contagem'})
    ax.set_title('Matriz de Confusão', fontsize=11, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=10)
    ax.set_xlabel('Classe Predita', fontsize=10)
    
    # ========== ROW 3: MÉTRICAS DE PERFORMANCE ==========
    # Curva ROC
    ax = fig.add_subplot(gs[2, 0:2])
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Classificador Aleatório')
    ax.set_xlabel('Taxa de Falso Positivo', fontsize=10)
    ax.set_ylabel('Taxa de Verdadeiro Positivo', fontsize=10)
    ax.set_title('Curva ROC', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    # Curva Precision-Recall
    ax = fig.add_subplot(gs[2, 2:4])
    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color='purple', lw=2,
            label=f'PR (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title('Curva Precision-Recall', fontsize=11, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    
    # Distribuição de probabilidades
    ax = fig.add_subplot(gs[2, 4:6])
    epb_probs = y_proba[y_test == 1, 1]
    no_epb_probs = y_proba[y_test == 0, 1]
    
    ax.hist(no_epb_probs, bins=20, alpha=0.6, label='Sem EPB (real)', 
            color='blue', density=True)
    ax.hist(epb_probs, bins=20, alpha=0.6, label='EPB (real)', 
            color='red', density=True)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
               label='Threshold (0.5)')
    ax.set_xlabel('Probabilidade de EPB', fontsize=10)
    ax.set_ylabel('Densidade', fontsize=10)
    ax.set_title('Distribuição de Probabilidades Preditas', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ========== ROW 4: INTERPRETAÇÃO DAS FEATURES ==========
    # Pega exemplos de cada classe
    epb_idx = np.where(y_test == 1)[0][0]
    no_epb_idx = np.where(y_test == 0)[0][0]
    
    for i, (idx, title_prefix) in enumerate([(epb_idx, 'EPB'), 
                                              (no_epb_idx, 'Sem EPB')]):
        # Imagem original
        ax = fig.add_subplot(gs[3, i*3])
        ax.imshow(X_test[idx], cmap='plasma')
        ax.set_title(f'{title_prefix}: Original', fontsize=10)
        ax.axis('off')
        
        # Mapa de importância
        importance_map, reconstructed = system.get_feature_importance(X_test[idx])
        
        ax = fig.add_subplot(gs[3, i*3+1])
        im = ax.imshow(importance_map, cmap='hot')
        ax.set_title(f'{title_prefix}: Importância', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Reconstrução
        ax = fig.add_subplot(gs[3, i*3+2])
        ax.imshow(reconstructed, cmap='plasma')
        ax.set_title(f'{title_prefix}: Reconstruído', fontsize=10)
        ax.axis('off')
    
    # ========== ROW 5: COMPONENTES PRINCIPAIS ==========
    # Visualiza primeiros componentes principais
    n_comp_to_show = min(6, system.n_components)
    for i in range(n_comp_to_show):
        ax = fig.add_subplot(gs[4, i])
        component = system.tdpca.projection_matrix[:, i].reshape(-1, 1)
        # Expande para visualização
        component_img = np.tile(component, (1, X_test.shape[1]))
        ax.imshow(component_img.T, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Comp {i+1}\n({variance_ratio[i]:.1%})', fontsize=9)
        ax.axis('off')
    
    plt.suptitle('🌌 Análise Completa: Sistema de Reconhecimento de EPBs', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def explain_feature_importance(system: 'EPBRecognitionSystem', X_sample: np.ndarray, y_sample: int,
                               filename: Optional[str] = None,
                               mode: str = 'full', image_folder: str = 'img-teste') -> np.ndarray:
    """
    Explica quais regiões da imagem contribuem para a classificação.
    
    Args:
        system: EPBRecognitionSystem treinado
        X_sample: Imagem pré-processada (2D array)
        y_sample: Label real (0 ou 1)
        filename: Nome do ficheiro da imagem
        mode: Modo de visualização:
              'full'       — Análise textual completa + 3 painéis (original processada,
                             mapa de importância, reconstrução 2DPCA)
              'simple'     — 2 painéis (imagem original do disco + mapa de importância)
              'comparison' — 3 painéis (imagem original do disco, processada, mapa de importância)
        image_folder: Pasta onde procurar a imagem original (usado por 'simple' e 'comparison')
    
    Returns:
        importance_map: Mapa de importância (2D array)
    """
    importance_map, reconstructed = system.get_feature_importance(X_sample)
    
    # Predição (uma só chamada)
    prob = system.predict_proba(X_sample.reshape(1, *X_sample.shape))[0]
    pred = np.argmax(prob)
    
    suptitle = (f'Interpretação: {"EPB" if y_sample == 1 else "Sem EPB"} '
                f'(Predição: {"EPB" if pred == 1 else "Sem EPB"}, {prob[1]*100:.1f}%)')
    
    # --- Modo FULL: análise textual + visualização completa ---
    if mode == 'full':
        print("\n" + "="*70)
        print("🔍 INTERPRETAÇÃO DAS FEATURES")
        print("="*70)
        
        print(f"\n📸 Imagem: {filename if filename else 'N/A'}")
        print(f"   • Classe real: {'EPB' if y_sample == 1 else 'Sem EPB'}")
        print(f"   • Predição: {'EPB' if pred == 1 else 'Sem EPB'}")
        print(f"   • Probabilidade EPB: {prob[1]*100:.2f}%")
        print(f"   • Confiança: {'Alta' if max(prob) > 0.8 else 'Média' if max(prob) > 0.6 else 'Baixa'}")
        
        threshold = np.percentile(importance_map, 90)
        important_pixels = importance_map > threshold
        
        h, w = importance_map.shape
        quadrants = {
            'Superior Esquerdo': important_pixels[:h//2, :w//2].sum(),
            'Superior Direito': important_pixels[:h//2, w//2:].sum(),
            'Inferior Esquerdo': important_pixels[h//2:, :w//2].sum(),
            'Inferior Direito': important_pixels[h//2:, w//2:].sum()
        }
        
        print(f"\n📊 Regiões mais importantes (pixels críticos):")
        total = sum(quadrants.values())
        for region, count in sorted(quadrants.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"   • {region}: {count} pixels ({percentage:.1f}%)")
        
        print(f"\n💡 Interpretação:")
        if pred == 1:
            print("   ✓ O modelo identificou padrões característicos de EPB:")
            print("     - Estruturas verticais/irregulares")
            print("     - Deplecções localizadas de plasma")
            print("     - Distribuição espacial típica de bolhas")
        else:
            print("   ✓ O modelo NÃO detectou padrões de EPB:")
            print("     - Distribuição uniforme de plasma")
            print("     - Ausência de estruturas irregulares")
            print("     - Características de ionosfera quieta")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(X_sample, cmap='plasma')
        axes[0].set_title('Imagem Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(importance_map, cmap='hot')
        axes[1].set_title('Mapa de Importância\n(Regiões críticas para classificação)',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, label='Importância')
        
        axes[2].imshow(reconstructed, cmap='plasma')
        axes[2].set_title('Reconstrução 2DPCA\n(Features capturadas)',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # --- Modo SIMPLE: imagem do disco + mapa de importância ---
    elif mode == 'simple':
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        img = _load_disk_image(filename, image_folder)
        if img is None:
            img = X_sample
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title(filename, fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        im = axes[1].imshow(importance_map, cmap='hot')
        axes[1].set_title('Mapa de Importância\n(Regiões críticas para classificação)',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, label='Importância')
        
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # --- Modo COMPARISON: disco + processada + mapa ---
    elif mode == 'comparison':
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        img_disk = _load_disk_image(filename, image_folder)
        if img_disk is None:
            img_disk = X_sample
        
        axes[0].imshow(img_disk, cmap='gray')
        axes[0].set_title(f'{filename}\n(Imagem Pré-Processada)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(X_sample, cmap='plasma')
        axes[1].set_title('Features extraídas pelo 2DPCA\n(Usada pelo modelo)',
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        im = axes[2].imshow(importance_map, cmap='hot')
        axes[2].set_title('Mapa de Importância\n(Regiões críticas para classificação)',
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, label='Importância')
        
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        plt.tight_layout()
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
                                 filename: Optional[str] = None, image_folder: str = 'img-teste') -> np.ndarray:
    return explain_feature_importance(system, X_sample, y_sample, filename,
                                      mode='simple', image_folder=image_folder)

def explain_feature_importance_3(system: 'EPBRecognitionSystem', X_sample: np.ndarray, y_sample: int,
                                 filename: Optional[str] = None, image_folder: str = 'img-teste') -> np.ndarray:
    return explain_feature_importance(system, X_sample, y_sample, filename,
                                      mode='comparison', image_folder=image_folder)


# ============================================================================
# PARTE 5: ANÁLISE TEMPORAL DE SEQUÊNCIA DE IMAGENS
# ============================================================================

def analyze_temporal_sequence(system: 'EPBRecognitionSystem', image_folder: str,
                              target_size: Tuple[int, int] = (64, 64),
                              show_thumbnails: bool = True, thumbnail_interval: int = 10) -> Optional[pd.DataFrame]:
    """
    Analisa uma sequência temporal de imagens e plota a evolução da probabilidade de EPB.
    
    Esta função é útil para:
    - Verificar se o modelo detecta EPBs antes de serem visíveis
    - Analisar a evolução temporal de eventos de EPB
    - Identificar o momento de início/fim de perturbações ionosféricas
    
    Args:
        system: Sistema EPBRecognitionSystem treinado
        image_folder: Pasta com imagens ordenadas cronologicamente
        target_size: Tamanho para redimensionar imagens (deve ser igual ao treino)
        show_thumbnails: Se True, mostra thumbnails das imagens no gráfico
        thumbnail_interval: Intervalo entre thumbnails (ex: 10 = mostra a cada 10 imagens)
    
    Returns:
        DataFrame com timestamps, probabilidades e predições
    """
    print("="*70)
    print("🕐 ANÁLISE TEMPORAL DE SEQUÊNCIA DE IMAGENS")
    print("="*70)
    
    image_folder = Path(image_folder)
    
    # Lista e ordena imagens
    image_files = sorted(list(image_folder.glob('*.png')))
    
    if len(image_files) == 0:
        print(f"⚠️  Nenhuma imagem encontrada em: {image_folder}")
        return None
    
    print(f"📂 Pasta: {image_folder}")
    print(f"📸 Total de imagens: {len(image_files)}")
    
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
            
            # Carrega e processa imagem
            img = Image.open(img_path).convert('L')
            img = np.array(img.resize(target_size))
            img_processed = preprocess_epb_image(img)
            
            # Faz predição
            prob = system.predict_proba(img_processed.reshape(1, *img_processed.shape))[0]
            pred = system.predict(img_processed.reshape(1, *img_processed.shape))[0]
            
            results.append({
                'filename': img_path.name,
                'filepath': str(img_path),
                'timestamp': timestamp,
                'prob_no_epb': prob[0],
                'prob_epb': prob[1],
                'prediction': pred,
                'prediction_label': 'EPB' if pred == 1 else 'Sem EPB'
            })
            
        except Exception as e:
            logger.warning(f"Erro em {img_path.name}: {str(e)}")
    
    # Cria DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        logger.error("Nenhuma imagem processada com sucesso.")
        return None
    
    # Estatísticas
    n_epb = (df['prediction'] == 1).sum()
    n_no_epb = (df['prediction'] == 0).sum()
    
    print(f"\n📊 Resumo das predições:")
    print(f"   • Com EPB: {n_epb} imagens ({n_epb/len(df)*100:.1f}%)")
    print(f"   • Sem EPB: {n_no_epb} imagens ({n_no_epb/len(df)*100:.1f}%)")
    print(f"   • Probabilidade média de EPB: {df['prob_epb'].mean()*100:.1f}%")
    print(f"   • Probabilidade máxima de EPB: {df['prob_epb'].max()*100:.1f}%")
    print(f"   • Probabilidade mínima de EPB: {df['prob_epb'].min()*100:.1f}%")
    
    # Identifica transições (mudanças de estado)
    df['state_change'] = df['prediction'].diff().fillna(0).abs()
    transitions = df[df['state_change'] == 1]
    
    if len(transitions) > 0:
        print(f"\n🔄 Transições detectadas ({len(transitions)}):")
        for _, row in transitions.iterrows():
            time_str = row['timestamp'].strftime('%H:%M:%S') if row['timestamp'] else 'N/A'
            print(f"   • {time_str} - Mudou para: {row['prediction_label']} ({row['prob_epb']*100:.1f}%)")
    
    # =========================================================================
    # VISUALIZAÇÃO ELEGANTE - Estilo moderno e profissional
    # =========================================================================
    
    # Configuração de estilo (contexto isolado para não afetar outros gráficos)
    _rc_backup = plt.rcParams.copy()
    plt.style.use('default')
    
    # Paleta de cores elegante
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
    }
    
    # Cria figura com fundo escuro
    fig = plt.figure(figsize=(18, 11), facecolor=COLORS['background'])
    
    if show_thumbnails:
        gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.8, 1], hspace=1,
                              left=0.06, right=0.94, top=0.88, bottom=0.08)
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 0.8], hspace=1,
                              left=0.06, right=0.94, top=0.88, bottom=0.08)
    
    # -------------------------------------------------------------------------
    # GRÁFICO PRINCIPAL: Probabilidade ao longo do tempo
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0], facecolor=COLORS['card_bg'])
    
    if df['timestamp'].notna().all():
        x_values = df['timestamp']
        x_label = 'Horário (UT)'
    else:
        x_values = range(len(df))
        x_label = 'Índice da Imagem'
    
    prob_values = df['prob_epb'] * 100
    
    # Gradiente de preenchimento usando múltiplas camadas
    from matplotlib.colors import LinearSegmentedColormap
    
    # Cria preenchimento com gradiente simulado
    ax1.fill_between(x_values, 0, prob_values, alpha=0.15, color=COLORS['fill_gradient_top'])
    ax1.fill_between(x_values, 0, prob_values * 0.7, alpha=0.15, color=COLORS['fill_gradient_bottom'])
    ax1.fill_between(x_values, 0, prob_values * 0.4, alpha=0.15, color=COLORS['fill_gradient_bottom'])
    
    # Linha principal com efeito de brilho (glow)
    ax1.plot(x_values, prob_values, color=COLORS['line_main'], linewidth=3.5, alpha=0.3)  # Glow
    ax1.plot(x_values, prob_values, color=COLORS['line_main'], linewidth=2, 
             label='Probabilidade de EPB', zorder=5)
    
    # Pontos nos dados com efeito sutil
    scatter_colors = [COLORS['epb_positive'] if p > 50 else COLORS['epb_negative'] for p in prob_values]
    ax1.scatter(x_values, prob_values, c=scatter_colors, s=8, alpha=0.6, zorder=6, edgecolors='none')
    
    # Linhas de referência com estilo elegante
    ax1.axhline(y=50, color=COLORS['threshold'], linestyle='--', linewidth=2, 
                label='Threshold (50%)', alpha=0.9)
    ax1.axhline(y=70, color=COLORS['confidence_medium'], linestyle=':', linewidth=1.8, 
                label='Confiança Média (70%)', alpha=0.8)
    ax1.axhline(y=90, color=COLORS['confidence_high'], linestyle=':', linewidth=1.8, 
                label='Confiança Alta (90%)', alpha=0.8)
    
    # Marca transições com estilo sutil
    for _, row in transitions.iterrows():
        if df['timestamp'].notna().all():
            x_pos = row['timestamp']
        else:
            x_pos = df[df['filename'] == row['filename']].index[0]
        ax1.axvline(x=x_pos, color=COLORS['transition'], linestyle='-', alpha=0.4, linewidth=1.5)
    
    # Estilização dos eixos
    ax1.set_xlabel(x_label, fontsize=12, color=COLORS['text'], fontweight='medium', labelpad=10)
    ax1.set_ylabel('Probabilidade de EPB (%)', fontsize=12, color=COLORS['text'], 
                   fontweight='medium', labelpad=10)
    ax1.set_ylim(-2, 105)
    ax1.set_xlim(x_values.iloc[0] if hasattr(x_values, 'iloc') else x_values[0], 
                 x_values.iloc[-1] if hasattr(x_values, 'iloc') else x_values[-1])
    
    # Grid elegante
    ax1.grid(True, linestyle='-', alpha=0.2, color=COLORS['grid'], linewidth=0.5)
    ax1.set_axisbelow(True)
    
    # Personaliza ticks
    ax1.tick_params(axis='both', colors=COLORS['text_secondary'], labelsize=10)
    for spine in ax1.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.5)
    
    # Rotaciona labels do eixo X se forem timestamps
    if df['timestamp'].notna().all():
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Legenda elegante - posicionada fora da área do gráfico
    legend = ax1.legend(loc='upper left', fontsize=10, framealpha=0.95,
                        facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'],
                        labelcolor=COLORS['text'], bbox_to_anchor=(0.01, 0.99))
    legend.get_frame().set_linewidth(0.5)
    
    # Título do subplot
    ax1.set_title('Evolução Temporal da Probabilidade de EPB', fontsize=14, 
                  color=COLORS['text'], fontweight='bold', pad=15, loc='left')
    
    # -------------------------------------------------------------------------
    # GRÁFICO SECUNDÁRIO: Timeline de classificação
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1], facecolor=COLORS['card_bg'])
    
    # Cria barras com cores baseadas na predição
    bar_colors = [COLORS['epb_negative'] if p == 0 else COLORS['epb_positive'] for p in df['prediction']]
    bars = ax2.bar(range(len(df)), [1] * len(df), color=bar_colors, alpha=0.85, width=1.0, 
                   edgecolor='none')
    
    # Adiciona indicador de intensidade (opacidade baseada na probabilidade)
    for i, (bar, prob) in enumerate(zip(bars, df['prob_epb'])):
        bar.set_alpha(0.3 + 0.6 * abs(prob - 0.5) * 2)  # Mais opaco = mais certeza
    
    ax2.set_ylabel('Status', fontsize=11, color=COLORS['text'], fontweight='medium', labelpad=10)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels([''])
    ax2.set_xlabel('Sequência de Imagens', fontsize=11, color=COLORS['text'], 
                   fontweight='medium', labelpad=10)
    ax2.set_xlim(-0.5, len(df) - 0.5)
    ax2.set_ylim(0, 1)
    
    # Estilização
    ax2.tick_params(axis='both', colors=COLORS['text_secondary'], labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.5)
    
    # Título com legenda integrada
    ax2.set_title('Classificação ao Longo do Tempo', fontsize=12, 
                  color=COLORS['text'], fontweight='bold', pad=10, loc='left')
    
    # Adiciona mini-legenda no gráfico
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['epb_negative'], alpha=0.8, label='Sem EPB'),
        Patch(facecolor=COLORS['epb_positive'], alpha=0.8, label='Com EPB')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9,
               facecolor=COLORS['card_bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    # -------------------------------------------------------------------------
    # THUMBNAILS (se habilitado)
    # -------------------------------------------------------------------------
    if show_thumbnails:
        ax3 = fig.add_subplot(gs[2], facecolor=COLORS['background'])
        ax3.axis('off')
        ax3.set_title('Amostra de Imagens', fontsize=12, 
                      color=COLORS['text'], fontweight='bold', pad=10, loc='left')
        
        # Seleciona imagens para mostrar
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
                
                # Cria sub-eixo para thumbnail
                left = i * thumb_width + 0.005
                bottom = 0.15
                width = thumb_width - 0.01
                height = 0.65
                
                ax_thumb = ax3.inset_axes([left, bottom, width, height])
                ax_thumb.imshow(img, cmap='gray', aspect='equal')
                ax_thumb.axis('off')
                
                # Borda colorida baseada na classificação
                border_color = COLORS['epb_positive'] if row_data['prediction'] == 1 else COLORS['epb_negative']
                for spine in ax_thumb.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)
                
                # Label com horário - estilo elegante
                time_label = row_data['timestamp'].strftime('%H:%M') if row_data['timestamp'] else str(idx)
                prob_label = f"{row_data['prob_epb']*100:.0f}%"
                
                ax_thumb.text(0.5, -0.08, time_label, transform=ax_thumb.transAxes,
                             fontsize=8, color=COLORS['text'], ha='center', va='top',
                             fontweight='medium')
                ax_thumb.text(0.5, -0.28, prob_label, transform=ax_thumb.transAxes,
                             fontsize=7, color=COLORS['text_secondary'], ha='center', va='top')
                
            except Exception:
                pass
    
    # -------------------------------------------------------------------------
    # TÍTULO GERAL
    # -------------------------------------------------------------------------
    if df['timestamp'].notna().all():
        date_str = df['timestamp'].iloc[0].strftime('%d/%m/%Y')
        time_range = f"{df['timestamp'].iloc[0].strftime('%H:%M')} – {df['timestamp'].iloc[-1].strftime('%H:%M')} UT"
        title_text = f'Análise Temporal de EPBs'
        subtitle_text = f'{date_str}  •  {time_range}  •  {len(df)} imagens'
    else:
        title_text = 'Análise Temporal de EPBs'
        subtitle_text = f'{len(df)} imagens analisadas'
    
    fig.text(0.5, 0.96, title_text, ha='center', va='top', fontsize=18, 
             color=COLORS['text'], fontweight='bold')
    fig.text(0.5, 0.925, subtitle_text, ha='center', va='top', fontsize=11, 
             color=COLORS['text_secondary'], fontweight='normal')
    
    # Adiciona estatísticas no canto
    stats_text = f"EPB: {(df['prediction'] == 1).sum()}/{len(df)}  |  Prob. média: {df['prob_epb'].mean()*100:.1f}%"
    fig.text(0.94, 0.96, stats_text, ha='right', va='top', fontsize=10, 
             color=COLORS['text_secondary'], fontweight='normal', 
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['card_bg'], 
                      edgecolor=COLORS['grid'], linewidth=0.5))
    
    plt.show()
    plt.rcParams.update(_rc_backup)
    
    print("\n" + "="*70)
    print("✅ ANÁLISE TEMPORAL CONCLUÍDA")
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
        print("⚠️  DataFrame vazio. Execute analyze_temporal_sequence primeiro.")
        return
    
    print("="*70)
    print("🔍 ANÁLISE DE PERÍODOS CRÍTICOS")
    print("="*70)
    
    # Detecta a data da observação
    if df_temporal['timestamp'].notna().all():
        obs_date = df_temporal['timestamp'].iloc[0].strftime('%d/%m/%Y')
        time_start = df_temporal['timestamp'].iloc[0].strftime('%H:%M')
        time_end = df_temporal['timestamp'].iloc[-1].strftime('%H:%M')
        print(f"📅 Data: {obs_date} | Período: {time_start} - {time_end} UT\n")
    
    # -------------------------------------------------------------------------
    # 1. Detecta TRANSIÇÕES (mudanças de estado Sem EPB <-> Com EPB)
    # -------------------------------------------------------------------------
    transitions = df_temporal[df_temporal['state_change'] == 1].copy()
    
    if len(transitions) > 0:
        print("🔄 TRANSIÇÕES DETECTADAS")
        print("-"*70)
        print("   Momentos onde o modelo mudou de classificação:\n")
        
        for i, (idx, row) in enumerate(transitions.iterrows()):
            time_str = row['timestamp'].strftime('%H:%M:%S') if row['timestamp'] else 'N/A'
            prob = row['prob_epb'] * 100
            
            if row['prediction'] == 1:
                direction = "Sem EPB → COM EPB 🔴"
            else:
                direction = "Com EPB → SEM EPB 🟢"
            
            print(f"   {i+1}. {time_str} UT | {direction} | Prob: {prob:.1f}%")
        
        print()
    
    # -------------------------------------------------------------------------
    # 2. Análise da PRIMEIRA TRANSIÇÃO para EPB (início do evento)
    # -------------------------------------------------------------------------
    first_epb_transition = transitions[transitions['prediction'] == 1]
    
    if len(first_epb_transition) > 0:
        first_epb = first_epb_transition.iloc[0]
        first_epb_time = first_epb['timestamp']
        
        print("⚡ ANÁLISE DO INÍCIO DO EVENTO EPB")
        print("-"*70)
        print(f"   Primeira detecção de EPB: {first_epb_time.strftime('%H:%M:%S')} UT\n")
        
        window_before = timedelta(minutes=15)
        window_after = timedelta(minutes=15)
        
        start_window = first_epb_time - window_before
        end_window = first_epb_time + window_after
        
        mask = (df_temporal['timestamp'] >= start_window) & (df_temporal['timestamp'] <= end_window)
        df_window = df_temporal[mask].copy()
        
        print(f"   Janela de análise: {start_window.strftime('%H:%M')} - {end_window.strftime('%H:%M')} UT")
        print(f"   Imagens na janela: {len(df_window)}\n")
        
        print("   Detalhamento:")
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
            
            marker = " ◄── TRANSIÇÃO" if row['timestamp'] == first_epb_time else ""
            
            print(f"   {time_str} UT | {emoji} {prob:5.1f}% | {pred}{marker}")
        
        print("   " + "-"*60)
        
        # Análise de detecção precoce
        pre_transition = df_temporal[df_temporal['timestamp'] < first_epb_time].tail(5)
        if len(pre_transition) > 0:
            avg_prob_before = pre_transition['prob_epb'].mean() * 100
            max_prob_before = pre_transition['prob_epb'].max() * 100
            
            print(f"\n   📊 Estatísticas pré-transição (últimas 5 imagens):")
            print(f"      • Probabilidade média: {avg_prob_before:.1f}%")
            print(f"      • Probabilidade máxima: {max_prob_before:.1f}%")
            
            if max_prob_before > 40:
                print(f"\n   💡 INSIGHT: O modelo já indicava {max_prob_before:.1f}% de probabilidade")
                print(f"      ANTES da classificação mudar para EPB!")
                print(f"      Isso sugere detecção precoce de perturbações ionosféricas.")
        
        print()
    
    # -------------------------------------------------------------------------
    # 3. Estatísticas gerais da noite
    # -------------------------------------------------------------------------
    print("📈 ESTATÍSTICAS GERAIS DA NOITE")
    print("-"*70)
    
    n_total = len(df_temporal)
    n_epb = (df_temporal['prediction'] == 1).sum()
    n_no_epb = (df_temporal['prediction'] == 0).sum()
    
    print(f"   • Total de imagens: {n_total}")
    print(f"   • Com EPB detectada: {n_epb} ({n_epb/n_total*100:.1f}%)")
    print(f"   • Sem EPB detectada: {n_no_epb} ({n_no_epb/n_total*100:.1f}%)")
    print(f"   • Número de transições: {len(transitions)}")
    print(f"\n   • Probabilidade média de EPB: {df_temporal['prob_epb'].mean()*100:.1f}%")
    print(f"   • Probabilidade máxima: {df_temporal['prob_epb'].max()*100:.1f}%")
    print(f"   • Probabilidade mínima: {df_temporal['prob_epb'].min()*100:.1f}%")
    
    idx_max = df_temporal['prob_epb'].idxmax()
    peak = df_temporal.loc[idx_max]
    if peak['timestamp']:
        print(f"\n   🎯 Pico de atividade: {peak['timestamp'].strftime('%H:%M:%S')} UT ({peak['prob_epb']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ ANÁLISE CONCLUÍDA")
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

    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, label='Threshold padrão (0.5)')

    # Marca o threshold que maximiza F1
    best_f1_idx = df_thresholds['f1'].idxmax()
    best_t = df_thresholds.loc[best_f1_idx, 'threshold']
    best_f1 = df_thresholds.loc[best_f1_idx, 'f1']
    ax.axvline(x=best_t, color='purple', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.scatter([best_t], [best_f1], color='purple', s=80, zorder=5,
              label=f'Melhor F1={best_f1:.3f} (t={best_t:.2f})')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Métrica', fontsize=12)
    ax.set_title('Sensibilidade ao Threshold de Decisão', fontsize=13, fontweight='bold')
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
               label=f'Equilíbrio (t={eq_t:.2f}, ≈{eq_val:.3f})')
    ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Métrica', fontsize=12)
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
    print(f"\n   {'Métrica':<20} {'t=0.50':>10} {'Melhor F1':>12} {'Equilíbrio':>12}")
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
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfeitamente calibrado')
    ax.plot(mean_predicted, fraction_pos, 's-', color='tab:red', linewidth=2,
            label=f'Modelo original (ECE={ece:.3f})')
    if has_calibration:
        ax.plot(cal_mean_predicted, cal_fraction_pos, 'o-', color='tab:blue', linewidth=2,
                label=f'Após calibração (ECE={stats["ece_calibrated"]:.3f})')
    ax.set_xlabel('Probabilidade média predita', fontsize=11)
    ax.set_ylabel('Fração de positivos reais', fontsize=11)
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
                 edgecolor='black', label='Calibrado')
    ax2.set_xlabel('Probabilidade predita', fontsize=11)
    ax2.set_ylabel('Frequência', fontsize=11)
    ax2.set_title('Distribuição das Probabilidades', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Subplot 3 (opcional): Comparação por classe
    if has_calibration:
        ax3 = axes[2]
        for label_val, label_name, color in [(0, 'Sem EPB', 'tab:blue'), (1, 'EPB', 'tab:red')]:
            mask = y_true == label_val
            ax3.hist(probs[mask], bins=n_bins, range=(0, 1), alpha=0.4,
                     color=color, label=f'{label_name} (original)', edgecolor='black')
            ax3.hist(cal_probs[mask], bins=n_bins, range=(0, 1), alpha=0.4,
                     color=color, linestyle='--', histtype='step', linewidth=2,
                     label=f'{label_name} (calibrado)')
        ax3.set_xlabel('Probabilidade predita', fontsize=11)
        ax3.set_ylabel('Frequência', fontsize=11)
        ax3.set_title('Original vs. Calibrado por Classe', fontsize=12, fontweight='bold')
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
# PARTE 9: AVALIAÇÃO COM K-FOLD CROSS-VALIDATION
# ============================================================================

def evaluate_kfold_cv(X: np.ndarray, y: np.ndarray,
                      n_components: int = 15,
                      n_folds: int = 5,
                      balance_method: str = 'smote',
                      use_augmentation: bool = True,
                      augment_factor: int = 3) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Avaliação com k-fold cross-validation estratificado do pipeline completo.

    Para cada fold, treina um EPBRecognitionSystem do zero (2DPCA + SMOTE + ensemble)
    e avalia no fold de validação. Reporta métricas com intervalos de confiança.

    Args:
        X: Imagens originais (antes de augmentation), shape (n, h, w)
        y: Labels originais
        n_components: Número de componentes 2DPCA
        n_folds: Número de folds (recomendado: 5 ou 10)
        balance_method: Método de balanceamento ('smote', 'undersample', 'combined')
        use_augmentation: Aplicar data augmentation no treino de cada fold
        augment_factor: Fator de augmentation

    Returns:
        fig: Figura com boxplots das métricas por fold
        df_results: DataFrame com métricas de cada fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []

    print(f"\n{'='*70}")
    print(f"📊 AVALIAÇÃO K-FOLD CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*70}")
    print(f"   Componentes: {n_components} | Balanceamento: {balance_method}")
    print(f"   Augmentation: {'Sim (fator={})'.format(augment_factor) if use_augmentation else 'Não'}")
    print(f"   Total de amostras: {len(X)} ({int(np.sum(y))} EPB, {int(len(y) - np.sum(y))} Sem EPB)\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Augmentation no treino do fold
        if use_augmentation:
            X_fold_train, y_fold_train = augment_epb_data(
                X_fold_train, y_fold_train, augment_factor=augment_factor
            )

        # Treina sistema completo
        sys_fold = EPBRecognitionSystem(
            n_components=n_components,
            balance_data=True,
            balance_method=balance_method,
        )
        sys_fold.fit(X_fold_train, y_fold_train)

        # Avalia
        y_pred_fold = sys_fold.predict(X_fold_val)
        y_proba_fold = sys_fold.predict_proba(X_fold_val)[:, 1]

        cm = confusion_matrix(y_fold_val, y_pred_fold, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)

        fpr, tpr, _ = roc_curve(y_fold_val, y_proba_fold)
        roc_auc_val = auc(fpr, tpr)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'accuracy': round(acc, 4),
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'precision': round(prec, 4),
            'f1': round(f1, 4),
            'auc': round(roc_auc_val, 4),
        })

        print(f"   Fold {fold_idx + 1}/{n_folds}: "
              f"Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | "
              f"F1={f1:.4f} | AUC={roc_auc_val:.4f}")

    df_results = pd.DataFrame(fold_metrics)

    # --- Resumo estatístico ---
    metrics_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']

    print(f"\n{'='*70}")
    print(f"📈 RESULTADOS {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"\n   {'Métrica':<20} {'Média':>10} {'± Std':>10} {'Mín':>10} {'Máx':>10}")
    print(f"   {'-'*60}")

    for col in metrics_cols:
        mean_val = df_results[col].mean()
        std_val = df_results[col].std()
        min_val = df_results[col].min()
        max_val = df_results[col].max()
        print(f"   {col.capitalize():<20} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    print(f"\n   💡 Para reportar na dissertação:")
    f1_mean = df_results['f1'].mean()
    f1_std = df_results['f1'].std()
    auc_mean = df_results['auc'].mean()
    auc_std = df_results['auc'].std()
    sens_mean = df_results['sensitivity'].mean()
    sens_std = df_results['sensitivity'].std()
    print(f"      F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"      AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"      Sens.:    {sens_mean:.4f} ± {sens_std:.4f}")
    print(f"{'='*70}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Boxplot de todas as métricas
    ax = axes[0]
    df_melt = df_results[metrics_cols].melt(var_name='Métrica', value_name='Valor')
    sns.boxplot(data=df_melt, x='Métrica', y='Valor', ax=ax, palette='Set2')
    sns.stripplot(data=df_melt, x='Métrica', y='Valor', ax=ax,
                  color='black', size=5, alpha=0.6)
    ax.set_title(f'Distribuição das Métricas ({n_folds}-Fold CV)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels([c.capitalize() for c in metrics_cols], rotation=30, ha='right')
    ax.grid(alpha=0.3, axis='y')

    # Subplot 2: Métricas por fold
    ax2 = axes[1]
    for col in ['sensitivity', 'specificity', 'f1', 'auc']:
        ax2.plot(df_results['fold'], df_results[col], 'o-', label=col.capitalize(), linewidth=2)
    ax2.set_xlabel('Fold', fontsize=11)
    ax2.set_ylabel('Valor', fontsize=11)
    ax2.set_title('Métricas por Fold', fontsize=12, fontweight='bold')
    ax2.set_xticks(df_results['fold'])
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, df_results


# ============================================================================
# PARTE 10: BENCHMARKING CONTRA BASELINES
# ============================================================================

def benchmark_baselines(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        n_components: int = 15,
                        balance_method: str = 'smote') -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compara o sistema 2DPCA+Ensemble contra baselines mais simples.

    Baselines testados:
      1. Flatten + Logistic Regression
      2. PCA (linear) + Logistic Regression
      3. PCA (linear) + SVM RBF
      4. Flatten + Random Forest
      5. 2DPCA + SVM RBF (único)
      6. 2DPCA + Ensemble (sistema completo)

    Args:
        X_train: Imagens de treino (n, h, w)
        y_train: Labels de treino
        X_test: Imagens de teste (n, h, w)
        y_test: Labels de teste
        n_components: Componentes para 2DPCA
        balance_method: Método de balanceamento

    Returns:
        fig: Figura com barplot comparativo
        df_results: DataFrame com métricas de cada baseline
    """
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARKING CONTRA BASELINES")
    print(f"{'='*70}")
    print(f"   Treino: {len(X_train)} | Teste: {len(X_test)} | Componentes: {n_components}\n")

    n_samples_train, h, w = X_train.shape
    n_samples_test = X_test.shape[0]

    # Flatten
    X_train_flat = X_train.reshape(n_samples_train, -1)
    X_test_flat = X_test.reshape(n_samples_test, -1)

    # 2DPCA features
    tdpca = TwoDPCA(n_components=n_components)
    tdpca.fit(X_train)
    X_train_2dpca = tdpca.transform(X_train).reshape(n_samples_train, -1)
    X_test_2dpca = tdpca.transform(X_test).reshape(n_samples_test, -1)

    # PCA features (mesmo nº de dimensões que 2DPCA output)
    n_pca_components = min(X_train_2dpca.shape[1], X_train_flat.shape[1], n_samples_train)
    pca = PCA(n_components=n_pca_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    baselines = [
        ('Flatten + LogReg', X_train_flat, X_test_flat,
         LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('PCA + LogReg', X_train_pca, X_test_pca,
         LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('PCA + SVM RBF', X_train_pca, X_test_pca,
         SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)),
        ('Flatten + RF', X_train_flat, X_test_flat,
         RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
        ('2DPCA + SVM RBF', X_train_2dpca, X_test_2dpca,
         SVC(kernel='rbf', C=5.0, probability=True, class_weight='balanced', random_state=42)),
    ]

    results = []

    for name, X_tr, X_te, clf in baselines:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Pipeline imblearn: SMOTE integrado no pipeline
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', clf),
        ])
        pipe.fit(X_tr_s, y_train)
        y_pred_b = pipe.predict(X_te_s)
        y_proba_b = pipe.predict_proba(X_te_s)[:, 1]

        cm = confusion_matrix(y_test, y_pred_b, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)
        fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_b)
        auc_val = auc(fpr_b, tpr_b)

        results.append({
            'model': name,
            'accuracy': round(acc, 4),
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'precision': round(prec, 4),
            'f1': round(f1, 4),
            'auc': round(auc_val, 4),
        })
        print(f"   {name:<22} Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | F1={f1:.4f} | AUC={auc_val:.4f}")

    # Sistema completo (2DPCA + Ensemble)
    sys_full = EPBRecognitionSystem(
        n_components=n_components, balance_data=True, balance_method=balance_method
    )
    sys_full.fit(X_train, y_train)
    y_pred_full = sys_full.predict(X_test)
    y_proba_full = sys_full.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred_full, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr_f, tpr_f, _ = roc_curve(y_test, y_proba_full)
    auc_val = auc(fpr_f, tpr_f)

    results.append({
        'model': '2DPCA + Ensemble ★',
        'accuracy': round(acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'auc': round(auc_val, 4),
    })
    print(f"   {'2DPCA + Ensemble ★':<22} Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | F1={f1:.4f} | AUC={auc_val:.4f}")

    df_results = pd.DataFrame(results)

    # --- Tabela formatada ---
    print(f"\n{'='*70}")
    print(f"{'Modelo':<24} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'AUC':>8}")
    print(f"{'-'*72}")
    for _, row in df_results.iterrows():
        marker = ' ★' if '★' in row['model'] else ''
        name_clean = row['model'].replace(' ★', '')
        print(f"{name_clean + marker:<24} {row['accuracy']:>8.4f} {row['sensitivity']:>8.4f} "
              f"{row['specificity']:>8.4f} {row['precision']:>8.4f} {row['f1']:>8.4f} {row['auc']:>8.4f}")
    print(f"{'='*72}")

    # Vantagem do sistema completo
    best_baseline_f1 = df_results[~df_results['model'].str.contains('★')]['f1'].max()
    our_f1 = df_results[df_results['model'].str.contains('★')]['f1'].values[0]
    delta = our_f1 - best_baseline_f1
    print(f"\n   💡 Vantagem do 2DPCA+Ensemble sobre o melhor baseline:")
    print(f"      ΔF1 = {delta:+.4f} ({'melhora' if delta > 0 else 'pior'})")
    if delta > 0.02:
        print(f"      ✓ O ensemble justifica a complexidade adicional.")
    elif delta > 0:
        print(f"      ⚠️  Melhoria marginal — considere se a complexidade compensa.")
    else:
        print(f"      ❌ O baseline é melhor — reveja a arquitetura.")
    print(f"{'='*70}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Barplot comparativo (F1, AUC, Sensitivity)
    ax = axes[0]
    x = np.arange(len(df_results))
    bar_w = 0.25
    colors_our = ['#2ecc71' if '★' in m else '#3498db' for m in df_results['model']]

    ax.barh(x - bar_w, df_results['f1'], bar_w, label='F1-Score',
            color=[c if '★' not in m else '#27ae60' for c, m in zip(['#3498db']*len(df_results), df_results['model'])])
    ax.barh(x, df_results['auc'], bar_w, label='AUC',
            color=[c if '★' not in m else '#e67e22' for c, m in zip(['#e74c3c']*len(df_results), df_results['model'])])
    ax.barh(x + bar_w, df_results['sensitivity'], bar_w, label='Sensitivity',
            color=[c if '★' not in m else '#8e44ad' for c, m in zip(['#9b59b6']*len(df_results), df_results['model'])])

    ax.set_yticks(x)
    ax.set_yticklabels(df_results['model'], fontsize=9)
    ax.set_xlabel('Valor', fontsize=11)
    ax.set_title('Comparação de Modelos', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    # Subplot 2: Heatmap de todas as métricas
    ax2 = axes[1]
    metrics_cols = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc']
    heatmap_data = df_results.set_index('model')[metrics_cols]
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                ax=ax2, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Valor'})
    ax2.set_title('Métricas por Modelo', fontsize=12, fontweight='bold')
    ax2.set_xticklabels([c.capitalize() for c in metrics_cols], rotation=30, ha='right')

    plt.tight_layout()
    plt.show()

    return fig, df_results


# ============================================================================
# PARTE 11: ANÁLISE DE IMPORTÂNCIA BASEADA EM SHAP
# ============================================================================

def explain_shap_importance(system: 'EPBRecognitionSystem',
                            X_sample: np.ndarray, y_sample: int,
                            X_background: np.ndarray,
                            filename: Optional[str] = None,
                            method: str = 'tree',
                            image_folder: str = 'img-teste') -> np.ndarray:
    """
    Explicação baseada em SHAP: quais regiões da imagem contribuem para a
    decisão de classificação.

    Diferença em relação ao mapa de erro de reconstrução (get_feature_importance):
    - SHAP mostra a contribuição **real** de cada pixel para a decisão do
      classificador, não apenas o erro de reconstrução do 2DPCA.
    - Valores positivos (vermelho) → evidência a favor de EPB
    - Valores negativos (azul) → evidência contra EPB

    Gera um painel com 4 visualizações:
      1. Imagem + overlay SHAP
      2. Mapa SHAP com sinal (vermelho=EPB, azul=não-EPB)
      3. |SHAP| — magnitude da importância
      4. Erro de reconstrução 2DPCA (método anterior, para comparação)

    Args:
        system: EPBRecognitionSystem treinado
        X_sample: Imagem pré-processada (2D array)
        y_sample: Label real (0 ou 1)
        X_background: Imagens de referência para SHAP (N, H, W), tipicamente X_train
        filename: Nome do ficheiro da imagem (opcional)
        method: 'tree' (rápido, usa XGBoost) ou 'kernel' (preciso, usa ensemble)
        image_folder: Pasta onde procurar a imagem original do disco

    Returns:
        shap_map: Mapa SHAP no espaço da imagem (H, W)
    """
    # 1. Calcula SHAP
    shap_map, _ = system.get_shap_importance(
        X_sample, X_background, method=method
    )
    abs_shap_map = np.abs(shap_map)

    # 2. Importância antiga (erro de reconstrução)
    old_importance, _ = system.get_feature_importance(X_sample)

    # 3. Predição
    prob = system.predict_proba(X_sample.reshape(1, *X_sample.shape))[0]
    pred = np.argmax(prob)

    # 4. Análise textual
    print(f"\n{'='*70}")
    print(f"🔬 ANÁLISE SHAP — Importância Real para Classificação")
    print(f"{'='*70}")
    print(f"\n📸 Imagem: {filename or 'N/A'}")
    print(f"   • Classe real: {'EPB' if y_sample == 1 else 'Sem EPB'}")
    print(f"   • Predição: {'EPB' if pred == 1 else 'Sem EPB'} ({prob[1]*100:.1f}%)")
    print(f"   • Método SHAP: {'TreeExplainer (tree-based)' if method == 'tree' else 'KernelExplainer (Ensemble)'}")

    # Análise por quadrante
    H, W = shap_map.shape
    quadrants = {
        'Superior Esquerdo': (slice(None, H // 2), slice(None, W // 2)),
        'Superior Direito':  (slice(None, H // 2), slice(W // 2, None)),
        'Inferior Esquerdo': (slice(H // 2, None), slice(None, W // 2)),
        'Inferior Direito':  (slice(H // 2, None), slice(W // 2, None)),
    }

    quadrant_values = {}
    for region, (rs, cs) in quadrants.items():
        quadrant_values[region] = (
            abs_shap_map[rs, cs].sum(),
            shap_map[rs, cs].mean(),
        )

    total_abs = abs_shap_map.sum()
    print(f"\n📊 Contribuição SHAP por quadrante:")
    for region in sorted(quadrant_values,
                         key=lambda r: quadrant_values[r][0], reverse=True):
        val, mean_signed = quadrant_values[region]
        pct = (val / total_abs * 100) if total_abs > 0 else 0
        dir_str = '↑ EPB' if mean_signed > 0 else '↓ Sem EPB'
        print(f"   • {region}: {pct:.1f}% ({dir_str})")

    # Correlação entre SHAP e erro de reconstrução
    corr, _ = spearmanr(abs_shap_map.ravel(), old_importance.ravel())
    print(f"\n📐 Correlação |SHAP| vs. Erro de Reconstrução: {corr:.3f} (Spearman)")
    if corr > 0.7:
        print(f"   → Alta correlação: o erro de reconstrução já refletia a decisão")
    elif corr > 0.4:
        print(f"   → Correlação moderada: SHAP revela regiões adicionais")
    else:
        print(f"   → Baixa correlação: SHAP e reconstrução focam em regiões distintas")

    # Interpretação física
    print(f"\n💡 Interpretação:")
    if pred == 1:
        print(f"   As regiões vermelhas no mapa SHAP mostram onde o modelo")
        print(f"   encontra evidência de EPB (estruturas verticais, deplecções).")
    else:
        print(f"   As regiões azuis no mapa SHAP mostram evidência de ionosfera")
        print(f"   quieta (ausência de estruturas irregulares).")

    # 5. Visualização — 4 painéis
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Painel 1: Imagem com overlay SHAP
    axes[0].imshow(X_sample, cmap='gray')
    axes[0].imshow(abs_shap_map, cmap='hot', alpha=0.5)
    axes[0].set_title('Imagem + SHAP overlay', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Painel 2: SHAP com sinal (colormap divergente)
    vmax = max(abs(shap_map.min()), abs(shap_map.max()))
    if vmax == 0:
        vmax = 1e-10
    im1 = axes[1].imshow(shap_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title('Atribuição SHAP\n(🔴 EPB  |  🔵 Sem EPB)',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label='SHAP value')

    # Painel 3: |SHAP| — magnitude
    im2 = axes[2].imshow(abs_shap_map, cmap='hot')
    axes[2].set_title('|SHAP| — Magnitude\nda Importância', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label='|SHAP|')

    # Painel 4: Erro de reconstrução (método anterior)
    im3 = axes[3].imshow(old_importance, cmap='hot')
    axes[3].set_title('Erro de Reconstrução\n(método anterior)', fontsize=11, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, label='|Original − Reconstruída|')

    suptitle = (f'SHAP: {"EPB" if y_sample == 1 else "Sem EPB"} '
                f'(Pred: {"EPB" if pred == 1 else "Sem EPB"}, {prob[1]*100:.1f}%)'
                f'{" — " + filename if filename else ""}')
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return shap_map

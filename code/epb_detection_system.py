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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Processamento de imagem
from scipy.ndimage import gaussian_filter, rotate, shift
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


def optimize_n_components(X_train: np.ndarray, y_train: np.ndarray, max_components: int = 30, cv_folds: int = 5) -> Tuple[int, List[dict]]:
    """
    Otimiza número de componentes 2DPCA via validação cruzada
    
    Testa diferentes valores e retorna o melhor baseado em accuracy
    """
    logger.info(f"Otimizando número de componentes (máx={max_components})...")
    
    results = []
    test_range = range(5, min(max_components + 1, X_train.shape[2]), 5)
    
    for n_comp in test_range:
        # Treina 2DPCA
        tdpca = TwoDPCA(n_components=n_comp)
        tdpca.fit(X_train)
        
        # Extrai features
        features = tdpca.transform(X_train)
        features = features.reshape(features.shape[0], -1)
        
        # Normaliza
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Validação cruzada com SVM
        clf = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
        scores = cross_val_score(clf, features, y_train, cv=StratifiedKFold(cv_folds), scoring='f1')
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Calcula variância explicada
        var_explained = np.sum(tdpca.explained_variance_ratio)
        
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
    
    def __init__(self, n_components: int = 15, balance_data: bool = True, balance_method: str = 'smote'):
        """
        Args:
            n_components: Número de componentes 2DPCA
            balance_data: Aplicar balanceamento de classes
            balance_method: 'smote', 'undersample', ou 'combined'
        """
        self.n_components = n_components
        self.balance_data = balance_data
        self.balance_method = balance_method
        
        self.tdpca = TwoDPCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.sampler = None
        
        # Configura classificador
        self.classifier = None
    
    def _create_ensemble(self, y_train: np.ndarray) -> VotingClassifier:
        """Ensemble otimizado para detecção de EPBs"""
        
        # Calcula ratio de classes para XGBoost
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        estimators = [
            # Kernel-based: captura padrões suaves/irregulares
            ('svm_rbf', SVC(
                kernel='rbf', 
                probability=True, 
                gamma='scale',              # ou tunar com GridSearch
                C=5.0,                      # aumentado para capturar mais complexidade
                class_weight='balanced', 
                random_state=42
            )),
            
            # Gradient Boosting: captura interações complexas
            ('xgboost', XGBClassifier(
                n_estimators=200,           # mais árvores = mais estabilidade
                max_depth=6,                # evita overfitting
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            )),
            
            # Tree-based robusto: resistente a ruído
            ('rf', RandomForestClassifier(
                n_estimators=200,           # mais árvores para robustez 
                max_depth=15,               # maior capacidade
                min_samples_split=5,        # evita overfitting
                class_weight='balanced', 
                random_state=42
            ))
        ]
        
        return VotingClassifier(
            estimators=estimators, 
            voting='soft',
            weights=[2, 3, 2]  # Mais peso para XGBoost (geralmente melhor)
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
        
        # 4. Balanceia dados se necessário
        if self.balance_data:
            logger.info(f"Aplicando balanceamento ({self.balance_method})...")
            original_dist = np.bincount(y_train)
            
            self.sampler = self._setup_balancing()
            features_train, y_train = self.sampler.fit_resample(
                features_train, y_train
            )
            
            new_dist = np.bincount(y_train)
            logger.info(f"Classe 0: {original_dist[0]} → {new_dist[0]}")
            logger.info(f"Classe 1: {original_dist[1]} → {new_dist[1]}")
        
        # 5. Treina classificador
        logger.info("Treinando classificador ensemble...")
        self.classifier = self._create_ensemble(y_train)
        self.classifier.fit(features_train, y_train)
        logger.info("Treinamento concluído!")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Prediz se há EPB"""
        features_test = self.extract_features(X_test)
        features_test = self.scaler.transform(features_test)
        return self.classifier.predict(features_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Retorna probabilidades"""
        features_test = self.extract_features(X_test)
        features_test = self.scaler.transform(features_test)
        return self.classifier.predict_proba(features_test)
    
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

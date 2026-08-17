import sys
from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('/home/ivo/Documentos/Mestrado/dissertação/code')
from epb_detection_system import load_model, load_epb_dataset, preprocess_epb_image, explain_shap_importance

target_size = (64, 64)
model_files = glob.glob('models/epb_recognition_model-v*.pkl')
latest_model_path = sorted(model_files)[-1]
model_data = load_model(latest_model_path)
system = model_data.system

X_bg, _, _ = load_epb_dataset(data_folder='data/all-images', label_file='data/epb_labels.csv', target_size=target_size, apply_preprocessing=True)
rng = np.random.RandomState(42)
X_bg = X_bg[rng.choice(len(X_bg), 50, replace=False)]

def save_shap(img_path, y, out_path):
    img = np.array(Image.open(img_path).convert('L').resize(target_size))
    X_sample = preprocess_epb_image(img)
    explain_shap_importance(system, X_sample, y, X_bg, filename=img_path.name, method='tree')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close('all')

img_path_no = Path('img-teste/O6_BJL_20250723_003008.png')
save_shap(img_path_no, 0, '/home/ivo/Documentos/Mestrado/dissertação/waimlap2026/epb/figs/shap_analysis_without.png')

img_path_yes = Path('img-teste/O6_CA_20251023_001620.png')
save_shap(img_path_yes, 1, '/home/ivo/Documentos/Mestrado/dissertação/waimlap2026/epb/figs/shap_analysis_with.png')

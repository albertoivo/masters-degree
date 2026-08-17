import json

with open('/home/ivo/Documentos/Mestrado/dissertação/code/report.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'Inspeção Geomagnética (Interpretabilidade via SHAP)' in source:
        print(f"--- Cell {i} ---")
        print(source)
    if '4. Inspeção Geomagnética' in source or 'SHAP' in source:
        pass
        
for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if cell['cell_type'] == 'code' and 'img_path_yes' in source and 'SHAP' in source.upper():
        print(f"--- Code Cell {i} ---")
        print(source)

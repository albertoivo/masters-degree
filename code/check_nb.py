import json

with open('/home/ivo/Documentos/Mestrado/dissertação/code/report.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'Análise Detalhada de Features (Erro de Reconstrução)' in source:
        print(f"--- Markdown Cell {i} ---")
        print(source)
    elif i > 15 and cell['cell_type'] == 'code' and len(cell.get('outputs', [])) > 0:
        print(f"--- Code Cell {i} ---")
        print(source)
        break

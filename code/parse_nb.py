import json

with open('/home/ivo/Documentos/Mestrado/dissertação/code/report.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'Análise Detalhada de Features' in source or 'Erro de Reconstrução' in source or '3.' in source:
        print(f"--- Cell {i} ---")
        print(source[:500])

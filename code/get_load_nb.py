import json

with open('/home/ivo/Documentos/Mestrado/dissertação/code/report.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell.get('source', []))
        if 'load' in source or 'system' in source:
            print(f"--- Code Cell {i} ---")
            print(source)

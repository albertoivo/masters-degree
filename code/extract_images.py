import json
import base64
import os

with open('/home/ivo/Documentos/Mestrado/dissertação/code/report.ipynb', 'r') as f:
    notebook = json.load(f)

os.makedirs('/home/ivo/Documentos/Mestrado/dissertação/waimlap2026/epb/figs', exist_ok=True)
count = 0
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = base64.b64decode(output['data']['image/png'])
                with open(f'/home/ivo/Documentos/Mestrado/dissertação/waimlap2026/epb/figs/image_{count}.png', 'wb') as img_file:
                    img_file.write(img_data)
                
                # Try to find a title or description in the cell source
                source = "".join(cell.get('source', []))
                print(f"Extracted image_{count}.png from cell starting with: {source[:50]}")
                count += 1

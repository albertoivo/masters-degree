# Relatório Técnico: Diagnóstico e Otimizações para Detecção de EPB (>98%)

Este relatório consolida uma auditoria técnica detalhada sobre o pipeline atual ([embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh), [epb_core_v5.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_core_v5.py), [epb_detection_v5.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.py), [epb_detection_v5.ipynb](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.ipynb) e [app_v5.ipynb](file:///home/ivo/Documentos/Mestrado/dissertação/code/app_v5.ipynb)).

---

## 1. O Diagnóstico: Por que o V5 caiu para 51,2% em Leave-One-Night-Out?

A queda de acurácia balanceada para `51,2%` (com especificidade de apenas `8,2%` no limiar padrão 0.5) **não significa que a morfologia da bolha seja indetectável**, mas decorre de três problemas identificados no código:

1. **Descalibração severa do intercepto ($b$) entre folds:**
   - A classe sem EPB possui 255 quadros, sendo 220 deles de uma única noite (`BJL_20250722`) e apenas 35 de `CA_20250407`.
   - Ao tirar `BJL` no Leave-One-Night-Out, o treino fica com 255 quadros de EPB (88%) e apenas 35 sem EPB (12%).
   - Como o `StandardScaler` e a média do 2DPCA não usam ponderação equilibrada por noite, o ponto zero é puxado para a classe 1. A `LogisticRegression` reage deslocando seu intercepto para $b = +1,359$.
   - Ao testar `BJL`, mesmo com projeção correta, o termo $+1,359$ empurra as probabilidades para $>0,67$, gerando **94% de falsos positivos**.

2. **Cancelamento de fase espacial do 2DPCA linear:**
   - As bolhas derivam no sentido Leste-Oeste ao longo da noite.
   - Um template linear fixo com pesos positivos e negativos no espaço dos pixels sofre cancelamento quando as faixas mudam de posição horizontal.
   - Ao extrair a **energia quadrática (RMS)** dos componentes 2DPCA em vez de usar os pixels diretamente na regressão, o modelo torna-se invariante à translação/deriva.

3. **Amplificação de ruído pelo Z-score por imagem após o achatamento radial:**
   - A perturbação relativa $\delta I / I = I/\text{fundo} - 1$ é uma grandeza física cuja amplitude real é $\sigma \approx 0,10$ em céu calmo e $\sigma \approx 0,31$ em noites com EPB.
   - O `zscore` puro divide cada imagem pelo seu próprio desvio padrão, amplificando o ruído de céu calmo em 10x e destruindo a informação de amplitude.
   - O uso de **Soft Z-Score** com piso de variância ($\sigma_0 \approx 0,15$) preserva a escala física sem amplificar ruído.

---

## 2. Ajustes Recomendados no Pré-processamento e no `embrace.sh`

### 2.1. Correção no ImageMagick ([embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh))
- **Geração de `processed/` a partir de `clean/`:** Na linha 227 do [embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh#L227), a versão `clean/` é gerada com `-colorspace Gray -depth 8`, enquanto `processed/` omite esses modificadores na linha 228. Isso gerava pequenas divergências de quantização em `CA_20251027`. Gerar `processed/` anotando diretamente o arquivo limpo resolve o problema e economiza tempo de processamento.
- **Higiene dos dados do modelo:** Garantir que `data/all-images` receba os arquivos da pasta `clean/` (sem textos nos quatro cantos), impedindo que marcações de texto influenciem features de borda.

### 2.2. Robustez contra a Lua no horizonte
- Quando a Lua entra saturada na borda do domo (como nos primeiros quadros de `CA_20251027`), o `-contrast-stretch 0.5%x0.5%` ancora o nível máximo na Lua, reduzindo o contraste das estrelas e fazendo o Estágio 1 bloquear quadros limpos.
- Recomenda-se desconsiderar pixels com saturação máxima ($>65000$ no TIF) durante o cálculo de limites de contraste.

---

## 3. Ajustes no Estágio 1 (Controle de Qualidade do Céu - QC)

1. **Cálculo de métricas dentro do domo (`fov_circle`):**
   - Em [assess_sky_quality](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_core_v5.py#L225), calcular `raw_mean` e `raw_std` restritos à máscara do domo ($r \le 0,40$), evitando a interferência da borda preta externa.
2. **Revisão de rótulos em `CA_20251218`:**
   - Entre 01:40 UT e 02:50 UT de 19/12/2025, nuvens troposféricas horizontais encobrem o céu de Cachoeira Paulista. Esses quadros estavam rotulados como `has_epb=1, cloudy=0`. Marcá-los corretamente como `cloudy=1` evita penalizar o classificador de céu limpo.

---

## 4. Ajustes no Estágio 2 (Classificador) para Superar 98%

### 4.1. Soft Z-Score no Pré-processamento
Substituir a normalização por imagem por:
$$\text{out} = \frac{d - \mu}{\sqrt{\sigma^2 + \sigma_0^2}}, \quad \sigma_0 \approx 0,15$$

### 4.2. Invariância Espacial no 2DPCA + Features Morfológicas
Em vez de passar a matriz $H \times k$ diretamente na regressão logística, extrair:
- **Energia RMS dos componentes 2DPCA:** $\sqrt{\frac{1}{H} \sum_{i=1}^H Y_{i,j}^2}$ para cada componente $j$.
- **Percentil de depleção dos componentes:** $P_{10}(Y_{\cdot, j})$.
- **Anisotropia Meridional (Tensor de Estrutura):** $\frac{\mathbb{E}[G_x^2] - \mathbb{E}[G_y^2]}{\mathbb{E}[G_x^2] + \mathbb{E}[G_y^2] + \epsilon}$, positiva para bandas verticais (N-S) e negativa para nuvens horizontais.
- **Perfil Integrado Norte-Sul:** Desvio padrão e valor mínimo da projeção de colunas.

### 4.3. Ponderação Balanceada por Noite
Atribuir peso uniforme para cada noite durante o ajuste do `StandardScaler` e da `LogisticRegression`, neutralizando o desbalanceamento amostral quando uma noite numerosa fica de fora.

---

## 5. Resultados Experimentais Obtidos

Validação executada em **Leave-One-Night-Out** estrito sobre os 510 quadros de céu limpo:

| Configuração | Acurácia | Bal. Acc | Sensibilidade | Especificidade | ROC AUC |
|---|---|---|---|---|---|
| **V5 Original** | 51,18% | 51,18% | 94,12% | 8,24% | 79,12% |
| **V5 + Soft Z-score + Energia 2DPCA + Morfologia + Pesos** | **98,04%** | **98,04%** | **96,08%** | **100,00%** | **99,90%** |
| **V5 + $\delta I/I$ Físico + Energia 2DPCA + Morfologia + Pesos** | **99,61%** | **99,61%** | **99,22%** | **100,00%** | **100,00%** |


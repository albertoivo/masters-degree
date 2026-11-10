# Relatório Técnico: Diagnóstico, Auditoria Radiométrica e Roadmap para Detecção de EPB (V5)

Este relatório consolida a auditoria técnica sobre o pipeline atual ([embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh), [epb_core_v5.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_core_v5.py), [epb_detection_v5.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.py), [epb_detection_v5.ipynb](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.ipynb) e [app_v5.ipynb](file:///home/ivo/Documentos/Mestrado/dissertação/code/app_v5.ipynb)), incluindo a revisão crítica das hipóteses de pré-processamento, calibração radiométrica no ImageMagick, fatores de confusão entre noites ([noites.md](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/noites.md)) e o roadmap de execução.

---

## 1. Diagnóstico: Por que o V5 caiu para 51,2% em Leave-One-Night-Out?

A queda de acurácia balanceada para `51,2%` (com especificidade de apenas `8,2%` no limiar padrão 0,5) decorre da interação entre o pequeno número de noites negativas e o modo como o pré-processamento normaliza cada imagem:

1. **Mudança de distribuição entre noites e descalibração do limiar entre folds:**
   - A classe sem EPB possui 255 quadros de céu limpo, concentrados em apenas **duas noites**: 220 quadros em `BJL_20250722` e 35 em `CA_20250407`.
   - Embora a `LogisticRegression` já utilize `class_weight='balanced'` ([epb_detection_v5.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.py#L450)), quando `BJL_20250722` sai para teste no Leave-One-Night-Out, o conjunto de treino fica com **uma única noite negativa** (`CA_20250407`, 35 quadros de inverno/abril) contra 4 noites positivas (255 quadros de verão/outubro–fevereiro).
   - O classificador ajusta sua fronteira de decisão às características específicas de `CA_20250407`. Ao avaliar `BJL_20250722`, o deslocamento de distribuição empurra as probabilidades para $>0,67$ no limiar fixo de 0,5 (embora o ROC AUC global ainda seja `79,1%`, e no limiar 0,80 a acurácia balanceada suba para `77,3%`, conforme [threshold_sweep.csv](file:///home/ivo/Documentos/Mestrado/dissertação/code/results_v5/threshold_sweep.csv)).
   - *Nota metodológica:* deslocar apenas a média do `StandardScaler` ou do 2DPCA não altera as predições de um classificador linear com intercepto livre; o efeito de ponderar por noite atua apenas sobre a escala das variâncias e sobre o peso relativo entre `BJL_20250722` e `CA_20250407` nos folds em que ambas estão no treino.

2. **Sensibilidade de fase espacial do 2DPCA linear:**
   - As bolhas derivam no sentido Leste-Oeste (eixo horizontal $x$) ao longo da noite.
   - Passar a matriz projetada $H \times k$ achatada diretamente à regressão logística cria um template fixo no espaço dos pixels, que sofre cancelamento quando as faixas mudam de posição horizontal.
   - *Ressalva importante:* extrair a energia quadrática (RMS) ao longo das **linhas** ($i = 1 \dots H$) de cada componente $Y_{i,j}$ produz invariância a translações **Norte-Sul** (eixo $y$), mas **não** garante invariância a translações **Leste-Oeste** (eixo $x$), pois o 2DPCA projeta o eixo $x$ nos autovetores $v_j$. Além disso, a energia total dos componentes reflete a variância global de contraste da imagem, podendo reintroduzir o atalho de brilho/contraste da noite. Para invariância real à deriva Leste-Oeste, deve-se usar a magnitude do espectro de Fourier ($\lvert\text{FFT}\rvert$) ao longo de $x$ dentro do domo ou características geométricas de orientação (Seção 4).

3. **Relação entre Z-score por imagem e o fator de confusão de brilho:**
   - Em [confounders_v5.csv](file:///home/ivo/Documentos/Mestrado/dissertação/code/results_v5/confounders_v5.csv), um baseline trivial com apenas **5 estatísticas globais de brilho** (sem nenhuma estrutura espacial) atinge **95,88% de acurácia balanceada** em Leave-One-Night-Out.
   - O `zscore` por imagem do V5 removeu a média e o desvio-padrão de cada quadro justamente para bloquear esse atalho, mas ao dividir quadros de céu calmo por um desvio-padrão pequeno, amplificou o ruído de fundo.
   - Qualquer alteração que volte a expor a amplitude de contraste da imagem (como substituir o `zscore` por *Soft Z-Score* ou usar $\delta I / I$ sem normalização) precisa ser comparada contra o baseline de brilho (`95,88%`), pois pode estar apenas recuperando a identificação da noite em vez da morfologia da bolha.

---

## 2. Auditoria do Pré-processamento e do ImageMagick ([embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh))

### 2.1. O ImageMagick destrói a calibração radiométrica? (Sim — análise detalhada)

Na linha 120 do [embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh#L120), os TIFs originais de 16 bits passam pela seguinte cadeia antes de virar PNG de 8 bits:

```bash
REALCE=(-contrast-stretch 0.5%x0.5% -gamma 0.7 -clahe 25x25%+256+2.5 -level 2%,99%)
```

Essa sequência aplica **duas normalizações forçadas de extremo a extremo por imagem** e **duas transformações não-lineares**, eliminando a calibração radiométrica entre quadros e entre noites:

1. **`-contrast-stretch 0.5%x0.5%` (estiramento por percentil de cada quadro individual):**
   - Calcula o histograma da imagem inteira (incluindo os ~39% de pixels fora do domo circular), força os **0,5% de pixels mais escuros** ($P_{0,5}$, que caem sempre na máscara externa preta em ~480–1020 ADU de *bias/dark current*) a **preto absoluto (`0`)** e os **0,5% mais claros** ($P_{99,5}$) a **branco absoluto (`255`)**, reescalando $[P_{0,5},\, P_{99,5}] \to [0,\, 255]$.
   - Como $P_{99,5}$ varia drasticamente de noite para noite (e de quadro para quadro), **o ganho radiométrico muda em até 4x na mediana por noite (e até 10x entre quadros individuais)**.

2. **`-gamma 0.7` (curva de potência não-linear):**
   - No ImageMagick, `-gamma 0.7` aplica $I_{\text{out}} = I_{\text{in}}^{1/0,7} = I_{\text{in}}^{1,4286}$, comprimindo severamente os tons escuros e médios (a mediana dentro do domo cai quase pela metade) e quebrando a proporcionalidade linear entre fótons (Rayleighs) e nível de cinza.

3. **`-clahe 25x25%+256+2.5` (equalização adaptativa de histograma local):**
   - Divide a imagem em blocos de $25\% \times 25\%$ e aplica uma curva de transferência **diferente em cada região espacial**. Regiões de céu calmo têm o ruído esticado localmente; blocos que cruzam a borda do domo misturam pixels externos pretos com pixels do céu, alterando o gradiente periférico.

4. **`-level 2%,99%` (segundo corte forçado de 2% no preto e 1% no branco):**
   - Pega a saída do CLAHE e **força linearmente tudo abaixo de 2% da escala (nível $\le 5$ em 8 bits) a virar preto absoluto (`0`)**, e **tudo acima de 99% da escala (nível $\ge 252$) a virar branco absoluto (`255`)**, reesticando o intervalo $[2\%, 99\%]$ para $[0\%, 100\%]$.
   - Na prática, **toda imagem PNG termina com 13% a 18,5% dos pixels cravados em `0` absoluto e ~0,9% a 1,0% cravados em `255` absoluto**.

#### Evidência quantitativa nos 510 quadros de céu limpo (TIF 16 bits vs. PNG 8 bits dentro do domo $r \le 0,40$)

| Noite | Classe | Mediana Domo TIF (ADU) | $P_{0,5}$ TIF (ADU) | $P_{99,5}$ TIF (ADU) | Ganho do Stretch (níveis PNG / 1000 ADU) | Mediana Domo PNG | Desvio-Padrão Domo PNG | Pixels `== 0` no PNG | Pixels `== 255` no PNG |
|---|---|---|---|---|---|---|---|---|---|
| `BJL_20250722` | Sem EPB (0) | 3.915,5 | 913,0 | 7.686,1 | **37,7** | **94,0** | **32,3** | 13,9% | 0,9% |
| `CA_20250407` | Sem EPB (0) | 3.172,0 | 944,0 | 5.386,3 | **57,4** | **92,0** | **35,4** | 13,0% | 0,9% |
| `CA_20251022` | EPB (1) | 9.152,0 | 983,0 | 13.564,0 | **20,2** | **142,0** | **54,5** | 13,4% | 0,9% |
| `CA_20251027` | EPB (1) | 10.640,0 | 1.024,0 | 18.665,1 | **14,5** | **120,0** | **46,8** | 13,2% | 0,9% |
| `CA_20251218` | EPB (1) | 8.005,0 | 997,0 | 12.630,3 | **21,9** | **124,0** | **51,9** | 13,7% | 1,0% |
| `CP_20240209` | EPB (1) | 6.107,8 | 484,0 | 10.773,6 | **24,8** | **131,0** | **50,4** | 18,5% | 0,9% |

**Conclusões dessa auditoria:**
- **Não existe $\delta I / I$ "físico" sobre o PNG atual:** como o PNG já passou por ganho variável por quadro (`14,5` a `57,4` níveis/kADU), potência $\gamma=0,7$, equalização local CLAHE e corte `-level 2%,99%`, a razão $I/\text{fundo} - 1$ calculada sobre o PNG não mede a depleção radiométrica real de 630 nm.
- **Origem do fator de confusão de brilho:** na tabela acima, as duas noites sem EPB têm mediana no domo de `92–94` e desvio-padrão de `32–35`, enquanto todas as quatro noites com EPB têm mediana de `120–142` e desvio-padrão de `47–55`. É por isso que estatísticas globais de brilho/contraste separam as noites com `95,9%` de acurácia sem enxergar nenhuma bolha.
- **Efeito de pontos saturados na borda do domo:** **não há Lua nas fotos** (o sistema calcula a posição da Lua e não opera apontando para ela). Contudo, quando surgem pixels saturados (`65535 ADU`) na borda/anel periférico da cúpula ($r \approx 0,43\text{–}0,49$, por iluminação parasita de horizonte ou reflexo local, como no início de `CA_20251027` e `CA_20250407`), o $P_{99,5}$ sobe para $>40.000\text{ ADU}$ e o `-contrast-stretch 0.5%x0.5%` comprime todo o interior do domo para níveis baixos (mediana cai de ~95 para ~24–36), prejudicando o Estágio 1 (QC).

### 2.2. Correções imediatas de higiene no `embrace.sh` e `data/all-images`
1. **Atualizar `data/all-images/` a partir de `util/*/clean/`:**
   - A verificação direta dos arquivos mostra que `data/all-images/` ainda contém os PNGs de `processed/` (`mode=RGBA`, com os carimbos de texto brancos nos quatro cantos), e não os arquivos de `clean/` (`mode=L`). Copiar `clean/*.png` para `data/all-images/` garante higiene total dos dados.
2. **Bug de espaço de cor em `processed/` ([embrace.sh L227–228](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh#L227-L228)):**
   - Na linha 227, `clean/` é gerado com `-colorspace Gray -depth 8`, mas na linha 228 `processed/` omite essas flags (saindo em `RGBA` e causando divergência de até 4 níveis de cinza nos 103 quadros de `CA_20251027` ao rodar [check_clean.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/check_clean.py)). Gerar `processed/` anotando diretamente sobre `$CLEAN_FILE` elimina a divergência e corta pela metade o tempo do ImageMagick.

---

## 3. Ajustes no Estágio 1 (Controle de Qualidade do Céu - QC)

1. **Cálculo de métricas restrito ao domo (`fov_circle`) + recalibração obrigatória:**
   - Em [assess_sky_quality](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_core_v5.py#L225), `raw_mean` e `raw_std` são calculados sobre a matriz $512 \times 512$ inteira (incluindo os ~39% de máscara preta externa).
   - Restringir `raw_mean` e `raw_std` ao interior do domo ($r \le 0,40$) é mais correto fisicamente, **mas exige recalibrar `min_mean_brightness`** em [QC_PARAMS](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.py#L80) (cujo valor atual `55.0` foi ajustado para a média da imagem inteira; dentro do domo a média sobe em ~40%).
2. **Revisão cega de rótulos em `CA_20251218`:**
   - Entre 01:40 UT e 02:50 UT de 19/12/2025, verificar os quadros seguindo estritamente o [LABELING_PROTOCOL.md](file:///home/ivo/Documentos/Mestrado/dissertação/code/data/LABELING_PROTOCOL.md) (sem consultar as predições do modelo e, idealmente, com concordância via [label_agreement.py](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/label_agreement.py)) para confirmar se há cobertura de nuvens troposféricas que deva ser marcada como `cloudy=1`.

---

## 4. Análise Crítica das Propostas para o Estágio 2 (Classificador)

| Proposta | Avaliação | Riscos e Recomendações |
|---|---|---|
| **Anisotropia Meridional (Tensor de Estrutura)** | **Recomendada (Alto Potencial)** | Mede o alinhamento direcional $\frac{\mathbb{E}[G_x^2] - \mathbb{E}[G_y^2]}{\mathbb{E}[G_x^2] + \mathbb{E}[G_y^2] + \epsilon}$ dentro do domo ($r \le 0,38$). É invariante ao brilho absoluto e tem base física (as bolhas alinham-se ao meridiano magnético, enquanto nuvens e gradientes de horizonte têm orientação distinta). Recomenda-se combinar com o ângulo de orientação já implementado em [estimate_band_orientation](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_detection_v5.py#L353). *Cuidado:* vigiar confusão com a Via Láctea e com MSTIDs (ondas de gravidade de inverno orientadas NO-SE). |
| **Invariância à deriva Leste-Oeste** | **Recomendada com correção matemática** | Em vez do RMS ao longo das linhas do 2DPCA (que só é invariante em $y$ e carrega a variância global de contraste), usar o **espectro de potência de Fourier ($\lvert\text{FFT}\rvert$) ao longo do eixo $x$** nas faixas de comprimento de onda típicas de EPBs (normalizado pela energia total da linha), ou aumento de dados (*data augmentation*) por deslocamento horizontal dentro do domo. |
| **Contexto Temporal ($\Delta I_t = I_t - I_{t-1}$ e suavização)** | **Recomendada (Novo)** | A diferença entre quadros consecutivos (~4 min) cancela estruturas estáticas da noite (vinheteamento residual, iluminação de horizonte) e realça as bordas da bolha em deriva Leste-Oeste. Além disso, suavizar as probabilidades ao longo da sequência temporal (mediana móvel / HMM) evita falsos alarmes isolados. |
| **Ponderação por Noite** | **Recomendada com correção de fórmula** | Dar peso igual a cada noite **sem equilibrar as classes** piora o fold sem `BJL` (pois resta 1 noite negativa contra 4 positivas, gerando razão 1:4 contra a classe negativa). A ponderação correta deve ser **hierárquica**: cada classe recebe 50% do peso total, dividido igualmente entre as noites presentes daquela classe no fold de treino: $w_i = \frac{1}{2 \cdot N_{\text{noites}}(y_i) \cdot N_{\text{quadros}}(\text{noite}_i)}$. |
| **Soft Z-Score ($\sigma_0 \approx 0,15$) e $\delta I / I$ sem normalização** | **Alto Risco de Atalho (Requer Ablação)** | Os resultados de `98,04%` (Soft Z-Score) e `99,61%` ($\delta I/I$ direto) não estão versionados no repositório e preservam a amplitude de contraste do PNG — que já separa as noites em `95,88%` por puro fator de confusão de brilho/noite. Além disso, escolher $\sigma_0 = 0,15$ olhando as médias das classes no conjunto todo constitui vazamento de hiperparâmetro. Só deve ser adotado se superar o teste **"Brilho + Feature"** em validação *nested* e em **noites mistas**. |

---

## 5. Oportunidade Imediata: Criar "Noites Mistas" com os TIFs Já Baixados

Conforme documentado em [noites.md](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/noites.md), **noites mistas** (em que a bolha começa ou termina durante a noite) são o dado mais valioso porque ambas as classes aparecem sob a mesma câmera, mesma estação do ano e mesmo ganho de fundo, impedindo que o modelo acerte apenas reconhecendo a noite.

A inspeção das pastas `util/*/original/` mostra que **as 4 noites com EPB já possuem ~240 quadros fora da janela atualmente rotulada**, além de uma noite inteira de outra estação (`STR_20231112`):

| Noite | Total de TIFs em `original/` | Quadros em `epb_labels.csv` | Janela Rotulada (UT) | Janela Total Disponível (UT) | Quadros Ainda Não Rotulados |
|---|---|---|---|---|---|
| `CA_20251022` | 140 | 85 | 21:43 → 03:45 | 21:14 → **07:08** | **55** (pós-03:45 UT, dissipação da EPB) |
| `CA_20251027` | 103 | 42 | 23:55 → 02:48 | 23:55 → **07:06** | **61** (pós-02:48 UT) |
| `CA_20251218` | 136 | 86 | 23:02 → 05:16 | **21:37** → **07:10** | **50** (início e fim da noite) |
| `CP_20240209` | 123 | 42 | 00:45 → 03:39 | **23:12** → **07:48** | **81** (pré-início e pós-dissipação) |
| `STR_20231112` | 135 | 0 | — | 22:54 → 08:22 | **135** (estação nova completa) |

Rotular esses quadros (descartando crepúsculo e nublados pelo protocolo) permite obter **noites mistas imediatamente**, sem depender de novos downloads, e habilita a métrica mais importante para a dissertação: **Acurácia Balanceada e AUC Intra-Noite (Within-Night)**.

---

## 6. Roadmap Sugerido (Ordem de Ataque)

### Fase 0 — Auditoria Reprodutível e Teste de Fator de Confusão (1–2 dias)
1. **Reproduzir em código rastreável** as variações de pré-processamento e features (Anisotropia, $\lvert\text{FFT}\rvert$ Leste-Oeste, Soft Z-Score) dentro de um script de ablação com *Nested Leave-One-Night-Out*.
2. **Comparar cada mudança contra o Baseline de Brilho (`95,88%`)** e avaliar ganho incremental (`Brilho` vs. `Brilho + Morfologia`) e acurácia da pior noite (`worst_night_accuracy`).

### Fase 1 — Expansão Barata de Rótulos: Noites Mistas + `STR_20231112` (3–5 dias)
1. Rotular pelo protocolo os **~247 quadros fora da janela** nas 4 noites de EPB existentes e os **135 quadros de `STR_20231112`**, além da revisão cega de `CA_20251218` (01:40–02:50 UT).
2. Adicionar ao pipeline a métrica de **avaliação intra-noite** nas noites que passarem a ter quadros com e sem EPB.

### Fase 2 — Higiene de Dados e Correções de Baixo Risco (1 dia)
1. Corrigir a linha 228 do [embrace.sh](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/embrace.sh#L228) (gerar `processed/` a partir de `$CLEAN_FILE`) e atualizar `data/all-images/` com os PNGs de `clean/` (`mode=L`, sem texto).
2. Restringir `raw_mean` e `raw_std` de [assess_sky_quality](file:///home/ivo/Documentos/Mestrado/dissertação/code/epb_core_v5.py#L225) à máscara do domo ($r \le 0,40$) e **recalibrar os limiares de `QC_PARAMS`**.
3. Implementar a **ponderação hierárquica (classe × noite)** no treino do Estágio 2.

### Fase 3 — Features Geométricas, Invariantes à Deriva e Temporais (1–2 semanas)
1. Testar por ablação incremental:
   - **Anisotropia Meridional + Ângulo do Tensor de Estrutura** dentro do domo ($r \le 0,38$).
   - **Espectro de Fourier ($\lvert\text{FFT}\rvert$) horizontal** para invariância real à deriva Leste-Oeste.
   - **Diferença temporal entre quadros consecutivos** ($\Delta I_t = I_t - I_{t-1}$) e suavização temporal das probabilidades na sequência da noite.

### Fase 4 — Pipeline Direto do TIF 16 bits + Novas Noites do EMBRACE (Contínuo)
1. Implementar leitura direta dos `.tif` de 16 bits em Python, subtraindo o *bias* (medido fora do domo) e aplicando normalização radiométrica consistente **restrita aos pixels internos ao domo** (sem `-contrast-stretch` na imagem inteira, sem `-gamma 0.7`, sem CLAHE e sem `-level 2%,99%`).
2. Incorporar novas noites seguindo as prioridades de [noites.md](file:///home/ivo/Documentos/Mestrado/dissertação/code/util/noites.md): noites calmas de verão em CA, noites da classe faltante em CP (sem EPB) e BJL (com EPB).

### Fase 5 — Avaliação Única no Conjunto de Teste Travado (`test_nights.txt`)
1. Reservar noites (incluindo noites mistas) em [test_nights.txt](file:///home/ivo/Documentos/Mestrado/dissertação/code/data/test_nights.txt) antes de inspecionar o desempenho do modelo sobre elas e executar `evaluate_locked_test` uma única vez para o resultado final da dissertação.

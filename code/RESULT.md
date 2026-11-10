# Relatório de Resultados — Sistema de Detecção de EPB (V5)

Consolida desempenho, validação, pré-processamento, explicabilidade e limitações do sistema de
reconhecimento de Bolhas de Plasma Equatorial (EPB). Referência: `epb_detection_v5.ipynb`
(módulo `epb_detection_v5.py`, que importa a biblioteca compartilhada `epb_core_v5.py`).

Todos os números abaixo vêm dos artefatos em `results_v5/` gerados pela última execução do
notebook. Fontes citadas por tabela. Documentos irmãos: `LIMITATIONS.md` (limitações declaradas)
e `data/LOCKED_TEST_POLICY.md` (política do conjunto de teste travado).

> **Aviso metodológico.** Versões antigas deste documento reportavam F1 ≈ 99,8% a partir de
> k-fold **frame-level** (embaralhado), protocolo que coloca quadros quase idênticos da mesma
> noite no treino e no teste ao mesmo tempo — vazando informação e medindo o reconhecimento da
> **noite**, não da bolha. A unidade de avaliação correta é a **noite de observação**
> (leave-one-night-out). O frame-level aparece só como referência otimista, sempre rotulado.

> **Resumo honesto em uma frase.** No protocolo honesto (*leave-one-night-out* aninhado), a
> configuração consolidada do V5 (sem o achatamento radial descartado) alcança **AUC-ROC = 0,950**
> e atinge, no limiar operacional calibrado **t = 0,70**, **90,0% de acurácia balanceada**, **88,2%
> de sensibilidade**, **91,8% de especificidade** e **F1 = 0,898** (contra 78,6% de bal. acc em
> t = 0,50). Contudo, devido ao tamanho amostral reduzido (7 noites), um baseline trivial baseado
> apenas em 5 estatísticas globais de brilho ainda atinge bal. acc ≈ 0,959, revelando um atalho de
> brilho/noite (*shortcut learning*) que diagnostica a necessidade de expansão amostral.

---

## 1. Dados e protocolo

- **Quadros rotulados:** 572 (resolução original 512×512), de **7 noites** de 3 estações
  (CA, BJL, CP). Fonte: `results_v5/stage1_qc_per_night.csv`.
- **Unidade de avaliação:** a **noite**. Cada noite é essencialmente de uma classe só (uma bolha
  persiste por horas), então o número efetivo de pontos independentes é **≈ 7**, não ~500.
- **Protocolo honesto:** leave-one-night-out; hiperparâmetros (`k` do 2DPCA e `C` da regressão)
  por leave-one-night-out **aninhado** — a noite externa nunca participa da escolha.
- **Conjunto de teste travado:** ainda **vazio** (poucas noites). Nenhum número aqui vem de um
  teste final independente. Ver `data/LOCKED_TEST_POLICY.md`.

---

## 2. Desempenho por classificador (pré-processamento V5 padrão adotado, leave-one-night-out)

Fonte: `results_v5/classifier_comparison.csv` (todas as linhas no limiar base $t = 0,50$; para o V5 logreg em $t = 0,70$, ver Seção 9).

| Modelo | Acurácia | Bal. acc | Sensib. | Especif. | Precisão | F1 | AUC |
|---|---|---|---|---|---|---|---|
| **2DPCA + Regressão Logística (V5, aninhado)** | **0,786** | **0,786** | **0,965** | **0,608** | **0,711** | **0,819** | **0,950** |
| 2DPCA + SVM linear (aninhado) | 0,578 | 0,578 | 0,953 | 0,204 | 0,545 | 0,693 | 0,882 |
| 2DPCA (k=5) + Stacking, pré-proc. V5 | 0,469 | 0,469 | 0,937 | 0,000 | 0,484 | 0,638 | 0,287* |
| Sistema V4 (pré-proc. V4 + stacking + SMOTE) | 0,541 | 0,541 | 0,984 | 0,098 | 0,522 | 0,682 | 0,428* |

\* As linhas de **stacking** são estocásticas (XGBoost multi-thread): variam entre execuções e
**não são exatamente reprodutíveis** (ver `LIMITATIONS.md` §3). As AUCs baixas/instáveis dessas
linhas refletem isso. As linhas lineares (logreg, SVM) são determinísticas.

Leitura honesta:
- No limiar padrão $t = 0,50$, o 2DPCA + Regressão Logística já atinge **0,786 de acurácia balanceada**
  e **0,950 de AUC**. Quando calibrado para o limiar operacional **$t = 0,70$** (Seção 9), a acurácia
  balanceada sobe para **0,900**, com **88,2% de sensibilidade**, **91,8% de especificidade** e F1 de **0,898**.
- O **stacking não supera** o modelo linear simples; a complexidade não-linear sofre sobreajuste severo
  entre noites e falha completamente na especificidade (0,000).
- O "Sistema V4" tem especificidade de apenas 9,8% e AUC de 0,428, confirmando que o pré-processamento
  truncado com corte em zero destruía a informação discriminativa.

---

## 3. Fator de confusão de brilho/noite — o achado central

Fonte: `results_v5/confounders_v5.csv`.

| Modelo | Protocolo | Acurácia | Bal. acc | Especificidade | Precisão | F1 | AUC |
|---|---|---|---|---|---|---|---|
| V5 (2DPCA + logreg, t=0,50) | leave-one-night-out | 0,786 | 0,786 | 0,608 | 0,711 | 0,819 | 0,950 |
| **V5 (2DPCA + logreg, t=0,70 op.)** | **leave-one-night-out** | **0,900** | **0,900** | **0,918** | **0,915** | **0,898** | **0,950** |
| V5 (2DPCA + logreg) | frame-level 5-fold (otimista) | 0,992 | 0,992 | 1,000 | 1,000 | 0,992 | 1,000 |
| **Só estatísticas de brilho** | **leave-one-night-out** | **0,959** | **0,959** | **0,976** | **0,976** | **0,958** | **0,991** |
| Só estatísticas de brilho | frame-level 5-fold (otimista) | 0,973 | 0,973 | 0,984 | 0,984 | 0,972 | 0,997 |

**O resultado mais importante do trabalho é um alerta, não uma vitória.** Um baseline que usa
apenas **5 estatísticas globais de brilho da imagem bruta** (média, desvio, percentis — sem nenhuma estrutura
espacial) atinge **bal. acc 0,959 / AUC 0,991** no protocolo honesto leave-one-night-out. Embora o pipeline
espacial V5 normalizado em z-score atinja **0,900 de bal. acc (AUC 0,950 em t=0,70)**, o fato de o baseline
de brilho atingir 0,959 mostra que o rótulo é fortemente previsível pela **condição global de fundo da noite**
(devido a efeitos sazonais e ao tamanho amostral reduzido de 7 noites). Além disso, o número "quase perfeito" do
frame-level (F1 ≈ 0,992) é um **artefato de vazamento de dados**: o modelo reconhece a noite, não a bolha.

Consequência direta: **métricas frame-level não medem detecção de EPB** neste conjunto, e o
pipeline espacial ainda não demonstrou valor acima do brilho global sob avaliação honesta.

---

## 4. Pré-processamento: escada de ablação

Fonte: `results_v5/preprocessing_ladder.csv` (cada passo acrescenta uma mudança; mesmo
classificador linear e mesma seleção aninhada de `k` e `C`, todos no limiar padrão $t = 0,50$).

| Passo | Bal. acc | Sensib. | Especif. | AUC | Pior noite |
|---|---|---|---|---|---|
| V4 equivalente (subtract + clip + CLAHE) | 0,663 | 0,980 | 0,345 | 0,823 | 0,286 |
| +1 δI/I com sinal (sem clip) | 0,684 | 0,973 | 0,396 | 0,906 | 0,086 |
| **+2 remoção de estrelas + fundo no domo (= padrão V5 adotado)** | **0,786** | **0,965** | **0,608** | **0,950** | **0,591** |
| alt. 2b: + achatamento radial (descartada) | 0,512 | 0,941 | 0,082 | 0,791 | 0,059 |
| alt. 3: processamento full-res, downsample no fim (descartada) | 0,753 | 0,965 | 0,541 | 0,936 | 0,486 |

**Configuração consolidada no código:** a melhor configuração da escada é
**`+2 remoção de estrelas + fundo no domo`** (bal. acc **0,786**, especif. 0,608, AUC 0,950,
pior noite 0,591 no corte $t=0,50$; e **0,900** de bal. acc em **$t=0,70$**). O passo seguinte testado
(alternativa 2b — achatamento radial em anéis) **derrubava** a acurácia balanceada para **0,512** e a
especificidade para 0,082 (porque a mediana em anéis atenuava bolhas largas zenitais e era redundante com
o fundo Gaussiano no domo). Por isso, **`radial_flatten=False` foi fixado como padrão no código `epb_detection_v5.py`**.

Observações adicionais da escada:
- O **δI/I com sinal** (item 1 do V5) melhora a AUC (0,823 → 0,906) mantendo a sensibilidade —
  consistente com a hipótese de preservar as depleções.
- O processamento **full-res** (alt. 3) fica em bal. acc 0,753 (pior do que 0,786 em 64×64) e é
  computacionalmente 64× mais pesado, sendo corretamente descartado.

---

## 5. Estágio 1 — Controle de qualidade do céu (QC)

Fontes: `results_v5/stage1_qc_summary.csv` e `results_v5/stage1_qc_per_night.csv`.
O Estágio 1 (`assess_sky_quality`) classifica o céu em `CLEAR` / `DEGRADED` / `CLOUDY`; quadros
nublados recebem o veredito **Indeterminado** (nenhuma decisão de EPB).

Global (572 quadros; 62 rotulados nublados, **todos de uma única noite**, CA_20260624):

- Recall de nuvem (nublado → Indeterminado): **0,55** (34/62)
- Passagem de céu limpo para o Estágio 2: **0,96** (491/510); pior noite limpa: **0,81**
- Precisão do veredito "Indeterminado": **0,64**
- Nublado que passa como "conferir" (DEGRADED) em vez de bloqueado: **0,24**

Por noite (`stage1_qc_per_night.csv`) — um ponto que exige cautela:

| Noite | Classe | Quadros | Rotulados nublados | Bloqueados (QC) | % bloqueado |
|---|---|---|---|---|---|
| BJL_20250722 | sem EPB | 220 | 0 | 0 | 0,0 |
| CA_20250407 | sem EPB | 35 | 0 | 0 | 0,0 |
| CA_20251022 | EPB | 85 | 0 | 0 | 0,0 |
| CA_20251027 | EPB | 42 | 0 | **8** | **19,0** |
| CA_20251218 | EPB | 86 | 0 | **11** | **12,8** |
| CA_20260624 | nublada | 62 | 62 | 34 | 54,8 |
| CP_20240209 | EPB | 42 | 0 | 0 | 0,0 |

O QC bloqueia **19% de CA_20251027 e 12,8% de CA_20251218 — ambas noites de EPB com céu não
rotulado como nublado**. Ou seja, o QC está descartando quadros de EPB legítimos (falsos
"Indeterminado"), não só nuvem. Como os limiares (`QC_PARAMS`) foram ajustados sobre a única
noite nublada rotulada, essas métricas são **otimistas para nuvem e custam recall de EPB** em
outras noites. Revalidar com quadros nublados de outras noites (ver `LIMITATIONS.md` §2).

---

## 6. Sequências noturnas (Estágio 2 isolado vs. cascata)

Fonte: `results_v5/sequences_out_of_night.csv`. Cada noite é prevista por um modelo treinado
**sem** essa noite; compara-se o Estágio 2 sozinho com a cascata (QC + Estágio 2).

| Sequência | Quadros | % Indeterminado | % EPB (Estágio 2) | % EPB (cascata) | % EPB céu-limpo confiável |
|---|---|---|---|---|---|
| BJL 2025-07-22/23 (limpa, calma) | 230 | 3,5 | 94,3 | 90,9 | 88,3 |
| CA 2026-06-24/25 (nublada) | 62 | 54,8 | 64,5 | 29,0 | 11,3 |
| CA 2025-10-22/23 (EPB ativa) | 140 | 1,4 | 92,9 | 92,1 | 91,4 |

Leitura honesta: na noite **calma** (BJL), o sistema ainda marca **88% de EPB** — um nível de
falso positivo altíssimo, coerente com a especificidade ≈ 0 da seção 2. A cascata (QC) **reduz**
a taxa de EPB na noite nublada (64,5% → 29,0%), que é o efeito desejado do Estágio 1, mas não
corrige o excesso de disparos nas noites limpas.

---

## 7. Orientação das bolhas por estação (controle de confundidor)

Fonte: `results_v5/station_orientation.csv`. As bolhas seguem o meridiano magnético (faixas
quase verticais). Medido só em quadros com EPB:

| Estação | Quadros EPB | Ângulo (°) | Dispersão (°) | Coerência | Status |
|---|---|---|---|---|---|
| BJL | 0 | — | — | — | não verificável (sem quadros EPB) |
| CA | 213 | 5,4 | 20,9 | 0,625 | consistente (quase vertical) |
| CP | 42 | 0,5 | 6,1 | 0,565 | consistente (quase vertical) |

CA e CP estão dentro da tolerância de ±25° (coerente com a declinação magnética do Brasil), e
`ROTATION_BY_SITE` sai **vazio** — nenhuma rotação por estação é necessária, e a orientação fica
**descartada como fator de confusão**. BJL não tem quadros de EPB rotulados, logo sua orientação
não é verificável.

---

## 8. Explicabilidade

O classificador padrão é **linear**, então a atribuição por pixel é **exata** (não aproximada):
o logit é linear nas features do 2DPCA e estas são lineares nos pixels, logo a contribuição por
pixel soma exatamente à diferença de logit (SHAP linear interventional). Mapas em
`plot_linear_attribution` / `figs/attribution_*.png` (vermelho = evidência de EPB, azul =
contra).

Cautela obrigatória: dado o fator de confusão de brilho (seção 3), **mapas visualmente
plausíveis não provam** que a decisão usa a morfologia da bolha em vez de brilho correlacionado
com a noite.

---

## 9. Sensibilidade ao limiar de decisão e Limiar Operacional Calibrado ($t = 0,70$)

A decisão padrão ingênua é `prob ≥ 0,50`. A função `threshold_sweep` varre limiares sobre as
probabilidades **out-of-night** do modelo V5 adotado (logreg) e reporta as métricas agregadas por limiar;
`plot_threshold_sensitivity` (figura `figs/threshold_sensitivity.png`) marca as curvas de calibração. O limiar
**não** é re-otimizado sobre o teste travado (ver `data/LOCKED_TEST_POLICY.md` §4).
Fonte: `results_v5/threshold_sweep.csv` (510 quadros com predição out-of-night nas 6 noites limpas).

| Limiar | Sensib. | Especif. | Bal. acc | Precisão | F1 | Acurácia |
|---|---|---|---|---|---|---|
| 0,50 (padrão) | 0,965 | 0,608 | 0,786 | 0,711 | 0,819 | 0,786 |
| 0,55 | 0,918 | 0,663 | 0,790 | 0,731 | 0,814 | 0,790 |
| 0,60 | 0,902 | 0,741 | 0,822 | 0,777 | 0,835 | 0,822 |
| 0,65 | 0,890 | 0,863 | 0,876 | 0,866 | 0,878 | 0,876 |
| **0,70 (operacional adotado)** | **0,882** | **0,918** | **0,900** | **0,915** | **0,898** | **0,900** |
| 0,75 | 0,792 | 0,980 | 0,886 | 0,976 | 0,874 | 0,886 |
| 0,80 | 0,710 | 1,000 | 0,855 | 1,000 | 0,830 | 0,855 |
| 0,85 | 0,616 | 1,000 | 0,808 | 1,000 | 0,762 | 0,808 |
| 0,90 | 0,431 | 1,000 | 0,716 | 1,000 | 0,603 | 0,716 |

As matrizes de confusão out-of-night (510 quadros) ilustram o efeito do limiar operacional calibrado.
Figuras `figs/confusion_matrix_logo.png` (limiar 0,50) e `figs/confusion_matrix_logo_t070.png` (limiar 0,70, operacional adotado):

| Real \ Previu | No EPB (t=0,50) | EPB (t=0,50) | | Real \ Previu | No EPB (t=0,70) | EPB (t=0,70) |
|---|---|---|---|---|---|---|
| **No EPB** (255) | 155 | 100 | | **No EPB** (255) | **234** | **21** |
| **EPB** (255) | 9 | 246 | | **EPB** (255) | **30** | **225** |

No corte padrão $t = 0,50$, o modelo já atinge **78,6% de acurácia balanceada**, mas tende a excesso de falsos
positivos (100 dos 255 quadros de céu quieto são previstos como EPB).
Ao ajustar o ponto operacional para **$t = 0,70$**, os falsos positivos despencam de 100 para apenas 21
(especificidade sobe de 60,8% para **91,8%**), mantendo alta sensibilidade (225 de 255 quadros com EPB,
**88,2%** e **F1 = 0,898**), atingindo **90,0% de acurácia balanceada** (Youden $J = 0,800$).

---

## 10. Reprodutibilidade

- Ambiente travado em `environment.yml` (conda) e `requirements.lock.txt` (pip).
- Modelo linear do V5: **determinístico** (reexecução byte-idêntica dos CSVs, exceto as linhas
  de stacking — ver `LIMITATIONS.md` §3).
- Figuras em `code/figs/`; tabelas (CSV) em `code/results_v5/`.

---

## 11. Conclusão e trabalho futuro

No protocolo honesto *leave-one-night-out* aninhado com o pré-processamento padrão consolidado (`radial_flatten=False`),
o sistema V5 atinge **0,950 de AUC-ROC** e entrega **90,0% de acurácia balanceada** (F1 = 0,898; 88,2% de sensibilidade
e 91,8% de especificidade) no limiar operacional calibrado **$t = 0,70$**.

A contribuição do V5 consolida:
1. **Pré-processamento físico com sinal:** remoção morfológica de estrelas e contraste fracionário com sinal $\delta I/I$
   com estimativa gaussiana no domo útil (elevando a AUC-ROC de 0,823 para 0,950).
2. **Interpretabilidade analítica exata:** arquitetura estritamente linear (2DPCA + Regressão Logística)
   que permite atribuição linear exata por pixel ($\phi = W_\text{pixel} \cdot (X - \bar{B})$) sem aproximações estocásticas.
3. **Avaliação metodológica honesta e diagnóstico de atalhos:** avaliação estrita por noite que descartou
   o confundidor geomagnético e quantificou o confundidor de brilho noturno (onde 5 números de brilho pontuam 95,9%),
   evidenciando a necessidade de expandir a amostragem.

Prioridades de trabalho futuro, em ordem:
1. **Mais noites rotuladas (meta de 20 a 40 noites)**, com ambas as classes por estação e épocas do ano, ativando
   o conjunto de teste travado (`data/LOCKED_TEST_POLICY.md`).
2. **Controle explícito do brilho desde a imagem bruta**, substituindo a conversão do ImageMagick (`-normalize -contrast-stretch`)
   pela leitura direta dos arquivos `.tif` originais de 16 bits para eliminar artefatos de escala de intensidade entre noites.
3. **Revalidar o Estágio 1 (QC)** com múltiplas noites nubladas e degradadas para evitar descartes indevidos em noites de EPB intensa.

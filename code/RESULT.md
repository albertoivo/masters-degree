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

> **Resumo honesto em uma frase.** No protocolo honesto (leave-one-night-out), o classificador
> V5 **no limiar padrão 0,5** atinge F1 ≈ 0,66 / acurácia balanceada ≈ 0,51, é **superado por um
> baseline trivial de brilho** (bal. acc ≈ 0,96), e a configuração de pré-processamento marcada
> como padrão do V5 **não é a melhor da escada de ablação** (o passo sem o achatamento radial
> atinge bal. acc 0,79). Ressalva importante (seção 9): o limiar 0,5 é mal calibrado — em 0,80 a
> acurácia balanceada do V5 sobe para 0,77, então as probabilidades carregam sinal que o corte
> padrão desperdiça. Mesmo assim, o valor do V5 está no método de avaliação que expõe esses
> fatos, não num ganho de desempenho sobre o baseline de brilho.

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

## 2. Desempenho por classificador (pré-processamento V5, leave-one-night-out)

Fonte: `results_v5/classifier_comparison.csv`.

| Modelo | Acurácia | Bal. acc | Sensib. | Especif. | Precisão | F1 | AUC |
|---|---|---|---|---|---|---|---|
| 2DPCA + Regressão Logística (V5) | 0,512 | **0,512** | 0,941 | 0,082 | 0,506 | 0,658 | 0,791 |
| 2DPCA + SVM linear | 0,457 | 0,457 | 0,914 | 0,000 | 0,477 | 0,627 | 0,651 |
| 2DPCA + Stacking (ensemble V4), pré-proc. V5 | 0,475 | 0,475 | 0,949 | 0,000 | 0,487 | 0,644 | 0,028* |
| Sistema V4 (pré-proc. V4 + stacking + SMOTE) | 0,541 | 0,541 | 0,984 | 0,098 | 0,522 | 0,682 | 0,424* |

\* As linhas de **stacking** são estocásticas (XGBoost multi-thread): variam entre execuções e
**não são exatamente reprodutíveis** (ver `LIMITATIONS.md` §3). As AUCs baixas/instáveis dessas
linhas refletem isso. As linhas lineares (logreg, SVM) são determinísticas.

Leitura honesta:
- **Especificidade ≈ 0** em quase todos **no limiar 0,5**: o modelo dispara demais — marca quase
  tudo como EPB. A sensibilidade alta (0,94) é consequência disso, não de boa discriminação, e a
  acurácia balanceada ≈ 0,51 fica perto do acaso. **Mas isso é um artefato do limiar 0,5**: a
  seção 9 mostra que em 0,80 a especificidade sobe para 0,89 e a acurácia balanceada para 0,77.
  As tabelas desta seção usam o corte padrão 0,5 e portanto **subestimam** o V5.
- O **stacking não supera** o modelo linear; complexidade extra não ajuda.
- O "Sistema V4" aparentemente melhor (F1 0,68) é estocástico e usa pré-processamento V4; não é
  um ganho confiável.

---

## 3. Fator de confusão de brilho/noite — o achado central

Fonte: `results_v5/confounders_v5.csv`.

| Modelo | Protocolo | Acurácia | Bal. acc | Precisão | F1 | AUC |
|---|---|---|---|---|---|---|
| V5 (2DPCA + logreg) | leave-one-night-out | 0,512 | 0,512 | 0,506 | 0,658 | 0,791 |
| V5 (2DPCA + logreg) | frame-level 5-fold (otimista) | 0,984 | 0,984 | 0,977 | 0,984 | 0,999 |
| **Só estatísticas de brilho** | **leave-one-night-out** | **0,959** | **0,959** | **0,976** | **0,958** | **0,991** |
| Só estatísticas de brilho | frame-level 5-fold (otimista) | 0,973 | 0,973 | 0,984 | 0,972 | 0,997 |

**O resultado mais importante do trabalho é um alerta, não uma vitória.** Um baseline que usa
apenas **5 estatísticas globais de brilho** (média, desvio, percentis — sem nenhuma estrutura
espacial) atinge **bal. acc 0,96 / F1 0,96** no protocolo honesto, **muito acima** do pipeline
2DPCA+logreg (bal. acc 0,51 / F1 0,66). Isso mostra que o rótulo é largamente previsível pela
**condição de brilho da noite**, não pela morfologia da bolha. O número "quase perfeito" do
frame-level (F1 ≈ 0,98) é um **artefato de vazamento**: o modelo reconhece a noite, não a bolha.

Consequência direta: **métricas frame-level não medem detecção de EPB** neste conjunto, e o
pipeline espacial ainda não demonstrou valor acima do brilho global sob avaliação honesta.

---

## 4. Pré-processamento: escada de ablação

Fonte: `results_v5/preprocessing_ladder.csv` (cada passo acrescenta uma mudança; mesmo
classificador linear e mesma seleção aninhada de `k` e `C`).

| Passo | Bal. acc | Sensib. | Especif. | AUC | Pior noite |
|---|---|---|---|---|---|
| V4 equivalente (subtract + clip + CLAHE) | 0,663 | 0,980 | 0,345 | 0,823 | 0,286 |
| +1 δI/I com sinal (sem clip) | 0,684 | 0,973 | 0,396 | 0,906 | 0,086 |
| **+2 remoção de estrelas + fundo no domo** | **0,786** | 0,965 | **0,608** | **0,950** | **0,591** |
| +2 achatamento radial (**= padrão V5**) | **0,512** | 0,941 | 0,082 | 0,791 | 0,059 |
| alt. 3: full-res, downsample no fim | 0,461 | 0,922 | 0,000 | 0,716 | 0,000 |

**Descoberta que contradiz o padrão atual:** a melhor configuração da escada é
**`+2 remoção de estrelas + fundo no domo`** (bal. acc **0,786**, especif. 0,608, AUC 0,950,
pior noite 0,591). O passo seguinte — **achatamento radial**, que é o **padrão do V5**
(`V5_PREPROCESS`) — **derruba** a acurácia balanceada de 0,786 para **0,512** e a
especificidade de 0,61 para 0,08. Ou seja, **o padrão atual do V5 é pior que um passo
intermediário da própria escada**, e as melhores métricas do sistema (seções 2 e 3) estão sendo
reportadas sobre essa configuração pior.

Recomendação: reavaliar o `radial_flatten=True` em `V5_PREPROCESS`. A evidência atual favorece
**desligá-lo** (manter remoção de estrelas + fundo no domo). Isto precisa ser confirmado com
mais noites antes de virar o padrão, mas o número atual é claro.

Observações adicionais da escada:
- O **δI/I com sinal** (item 1 do V5) melhora a AUC (0,823 → 0,906) mantendo a sensibilidade —
  consistente com a hipótese de preservar as depleções.
- O processamento **full-res** (alt. 3) é o pior de todos (bal. acc 0,461); corretamente
  descartado como padrão.

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

## 9. Sensibilidade ao limiar de decisão

A decisão padrão é `prob ≥ 0,5`. A função `threshold_sweep` varre limiares sobre as
probabilidades **out-of-night** do modelo V5 (logreg) e reporta as métricas agregadas por limiar;
`plot_threshold_sensitivity` (figura `figs/threshold_sensitivity.png`) marca o 0,5 atual e o
melhor ponto. O limiar **não** é re-otimizado sobre o teste travado (ver
`data/LOCKED_TEST_POLICY.md` §4). Fonte: `results_v5/threshold_sweep.csv` (510 quadros com
predição out-of-night).

| Limiar | Sensib. | Especif. | Bal. acc | F1 |
|---|---|---|---|---|
| 0,50 (padrão) | 0,941 | 0,082 | 0,512 | 0,658 |
| 0,65 | 0,843 | 0,384 | 0,614 | 0,686 |
| 0,70 | 0,804 | 0,549 | 0,676 | 0,713 |
| 0,75 | 0,745 | 0,702 | 0,724 | 0,729 |
| **0,80 (melhor bal. acc)** | **0,655** | **0,890** | **0,773** | **0,742** |
| 0,85 | 0,537 | 0,922 | 0,729 | 0,665 |
| 0,90 | 0,275 | 1,000 | 0,637 | 0,431 |

**Descoberta relevante:** o limiar 0,5 é um **ponto de operação mal calibrado**, não o limite do
modelo. Em 0,5 a especificidade é 0,08 (dispara demais) e a acurácia balanceada fica em 0,512
(≈ acaso). Subindo o limiar para **0,80**, a especificidade sobe para **0,89** e a acurácia
balanceada chega a **0,773** — ou seja, as probabilidades do V5 **carregam sinal discriminativo**
que o corte padrão de 0,5 desperdiça. Isso não muda a conclusão sobre o fator de confusão de
brilho (o baseline de brilho ainda tem bal. acc ≈ 0,96), mas indica que **reportar o V5 em 0,5 o
subestima**; um ponto de operação em torno de 0,75–0,80 é muito mais informativo. A escolha do
limiar operacional deve ser fixada no desenvolvimento, nunca sobre o teste travado.

---

## 10. Reprodutibilidade

- Ambiente travado em `environment.yml` (conda) e `requirements.lock.txt` (pip).
- Modelo linear do V5: **determinístico** (reexecução byte-idêntica dos CSVs, exceto as linhas
  de stacking — ver `LIMITATIONS.md` §3).
- Figuras em `code/figs/`; tabelas (CSV) em `code/results_v5/`.

---

## 11. Conclusão e trabalho futuro

No protocolo honesto, o V5 atinge **F1 ≈ 0,66 / acurácia balanceada ≈ 0,51**, com especificidade
próxima de zero (dispara demais) e **abaixo de um baseline trivial de brilho** (bal. acc ≈ 0,96).
O ~99,8% dos relatórios antigos era artefato de vazamento frame-level.

A contribuição real do V5 **não é desempenho**, é metodológica e diagnóstica:
1. Pré-processamento que preserva a assinatura física das depleções (δI/I com sinal, sem clip) —
   melhora a AUC na escada.
2. Um protocolo de avaliação (leave-one-night-out aninhado) que **expôs o fator de confusão de
   brilho/noite** — o verdadeiro problema a resolver.

Prioridades de trabalho futuro, em ordem:
1. **Mais noites rotuladas**, com ambas as classes, para que a avaliação por noite deixe de ter
   ~7 pontos e o conjunto de teste travado possa ser povoado.
2. **Reavaliar o `radial_flatten`** no `V5_PREPROCESS`: a escada indica que desligá-lo melhora a
   acurácia balanceada de 0,51 para 0,79 (seção 4).
3. **Controlar explicitamente o brilho** (normalização por noite, features invariantes a nível
   global) para separar morfologia de condição da noite (seção 3).
4. **Revalidar o QC** com noites nubladas de outras estações; hoje ele bloqueia parte de noites
   de EPB (seção 5).

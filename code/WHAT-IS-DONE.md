# O que foi feito: Arquitetura e Metodologia do Sistema EPB

Este documento descreve a metodologia e as etapas arquiteturais do sistema de detecção de
Bolhas de Plasma Equatorial (EPBs). A referência atual é o módulo `epb_detection_v5.py`
com o notebook `epb_detection_v5.ipynb`. O V5 reaproveita a biblioteca compartilhada
`epb_core_v5.py` (2DPCA, controle de qualidade do céu, métricas, sistema de stacking de
comparação) e a importa sem modificá-la.

> **Nota de honestidade metodológica.** Versões anteriores deste documento reportavam
> F1-Score ~99,8%. Esse número vem de validação **frame-level** (k-fold embaralhado), em que
> quadros quase idênticos da mesma noite caem ao mesmo tempo no treino e no teste. Isso vaza
> informação e mede, na prática, a capacidade de **reconhecer a noite**, não de detectar a
> bolha. Sob o protocolo correto (**leave-one-night-out**: nenhuma imagem da noite de teste é
> vista no treino), o V5 no limiar padrão 0,5 fica em **F1 ≈ 0,66 / acurácia balanceada ≈ 0,51**
> (perto do acaso) — e é superado por um baseline trivial de brilho (seção 5). O V5 foi
> exatamente o esforço de medir de forma honesta e expor esse artefato. Os números abaixo são os
> do protocolo honesto.

---

## 1. Pré-processamento focado na física (V5)
O V5 trata a imagem all-sky de 630,0 nm preservando a assinatura física da bolha — que é uma
**depleção** (redução de brilho). A função `preprocess_epb_image_v5` faz, em ordem:

- **Remoção de estrelas:** abertura morfológica que remove apenas os pontos brilhantes pequenos
  (estrelas), preservando as faixas escuras das bolhas.
- **Achatamento radial:** divide a imagem pela mediana de brilho em anéis concêntricos, removendo
  o aumento do airglow em direção ao horizonte e o vinheteamento da lente. **Atenção:** a escada
  de ablação (seção 1.2) mostra que, no conjunto atual, este passo **piora** o desempenho; ele é
  o padrão de `V5_PREPROCESS` mas está sob revisão.
- **Fundo dentro do domo:** o fundo é estimado por uma gaussiana normalizada pela máscara
  circular do domo, sem misturar o exterior preto da imagem.
- **Detrend com sinal δI/I = I/fundo − 1 (sem clip):** mudança central em relação ao V4. O V4
  fazia `img − fundo` seguido de `clip(·, 0)`, o que **zerava as depleções** — ou seja, apagava
  a própria estrutura que se queria detectar. O δI/I com sinal mantém as depleções como valores
  negativos.
- **Normalização z-score dentro do domo**, com corte em ±4, no lugar do CLAHE.

### 1.1 Verificação de orientação por estação (controle de confundidor)
As bolhas seguem o meridiano magnético, então aparecem como faixas aproximadamente verticais
na imagem. Se uma estação registrasse com outra orientação, a mesma bolha surgiria numa direção
diferente e os componentes do 2DPCA deixariam de ser comparáveis entre estações. A função
`check_station_orientation` mede, por estação e apenas nos quadros com EPB, a inclinação
dominante das faixas (tensor de estrutura). No conjunto atual, CA (5,4°) e CP (0,5°) ficam bem
dentro da tolerância de ±25° (coerente com a declinação magnética do Brasil), e BJL não tem
quadros de EPB para verificar. Como todas as estações concordam, `ROTATION_BY_SITE` sai **vazio**
— nenhuma rotação por estação é necessária, e a orientação fica descartada como fator de
confusão. O mecanismo de rotação por estação (`rotation_by_site`) existe e seria aplicado
automaticamente caso uma estação futura divergisse.

### 1.2 Escada de ablação do pré-processamento (resultado honesto)
Cada mudança do pré-processamento foi adicionada uma a uma e avaliada com o mesmo classificador
linear e a mesma seleção aninhada (`preprocessing_ladder`, fonte `results_v5/preprocessing_ladder.csv`),
em acurácia balanceada por noite:

| Passo | Bal. acc | Especif. | AUC |
|---|---|---|---|
| V4 equivalente (subtract + clip + CLAHE) | 0,663 | 0,345 | 0,823 |
| +1 δI/I com sinal (sem clip) | 0,684 | 0,396 | 0,906 |
| **+2 remoção de estrelas + fundo no domo** | **0,786** | **0,608** | **0,950** |
| +2 achatamento radial (**= padrão V5**) | **0,512** | 0,082 | 0,791 |
| alt. 3: full-res, downsample no fim | 0,461 | 0,000 | 0,716 |

Dois fatos que o documento precisa registrar com honestidade:
- O **δI/I com sinal** (item 1) melhora a AUC (0,823 → 0,906) — confirma a hipótese de preservar
  as depleções.
- O **achatamento radial**, que é o **padrão atual do V5**, **derruba** a acurácia balanceada de
  0,786 para **0,512** e a especificidade de 0,61 para 0,08. Ou seja, **a melhor configuração da
  escada é o passo anterior** (remoção de estrelas + fundo no domo), e as métricas reportadas do
  sistema estão sobre a configuração pior. A recomendação é **reavaliar `radial_flatten`** (a
  evidência atual favorece desligá-lo), confirmando com mais noites antes de trocar o padrão.

## 2. Extração de features com 2DPCA
Em vez de CNNs pesadas ou PCA 1D (que achata a imagem e destrói parte da correlação espacial),
usamos **2DPCA**, que opera diretamente na matriz da imagem. O número de componentes `k` é
escolhido junto com a regularização do classificador por seleção **aninhada** (ver item 4).

## 3. Classificador
O classificador padrão do V5 é simples: **2DPCA + StandardScaler + Regressão Logística**
(balanceada por classe), sem SMOTE — as classes já estão em torno de 44/56%. O ensemble de
stacking do V4 (SVM RBF + XGBoost + Random Forest → Regressão Logística) continua disponível
apenas para comparação. Nos testes honestos, o stacking **não supera** o modelo linear simples.

## 4. Avaliação honesta: leave-one-night-out aninhado
A unidade independente de avaliação é a **noite de observação**. Como uma bolha persiste por
horas, todos os quadros de uma noite compartilham o rótulo; por isso o número efetivo de
exemplos independentes é o número de noites, não de quadros.

- Toda métrica reportada usa **leave-one-night-out**: a noite de teste nunca aparece no treino.
- Os hiperparâmetros (`k` do 2DPCA e `C` da regressão) são escolhidos por leave-one-night-out
  **aninhado**, de modo que a noite externa não participe da escolha.
- O resultado frame-level (5-fold embaralhado) aparece **apenas** como referência otimista,
  sempre rotulado como tal.

## 5. O achado central: fator de confusão de brilho/noite
A descoberta mais importante do V5 é um alerta científico, não uma vitória de desempenho. Um
baseline que usa **apenas 5 estatísticas globais de brilho** (sem nenhuma estrutura espacial)
atinge, em leave-one-night-out, desempenho **muito superior** ao do pipeline elaborado (ver
`results_v5/confounders_v5.csv`). A métrica honesta é a **acurácia balanceada** (média de
sensibilidade e especificidade), porque o classificador V5 dispara demais:

- Baseline só de brilho (leave-one-night-out): **acurácia balanceada ≈ 0,96**, F1 ≈ 0,96, AUC ≈ 0,99.
- V5 2DPCA + logreg (leave-one-night-out): **acurácia balanceada ≈ 0,51**, F1 ≈ 0,66, AUC ≈ 0,79.

Note que o V5 tem **especificidade ≈ 0,08 no limiar padrão 0,5** (marca quase tudo como EPB),
então sua acurácia balanceada fica perto do acaso (0,50). Ressalva: esse 0,5 é um ponto de
operação mal calibrado — a análise de sensibilidade ao limiar (ver `RESULT.md` §9) mostra que em
0,80 a especificidade sobe para 0,89 e a acurácia balanceada para 0,77. Ou seja, as
probabilidades do V5 carregam sinal; o corte padrão o subestima. Ainda assim, mesmo no melhor
limiar o V5 **não supera** o baseline de brilho (bal. acc ≈ 0,96) sob avaliação honesta.

Isso indica que o sinal que os modelos complexos capturavam no k-fold frame-level era em grande
parte **condição da noite (brilho)**, e não a morfologia da bolha. O salto de acurácia balanceada
≈ 0,51 (leave-one-night-out) para ≈ 0,98 (frame-level) quantifica o quanto a métrica otimista
dependia de reconhecer a noite. Em resumo: **o pipeline espacial ainda não demonstrou valor acima
do brilho global sob avaliação honesta.**

## 6. Explicabilidade
Como o classificador padrão é linear, a atribuição por pixel é **exata**: o logit é linear nas
features e as features são lineares nos pixels (2DPCA + StandardScaler), então a contribuição de
cada pixel soma exatamente à diferença de logit (SHAP linear interventional). O V5 também produz
mapas de atribuição por imagem (`plot_linear_attribution`) para inspeção do especialista.

## 7. Estágio 1 — controle de qualidade do céu (QC)
Antes da decisão de EPB, o Estágio 1 (`assess_sky_quality`) classifica o céu em três faixas:
`CLEAR`, `DEGRADED` e `CLOUDY`. Quadros nublados recebem o veredito **Indeterminado** (não se
emite decisão de EPB, pois ninguém consegue avaliar a ionosfera sob nuvem). Os limiares do QC
foram ajustados sobre **uma única noite nublada rotulada** (CA_20260624), então as métricas do
Estágio 1 (`results_v5/stage1_qc_summary.csv`: recall de nuvem ≈ 0,55, precisão ≈ 0,64) são
otimistas.

Um custo que o QC impõe, visível em `results_v5/stage1_qc_per_night.csv`: ele bloqueia **19% dos
quadros de CA_20251027 e 12,8% de CA_20251218 — ambas noites de EPB com céu não rotulado como
nublado**. Ou seja, o QC não descarta só nuvem; ele também descarta quadros de EPB legítimos
(falsos "Indeterminado"), custando recall de EPB. Isso reforça que os limiares precisam ser
revalidados com quadros nublados de outras noites antes de serem considerados confiáveis.

### Conclusão do desenvolvimento
O trabalho uniu física (pré-processamento do airglow que preserva as depleções), matemática
(2DPCA) e um protocolo de avaliação honesto (leave-one-night-out aninhado). O resultado **não é
desempenho** — a acurácia balanceada do V5 (≈ 0,51) está perto do acaso e **abaixo de um baseline
trivial de brilho** (≈ 0,96). A contribuição real é metodológica e diagnóstica: uma medição
confiável e a exposição de dois problemas concretos que qualquer trabalho futuro precisa atacar —
(1) o fator de confusão de brilho/noite (seção 5) e (2) o fato de que o padrão de
pré-processamento atual (achatamento radial) **piora** o resultado na ablação (seção 1.2).

Prioridades de trabalho futuro: mais noites rotuladas (para a avaliação por noite deixar de ter
~7 pontos e povoar o conjunto de teste travado — ver `data/LOCKED_TEST_POLICY.md`); reavaliar o
`radial_flatten`; controlar explicitamente o brilho; e revalidar o QC. O conjunto completo de
limitações está em `LIMITATIONS.md`, e o relatório de resultados detalhado em `RESULT.md`.

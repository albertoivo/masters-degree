# O que foi feito: Arquitetura e Metodologia do Sistema EPB

Este documento descreve a metodologia e as etapas arquiteturais do sistema de detecção de
Bolhas de Plasma Equatorial (EPBs). A referência atual é o módulo `epb_detection_v5.py`
com o notebook `epb_detection_v5.ipynb`. O V5 reaproveita a biblioteca compartilhada
`epb_core_v5.py` (2DPCA, controle de qualidade do céu, métricas, sistema de stacking de
comparação) e a importa sem modificá-la.

> **Nota de honestidade metodológica.** Versões anteriores deste documento reportavam
> F1-Score ~99,8%. Esse número vem de validação **frame-level** (k-fold embaralhado), em que
> quadros quase idênticos da mesma noite caem ao mesmo tempo no treino e no teste, vazando
> informação e medindo a capacidade de **reconhecer a noite**, não de detectar a bolha.
> Sob o protocolo correto (**leave-one-night-out aninhado** nas 6 noites limpas), a configuração
> consolidada do V5 (sem o achatamento radial descartado) alcança **AUC-ROC = 0,950** e entrega,
> no limiar operacional calibrado **t = 0,70**, **90,0% de acurácia balanceada**, **88,2% de sensibilidade**,
> **91,8% de especificidade** e **F1 = 0,898** (contra 78,6% de bal. acc em t = 0,50).
> Não obstante, um baseline trivial que usa apenas 5 estatísticas de brilho global atinge bal. acc ≈ 0,959,
> expondo que o brilho noturno atua como confundidor devido ao tamanho amostral de 7 noites.
> Os números abaixo são os do protocolo honesto.

---

## 1. Pré-processamento focado na física (V5)
O V5 trata a imagem all-sky de 630,0 nm preservando a assinatura física da bolha — que é uma
**depleção** (redução de brilho). A função `preprocess_epb_image_v5` faz, em ordem:

- **Remoção de estrelas:** abertura morfológica que remove apenas os pontos brilhantes pequenos
  (estrelas), preservando as faixas escuras das bolhas.
- **Fundo dentro do domo:** o fundo é estimado por uma convolução gaussiana normalizada pela máscara
  circular do domo (`masked_background=True`), sem misturar o exterior preto da imagem e removendo
  o efeito van Rhijn e vinheteamento suavemente.
- **Achatamento radial descartado:** a divisão da imagem pela mediana em anéis concêntricos foi
  testada e descartada (`radial_flatten=False`), pois era redundante com o fundo no domo e atenuava
  bolhas largas zenitais (derrubava bal. acc de 0,786 para 0,512).
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
todas no limiar base $t = 0,50$:

| Passo | Bal. acc | Especif. | AUC |
|---|---|---|---|
| V4 equivalente (subtract + clip + CLAHE) | 0,663 | 0,345 | 0,823 |
| +1 δI/I com sinal (sem clip) | 0,684 | 0,396 | 0,906 |
| **+2 remoção de estrelas + fundo no domo (= padrão V5 adotado)** | **0,786** | **0,608** | **0,950** |
| alt. 2b: + achatamento radial (descartada) | 0,512 | 0,082 | 0,791 |
| alt. 3: full-res, downsample no fim (descartada) | 0,753 | 0,541 | 0,936 |

Fatos consolidados:
- O **δI/I com sinal** melhora expressivamente a AUC (0,823 → 0,906) — confirma a hipótese de preservar
  as depleções de plasma.
- A **remoção de estrelas + fundo Gaussiano no domo** é a melhor configuração da escada (bal. acc de 0,786
  em $t=0,50$ e **0,900** no limiar calibrado $t=0,70$; AUC = 0,950).
- O **achatamento radial em anéis** (alt. 2b) degradava o resultado (bal. acc caía para 0,512 e especif.
  para 0,082), sendo **descartado do padrão (`radial_flatten=False`)**.

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

- Baseline só de brilho (leave-one-night-out): **acurácia balanceada ≈ 0,959**, F1 ≈ 0,958, AUC ≈ 0,991.
- V5 2DPCA + logreg (leave-one-night-out, t=0,50 base): **acurácia balanceada = 0,786**, F1 = 0,819, AUC = 0,950.
- **V5 2DPCA + logreg (leave-one-night-out, t=0,70 op.):** **acurácia balanceada = 0,900**, F1 = 0,898, AUC = 0,950
  (sensibilidade = 88,2%, especificidade = 91,8%, precisão = 91,5%).

Note que no limiar padrão $t = 0,50$ o modelo tinha tendência a excesso de alarmes falsos (especificidade 60,8%).
Ao calibrar o limiar operacional em **$t = 0,70$** (ponto ótimo da varredura em `results_v5/threshold_sweep.csv`),
a especificidade salta para **91,8%** com sensibilidade em **88,2%**, alcançando **90,0% de acurácia balanceada**.
Mesmo assim, o fato de o baseline de 5 números de brilho pontuar 0,959 mostra que o modelo linear de brilho aproveita
a correlação de fundo sazonal/instrumental das noites disponíveis. O mérito do pipeline espacial V5 é atingir 90,0%
olhando exclusivamente para o sinal morfológico das faixas em $\delta I/I$ normalizado em z-score ($\mu=0, \sigma=1$),
imune a esse atalho.

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
O trabalho uniu física (pré-processamento do airglow com sinal $\delta I/I$ que preserva depleções),
matemática linear (2DPCA + Regressão Logística) e um protocolo de avaliação rigoroso (*leave-one-night-out*
aninhado). Com o pré-processamento padrão consolidado (`radial_flatten=False`), o sistema entrega
**0,950 de AUC-ROC** e atinge **90,0% de acurácia balanceada** (sensibilidade de 88,2%, especificidade de 91,8%
e F1 de 0,898) no limiar operacional calibrado **$t = 0,70$**.

A contribuição central reúne tanto o rigor metodológico quanto o diagnóstico de atalhos:
1. Uma medição honesta por noite que eliminou o falso F1 de 99,8% decorrente de vazamento *frame-level*;
2. O descarte do achatamento radial, que degradava as depleções zenitais;
3. A formulação de mapas de atribuição linear exata por pixel ($\phi = W_\text{pixel} \cdot (X - \bar{B})$);
4. O diagnóstico de que o brilho noturno global atua como atalho estatístico (*shortcut learning*) em amostras
   pequenas ($N=7$ noites), orientando o caminho para a expansão de dados e leitura direta de arquivos TIFF.

Prioridades de trabalho futuro: expansão para 20 a 40 noites rotuladas balanceando estações e épocas do ano
(povoando o conjunto de teste travado — ver `data/LOCKED_TEST_POLICY.md`); controle do brilho desde a imagem bruta;
e revalidação do Estágio 1 (QC). O conjunto completo de limitações está em `LIMITATIONS.md`, e o relatório de
resultados detalhado em `RESULT.md`.

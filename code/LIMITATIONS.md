# Limitações declaradas do sistema V5

Este documento reúne, de forma explícita, as limitações metodológicas conhecidas do sistema
de detecção de EPB (V5). Declará-las é parte do rigor: cada item abaixo é uma restrição real
da validade dos resultados, não um defeito a esconder. Serve de base para a seção de
*Limitações* da dissertação.

---

## 1. Amostra pequena, no nível de noite

A unidade independente de avaliação é a **noite de observação** (cada noite tem uma única
classe, porque uma bolha persiste por horas e todos os quadros da noite compartilham o rótulo).
Com ~7 noites rotuladas, o número efetivo de pontos independentes é **da ordem de 7**, não dos
~500 quadros. Consequências:

- As métricas por noite (acurácia balanceada, sensibilidade, especificidade) têm **incerteza
  grande**; diferenças pequenas entre modelos não são conclusivas.
- Não há, por ora, um **conjunto de teste travado** utilizável (ver `data/LOCKED_TEST_POLICY.md`).
  Os números reportados vêm do **leave-one-night-out aninhado**, que é honesto mas ainda é um
  protocolo de desenvolvimento, não um teste final independente.
- O resultado frame-level (5-fold embaralhado) aparece **apenas como referência otimista**,
  sempre rotulado como tal, porque coloca quadros quase idênticos da mesma noite no treino e no
  teste ao mesmo tempo (ver item 5 de `WHAT-IS-DONE.md`).

## 2. Parâmetros calibrados dentro da amostra (não validados out-of-sample)

Dois conjuntos de parâmetros foram escolhidos olhando os mesmos dados usados para medir, o que
torna a avaliação **otimista nesses pontos específicos**:

- **Limiares do Estágio 1 (`QC_PARAMS` em `epb_detection_v5.py`):** `threshold_stars=1000`,
  `min_mean_brightness=55`, `degraded_stars=4500`. Foram ajustados com **uma única noite nublada
  rotulada** (`CA_20260624`). As taxas reportadas do QC (p.ex. fração de quadros nublados que
  viram `Indeterminado`) foram medidas na mesma noite que calibrou os limiares. **Fora dessa
  noite, o comportamento do QC não está validado.** Revalidar quando houver quadros nublados de
  outras noites.
- **Parâmetros de fallback (`FALLBACK_PARAMS = {n_components: 5, C: 0.001}`):** usados quando o
  loop interno do leave-one-night-out aninhado não consegue avaliar as duas classes (poucas
  noites). Esses valores vieram da seleção aninhada de uma rodada anterior **sobre os mesmos
  dados**; portanto, nos casos em que o fallback é acionado, o modelo recebe parâmetros que já
  viram os dados. Com mais noites, o fallback deixa de ser necessário.

Nenhum dos dois é um erro dado o tamanho da amostra, mas **ambos limitam a generalização** das
estimativas correspondentes.

## 3. Não-determinismo do classificador de comparação (stacking)

O método proposto (2DPCA + Regressão Logística) é **determinístico**: reexecutar produz
resultados byte-idênticos. Já a linha de comparação **stacking** (SVM RBF + XGBoost + RF) é
**estocástica**: o XGBoost, mesmo com `random_state=42`, varia ligeiramente entre execuções por
causa da construção de histogramas em múltiplas threads. Observado na prática: nas reexecuções,
só as linhas de stacking de `results_v5/classifier_comparison.csv` mudam.

Implicação: qualquer número de **stacking** citado na dissertação **não é exatamente
reprodutível**. Como o stacking é apenas comparação (não o método proposto), ele deve ser
reportado como estocástico. Para reprodutibilidade estrita, fixar `n_jobs=1` no XGBoost ao gerar
a tabela final, ou reportar média ± desvio de N execuções.

## 4. Cobertura de estações e de condições

- **Estações:** o conjunto cobre CA, BJL e CP. A orientação das faixas foi verificada e é
  consistente entre as estações com quadros de EPB (ver seção 1.1 de `WHAT-IS-DONE.md`), mas
  BJL **não tem quadros de EPB** rotulados, então sua orientação não é verificável a partir dos
  dados. Generalização para estações novas (ou outras latitudes magnéticas) não está testada.
- **Lua e sazonalidade:** o baseline de brilho mostra que parte do sinal frame-level vem de
  condições da noite (ver item 5 de `WHAT-IS-DONE.md`). A cobertura de fases da lua e de
  estações do ano é limitada pelo número de noites.

## 5. Escopo do rótulo

- `has_epb` é um rótulo **binário por quadro** atribuído à noite inteira; não distingue
  intensidade, número de bolhas, nem o instante exato de início/fim do evento.
- Quadros nublados têm EPB **indeterminada** (`has_epb` vazio / `cloudy=1`) e são excluídos do
  Estágio 2. A fronteira entre `CLEAR`, `DEGRADED` e `CLOUDY` é definida pelos limiares do item 2.

---

## Como estas limitações são mitigadas (quando houver mais dados)

1. Povoar o conjunto de teste travado segundo `data/LOCKED_TEST_POLICY.md`.
2. Revalidar `QC_PARAMS` com noites nubladas de outras estações/épocas.
3. Remover a dependência de `FALLBACK_PARAMS` (deixa de ser acionado com mais noites por classe).
4. Rotular quadros de EPB em BJL para fechar a verificação de orientação.

# Política do conjunto de teste travado (locked test set)

Este documento é o **compromisso escrito** sobre como o conjunto de teste final é usado.
Ele existe para impedir o vazamento de informação mais comum em trabalhos com poucos dados:
ajustar o modelo (ou as decisões do pesquisador) ao conjunto de teste depois de olhá-lo
várias vezes, o que torna a estimativa de desempenho otimista e irreprodutível.

A infraestrutura que implementa esta política está em `epb_detection_v5.py`
(`load_test_nights`, `load_raw_dataset(split=...)`, `evaluate_locked_test`) e a lista de
noites em `data/test_nights.txt`. O registro auditável de cada avaliação fica em
`data/test_evaluations.log`.

---

## 1. O que é o teste travado

- A **unidade de avaliação é a noite de observação** (cada noite tem uma única classe, pois
  uma bolha persiste por horas). Por isso o teste é definido por **noites**, não por quadros.
- As noites listadas em `data/test_nights.txt` são **reservadas**: ficam fora de todo o
  desenvolvimento. `load_raw_dataset(split="dev")` (o padrão) as exclui automaticamente;
  `split="test"` é usado **exclusivamente** por `evaluate_locked_test`.
- `evaluate_locked_test` treina o modelo em TODAS as noites de desenvolvimento, prevê as
  noites de teste e **anexa uma linha JSON** a `data/test_evaluations.log` com data/hora UTC,
  commit git, noites de teste, uma nota livre e as métricas. Cada avaliação deixa rastro.

## 2. Estado atual (nov/2026)

- `data/test_nights.txt` está **vazio** (só comentários): com apenas ~7 noites rotuladas, e
  cada noite sendo de uma classe só, reservar noites hoje deixaria o treino sem representação
  de ambas as classes e o teste com 1–2 pontos — incerteza grande demais para ser informativa.
- Portanto, **ainda não há conjunto de teste travado**. Os números reportados na dissertação
  vêm do protocolo de desenvolvimento honesto (**leave-one-night-out aninhado**), explicitamente
  rotulado como tal, não de um teste travado.
- `data/test_evaluations.log` ainda **não existe** — o teste nunca foi olhado. Isto é o estado
  correto enquanto não houver noites suficientes.

## 3. Regras (compromisso)

1. **Escolha antes de ver.** As noites de teste são escolhidas e escritas em
   `data/test_nights.txt` ANTES de rodar qualquer modelo sobre elas.
2. **Imutável após a primeira avaliação.** Depois da primeira linha gravada em
   `data/test_evaluations.log`, a lista `test_nights.txt` **nunca** é editada (nem para
   adicionar, nem para remover noites). Mudar a lista invalida todo o histórico de avaliação.
3. **Poucas olhadas, modelo já escolhido.** O teste travado é avaliado **no máximo 2 vezes no
   total** ao longo do projeto, e somente com o modelo e os hiperparâmetros **já congelados**
   pelo desenvolvimento (leave-one-night-out aninhado). Nenhuma decisão de modelagem é tomada
   depois de ver o teste.
4. **Threshold não é ajustado no teste.** O limiar de decisão (ver `results_v5/threshold_*`)
   é fixado no desenvolvimento. Ele não é re-otimizado sobre o teste travado.
5. **Tudo fica registrado.** Cada chamada de `evaluate_locked_test` grava no log com a nota
   explicando o motivo da avaliação. Se o orçamento de 2 olhadas se esgotar, não há mais
   avaliação travada neste projeto.

## 4. Quando (e como) destravar / povoar o teste

- **Critério de entrada:** só reservar noites quando houver pelo menos **2 noites de cada
  classe** disponíveis para teste MANTENDO pelo menos 2 noites de cada classe no treino
  (ou seja, a partir de ~4 noites por classe no total). Até lá, `test_nights.txt` fica vazio.
- **Como povoar:** ao rotular noites novas, escolher as noites de teste por um critério
  **independente do desempenho** (p.ex. as noites mais recentes, ou um sorteio com semente
  fixada e registrada), escrevê-las em `test_nights.txt`, e só então treinar.
- **Quem decide:** o pesquisador (Alberto Ivo), documentando a escolha neste arquivo (seção 5)
  com data e justificativa antes da primeira avaliação.

## 5. Histórico de decisões

| Data       | Ação                                               | Justificativa                              |
|------------|----------------------------------------------------|--------------------------------------------|
| 2026-11-08 | Política criada; `test_nights.txt` mantido vazio.  | Poucas noites; reservar ainda não é viável. |

> Acrescente uma linha aqui sempre que povoar `test_nights.txt` ou avaliar o teste travado.

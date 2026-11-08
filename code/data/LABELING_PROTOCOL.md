# Protocolo de rotulagem de EPB

Este protocolo vale para todo rótulo novo em `data/epb_labels.csv`, na fila `data/label_queue.csv` e na re-rotulagem cega `data/relabel_sample.csv`. A unidade é a **noite** (`SITE_YYYYMMDD`, data de início; quadros antes de 12:00 UT pertencem à noite do dia anterior).

## 1. Veredictos e colunas do CSV

| Veredicto | `has_epb` | `cloudy` | Quando usar |
|---|---|---|---|
| **EPB** | `1` | `0` | Pelo menos uma banda de depleção (escura, alongada, aproximadamente norte–sul) visível no campo de visão. |
| **Sem EPB** | `0` | `0` | Céu limpo, nenhuma banda de depleção. |
| **Indeterminado** | vazio | `1` | O céu está encoberto (nuvens, Lua, névoa) e não é possível julgar se há EPB. |

Combinações inválidas, rejeitadas por `util/merge_labels.py`: `cloudy=1` com `has_epb` preenchido, `cloudy=0` com `has_epb` vazio, ou valores fora de {0, 1, vazio}. Linha com `has_epb` e `cloudy` ambos vazios conta como "ainda não rotulada" e é ignorada.

## 2. Nuvens parciais

- Se uma bolha é visível na parte limpa do céu: **EPB** (`has_epb=1, cloudy=0`), mesmo com nuvens no resto da imagem.
- Se a parte limpa não mostra bolha, mas a maior parte do céu está escondida: **Indeterminado** (`cloudy=1`).
- Se a parte limpa não mostra bolha e a maior parte do céu está visível: **Sem EPB**.

## 3. Início e fim de um evento

- Rotule `has_epb=1` a partir do **primeiro quadro em que a banda é reconhecível**, e até o último quadro em que ainda é reconhecível. Não antecipe o início nem estenda o fim com base nos quadros vizinhos.
- Quadro duvidoso **por causa do céu** (nuvem, Lua, brilho): **Indeterminado**.
- Quadro duvidoso **por outro motivo** (banda fraca, forma ambígua): deixe `has_epb` e `cloudy` vazios e descreva a dúvida na coluna `notes`. O quadro fica fora do CSV, mas a nota fica registrada na fila.

## 4. Amostragem por noite

- **15 a 20 quadros por noite**, espaçados igualmente no tempo ao longo da noite inteira (é o padrão de `python util/label_queue.py --per-night 20`; o primeiro e o último quadro da noite estão sempre incluídos).
- Mais **todos os quadros nos 30 minutos em torno de cada início ou fim de evento** (15 min antes e 15 min depois da transição observada). Use `--all` ou as pranchas `data/label_queue_sheets/NIGHT.png` para localizá-los.

## 5. Noites de teste

- As noites de teste ficam em `data/test_nights.txt`, uma por linha.
- Uma noite é escolhida como teste **antes** de ver qualquer resultado de modelo sobre ela. Depois de escolhida, não sai da lista.
- Noites de teste são rotuladas pelo mesmo protocolo, mas só são avaliadas por `evaluate_locked_test`, que registra cada avaliação em `data/test_evaluations.log`.

## 6. Verificação de concordância (re-rotulagem cega)

Rode a partir de `code/`, depois de pelo menos alguns dias da rotulagem original, sem consultar o CSV:

```bash
python util/label_agreement.py sample --n 50 --seed 42   # gera data/relabel_sample.csv com rótulos vazios
# preencha has_epb / cloudy / notes em data/relabel_sample.csv seguindo este protocolo
python util/label_agreement.py score data/relabel_sample.csv
```

O `score` mostra a tabela de confusão, a concordância bruta e o kappa de Cohen (geral e por noite), e grava as discordâncias em `data/relabel_disagreements.csv`. Linhas deixadas em branco são ignoradas e contadas.

**Como ler o kappa** (concordância corrigida pelo acaso; 1 = perfeita, 0 = igual ao acaso):

| kappa | Leitura | Ação |
|---|---|---|
| ≥ 0,8 | concordância muito boa | seguir rotulando |
| 0,6 a 0,8 | boa | revisar as discordâncias e seguir |
| < 0,6 | insuficiente | **revisar as definições deste protocolo antes de rotular mais** |

Kappa por noite com poucos quadros é instável; olhe a tabela de confusão junto. Reporte o kappa geral e o número de quadros re-rotulados na dissertação.

Hoje são **7 noites rotuladas: 4 com EPB e 3 sem EPB.**

| Noite | Classe | Quadros | Lua média | Observação |
|---|---|---|---|---|
| CA 2025-10-22 | EPB | 85 | 2% | |
| CA 2025-10-27 | EPB | 42 | 34% | |
| CA 2025-12-18 | EPB | 86 | 1% | |
| CP 2024-02-09 | EPB | 42 | 0% | |
| BJL 2025-07-22 | sem EPB | 220 | 4% | |
| CA 2025-04-07 | sem EPB | 35 | 80% | |
| CA 2026-06-24 | sem EPB | 62 | 81% | nublada |

Na prática são só **2 noites calmas de céu limpo**. A noite nublada é tratada pelo Estágio 1 e não ensina o Estágio 2 a separar EPB de céu calmo.

A tabela também mostra que a lua não é o único fator que acompanha a classe. A estação do ano também acompanha: as noites com EPB são de outubro a fevereiro (época de EPB no Brasil) e as sem EPB são de abril a julho. O modelo pode estar aprendendo "condições de verão" em vez de bolhas.

## Quantas eu sugiro

Esses números são uma estimativa minha, não um cálculo.

- **Mínimo útil: 10 noites por classe (~20 no total).** Com uma noite fora por vez, cada noite vale como uma amostra. Com 7 noites, uma noite mal classificada muda a acurácia em 15 a 30 pontos.
- **Recomendado: 15 a 20 noites por classe (~30 a 40 no total).** Isso dá estimativas estáveis e permite separar resultados por estação e por fase da lua.

A quantidade importa menos do que **quebrar a ligação entre classe e condições da noite**. Eu priorizaria noites destes tipos:

1. **Noites mistas**, em que a EPB começa ou termina no meio da noite. São as mais valiosas: as duas classes aparecem com o mesmo céu, a mesma lua e o mesmo equipamento, e o modelo não consegue acertar só reconhecendo a noite. Hoje não há nenhuma.
2. **Noites calmas sem lua na época de EPB (outubro a março)**, de preferência em CA. Elas testam exatamente o caso em que o modelo falha hoje.
3. **Noites com EPB com lua forte (≥ 50%).** Hoje não existe nenhuma.
4. **As duas classes em cada estação.** CP tem só EPB e BJL tem só noites calmas. Basta 1 a 2 noites da classe que falta em cada uma.

Também vi em `util/` pastas que ainda não estão no CSV: `STR_20231112` (outra estação) e a pasta `original` de cada noite, que tem mais quadros. Vale conferir se alguma dessas noites tem o perfil acima antes de buscar dados novos.

## Sobre o script que pre-processa as fotos originais do EMBRACE:

Recomendação: não trocar o magick agora, porque ele melhora o resultado atual. Eu só faria duas mudanças: corrigir o bug e gerar uma versão sem texto para o modelo. Depois, com mais noites, vale testar um pipeline Python direto sobre o TIF, com ajuste de brilho igual para todas as imagens. Esse teste é o que pode confirmar ou derrubar a hipótese do brilho.


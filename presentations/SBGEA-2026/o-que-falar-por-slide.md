# 🎤 Guia Mestre de Apresentação e Defesa — SBGEA 2026

**Título do Trabalho:** *Detecção Automática e Explicável de Bolhas de Plasma Equatoriais em Imagens All-Sky OI 630,0 nm via 2DPCA e Classificação Linear com Atribuição Exata por Pixel*  
**Autores:** Ivo (UFPB) & Prof. Dr. Paulo Roberto (UFCG)  
**Evento:** IX SBGEA — Simpósio Brasileiro de Geofísica Espacial e Aeronomia (Rio de Janeiro, 2026)  
**Tempo Alvo da Apresentação Oral:** **~10 minutos** (deixando 2 a 5 minutos para perguntas da banca/platéia)

---

## ⏱️ Mapa de Tempo e Estrutura (10 Minutos — 14 Slides)

Para manter o ritmo dinâmico, honesto e não estourar o tempo, siga este orçamento por slide:

| Slide | Título no Slide | Tempo Alvo | Tempo Acumulado | Ação no Teclado / Mouse |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Capa / Título | `0:25` | `0:25` | — |
| **2** | O que são Bolhas de Plasma Equatoriais? | `0:45` | `1:10` | Vídeo roda sozinho (pode pausar se quiser apontar depleção) |
| **3** | O gargalo da identificação manual | `0:45` | `1:55` | Contadores animam sozinhos (`572`, `255/255`, `7 noites · 3 estações`) |
| **4** | Pipeline em cascata guiado pela física | `1:10` | `3:05` | ⚠️ **Aperte `→` 5 vezes** durante a fala para revelar as 5 etapas (`.frag`) |
| **5** | Extração de características via 2DPCA | `0:45` | `3:50` | — |
| **6** | Desempenho em Leave-One-Night-Out (`t = 0,80`) | `1:15` | `5:05` | 🖱️ **Clique nas Abas 2 e 3** (ou comente os números da ablação/confundidor) |
| **7** | O modelo olha para a física? Atribuição linear exata | `0:55` | `6:00` | — |
| **8** | Uma noite inteira, quadro a quadro (3 vereditos) | `0:50` | `6:50` | 🖱️ **Clique nas Abas 2 e 3** para mostrar noite limpa e avaliação *out-of-night* |
| **9** | E quando o céu está nublado? Decisão em 3 vias (QC) | `0:45` | `7:35` | — |
| **10** | Amostra efetiva: 572 quadros, mas apenas 7 noites | `0:55` | `8:30` | — |
| **11** | Onde estamos — síntese honesta | `0:40` | `9:10` | — |
| **12** | Próximos passos | `0:35` | `9:45` | ⚠️ **Aperte `→` 5 vezes** rápido para revelar os 5 passos (`.frag`) |
| **13** | Referências selecionadas | `0:05` | `9:50` | Passagem rápida de reconhecimento |
| **14** | Obrigado! / Perguntas? | `0:10` | `10:00` | Encerramento |

> **💡 Regra de Ouro para os Slides 4 e 12:** Esses dois slides possuem elementos `.frag` (fragmentos animados). O teclado **não avança para o próximo slide** enquanto os 5 blocos daquele slide não tiverem aparecido. Lembre-se de ir apertando a seta para a direita (`→`) conforme fala cada etapa!

---

## 📋 Como usar este documento

Para cada slide, você encontrará **3 camadas**:
1. **🎙️ O que falar (Roteiro de Fala):** Texto corrido, natural e direto, pronto para ser falado no palco dentro do tempo cronometrado.
2. **🧠 O que você precisa saber a fundo (Bastidor Técnico e Físico):** Explicação detalhada da física ionosférica, da matemática e dos detalhes de implementação do código caso você queira improvisar ou aprofundar.
3. **❓ Perguntas Prováveis deste Slide (Defesa Rápida):** O que um avaliador pode questionar olhando especificamente para aquela tela e como responder com segurança.

Ao final do documento, há também um **Banco Geral de Perguntas Difíceis da Banca** cobrindo validação por noite, confundidores, física, pré-processamento e trabalhos futuros.

---

# SLIDE 1 — Capa / Título
*(Tempo sugerido: 25 segundos)*

### 🎙️ O que falar
> "Bom dia / boa tarde a todos. Meu nome é Ivo, sou mestrando na UFPB sob orientação do professor Paulo Roberto, da UFCG. Hoje vou apresentar nosso trabalho sobre detecção automática e explicável de Bolhas de Plasma Equatoriais em imagens *all-sky* na linha do oxigênio atômico em 630 nanômetros, combinando um controle de qualidade do céu em cascata, pré-processamento guiado pela física que preserva o contraste negativo das depleções, redução matricial via 2DPCA e um classificador linear com atribuição exata por pixel — avaliado de forma rigorosa sob validação cruzada deixando uma noite inteira de fora por vez."

### 🧠 O que você precisa saber a fundo
- **Por que cada termo do título importa:**
  - *Controle de qualidade em cascata (Estágio 1):* Antes de tentar detectar bolha, o sistema decide se o céu é confiável (`Limpo`, `Degradado/conferir` ou `Nublado/Indeterminado`), evitando classificar nuvens como eventos geofísicos.
  - *Pré-processamento guiado pela física ($\delta I/I$):* A bolha é uma depleção de emissão óptica (queda de brilho). O pipeline preserva desvios negativos sem cortar em zero.
  - *2DPCA (Two-Dimensional Principal Component Analysis):* Opera diretamente sobre a matriz 2D da imagem ($64 \times 64$) sem achatar para um vetor 1D gigante, preservando correlações espaciais entre linhas e colunas com pouquíssimos parâmetros.
  - *Classificador linear com atribuição exata por pixel:* Como a projeção 2DPCA é linear nos pixels e a Regressão Logística é linear nas características, o *logit* final é uma função 100% linear da imagem pré-processada — permitindo calcular a contribuição exata de cada pixel sem aproximações estocásticas.

---

# SLIDE 2 — O que são Bolhas de Plasma Equatoriais?
*(Tempo sugerido: 45 segundos)*

### 🎙️ O que falar
> "Para contextualizar rapidamente: após o pôr do sol, o forte gradiente vertical de densidade na base da camada F equatorial, combinado à deriva eletrodinâmica vertical pré-reversão ($\mathbf{E} \times \mathbf{B}$), torna a ionosfera susceptível à instabilidade de Rayleigh–Taylor. Bolhas de plasma menos denso sobem para o topo da camada F e deixam assinaturas ópticas claras na aeroluminescência noturna do oxigênio atômico em 630 nanômetros, por volta de 250 quilômetros de altitude.
>
> Aqui à direita vemos uma sequência real da estação de São João do Cariri, da rede EMBRACE do INPE. As bolhas aparecem como essas faixas escuras — depleções de emissão — alinhadas ao meridiano magnético e derivando para leste. Monitorar isso é crítico porque essas irregularidades causam cintilação severa em sinais GNSS e comunicações transionosféricas na região da Anomalia de Ionização Equatorial."

### 🧠 O que você precisa saber a fundo
- **Mecanismo químico da emissão OI 630,0 nm:**
  1. Troca de carga na base da camada F (~220–300 km): $\text{O}^+ + \text{O}_2 \rightarrow \text{O}_2^+ + \text{O}$.
  2. Recombinação dissociativa: $\text{O}_2^+ + e^- \rightarrow \text{O}^*({}^1\text{D}) + \text{O}$.
  3. O átomo de oxigênio no estado metastável ${}^1\text{D}$ (tempo de vida radiativo ~110 s) decai emitindo um fóton em $630,0\text{ nm}$: $\text{O}({}^1\text{D}) \rightarrow \text{O}({}^3\text{P}) + h\nu_{630\text{ nm}}$.
- **Por que a bolha é escura:** Dentro da bolha, a densidade de elétrons ($n_e$) e de íons $\text{O}^+$ cai de 1 a 3 ordens de grandeza. Sem $e^-$ e $\text{O}^+$, a taxa de recombinação dissociativa despenca e a emissão em 630,0 nm desaparece naquela coluna integrada, gerando uma faixa escura (*depletion*) alinhada às linhas de campo geomagnético (norte–sul magnético).
- **Deriva zonal:** Durante a noite quieta, o dínamo da camada F dirige o plasma para leste com velocidades típicas entre $50\text{ e }150\text{ m/s}$.

### ❓ Perguntas Prováveis deste Slide
- **P: Por que usar imagens em 630,0 nm e não apenas receptores GNSS?**  
  **R:** O receptor GNSS mede a cintilação (S4) ou o ROTI apenas ao longo das linhas de visada pontuais dos satélites visíveis. O imageador *all-sky* fornece um mapa bidimensional contínuo de ~1.000 km de diâmetro sobre a estação, permitindo visualizar diretamente a morfologia espacial, bifurcações, inclinação magnética e velocidade zonal das estruturas.

---

# SLIDE 3 — O gargalo da identificação manual
*(Tempo sugerido: 45 segundos)*

### 🎙️ O que falar
> "O problema prático é que cada imageador *all-sky* gera centenas de imagens por noite em múltiplas estações. Fazer a triagem visual disso é lento, cansativo e subjetivo, especialmente em bordas de eventos ou sob luar e nebulosidade parcial.
>
> Por outro lado, quando temos poucas noites rotuladas, treinar redes neurais profundas diretamente nos pixels brutos é perigoso: uma CNN facilmente memoriza o brilho médio da noite, a fase da lua ou o fundo instrumental de cada câmera em vez de aprender a geometria da bolha.
>
> Nosso conjunto rotulado atual possui **572 quadros** em 512 por 512 pixels, sendo **255 com EPB e 255 sem EPB** em céu limpo, além de **62 quadros nublados** para o controle de qualidade. Porém — e esta é uma premissa central de todo o nosso trabalho — esses 572 quadros vêm de **7 noites de observação em 3 estações** (São João do Cariri, Cachoeira Paulista e Bom Jesus da Lapa). Como quadros da mesma noite são altamente autocorrelacionados, a verdadeira unidade estatística independente é a **noite de observação**, e não o quadro isolado."

### 🧠 O que você precisa saber a fundo
- **As 3 estações no dataset atual:**
  - `CA`: São João do Cariri – PB ($7,4^\circ\text{S}, 36,5^\circ\text{O}$) — 5 noites no dataset (3 com EPB, 1 sem EPB, 1 nublada).
  - `CP`: Cachoeira Paulista – SP ($22,7^\circ\text{S}, 45,0^\circ\text{O}$) — 1 noite no dataset (com EPB).
  - `BJL`: Bom Jesus da Lapa – BA ($13,3^\circ\text{S}, 43,4^\circ\text{O}$) — 1 noite no dataset (sem EPB).
- **Por que a unidade independente é a noite (`night_key`):** Imagens consecutivas são capturadas a cada ~2 a 5 minutos. Uma bolha atravessa o campo de visão ao longo de 1 a 3 horas, e o fundo de emissão, a fase da lua e a transparência atmosférica mudam muito pouco de um quadro para o seguinte. Avaliar um classificador misturando quadros da mesma noite no treino e no teste vazaria o contexto noturno (inflando artificialmente qualquer métrica para perto de 100%).

---

# SLIDE 4 — Pipeline em cascata guiado pela física
*(Tempo sugerido: 1 minuto e 10 segundos)*  
⚠️ **ATENÇÃO:** Neste slide você precisa apertar a seta `→` **5 vezes** para fazer aparecer os 5 blocos numerados do pipeline na parte inferior!

### 🎙️ O que falar
> "Para lidar com esse cenário de forma fisicamente coerente e auditável, desenhamos uma arquitetura enxuta baseada em duas decisões físicas principais, mostradas nos cartões superiores:
>
> A primeira é a **preservação das depleções com sinal ($\delta I/I$)**: removemos estrelas pontuais por abertura morfológica na resolução original de 512 pixels e estimamos o fundo suave de emissão **exclusivamente dentro do domo circular**, evitando que a borda preta da máscara contamine a periferia. Em seguida, calculamos o contraste fracionário assinado $\delta I/I = I/\text{fundo} - 1$, **sem cortar valores em zero**. Isso é fundamental porque bolhas são depleções escuras — portanto valores negativos — e truncar em zero apagaria exatamente o sinal físico que queremos detectar.
>
> A segunda decisão é usar **decomposição 2DPCA acoplada a um classificador linear** (Regressão Logística balanceada), onde o número de componentes $k$ e a regularização $C$ são escolhidos por validação aninhada por noite.
>
> *(Aperte `→` — Passo 1)* O fluxo completo começa na imagem *all-sky*, que após o pré-processamento é reamostrada para 64 por 64.  
> *(Aperte `→` — Passo 2)* No **Estágio 1**, um porteiro de controle de qualidade avalia nitidez estelar, brilho e contraste para classificar o quadro como **Limpo**, **Degradado** (sinalizado com *conferir*) ou **Nublado**, que recebe o veredito **Indeterminado**.  
> *(Aperte `→` — Passo 3)* Quadros válidos passam pela extração do contraste físico $\delta I/I$ normalizado em z-score dentro do domo.  
> *(Aperte `→` — Passo 4)* O **2DPCA** projeta a matriz 64 por 64 nos seus $k$ autovetores principais, gerando $64 \times k$ características compactas.  
> *(Aperte `→` — Passo 5)* E, no **Estágio 2**, a Regressão Logística emite a probabilidade calibrada no limiar operacional de **0,80**, junto com o mapa de atribuição exata por pixel."

### 🧠 O que você precisa saber a fundo
- **Parâmetros exatos do pré-processamento:**
  - **Remoção de estrelas (`remove_stars`):** Operação morfológica de abertura (`cv2.MORPH_OPEN`) com elemento estruturante elíptico de $7 \times 7$ pixels na imagem $512 \times 512$ (`star_size=7`). Como estrelas são picos locais menores que 7 pixels e bolhas têm dezenas de pixels de largura, a abertura remove as estrelas preservando a camada de airglow.
  - **Máscara do domo (`fov_radius_frac=0.44`):** Raio útil de $44\%$ do lado da imagem em torno do centro $(H/2, W/2)$, descartando o anel externo onde há distorção severa da lente olho-de-peixe, árvores, antenas e textos gravados na borda.
  - **Estimativa do fundo (`estimate_background`):** Borramento Gaussiano com $\sigma = 10$ na escala de $64\times 64$ (equivalente a $\sigma = 80$ em $512\times 512$). **Detalhe crucial:** a convolução Gaussiana é calculada dividindo `GaussianBlur(img * mask)` por `GaussianBlur(mask)`. Assim, os zeros fora do domo não puxam o fundo da borda para baixo.
  - **Contrasto $\delta I/I$ assinado + z-score:** $\text{dI} = (I_\text{sem estrelas} - \text{fundo}) / \max(\text{fundo}, \varepsilon)$. Depois, subtrai-se a média e divide-se pelo desvio-padrão calculados apenas sobre os pixels dentro da máscara do domo, limitando valores extremos em $[-4, +4]$ desvios-padrão (`zclip=4.0`). Fora do domo, os pixels valem `0.0`.
- **Matemática do 2DPCA (Yang et al., 2004):**
  - Dadas $N$ imagens de treino $X_i \in \mathbb{R}^{64 \times 64}$ e a imagem média $\bar{X}$, a matriz de dispersão (covariância das linhas) é:
    $$G_t = \frac{1}{N}\sum_{i=1}^{N} (X_i - \bar{X})^\top (X_i - \bar{X}) \in \mathbb{R}^{64 \times 64}$$
  - Extraímos os $k$ autovetores ortonormais $V = [v_1, \dots, v_k] \in \mathbb{R}^{64 \times k}$ correspondentes aos $k$ maiores autovalores de $G_t$.
  - A projeção de uma imagem $X$ é $Y = (X - \bar{X})V \in \mathbb{R}^{64 \times k}$, que achatada gera um vetor de $64k$ características. No modelo final selecionado pelo loop aninhado ($k=5, C=0,001$), são $64 \times 5 = 320$ características.

### ❓ Perguntas Prováveis deste Slide
- **P: Por que usar Regressão Logística em vez de SVM com kernel RBF, Random Forest ou XGBoost?**  
  **R:** Dois motivos fundamentais quando se tem apenas 7 noites: (1) regularização forte contra sobreajuste ao brilho da noite — modelos não-lineares complexos memorizam facilmente limiares de intensidade específicos de cada noite; e (2) interpretabilidade exata — como a combinação 2DPCA + Regressão Logística é inteiramente linear, sabemos exatamente qual peso espacial cada pixel do domo recebe na decisão.

---

# SLIDE 5 — Extração de características via 2DPCA: onde o modelo enxerga a anomalia
*(Tempo sugerido: 45 segundos)*

### 🎙️ O que falar
> "Este slide ilustra visualmente o que o pré-processamento físico e o 2DPCA fazem com a imagem. Na linha de cima temos uma noite de céu quieto em Bom Jesus da Lapa, e na linha de baixo uma noite com bolha de plasma ativa em São João do Cariri.
>
> Da primeira para a segunda coluna, vemos o efeito da remoção de estrelas e do cálculo de $\delta I/I$ dentro do domo útil: na noite quieta o campo fica suave, enquanto na noite com EPB as bandas escuras de depleção norte–sul saltam com alto contraste negativo.
>
> Na terceira coluna temos o erro absoluto de reconstrução do 2DPCA, $|X - \hat{X}|$. Enquanto o céu quieto é reconstruído quase sem resíduo, as bordas abruptas das faixas de depleção da bolha concentram energia de reconstrução exatamente ao longo dos canais de plasma depletado."

### 🧠 O que você precisa saber a fundo
- **Como a reconstrução $\hat{X}$ é calculada:**
  - Como $V \in \mathbb{R}^{64 \times k}$ tem colunas ortonormais ($V^\top V = I_k$), a imagem reconstruída a partir das $k$ componentes é $\hat{X} = \bar{X} + (X - \bar{X}) V V^\top$.
  - O mapa de erro $|X - \hat{X}|$ mostra a informação de alta frequência azimutal/zonal que foge do fundo médio suave. Note que o classificador recebe as projeções $Y = (X - \bar{X})V$, e o mapa de erro serve como verificação visual de que o subespaço 2DPCA distingue a morfologia das bandas.

---

# SLIDE 6 — Desempenho em Leave-One-Night-Out (limiar operacional = 0,80)
*(Tempo sugerido: 1 minuto e 15 segundos)*  
🖱️ **AÇÃO INTERATIVA:** Comece na **Aba 1** (padrão), comente rapidamente a **Aba 2** e clique na **Aba 3** para mostrar a tabela de ablação e o achado central sobre o confundidor de brilho!

### 🎙️ O que falar
> "Passando aos resultados quantitativos: toda a avaliação foi feita sob **Leave-One-Night-Out aninhado** nas 6 noites de céu limpo — ou seja, em cada rodada, uma noite inteira fica 100% fora do treino, do ajuste do 2DPCA e da seleção de hiperparâmetros.
>
> No limiar operacional calibrado de **0,80**, obtemos **77,3% de acurácia balanceada** em noites inéditas, com **65,5% de sensibilidade** (167 de 255 quadros com bolha detectados, F1 de 0,742), **89,0% de especificidade** (227 de 255 quadros de céu limpo corretos, apenas 28 falsos positivos) e **AUC-ROC de 0,791**.
>
> *(Opcional: clique na Aba 2)* Na segunda aba vemos por que calibramos o limiar em 0,80: como cada noite tem um nível basal de airglow diferente, o limiar padrão de 0,50 gera muitos falsos positivos em noites inéditas escuras (especificidade de apenas 8%); elevar o limiar para 0,80 sobe a especificidade para 89% e maximiza a acurácia balanceada.
>
> *(Clique na Aba 3: Ablação & Confundidor)* Mas o resultado científico mais importante do nosso diagnóstico está nesta terceira aba, na tabela de ablação do pré-processamento e no card ao lado:
> Primeiro, na tabela (toda medida no limiar padrão 0,5), descobrimos que o **pré-processamento físico com remoção de estrelas e fundo no domo atinge 78,6% de acurácia balanceada e 0,950 de AUC-ROC** — a melhor configuração da escada. O passo seguinte, o achatamento radial, que é a **configuração atual do sistema**, reduz a AUC para 0,791. Isso sugere que **desligar o achatamento radial seria uma melhoria** — uma hipótese promissora que ainda precisamos confirmar com mais noites antes de trocar o padrão.
> E segundo: quando testamos um *baseline* cego que usa apenas **5 estatísticas globais de brilho da imagem bruta** (média e percentis, sem nenhuma informação espacial), ele atinge **95,9% de acurácia balanceada e 0,991 de AUC** no mesmo protocolo *leave-one-night-out*! Isso acontece porque, com apenas 6 noites de céu limpo, o nível de brilho da noite separa quase perfeitamente as noites do nosso dataset atual — um achado diagnóstico fundamental que guia nossos próximos passos."

### 🧠 O que você precisa saber a fundo
- **Números exatos da Escada de Ablação (`preprocessing_ladder.csv`, todos no limiar padrão `t = 0,5`):**
  1. **V4 equivalente** (subtract + clip + CLAHE, sem $\delta I < 0$): Bal. Acc = **66,3%**, Spec = 34,5%, AUC = **0,823** — base de comparação com o pipeline antigo.
  2. **+ $\delta I/I$ com sinal, sem corte em zero** (contraste assinado em 64×64): Bal. Acc = **68,4%**, Spec = 39,6%, AUC = **0,906** — preservar o sinal negativo salta a AUC.
  3. **+ Remoção de estrelas + fundo no domo** (abertura morfológica + fundo gaussiano só no domo): Bal. Acc = **78,6%**, Sens = **96,5%**, Spec = **60,8%**, AUC = **0,950** — o **melhor desempenho de toda a escada**.
  4. **+ Achatamento radial** (= configuração padrão atual do V5): Bal. Acc = **51,2%**, Sens = 94,1%, Spec = **8,2%**, AUC = **0,791**. Ou seja, no mesmo limiar das outras linhas, o achatamento radial **derruba** a acurácia balanceada (de 78,6% para 51,2%) e a especificidade (de 60,8% para 8,2%). *(O desempenho forte do sistema no Slide — especif. 89,0% — vem de calibrar o limiar em 0,80, não do achatamento radial.)*
  5. **Pré-processamento em 512×512 completo** (antes de reduzir para 64×64): Bal. Acc = **46,1%**, Spec = 0,0%, AUC = **0,716** — pior configuração; corretamente descartada.
- **Como funciona o Baseline de Brilho de 5 características (`run_brightness_confounder_baseline`):**
  - Extrai apenas `[mean, std, p10, p50, p90]` dos pixels brutos dentro do domo (`fov_radius_frac=0.44`) e treina uma Regressão Logística (`C=1.0, class_weight="balanced"`).
  - Por que ele dá 95,9% de acurácia balanceada e 0,991 de AUC? Porque as 4 noites com EPB têm distribuição de intensidade global diferente das 2 noites sem EPB (uma delas tem 80% de lua em CA e a outra é em BJL). O classificador de brilho simplesmente "adivinha a noite" pelo brilho global.

### ❓ Perguntas Prováveis deste Slide
- **P: Se o baseline de 5 números de brilho dá 95,9% e o seu pipeline espacial dá 77,3%, por que não usar o modelo de brilho?**  
  **R:** Porque o modelo de brilho está acertando pelo motivo errado (aprendizado por atalho / *shortcut learning*). Ele não enxerga nenhuma bolha; ele apenas memorizou o brilho médio das 6 noites disponíveis. Se colocarmos uma noite limpa escura ou uma noite de bolha com lua, o modelo de brilho vai errar 100% da noite. O nosso pipeline espacial normaliza cada quadro em z-score ($\mu=0, \sigma=1$ dentro do domo) justamente para impedir esse atalho e forçar o classificador a olhar para a morfologia espacial.

---

# SLIDE 7 — O modelo olha para a física? Atribuição linear exata
*(Tempo sugerido: 55 segundos)*

### 🎙️ O que falar
> "Para verificar onde o classificador espacial apoia sua decisão, calculamos o mapa de **atribuição linear exata por pixel**, mostrado à esquerda. Como a projeção 2DPCA é linear nos pixels e a Regressão Logística é linear nas características, o *logit* do modelo é o produto escalar direto entre um mapa de pesos $W_\text{pixel}$ e o desvio de cada pixel em relação ao céu médio. Não há aproximação estocástica: a soma dos valores do mapa mais o intercepto reproduz exatamente o *logit* da predição.
>
> Na imagem superior, em São João do Cariri (probabilidade de 87,6%), os pixels em vermelho que empurram a decisão para EPB alinham-se ao longo das paredes e canais norte–sul da bolha. Na imagem inferior, em Bom Jesus da Lapa (probabilidade de 12,5%), o interior do domo permanece neutro e azulado, empurrando para céu sem bolha.
>
> Além disso, à direita, auditamos um segundo potencial confundidor: a **orientação magnética das faixas por estação** via tensor de estrutura. Tanto São João do Cariri ($+5,4^\circ \pm 20,9^\circ$, coerência 0,625) quanto Cachoeira Paulista ($+0,5^\circ \pm 6,1^\circ$, coerência 0,565) ficaram dentro da tolerância de $\pm 25^\circ$ do eixo vertical, descartando a declinação magnética entre sítios como confundidor no conjunto atual."

### 🧠 O que você precisa saber a fundo
- **Equação da Atribuição Linear Exata (`linear_pixel_attribution`):**
  - Seja $X \in \mathbb{R}^{64 \times 64}$ a imagem pré-processada, $\bar{X}$ a imagem média do 2DPCA, $V \in \mathbb{R}^{64 \times k}$ os autovetores do 2DPCA, $\mu_s, \sigma_s \in \mathbb{R}^{64k}$ a média e escala do `StandardScaler`, e $w \in \mathbb{R}^{64k}, b \in \mathbb{R}$ os pesos e intercepto da Regressão Logística.
  - Definimos os pesos efetivos no espaço projetado como $\tilde{w} = w / \sigma_s$, reorganizados como matriz $\tilde{W} \in \mathbb{R}^{64 \times k}$.
  - O mapa de pesos no espaço de pixels é:
    $$W_\text{pixel} = \tilde{W} V^\top \in \mathbb{R}^{64 \times 64}$$
  - A atribuição exata de cada pixel $(r, c)$ para uma imagem $X$, usando uma amostra de fundo de referência $B = \{B_1, \dots, B_M\}$ (média $\bar{B}$), é:
    $$\phi(r, c) = W_\text{pixel}(r, c) \cdot \big(X(r, c) - \bar{B}(r, c)\big)$$
  - Isso é matematicamente idêntico ao *Linear interventional SHAP* para cadeias lineares, com custo computacional instantâneo e erro numérico zero.
- **Verificação de Orientação via Tensor de Estrutura (`band_orientation_check.csv`):**
  - Calcula gradientes de Sobel $g_x, g_y$ na imagem $\delta I/I$ dentro de um domo interno (`fov_radius_frac=0.38`) e monta o tensor de estrutura médio $J_{xx}, J_{yy}, J_{xy}$ para extrair o ângulo dominante das faixas em relação ao eixo vertical da imagem.
  - Como $|\text{mediana}| \le 25^\circ$ e coerência $\ge 0,15$ tanto em CA quanto em CP, o dicionário `ROTATION_BY_SITE` permaneceu vazio (`{}`), provando que o classificador não está separando estações pela inclinação das faixas.

### ❓ Perguntas Prováveis deste Slide
- **P: Se o mapa de atribuição mostra claramente as faixas norte–sul da bolha, por que vocês colocaram uma ressalva no terceiro card dizendo que isso não prova independência total de confundidores?**  
  **R:** Por rigor metodológico. Um mapa de atribuição linear mostra *quais pixels multiplicados pelos seus desvios mais contribuíram para o logit naquele quadro*, mas se os pesos $W_\text{pixel}$ tiverem sido influenciados pelo contraste global das poucas noites de treino, o modelo ainda pode sofrer deslocamento de limiar em uma noite inédita. Por isso tratamos a atribuição por pixel como uma evidência física necessária, mas complementar à validação *out-of-night*.

---

# SLIDE 8 — Uma noite inteira, quadro a quadro (3 vereditos)
*(Tempo sugerido: 50 segundos)*  
🖱️ **AÇÃO INTERATIVA:** Mostre a **Aba 1** (noite com EPB), clique na **Aba 2** (noite sem EPB) e clique na **Aba 3** (tabela de avaliação cega *Out-of-Night*)!

### 🎙️ O que falar
> "Na operação prática ao longo de uma noite inteira, o sistema não força uma decisão binária cega: ele emite **três vereditos possíveis** — **EPB** (vermelho), **Sem EPB** (azul) ou **Indeterminado** (cinza, quando o Estágio 1 bloqueia por nuvem ou falha de sensor) — além de uma faixa amarela de atenção (*conferir*) quando o céu está parcialmente degradado.
>
> Na **Aba 1**, vemos o relatório sobre a noite de 22–23 de outubro de 2025 em São João do Cariri (140 quadros): o sistema acompanha a evolução temporal da bolha com 98,6% de cobertura útil e apenas 2 quadros bloqueados como indeterminados no final do crepúsculo.  
> *(Clique na Aba 2)* Na **Aba 2**, para a noite limpa de 22–23 de julho de 2025 em Bom Jesus da Lapa (230 quadros), a curva permanece estavelmente abaixo do limiar de 0,80 durante toda a madrugada (222 quadros Sem EPB e 8 Indeterminados).
>
> *(Clique na Aba 3: Avaliação Cega Out-of-Night)* Porém, na **Aba 3**, fazemos questão de mostrar a diferença entre esse relatório visual com o modelo final e a **avaliação cega *out-of-night***: quando a noite de São João do Cariri com bolha é avaliada por um modelo que nunca viu aquela noite, o sistema mantém **91,4% de detecção em céu limpo**; mas quando a noite limpa de Bom Jesus da Lapa fica fora do treino, ela sofre **88,3% de falsos positivos**. Por quê? Porque tirando Bom Jesus da Lapa do treino, resta **apenas uma única noite sem bolha** no conjunto de treinamento — e é uma noite em Cariri com 80% de lua! Ou seja, o modelo nunca viu uma noite limpa e sem lua durante o treino."

### 🧠 O que você precisa saber a fundo
- **Números exatos de `sequences_in_sample.csv` (Aba 1 e Aba 2 — Modelo Final):**
  - `CA 22-23/10/2025 (com EPB, 140 quadros)`: 2 Indeterminados (1,4%), 2 Degradados/*conferir* (1,4%), 138 EPB (98,6% do total; **100% dos quadros de céu limpo**).
  - `BJL 22-23/07/2025 (sem EPB, 230 quadros)`: 8 Indeterminados (3,5%), 6 Degradados/*conferir* (2,6%), 0 EPB (**0,0% de falso alarme**).
  - `CA 24-25/06/2026 (nublado, 62 quadros)`: 34 Indeterminados (54,8%), 15 Degradados/*conferir* (24,2%), 18 EPB na cascata total (29,0%), mas apenas **11,3% de EPB em quadros considerados de céu limpo**.
- **Números exatos de `sequences_out_of_night.csv` (Aba 3 — Avaliação Cega Out-of-Night):**
  - `CA 22-23/10/2025 (com EPB)`: 1,4% Indeterminado, 92,9% EPB se usasse só o Estágio 2, **91,4% EPB na cascata (e 91,4% em céu limpo)** — excelente generalização.
  - `CA 24-25/06/2026 (nublado)`: 54,8% Indeterminado, 64,5% EPB se usasse só o Estágio 2 $\rightarrow$ cai para **29,0% na cascata e 11,3% em céu limpo** graças ao Estágio 1.
  - `BJL 22-23/07/2025 (sem EPB)`: 3,5% Indeterminado, 94,3% EPB no Estágio 2 $\rightarrow$ **88,3% na cascata (86,5% em céu limpo)**. É a prova concreta de que 2 noites sem EPB são insuficientes para cobrir a variação de fundo entre estações e fases lunares.

---

# SLIDE 9 — E quando o céu está nublado? Decisão em 3 vias (Estágio 1 - QC)
*(Tempo sugerido: 45 segundos)*

### 🎙️ O que falar
> "E o que acontece quando entram nuvens densas ou iluminação lunar forte sobre a cúpula? Como nuvens criam manchas escuras e brilhantes que enganariam qualquer classificador de Estágio 2, nosso **Estágio 1 de controle de qualidade** atua antes da classificação física.
>
> Ele mede três indicadores simples dentro do domo: a energia Laplaciana de objetos pontuais (contagem de estrelas nítidas), o brilho médio e o contraste global. Com isso, divide cada imagem em três zonas: **Céu Limpo** (`CLEAR`), **Degradado** (`DEGRADED`, que passa ao Estágio 2 mas recebe a faixa amarela *conferir* no gráfico) e **Nublado** (`CLOUDY`, que é bloqueado como **Indeterminado** em cinza).
>
> Na noite nublada de junho de 2026 à esquerda, sem o porteiro de qualidade, o classificador marcaria 64,5% da noite como bolha; com o porteiro, **54,8% dos quadros são barrados como Indeterminados**, outros 24,2% são sinalizados na zona amarela *conferir*, e os falsos alarmes em céu dado como limpo caem para **11,3%**.
>
> Sendo transparente sobre o custo atual: como esses limiares foram calibrados na única noite nublada rotulada que temos, o porteiro hoje acaba bloqueando como indeterminados **19,0% dos quadros legítimos de bolha em Cariri na noite de 27/10** e **12,8% na noite de 18/12** — porque uma bolha muito larga e escura também reduz temporariamente a contagem de estrelas e o brilho médio."

### 🧠 O que você precisa saber a fundo
- **Regras exatas do Estágio 1 (`ImageQualityGatekeeper`):**
  - Calcula dentro de um raio central (`0.38` do lado):
    1. `star_score`: número de pixels onde $|\nabla^2 I| > 30$ (Laplaciano 3×3 detectando bordas agudas de estrelas).
    2. `mean_brightness`: média de intensidade dos pixels no domo.
    3. `contrast_std`: desvio-padrão de intensidade no domo.
  - **Bloqueio como `CLOUDY` (veredito `Indeterminado`):**
    - `mean_brightness < 4.0` ou `contrast_std < 2.0` (tampa fechada/falha de sensor), **OU**
    - `star_score < 1000` (`threshold_stars=1000`) **OU** `mean_brightness < 55.0` (`min_mean_brightness=55.0`).
  - **Alerta como `DEGRADED` (flag `conferir = True`):**
    - Passou do corte de `CLOUDY`, mas `star_score < 4500` (`degraded_stars=4500`) **OU** `mean_brightness < 66.0` (`1.2 * min_mean_brightness`).
  - **Aprovação como `CLEAR`:** `star_score >= 4500` e `mean_brightness >= 66.0`.
- **Desempenho do QC (`qc_stage1_metrics.csv`):**
  - Recall em nuvens = **54,8%** (34/62 quadros da noite nublada barrados como `CLOUDY`; se somar os 15 `DEGRADED`, **79,0%** da noite nublada é barrada ou marcada com *conferir*).
  - Precisão em nuvens = **64,2%** (dos 53 quadros barrados no dataset inteiro de 572 imagens, 34 são da noite nublada e 19 são falsos bloqueios nas duas noites de EPB: 8 em `CA_20251027` [19,0%] e 11 em `CA_20251218` [12,8%]).

---

# SLIDE 10 — Amostra efetiva: 572 quadros, mas apenas 7 noites
*(Tempo sugerido: 55 segundos)*

### 🎙️ O que falar
> "Isso nos leva diretamente à discussão mais honesta desta apresentação: por que temos esses confundidores de brilho e esses falsos positivos quando Bom Jesus da Lapa sai do treino? A resposta está nesta tabela: embora tenhamos **572 quadros rotulados**, nossa amostra efetiva no nível de noite é de **apenas 7 noites** — sendo 4 noites com EPB, apenas **2 noites de céu limpo sem EPB**, e 1 noite nublada.
>
> Olhando a tabela à esquerda, vemos três armadilhas estatísticas clássicas de amostras pequenas:
> 1. **Estação vs. Classe:** Bom Jesus da Lapa só tem noite sem EPB, e Cachoeira Paulista só tem noite com EPB;
> 2. **Sazonalidade vs. Classe:** todas as noites com bolha são de primavera/verão (outubro a fevereiro, estação de bolhas no Brasil), enquanto as noites sem bolha são de outono/inverno (abril a julho);
> 3. **Fase da Lua:** das duas únicas noites sem bolha, uma tem 80% de lua e a outra tem 4%.
>
> Por isso, como detalhamos à direita, nossa meta prioritária é expandir o catálogo para **20 a 40 noites independentes** (mínimo de 20 para validação cruzada estável e um conjunto de teste travado, e idealmente 30 a 40 noites). Mais importante do que apenas somar quadros é buscar **quatro perfis prioritários de noites** que quebrem os confundidores:
> 1. **Noites mistas**, onde a bolha nasce ou morre no meio da mesma noite (mesma câmera e mesmo fundo com os dois rótulos);
> 2. **Noites sem EPB e sem lua na estação de bolhas** (outubro a março em Cariri);
> 3. **Noites com EPB sob lua moderada ou forte** ($\ge 50\%$); e
> 4. **Ambas as classes representadas em cada estação** (CP, BJL, CA e Santa Maria)."

### 🧠 O que você precisa saber a fundo
- **Por que 20 a 40 noites? (Baseado em `code/util/noites.md`):**
  - Com $N = 6$ noites de céu limpo (4 EPB + 2 Sem EPB), cada noite sem EPB representa **50% da especificidade** de todo o estudo! Além disso, quando fazemos *leave-one-night-out* aninhado e tiramos uma das 2 noites sem EPB no loop externo, o loop interno de seleção de hiperparâmetros fica com apenas 1 noite negativa, produzindo *splits* internos de classe única onde a AUC não é definida (acionando `FALLBACK_PARAMS = {k: 5, C: 0.01}`).
  - Com **20 noites** (~10 com EPB e ~10 sem EPB): cada noite passa a valer 10% da sua classe, permitindo reservar 4 a 6 noites em um `LOCKED_TEST_NIGHTS` intocado e rodar validação cruzada aninhada sem *splits* degenerados.
  - Com **30 a 40 noites** (15–20 por classe): conseguimos preencher uma grade fatorial cruzando `Estação × Fase Lunar (nova vs. >50%) × Estação do Ano`, reduzindo o erro-padrão da acurácia por noite para uma faixa publicável em periódico internacional (~7 a 9 pontos percentuais).
- **Detalhe técnico do pré-processamento de arquivo (`code/util/noites.md`):**
  - No fluxo atual de catalogação manual, as imagens originais `.tif` (16 bits) foram convertidas para `.png` usando `magick *.tif -normalize -contrast-stretch 2%x1% ...`. Como o `-normalize` reescala o histograma de cada quadro entre 0 e 255 independentemente, um quadro limpo escuro tem o fundo esticado para o meio da escala, enquanto um quadro com estrelas saturadas ou lua tem o fundo comprimido. Ler diretamente os `.tif` originais em Python (com normalização robusta por percentis dentro do domo) eliminará essa distorção de ganho antes mesmo do 2DPCA.

### ❓ Perguntas Prováveis deste Slide
- **P: O banco de dados do EMBRACE tem milhares de noites gravadas. Por que vocês usaram apenas 7 noites até agora?**  
  **R:** Os dados brutos no portal do EMBRACE existem em grande volume, mas **não possuem rótulos quadro a quadro** dizendo onde começa e onde termina cada bolha ou quando há entrada de nuvens. A curadoria visual quadro a quadro por especialista — verificando keogramas, condições lunares e bordas de eventos — foi feita inicialmente neste conjunto piloto de 572 quadros para construir e auditar toda a infraestrutura de pré-processamento, explicabilidade e detecção de confundidores. Agora que o diagnóstico revelou exatamente *quais* tipos de noites faltam para quebrar os confundidores, estamos fazendo a rotulagem direcionada dessas 20 a 40 noites.

---

# SLIDE 11 — Onde estamos — síntese honesta
*(Tempo sugerido: 40 segundos)*

### 🎙️ O que falar
> "Resumindo de forma bastante franca onde estamos hoje:
>
> Do lado das **contribuições metodológicas e diagnósticas** à esquerda:
> - Mostramos que o pré-processamento guiado pela física ($\delta I/I$ sem corte em zero, remoção morfológica de estrelas e fundo restrito ao domo) eleva a separabilidade das depleções (chegando a **0,950 de AUC-ROC** na ablação sem achatamento radial) e permite **atribuição linear exata por pixel**;
> - Estabelecemos um protocolo rigoroso de validação **Leave-One-Night-Out aninhada**, alcançando **77,3% de acurácia balanceada e 89,0% de especificidade** no limiar operacional de 0,80; e
> - Auditamos explicitamente os confundidores: descartamos a orientação magnética entre estações e expusemos o confundidor de brilho noturno.
>
> E do lado das **limitações declaradas** à direita:
> - Nosso classificador espacial ainda fica abaixo do *baseline* de 5 estatísticas de brilho (95,9%) porque a amostra atual no nível de noite é pequena ($N=7$, com apenas 2 noites sem bolha);
> - O conjunto de teste travado (`LOCKED_TEST_NIGHTS`) ainda está vazio aguardando novas noites; e
> - Os limiares do porteiro de nuvens foram calibrados *in-sample* em uma única noite nublada, e a ablação sugere que o achatamento radial atual pode estar penalizando o desempenho — algo a confirmar com mais noites."

### 🧠 O que você precisa saber a fundo
- Este slide é a sua maior proteção perante a banca. Ao colocar lado a lado o que funciona e o que ainda é limitação estatística do tamanho amostral, você demonstra maturidade científica e antecipa qualquer crítica sobre generalização.

---

# SLIDE 12 — Próximos passos
*(Tempo sugerido: 35 segundos)*  
⚠️ **ATENÇÃO:** Aperte a seta `→` **5 vezes** ao longo da fala para revelar os Passos 1 a 5!

### 🎙️ O que falar
> "A partir desse diagnóstico, nosso roteiro para a dissertação está dividido em 5 passos diretos:
> *(Aperte `→` — Passo 1)* **Passo 1:** Expandir o dataset para **30 a 40 noites** seguindo os 4 perfis prioritários (noites mistas, sem lua no verão, com lua e multi-estação) e congelar o conjunto de teste travado.  
> *(Aperte `→` — Passo 2)* **Passo 2:** Revisar o pré-processamento padrão, **testando desativar o achatamento radial** (a ablação indica que isso levaria a AUC de 0,791 para 0,950, a confirmar com mais noites) e mascarando os textos de carimbo nas bordas da imagem.  
> *(Aperte `→` — Passo 3)* **Passo 3:** Eliminar o confundidor de brilho na origem, lendo os arquivos `.tif` originais em Python sem o *contrast-stretch* quadro a quadro do ImageMagick, e testar regularização adversarial/ortogonal ao brilho.  
> *(Aperte `→` — Passo 4)* **Passo 4:** Revalidar o Estágio 1 de controle de qualidade em múltiplas noites nubladas e parcialmente nubladas para eliminar os falsos bloqueios em noites de bolha intensa.  
> *(Aperte `→` — Passo 5)* E **Passo 5:** Integrar a saída probabilística do imageador óptico com índices de cintilação GNSS (S4 e ROTI) e digissondas da própria rede EMBRACE."

### 🧠 O que você precisa saber a fundo
- **Por que o Passo 2 fala em mascarar textos nas bordas?** Algumas estações gravam carimbos de data/hora/estação perto da borda do domo. Embora `fov_radius_frac=0.44` já corte a maior parte, uma máscara explícita sobre retângulos de texto garante zero interferência no cálculo do fundo e dos gradientes.
- **Como funciona a leitura direta dos `.tif` no Passo 3 (`code/util/noites.md`):** Em vez de converter `.tif` para `.png` com `magick -normalize -contrast-stretch 2%x1%` (que altera o ganho de cada quadro conforme a presença de estrelas brilhantes ou nuvens), leremos a matriz 16-bit original e faremos a normalização usando apenas os percentis $P_2$ e $P_{99}$ calculados *estritamente dentro da máscara do domo após a remoção morfológica de estrelas*.

---

# SLIDE 13 — Referências selecionadas
*(Tempo sugerido: 5 segundos — apenas transição visual)*

### 🎙️ O que falar
> "Aqui estão as principais referências que fundamentam a física das bolhas, a redução via 2DPCA, a validação em blocos agrupados e a interpretabilidade linear, além do nosso agradecimento ao programa EMBRACE do INPE pelos dados abertos e à UFPB, UFCG, CAPES e CNPq."

---

# SLIDE 14 — Obrigado! / Perguntas?
*(Tempo sugerido: 10 segundos)*

### 🎙️ O que falar
> "Muito obrigado pela atenção de todos! O código, os relatórios de ablação e os testes de confundidores estão integralmente documentados e fico à disposição para perguntas e sugestões."

---

# 🛡️ Banco Completo de Perguntas e Respostas para a Defesa (Q&A)

Abaixo estão perguntas adicionais que pesquisadores de aeronomia (físicos do INPE, observadores ópticos) ou de computação/aprendizado de máquina podem fazer na sessão de perguntas:

### 1. Sobre Validação e Estatística
- **P: O que acontece na validação aninhada (*nested Leave-One-Night-Out*) quando o loop interno fica sem uma das classes?**  
  **R:** Excelente pergunta. Como temos apenas 2 noites sem EPB no total (`BJL_20250722` e `CA_20250407`), quando uma delas é a noite de teste externo e a outra cai como noite de validação interna, os 4 *folds* restantes de treino interno passam a ter apenas noites com EPB (classe única). Nosso código detecta automaticamente quando um *split* interno não tem as duas classes, ignora esse *fold* indefinido e, se nenhum par $(k, C)$ puder ser pontuado naquele loop interno, aciona de forma determinística parâmetros de segurança pré-definidos (`FALLBACK_PARAMS`: $k=5, C=0,01$). Com a expansão para 20–40 noites, esse problema deixa de existir.

- **P: Por que vocês reportam o limiar operacional de 0,80 em vez do limiar padrão de 0,50 da Regressão Logística?**  
  **R:** Porque no protocolo *leave-one-night-out*, cada noite tem um nível basal de emissão 630,0 nm e de espalhamento lunar ligeiramente diferente das noites de treino. Isso desloca a distribuição das probabilidades preditas para cima nas noites escuras. No limiar não-calibrado de 0,50, o modelo tem sensibilidade altíssima (94,1%), mas a especificidade despenca para apenas 8,2% (acurácia balanceada de 51,2%, perto do acaso). Ao calibrar o limiar operacional na curva de sensibilidade/especificidade em `t = 0,80`, contemos os falsos alarmes noturnos, subindo a especificidade para **89,0%** e a acurácia balanceada para **77,3%**.

### 2. Sobre o Pré-processamento Físico e o Achatamento Radial
- **P: Como funciona exatamente o achatamento radial e por que ele piorou a AUC de 0,950 para 0,791?**  
  **R:** Na configuração atual, achatamos o perfil radial (dividindo pela mediana de brilho em anéis concêntricos) para remover residual de *van Rhijn* e vinhetagem da lente. Porém, a tabela de ablação revelou que, no mesmo limiar (0,5) das demais linhas, o achatamento radial derruba a acurácia balanceada de 78,6% para 51,2% e a especificidade de 60,8% para 8,2% (AUC de 0,950 para 0,791). A hipótese física é que, quando uma bolha larga passa pelo zênite (centro do domo, $r \approx 0$), ela escurece os raios pequenos; o achatamento radial pode interpretar esse escurecimento central como perfil instrumental da lente e subtraí-lo, atenuando a própria bolha. Como a divisão pelo fundo Gaussiano ($\sigma = 10/64$) já corrige a vinhetagem suave, a evidência atual sugere que **desligar o achatamento radial melhoraria o sistema** — mas, dado que temos apenas ~7 noites, tratamos isso como uma hipótese a confirmar com mais dados antes de trocar o padrão, e não como decisão já tomada.

- **P: O filtro Gaussiano usado para estimar o fundo ($\sigma = 10$ em 64×64) não acaba entrando dentro da bolha se a bolha for muito larga?**  
  **R:** Sim, se uma depleção ocupar metade do domo, um filtro Gaussiano isotrópico pode sofrer uma leve queda no centro da bolha. Ainda assim, como $\sigma = 10$ em $64\times 64$ equivale a ~16% do tamanho da imagem (cerca de 150–200 km na altura da camada F), ele é mais largo que a maioria das ramificações individuais das bolhas, preservando alto contraste negativo nas bordas e no canal principal, como provado pela AUC de 0,950 atingida no passo de remoção de estrelas + fundo no domo. Uma melhoria futura simples para noites com múltiplas bolhas largas é estimar o fundo usando um filtro de percentil alto (ex.: percentil 75 local ou fechamento morfológico largo) ou uma mediana deslizante no tempo ao longo de ~1 hora.

### 3. Sobre Geofísica Espacial e Operação no EMBRACE
- **P: O sistema consegue diferenciar uma Bolha de Plasma Equatorial (EPB) de uma MSTID (Distúrbio Ionosférico Propagante de Média Escala) ou de ondas de gravidade?**  
  **R:** Excelente questão geofísica. No conjunto atual de 7 noites, os rótulos são binários (presença ou ausência de EPB em céu limpo). Porém, geometricamente, nosso pipeline já tem a ferramenta para diferenciá-las: na verificação de orientação por tensor de estrutura (Slide 7), mostramos que as EPBs em CA e CP são estritamente alinhadas ao meridiano magnético norte–sul (ângulos médios de $+5,4^\circ$ e $+0,5^\circ$). Já as MSTIDs eletrificadas noturnas no setor brasileiro tipicamente apresentam frentes de fase alinhadas de **noroeste para sudeste** ($\sim -30^\circ \text{ a } -60^\circ$) propagando-se para sudoeste. Portanto, acoplar o ângulo do tensor de estrutura como filtro físico ou característica explícita permitirá separar EPBs de MSTIDs quando incluirmos noites com ondas e MSTIDs no dataset.

- **P: Qual é o custo computacional desse pipeline? Ele pode rodar em tempo real nas estações remotas do EMBRACE?**  
  **R:** Sim, ele é extremamente leve e roda inteiramente em CPU comum em poucos milissegundos por imagem. Não exige GPU: o pré-processamento usa apenas operações morfológicas e convoluções Gaussianas rápidas do OpenCV, a projeção 2DPCA é uma única multiplicação de matrizes $(64 \times 64) \times (64 \times 5)$, e tanto a predição da Regressão Logística quanto o mapa de atribuição exata por pixel são produtos escalares diretos. Uma noite inteira de 230 imagens é processada e auditada em poucos segundos.

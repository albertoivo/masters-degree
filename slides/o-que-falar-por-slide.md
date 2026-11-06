# Guia Mestre de Apresentação e Defesa — SBGEA
## Detecção Automática de Bolhas de Plasma Equatoriais em Imagens All-Sky com Machine Learning

> **Propósito deste documento:** Este é o seu manual completo de estudo, roteiro de fala e preparação para perguntas da banca/plateia no **SBGEA (Sociedade Brasileira de Geofísica Espacial e Aeronomia)**. Ele foi escrito especificamente para acompanhar o arquivo `slides/index.html` (13 slides, planejados para ~10 minutos de apresentação + perguntas).
>
> **Como estudar com este arquivo:**
> 1. **Nível 1 (Roteiro de Fala — ~10 min):** Em cada slide há um bloco **"🎙️ O que falar (Roteiro palavra por palavra)"**. Se você falar exatamente aquilo em ritmo natural (~130 palavras por minuto), sua apresentação inteira durará cravados **9 a 10 minutos**.
> 2. **Nível 2 (Domínio Profundo — Físico e Computacional):** Em cada slide há a seção **"🧠 O que você precisa saber a fundo"**, explicando toda a física da ionosfera e toda a matemática/engenharia do seu código (`epb_detection_system.py`, `epb_detection_system_v3.ipynb`, `app_v3.ipynb`), traduzida para a linguagem de físicos.
> 3. **Nível 3 (Blindagem contra Perguntas):** Ao final de cada slide (e em uma seção especial no fim do documento), estão as **perguntas mais prováveis e difíceis** que geofísicos espaciais ou especialistas em IA podem fazer, com a resposta pronta na ponta da língua.

---

## ⏱️ Mapa de Tempo da Apresentação (Meta: 10 Minutos)

| Slide | Título no Slide | Tempo Alvo | Tempo Acumulado | Atalho / Ação no HTML |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Capa / Título | 0:30 | 0:30 | Tecla `F` para tela cheia |
| **2** | O que são Bolhas de Plasma Equatoriais? | 1:15 | 1:45 | Mostrar animação all-sky rodando |
| **3** | O gargalo da identificação manual | 0:45 | 2:30 | Apontar os contadores (572 imgs) |
| **4** | Pipeline de detecção | 1:30 | 4:00 | **Apertar `→` 5 vezes** (ativa os 5 blocos `.frag`) |
| **5** | Extração de características via 2DPCA: onde o modelo enxerga a anomalia | 1:00 | 5:00 | Apontar o mapa quente (erro de reconstrução) |
| **6** | Validação cruzada (5 folds) | 1:00 | 6:00 | Destacar Acurácia (96,9%), AUC (0,998) e Matriz |
| **7** | O modelo olha para a física? (SHAP) | 1:00 | 7:00 | Comparar vermelho (EPB) vs azul + barras (73,1%) |
| **8** | Uma noite inteira, quadro a quadro | 1:00 | 8:00 | **Clicar nas 3 abas** (Ativa CA / Dinâmica STR / Calma BJL) |
| **9** | E quando o céu está nublado? | 0:50 | 8:50 | Apontar as faixas cinzas (QC) e picos residuais |
| **10** | Onde estamos — e o que falta (Ressalvas) | 0:45 | 9:35 | Enfatizar a frase final ("triagem → operacional") |
| **11** | Rumo a um sistema operacional (Próximos passos) | 0:20 | 9:55 | **Apertar `→` 5 vezes** rápido para acender os 5 passos |
| **12** | Referências selecionadas | 0:05 | 10:00 | Passar rapidamente mencionando EMBRACE/CAPES/CNPq |
| **13** | Obrigado! / Perguntas? | — | — | Deixar na tela durante a sessão de perguntas |

---

# SLIDE 1 — Capa / Título

### 🖥️ O que está na tela
- **Kicker:** `SBGEA · Geofísica Espacial e Aeronomia`
- **Título:** *Detecção automática de Bolhas de Plasma Equatoriais em imagens all-sky com Machine Learning*
- **Subtítulo:** *Pré-processamento informado pela física + 2DPCA + ensemble de classificadores, com interpretabilidade via SHAP*
- **Autores:** Alberto I. C. Vieira · Cosme A. O. B. Figueiredo (UFPB · UFCG)
- **Logos:** INPE, EMBRACE, UFPB, UFCG.

### 🎬 Ação de palco e controles
- Antes de começar a falar, aperte a tecla **`F`** para colocar o navegador em tela cheia.
- O fundo tem um canvas HTML5 com 260 estrelas cintilando e derivando lentamente para a esquerda e uma aurora sutil nas cores do airglow.

### 🎙️ O que falar (Roteiro — ~30 segundos)
> *"Bom dia / Boa tarde a todos. Meu nome é Alberto Vieira, e este trabalho foi desenvolvido em parceria com o professor Cosme Figueiredo, entre a UFPB e a UFCG, utilizando dados ópticos da rede EMBRACE do INPE.*
> *Hoje eu vou apresentar para vocês como nós construímos um sistema de Aprendizado de Máquina leve, transparente e guiado pela física para detectar automaticamente Bolhas de Plasma Equatoriais em imagens all-sky — e discutir de forma muito honesta o que já funciona muito bem e o que ainda falta para chegarmos a um sistema operacional em tempo real."*

### 🧠 O que você precisa saber a fundo (Backstage deste slide)
- **Por que o subtítulo fala em "Pré-processamento informado pela física"?**
  Porque você não jogou uma imagem crua (raw) dentro de uma caixa-preta de Deep Learning. Você usou conhecimento óptico e aerônomico: sabe que o airglow tem variações suaves de fundo (efeito Van Rhijn, luar, poluição luminosa), que as bolhas são depleções locais de contraste na emissão de 630 nm, e que estrelas na troposfera limpa funcionam como fontes pontuais de alta frequência espacial.
- **Qual é a diferença entre IA, Machine Learning e Deep Learning (caso alguém da física pergunte)?**
  - **Inteligência Artificial (IA):** área ampla de sistemas que automatizam tarefas cognitivas/decisórias.
  - **Machine Learning (Aprendizado de Máquina):** algoritmos que aprendem padrões estatísticos a partir de dados rotulados. O seu trabalho usa **Machine Learning clássico estruturado** (2DPCA + Stacking de SVM, XGBoost e Random Forest).
  - **Deep Learning (Aprendizado Profundo):** redes neurais profundas com milhões de parâmetros (como CNNs — ResNet, YOLO, U-Net). Nós evitamos Deep Learning pesado de propósito porque o nosso conjunto rotulado tem centenas de imagens (não centenas de milhares) e queríamos baixa latência (rodar em CPU comum em milissegundos) e alta interpretabilidade matemática.

---

# SLIDE 2 — O que são Bolhas de Plasma Equatoriais?

### 🖥️ O que está na tela
- **Coluna da esquerda:**
  - 4 pontos explicando a formação física das EPBs (pós-pôr do sol, recombinação da camada F inferior, instabilidade de Rayleigh–Taylor, deriva para leste, cintilação em GNSS/satélites).
  - Card *"Como enxergamos?"*: Imageadores *all-sky* da rede EMBRACE/INPE registram o airglow **OI 630,0 nm** (~250 km). As bolhas aparecem como **faixas escuras** no céu.
- **Coluna da direita:**
  - Player circular interativo com **28 quadros reais** da estação de **Cachoeira Paulista (CA)** na noite de **30/01/2025**, das **00:03 UT às 01:58 UT**, em emissão OI 630 nm.

### 🎬 Ação de palco e controles
- A animação já começa rodando sozinha (troca de quadro a cada 280 ms).
- Se alguém quiser ver um instante específico ou se você quiser mostrar a imagem original em tons de cinza (como sai do sensor CCD), você pode clicar no botão **`cor`** abaixo da animação ou pausar no botão **`❚❚`** e arrastar a barra de tempo.

### 🎙️ O que falar (Roteiro — ~1 minuto e 15 segundos)
> *"Para nivelar todo mundo, especialmente quem é de outras áreas da física ou da graduação: o que são as Bolhas de Plasma Equatoriais, ou EPBs?*
> *Quando o Sol se põe sobre o equador magnético, a radiação extremo-ultravioleta cessa rapidamente. Na base da região F da ionosfera, onde a densidade neutra é maior, os íons recombinam muito rápido, enquanto no topo da camada F o plasma continua denso. Isso cria um gradiente vertical muito forte: um fluido de plasma rarefeito e leve embaixo sustentando um plasma denso e pesado em cima.*
> *Fisicamente, essa configuração é classicamente instável — é a **instabilidade generalizada de Rayleigh–Taylor**. Qualquer perturbação inicial na base (como uma onda de gravidade atmosférica) cresce de forma não-linear, fazendo com que grandes 'bolhas' de plasma rarefeito subam para o topo da ionosfera, alinhadas às linhas do campo geomagnético, e derivem para leste ao longo da noite.*
> *Por que isso importa na prática? Porque dentro dessas bolhas há forte turbulência de densidade eletrônica que refrata e difrata ondas de rádio trans-ionosféricas, causando **cintilação** severa de fase e amplitude em sinais de GPS/GNSS, agricultura de precisão, aviação e telecomunicações por satélite.*
> *E como nós enxergamos isso do chão? Aqui à direita temos uma sequência real da rede EMBRACE/INPE em Cachoeira Paulista. O imageador all-sky com lente olho-de-peixe observa 180 graus do céu na linha de emissão do oxigênio atômico em **630 nanômetros**, a cerca de 250 km de altitude. Onde há plasma normal, a recombinação dissociativa emite luz; onde passa a bolha — como falta plasma — falta emissão, e nós vemos essas grandes faixas escuras ramificadas cruzando o céu."*

### 🧠 O que você precisa saber a fundo (Física Ionosférica e Aeronomia)

1. **A Química da Emissão OI 630,0 nm (Por que a bolha é escura?):**
   - Na região F noturna (~220–300 km, com pico em ~250 km), o íon dominante é o $\text{O}^+$.
   - Ele não recombina diretamente com elétrons de forma eficiente; primeiro ocorre transferência de carga com o oxigênio molecular neutro ($\text{O}_2$):
     $$\text{O}^+ + \text{O}_2 \rightarrow \text{O}_2^+ + \text{O}$$
   - Em seguida, ocorre a **recombinação dissociativa** do $\text{O}_2^+$ com um elétron ($e^-$):
     $$\text{O}_2^+ + e^- \rightarrow \text{O}(^1\text{D}) + \text{O}(^3\text{P})$$
   - O átomo de oxigênio no estado metaestável excitado $\text{O}(^1\text{D})$ tem tempo de vida radiativo de ~110 segundos e decai para o estado fundamental $\text{O}(^3\text{P})$ emitindo um fóton no vermelho:
     $$\text{O}(^1\text{D}) \rightarrow \text{O}(^3\text{P}) + h\nu \quad (\lambda = 630,0\text{ nm})$$
   - **Conclusão crucial:** A intensidade integrada de 630,0 nm é diretamente proporcional à integral da concentração de elétrons ($n_e$) e de $\text{O}_2$ na base da camada F. Quando uma EPB passa, $n_e$ cai em até 2 ou 3 ordens de grandeza dentro da bolha $\Rightarrow$ a reação para $\Rightarrow$ a emissão de 630 nm apaga localmente $\Rightarrow$ vemos uma **banda escura (depleção óptica)**.

2. **A Física da Instabilidade de Rayleigh–Taylor (IRT) e o Pico Pré-Reversão (PRE):**
   - A taxa de crescimento linear ($\gamma$) da instabilidade de Rayleigh–Taylor na ionosfera equatorial depende de:
     $$\gamma \approx \frac{1}{L} \left( \frac{g}{\nu_{in}} + \frac{E_x}{B} \right) - \beta$$
     onde $L = \left(\frac{1}{n_e}\frac{\partial n_e}{\partial z}\right)^{-1}$ é a escala de comprimento do gradiente vertical de densidade, $g$ é a gravidade, $\nu_{in}$ é a frequência de colisão íon-neutro, $E_x/B$ é a deriva vertical eletrodinâmica ($\mathbf{E} \times \mathbf{B}$ com campo elétrico zonal $E_x$ para leste), e $\beta$ é a taxa de recombinação química.
   - Logo após o pôr do sol, o dínamo da camada F gera a **intensificação pré-reversão (Pre-Reversal Enhancement — PRE)** do campo elétrico zonal para leste ($E_x$). A deriva $\mathbf{E} \times \mathbf{B}$ levanta a camada F para altitudes maiores (~350–400 km).
   - Como $\nu_{in}$ cai exponencialmente com a altitude, levantar a camada F faz o termo $g/\nu_{in}$ crescer enormemente, disparando a instabilidade de Rayleigh–Taylor.

3. **Por que a estrutura se alinha Norte–Sul e estende para latitudes baixas (como Cachoeira Paulista)?**
   - A condutividade paralela às linhas do campo geomagnético ($\sigma_\parallel$) é altíssima. Assim, os campos elétricos de polarização dentro da bolha mapeiam ao longo de todo o tubo de fluxo magnético.
   - À medida que a bolha sobe no plano equatorial (sobre o equador magnético, ex: até 800–1500 km de altitude de ápice), a "assinatura" ou os "pés" desse tubo de fluxo magnético na altitude do airglow (250 km) se afastam para o norte e para o sul do equador magnético, chegando a latitudes tropicais como Cachoeira Paulista (~22,7° S, ~16–18° S de latitude magnética).

### ❓ Perguntas Prováveis neste Slide
- **P: "Na animação de Cachoeira Paulista, por que as bolhas parecem vir inclinadas e se estendendo de norte para sul?"**
  - **R:** *"Porque Cachoeira Paulista está fora do equador magnético (na região da Anomalia de Ionização Equatorial sul). O que vemos a 250 km de altitude é a interseção da camada emissora de 630 nm com o tubo de fluxo magnético depletado que subiu lá no plano equatorial. Além disso, o cisalhamento vertical do vento neutro zonal faz a parte superior da pluma inclinar para oeste (tilt para oeste) enquanto a estrutura inteira deriva para leste."*
- **P: "Por que usar 630 nm e não 557,7 nm (linha verde) ou 777,4 nm?"**
  - **R:** *"A linha de 630,0 nm em ~250 km tem a melhor relação sinal-ruído na rede EMBRACE para monitorar a base da camada F. A linha de 557,7 nm tem uma forte componente mesosférica em ~95 km que mascara a componente termal da região F, e a 777,4 nm (recombinação radiativa no pico da camada F, ~350 km) é muito mais fraca, exigindo tempos de exposição maiores ou céus perfeitamente escuros."*

---

# SLIDE 3 — O Gargalo da Identificação Manual

### 🖥️ O que está na tela
- **Esquerda:** 3 cards:
  1. 👁️ **Inspeção visual:** especialistas analisando quadro a quadro milhares de imagens por noite.
  2. ☁️ **Ruído ambiental:** Lua, nuvens cirrus, luzes urbanas e brilho do sensor confundem depleções tênues.
  3. 🧠 **Por que não uma rede profunda "crua"?** Custo computacional alto e tendência a aprender artefatos (estrelas, cidades, borda da lente) em vez da física.
- **Direita:**
  - **Pergunta central:** *É possível ensinar um algoritmo a reconhecer a morfologia física de uma bolha?*
  - **Contadores animados:** **572** imagens rotuladas (EMBRACE) | **255** com EPB / **317** sem EPB.

### 🎙️ O que falar (Roteiro — ~45 segundos)
> *"Qual é a motivação prática do nosso trabalho? O programa EMBRACE opera vários imageadores all-sky espalhados pelo Brasil, gerando uma imagem a cada 3 a 5 minutos durante toda a noite. Multiplicando isso por anos de observação e várias estações, o volume de dados torna a inspeção visual humana um gargalo severo — além de ser subjetiva quando a bolha é muito tênue ou há interferência de luar e nuvens finas.*
> *Quando olhamos para a literatura recente de Inteligência Artificial, a tentação imediata é jogar as imagens brutas dentro de uma rede neural convolucional profunda (CNN). O problema é duplo: primeiro, redes profundas precisam de dezenas de milhares de exemplos e GPUs pesadas; segundo, sem tratamento físico prévio, elas costumam trapacear — decoram a posição da Via Láctea, o brilho da cidade na borda do domo ou artefatos da câmera.*
> *Então a nossa pergunta científica foi: partindo de um conjunto enxuto de **572 imagens rotuladas** do EMBRACE (255 com bolha e 317 sem bolha), nós conseguimos guiar o algoritmo com física para que ele aprenda a **morfologia real** da depleção?"*

### 🧠 O que você precisa saber a fundo (Backstage deste slide)
- **De onde vieram as 572 imagens?**
  Do conjunto rotulado (`data/all-images` e `data/epb_labels.csv`) de observações da rede EMBRACE/INPE, abrangendo estações brasileiras como São João do Cariri (BJL) e Cachoeira Paulista (CA), com **255 imagens positivas (`has_epb = 1`, 44,6%)** e **317 imagens negativas (`has_epb = 0`, 55,4%)**.
- **O efeito "Clever Hans" em Visão Computacional Científica:**
  Em Machine Learning, quando você alimenta imagens all-sky brutas para uma CNN com poucas noites de observação, a rede rapidamente percebe que "toda imagem onde a Lua está no canto esquerdo não tem bolha" ou "toda imagem com a árvore/antena da estação X tem bolha". Isso é chamado de *shortcut learning* (aprendizado por atalho). É por isso que o pré-processamento físico + 2DPCA + verificação via SHAP são o coração da sua dissertação.

### ❓ Perguntas Prováveis neste Slide
- **P: "572 imagens não é um conjunto de dados pequeno para Machine Learning?"**
  - **R:** *"Para Deep Learning puro (redes neurais com milhões de pesos), 572 imagens seria pequeno e causaria overfitting severo. É exatamente por isso que nós desenhamos uma arquitetura baseada em extração analítica de modos espaciais (2DPCA), que reduz cada imagem a apenas 5 componentes principais, exigindo ordens de grandeza menos parâmetros livres. Mesmo assim, como vou mostrar no slide 10, expandir esse banco para milhares de imagens ao longo do ciclo solar é o nosso principal próximo passo."*
- **P: "Quem rotulou essas 572 imagens? Qual é o 'ground truth' (verdade de referência)?"**
  - **R:** *"A rotulagem foi feita por inspeção visual especialista nas imagens OI 630 nm do EMBRACE, identificando a presença clara das bandas escuras alinhadas magneticamente (classe 1) versus céu limpo sem depleções (classe 0)."*

---

# SLIDE 4 — Pipeline de Detecção (Metodologia)

### 🖥️ O que está na tela
- **2 Cards superiores (já visíveis ao abrir o slide):**
  - **🧭 Intuição para físicos:** O 2DPCA funciona como uma *decomposição em modos* (análoga a EOFs): poucas "imagens-base" descrevem o céu típico. O que *não* é bem descrito por elas — o resíduo — é justamente a estrutura anômala.
  - **⚖️ Desbalanceamento de classes:** Noites calmas são muito mais frequentes. Usamos *data augmentation* + *SMOTE* (amostras sintéticas no espaço de features) para o modelo não "chutar céu limpo" sempre.
- **Fluxograma inferior animado com 5 blocos (`.frag`):**
  1. 🌌 **Imagem all-sky:** OI 630 nm, rede EMBRACE/INPE (64×64).
  2. 🔬 **Pré-processamento físico:** Remove iluminação difusa (filtro gaussiano $\sigma=10$) + realce local de contraste (CLAHE).
  3. 🧩 **2DPCA:** Extrai os 5 modos espaciais dominantes *sem destruir a estrutura 2D*.
  4. 🗳️ **Comitê de classificadores:** SVM + XGBoost + Random Forest $\rightarrow$ regressão logística decide.
  5. 📈 **Probabilidade de EPB:** + mapa SHAP explicando a decisão.

### 🎬 Ação de palco e controles (MUITO IMPORTANTE!)
- ⚠️ **Atenção:** Quando você entra no Slide 4, os 2 cards conceituais no topo já aparecem, enquanto os 5 blocos do pipeline embaixo estão **apagados/cinzas** (`opacity: 0.12`).
- Cada vez que você aperta a **seta para a direita (`→`)**, a **barra de espaço** ou **clica no pipeline**, **um bloco se acende** da esquerda para a direita!
- Portanto, vá apertando `→` conforme for falando de cada etapa (5 toques no total). No 6º toque, ele vai para o Slide 5.

### 🎙️ O que falar (Roteiro — ~1 minuto e 30 segundos)
> *(Aperte `→` para acender o Bloco 1: Imagem all-sky)*
> *"Como funciona a nossa metodologia passo a passo? Tudo começa com a imagem all-sky em escala de cinza, redimensionada para uma grade padronizada de 64 por 64 pixels.*
> *(Aperte `→` para acender o Bloco 2: Pré-processamento físico)*
> *Na segunda etapa, aplicamos um pré-processamento guiado pela física. Primeiro, estimamos o fundo luminoso lento — causado pelo efeito Van Rhijn, luar ou brilho urbano — passando um filtro Gaussiano largo, com desvio padrão igual a 10, e subtraímos esse fundo da imagem original. Isso age como um filtro passa-altas espacial. Em seguida, aplicamos o CLAHE (equalização adaptativa de histograma com limite de contraste de 0,03), que realça localmente as bordas das faixas escuras das bolhas sem estourar o ruído.*
> *(Aperte `→` para acender o Bloco 3: 2DPCA)*
> *Na terceira etapa, em vez de 'achatar' a imagem 2D em um vetor 1D gigante — o que destruiria a correlação espacial entre pixels vizinhos — nós usamos o **2DPCA** (Análise de Componentes Principais Bidimensional). Para os físicos na plateia, a intuição é direta: o 2DPCA é uma decomposição em modos ortogonais, exatamente como as Funções Ortogonais Empíricas (EOFs) que usamos em geofísica, mas operando direto na matriz da imagem.*
> *(Aperte `→` para acender o Bloco 4: Comitê de classificadores)*
> *Na quarta etapa, para contornar o desbalanceamento natural entre noites calmas e noites com bolha, aplicamos aumento de dados geométrico e a técnica SMOTE no espaço de features. E para classificar, em vez de confiar em um único algoritmo, montamos um comitê — um **Stacking Ensemble** — que combina três famílias matemáticas diferentes: uma máquina de vetores de suporte (SVM com kernel RBF), árvores com gradient boosting (XGBoost) e uma floresta aleatória (Random Forest). Uma regressão logística atua como 'juiz' final combinando os votos.*
> *(Aperte `→` para acender o Bloco 5: Probabilidade + SHAP)*
> *Por fim, o sistema entrega não só a probabilidade contínua de haver uma bolha, mas também um mapa espacial SHAP que mostra onde o modelo olhou para tomar aquela decisão."*

### 🧠 O que você precisa saber a fundo (Matemática e Código desta Metodologia)

Se alguém da banca ou plateia quiser detalhes técnicos de qualquer bloco, você tem todos os números exatos do seu código (`epb_detection_system.py`):

#### 1. Redimensionamento (`target_size = (64, 64)`)
- As imagens originais têm $512 \times 512$ ou $1280 \times 1280$ pixels. Por que reduzir para $64 \times 64$?
- Porque as Bolhas de Plasma Equatoriais são estruturas de **grande escala espacial** (dezenas a centenas de quilômetros na ionosfera, ocupando frações grandes da lente all-sky). Em $64 \times 64$, uma faixa de depleção ainda tem vários pixels de largura, enquanto o ruído de pixel (shot noise do CCD e estrelas individuais finas) é suavizado e o custo computacional cai por um fator de $>60\times$.

#### 2. Equações do Pré-processamento Físico (`preprocess_epb_image`, linhas 350–385)
- Seja $I_0(x,y) \in [0, 1]$ a imagem original normalizada em ponto flutuante (`float32`).
- **Remoção de fundo difuso:** Calculamos a convolução bidimensional com um núcleo Gaussiano $G_\sigma(x,y)$ com $\sigma = 10$ pixels:
  $$B(x,y) = (I_0 * G_{\sigma=10})(x,y)$$
  Como $\sigma = 10$ em uma imagem $64 \times 64$ cobre cerca de $\sim 1/6$ do diâmetro do domo, $B(x,y)$ captura apenas gradientes lentos de iluminação (luar, gradiente do horizonte/Van Rhijn). Subtraímos e truncamos em zero:
  $$I_1(x,y) = \max\big(0,\, I_0(x,y) - B(x,y)\big)$$
- **CLAHE (`equalize_adapthist`, `clip_limit = 0.03`):**
  A equalização de histograma global comum amplificaria o ruído de fundo onde o céu é uniforme. O CLAHE divide a imagem em pequenos blocos locais (tiles) e equaliza o histograma dentro de cada bloco, mas **corta (clips)** qualquer pico do histograma em `0.03` antes de redistribuir. Isso aumenta o contraste local das paredes da bolha de plasma sem transformar ruído térmico em manchas falsas.
- **Normalização Min-Max:**
  $$I_{\text{final}}(x,y) = \frac{I_2(x,y) - \min(I_2)}{\max(I_2) - \min(I_2)} \in [0, 1]$$

#### 3. Matemática do 2DPCA vs. PCA 1D Clássico (`TwoDPCA`, linhas 460–515)
- **O problema do PCA 1D tradicional:** Uma imagem $64 \times 64$ vira um vetor coluna $\mathbf{x} \in \mathbb{R}^{4096}$. A matriz de covariância tem tamanho $4096 \times 4096$ (mais de 16 milhões de entradas!). Com apenas ~450 imagens de treino, estimar uma matriz $4096 \times 4096$ sofre da *maldição da dimensionalidade* (matriz singular, ruído numérico gigante) e perde a noção de que o pixel $(i,j)$ é vizinho vertical do pixel $(i+1,j)$.
- **Como o 2DPCA resolve isso (Yang et al., 2004):**
  Mantemos cada imagem $k$ como uma matriz $A_k \in \mathbb{R}^{H \times W}$ ($64 \times 64$).
  1. Calculamos a imagem média $\bar{A} = \frac{1}{N}\sum_{k=1}^N A_k$ e centralizamos: $\tilde{A}_k = A_k - \bar{A}$.
  2. Construímos a **matriz de covariância de dispersão de imagem** $G_t \in \mathbb{R}^{W \times W}$ ($64 \times 64$, apenas 4.096 entradas!):
     $$G_t = \frac{1}{N} \sum_{k=1}^{N} \tilde{A}_k^T \tilde{A}_k$$
  3. Resolvemos o problema de autovalores simétrico (`np.linalg.eigh`):
     $$G_t \mathbf{u}_j = \lambda_j \mathbf{u}_j$$
  4. Escolhemos os $d = 5$ autovetores associados aos 5 maiores autovalores, formando a matriz de projeção ortogonal $P = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_5] \in \mathbb{R}^{64 \times 5}$.
  5. A projeção de uma imagem $A$ é simplesmente:
     $$Y = (A - \bar{A}) P \in \mathbb{R}^{64 \times 5}$$
  - **Qual é o tamanho final do vetor de features?** $64 \times 5 = \mathbf{320\text{ features}}$! Reduzimos de $4096$ pixels para $320$ coeficientes preservando a relação entre linhas e colunas.

#### 4. Data Augmentation Estratificado + SMOTE (`augment_epb_data` e `_setup_balancing`)
- **Importante para defender contra pergunta de "Data Leakage" (vazamento de dados):** No seu código (`evaluate_kfold_cv`, linha 2405), o Data Augmentation e o SMOTE são aplicados **estritamente dentro do conjunto de treino de cada fold**, nunca nos dados de validação/teste!
- **Quais transformações o Data Augmentation faz (`_apply_random_augmentation`, linhas 438–454)?**
  - Espelhamento horizontal (`fliplr`);
  - Rotação suave aleatória entre $-10^\circ$ e $+10^\circ$ (preserva o alinhamento aproximadamente Norte-Sul das bolhas);
  - Translação espacial (`shift`) de $-5$ a $+5$ pixels (simula a deriva zonal da bolha pelo campo de visão);
  - Ruído Gaussiano aditivo ($\sigma = 0.02$).
  - Com `augment_factor = 3`, a classe majoritária (Sem EPB) é multiplicada por $\times 3$ (de ~253 para ~759 no fold) e a minoritária (Com EPB) recebe o dobro de atenção ($\times 5$, de ~204 para ~1020).
- **O que o SMOTE faz depois?** No espaço das 320 features do 2DPCA, já normalizadas pelo `StandardScaler`, o SMOTE (*Synthetic Minority Over-sampling Technique*) iguala exatamente as duas classes (1020 vs 1020) interpolando linearmente entre vizinhos próximos da classe minoritária.

#### 5. O Comitê Stacking Ensemble (`_create_ensemble`, linhas 858–914)
- **Nível 0 (Base Learners):**
  1. **SVM com Kernel RBF (`C=5.0`, `gamma='scale'`, `class_weight='balanced'`):** Excelente para encontrar hiperplanos curvos suaves em espaços de dimensão média (320 features).
  2. **XGBoost (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`):** Árvores de decisão impulsionadas por gradiente; captura interações não-lineares entre modos do 2DPCA.
  3. **Random Forest (`n_estimators=200`, `max_depth=15`, `min_samples_split=5`):** Bagging de árvores independentes; reduz a variância e evita overfitting.
- **Nível 1 (Meta-classificador):**
  - **Regressão Logística (`C=1.0`):** Treinada com validação cruzada interna (`cv=5`) sobre as probabilidades previstas pelos 3 modelos de base. Ela aprende o peso ótimo de cada especialista.

---

# SLIDE 5 — Resultados: Extração de Características via 2DPCA: Onde o Modelo Enxerga a Anomalia

### 🖥️ O que está na tela
- **Título:** *Extração de características via 2DPCA: onde o modelo enxerga a anomalia*
- **Texto no topo:** *Os 5 primeiros componentes concentram a informação estrutural; além disso a curva se achata (ruído). O erro de reconstrução se concentra exatamente onde está a depleção.*
- **Figura superior (`recon_error_without.png`):** Céu sem bolha (Imagem pré-processada | Features 2DPCA | Mapa de Importância/Erro de reconstrução baixo e espalhado sem estrutura coerente).
- **Figura inferior (`recon_error_with.png`):** Céu com bolha (`O6_CA_20251023_001620.png` — Probabilidade predita: 97,4%) mostrando que o mapa de importância (erro de reconstrução) brilha intensamente exatamente sobre as faixas verticais da bolha.

### 🎬 Ação de palco e controles
- Se quiser destacar os detalhes da figura para a plateia, basta **clicar sobre qualquer uma das duas figuras** para abrir o *lightbox* em tela cheia; clique novamente (ou aperte `Esc`) para fechar.

### 🎙️ O que falar (Roteiro — ~1 minuto)
> *"Vamos ver o que acontece dentro do 2DPCA quando aplicamos isso às imagens reais. Nós descobrimos que apenas os **5 primeiros componentes principais** — que retêm cerca de 56% da variância global da matriz — são suficientes para capturar toda a assinatura estrutural necessária para classificar as bolhas, reduzindo cada imagem a 320 números.*
> *E aqui surge uma propriedade física muito elegante quando projetamos a imagem nesses 5 modos e a reconstruímos de volta: olhem para a diferença entre a imagem original pré-processada e a reconstrução, que nós chamamos de mapa de erro de reconstrução.*
> *Na figura de cima, temos uma noite de céu limpo sem bolha: o erro de reconstrução é mínimo e espalhado aleatoriamente.*
> *Já na figura de baixo, quando uma bolha de plasma entra no campo de visão, a estrutura fina e escura da depleção quebra a simetria suave do fundo. O erro de reconstrução explode e desenha perfeitamente a geometria das ramificações da bolha. Ou seja, antes mesmo do classificador dar o veredito, o subespaço do 2DPCA já isola espacialmente a anomalia ionosférica."*

### 🧠 O que você precisa saber a fundo (Backstage deste slide)
- **Como o mapa de erro de reconstrução é calculado no código (`get_feature_importance`, linhas 1010–1026)?**
  1. Projetamos a imagem $X \in \mathbb{R}^{64 \times 64}$ nos 5 autovetores: $Y = (X - \bar{X}) P$.
  2. Reconstruímos a imagem de volta para $64 \times 64$: $\hat{X} = Y P^T + \bar{X}$.
  3. O mapa é o módulo da diferença pixel a pixel:
     $$E(x,y) = |X(x,y) - \hat{X}(x,y)|$$
- **Por que 5 componentes explicam ~55,8% da variância e não 95%?**
  - Porque após o pré-processamento (que já removeu o fundo gaussiano $\sigma=10$ — onde ficava 90% da energia contínua da imagem!), a imagem $X$ contém apenas variações de alta frequência e estruturas locais. Os 5 primeiros modos capturam toda a estrutura coerente de grande/média escala (~56% da variância pós-filtro). Os modos de ordem superior ($>5$) são dominados por ruído granular de alta frequência e cintilação de estrelas individuais. No teste de otimização (`optimize_n_components`), usar 5 componentes maximizou o F1-score sem sobreajustar ao ruído.

---

# SLIDE 6 — Resultados: Desempenho na Validação Cruzada (5 Folds)

### 🖥️ O que está na tela
- **4 Cartões de Métricas (com animação de contagem):**
  - **96,9%** Acurácia (`0.9685 ± 0.0119`)
  - **95,7%** Sensibilidade — bolhas detectadas (`0.9569 ± 0.0402`)
  - **97,8%** Especificidade — céu limpo correto (`0.9779 ± 0.0180`)
  - **0,998** AUC-ROC (`0.9976 ± 0.0016`)
- **Esquerda (`roc_pr_curves.png`):** Curva ROC (AUC = 1,000 no conjunto de teste), Curva Precision-Recall (AUC = 1,000) e Histograma de distribuição de probabilidades preditas (azul concentrado em 0, vermelho concentrado em 1).
- **Direita (`image_1.png`):** Matriz de Confusão no conjunto de teste (81 Verdadeiros Negativos, 0 Falsos Positivos, 0 Falsos Negativos, 56 Verdadeiros Positivos).

### 🎙️ O que falar (Roteiro — ~1 minuto)
> *"Como nós avaliamos se o modelo realmente generaliza e não apenas decorou os exemplos? Nós executamos uma **validação cruzada estratificada de 5 folds** sobre todo o conjunto de 572 imagens. Em cada um dos 5 folds, o modelo é treinado do zero — incluindo o cálculo do 2DPCA e o balanceamento — e testado em uma partição totalmente invisível.*
> *Na média dos 5 folds, o sistema alcançou **96,9% de acurácia**, com **95,7% de sensibilidade** — que é a nossa capacidade de não deixar passar uma bolha real — e **97,8% de especificidade**, que garante que não vamos gerar alarmes falsos em noites de céu calmo. O F1-Score médio ficou em **96,4%** e a área sob a curva ROC em **0,998**.*
> *Nos gráficos abaixo, vemos o comportamento no conjunto de teste independente: tanto a curva ROC quanto a curva Precision-Recall — que é a métrica mais rigorosa para dados desbalanceados — ficam coladas no topo, e o histograma mostra uma separação bimodal limpa: imagens sem bolha recebem probabilidade próxima de zero, e imagens com bolha ficam próximas de 100%."*

### 🧠 O que você precisa saber a fundo (ATENÇÃO AOS NÚMEROS E PERGUNTAS DA BANCA!)

1. **Entenda perfeitamente a diferença entre os dois conjuntos de números do seu trabalho (para nunca se confundir):**
   - **No Conjunto de Teste Único (Hold-out / Split de Teste — gráficos na parte de baixo do slide):**
     - Matriz de confusão (`image_1.png`): **81 Sem EPB** acertados, **56 Com EPB** acertados, **0 erros** (100% no split de teste estático; ou 99,80% reportado na versão preliminar do texto do artigo WAIMLAP).
   - **Na Validação Cruzada de 5 Folds (`evaluate_kfold_cv` no notebook `epb_detection_system_v3.ipynb` — números grandes no topo do slide):**
     - **Acurácia:** $96,85\% \pm 1,19\%$ (mín: $95,61\%$, máx: $98,26\%$)
     - **Sensibilidade (Recall):** $95,69\% \pm 4,02\%$ (mín: $92,16\%$, máx: $100,0\%$)
     - **Especificidade:** $97,79\% \pm 1,80\%$ (mín: $95,24\%$, máx: $100,0\%$)
     - **Precisão (PPV):** $97,30\% \pm 2,08\%$ (mín: $94,44\%$, máx: $100,0\%$)
     - **F1-Score:** $96,42\% \pm 1,41\%$ (mín: $94,95\%$, máx: $98,08\%$)
     - **AUC-ROC:** $0,9976 \pm 0,0016$ (mín: $0,9956$, máx: $0,9994$)
   - **Por que é muito melhor falar os ~96,9% da validação cruzada de 5 folds na apresentação?**
     Porque qualquer avaliador experiente desconfia de "100%" em um único conjunto de teste de 137 imagens, mas respeita profundamente quando você apresenta **média e desvio padrão em 5-fold cross-validation** ($96,9\% \pm 1,2\%$) e mostra que em um dos splits o modelo chega a gabaritar a matriz de confusão.

2. **Fórmulas das Métricas na Ponta da Língua:**
   - **Sensibilidade (Recall / Taxa de Verdadeiros Positivos):** $\frac{VP}{VP + FN}$ $\rightarrow$ *"Das bolhas reais que aconteceram na ionosfera, quantas % nós detectamos?"*
   - **Especificidade (Taxa de Verdadeiros Negativos):** $\frac{VN}{VN + FP}$ $\rightarrow$ *"Das noites de céu calmo, quantas % o sistema deixou em paz sem tocar alarme falso?"*
   - **Precisão (Valor Preditivo Positivo):** $\frac{VP}{VP + FP}$ $\rightarrow$ *"Quando o sistema apita 'Tem Bolha!', qual a chance de ser verdade?"*
   - **F1-Score:** Média harmônica entre Precisão e Sensibilidade: $2 \cdot \frac{\text{Precisão} \times \text{Sensibilidade}}{\text{Precisão} + \text{Sensibilidade}}$.

3. **E se perguntarem: "Por que usar o Ensemble se o SVM sozinho teve resultado parecido no benchmark?"**
   - No seu notebook (célula 10), você rodou um benchmark honesto onde `2DPCA + SVM RBF` teve F1 = `0.9767` e `2DPCA + Ensemble` teve F1 = `0.9688` (diferença de 1 única imagem em 143 do teste).
   - **Resposta de mestre:** *"Em um conjunto de teste de 143 imagens, 0,7% de diferença equivale a literalmente 1 imagem de fronteira. Nós mantivemos o Stacking Ensemble por duas razões arquiteturais importantes: primeiro, **redução de variância e calibração de probabilidades** ao combinar modelos de naturezas distintas (margem máxima do SVM + partições ortogonais de árvores no XGBoost/RF) ao longo de sequências noturnas inteiras; e segundo, porque ter modelos baseados em árvores dentro do comitê nos permite usar o **TreeExplainer do SHAP** para gerar mapas de interpretabilidade em milissegundos, algo que um SVM RBF puro exigiría aproximação lenta por KernelExplainer."*

---

# SLIDE 7 — Resultados: Interpretabilidade (O modelo olha para a física?)

### 🖥️ O que está na tela
- **Esquerda superior (`shap_analysis_with.png`):** Imagem original com EPB ao lado do mapa de atribuição SHAP (`RdBu_r`), mostrando regiões **vermelhas intensas** exatamente sobre as estruturas verticais de depleção nos quadrantes superiores.
- **Esquerda inferior (`shap_analysis_without.png`):** Imagem sem EPB ao lado do mapa SHAP predominantemente **azul e difuso**, indicando evidência a favor de ionosfera quieta.
- **Direita (3 Cards estruturados):**
  1. **🎯 O que o mapa SHAP mede?** Calcula a contribuição marginal de cada feature (valores de Shapley) e a retro-projeta pelos autovetores do 2DPCA para os pixels da imagem (`🔵 Empurra p/ Céu Limpo` $\leftrightarrow$ `🔴 Empurra p/ Bolha (EPB)`).
  2. **📐 Onde o modelo concentra a decisão?** Barras quantitativas por região do domo na noite ativa: **Quadrante Superior Esquerdo (38,1%)**, **Quadrante Superior Direito (35,0%)** e **Quadrantes Inferiores (26,9%)**.
  3. **🛡️ Prova contra "efeito caixa-preta":** Mais de **73% da decisão** vem das faixas de depleção que sobem para os quadrantes superiores — ignorando bordas da lente, árvores e estrelas.

### 🎙️ O que falar (Roteiro — ~1 minuto)
> *"Para um físico, ter 97% de acurácia não basta se o modelo for uma caixa-preta que acerta pelos motivos errados. Nós precisávamos provar que o classificador está enxergando a morfologia da depleção de plasma, e não artefatos de borda da lente ou estrelas.*
> *Para isso, acoplamos ao sistema o framework **SHAP**, que vem da teoria dos jogos cooperativos de Shapley e calcula a contribuição marginal exata de cada feature para a decisão do modelo, retro-projetando esses pesos de volta para o espaço de pixels da imagem.*
> *O código de cores aqui é simples: **vermelho** significa evidência que empurra a probabilidade para 'Bolha de Plasma', e **azul** empurra para 'Ionosfera Quieta'.*
> *Na imagem de cima, com atividade intensa de EPB, o mapa SHAP acende em vermelho vivo exatamente em cima das estruturas verticais e ramificadas de depleção que sobem para os quadrantes superiores — que somam mais de 73% da contribuição total da decisão. Na imagem de baixo, em uma noite calma, o mapa inteiro fica azul e difuso. Isso confirma que o modelo aprendeu a assinatura física real do fenômeno."*

### 🧠 O que você precisa saber a fundo (SHAP + Retro-projeção 2DPCA)

1. **O que é SHAP (SHapley Additive exPlanations)?**
   - Criado por Lundberg & Lee (2017) com base nos Valores de Shapley (Nobel de Economia, Lloyd Shapley).
   - Imagine que as 320 features do 2DPCA são "jogadores" cooperando em um time para mudar a previsão do modelo desde o valor base médio $E[f(Y)]$ (probabilidade média de um céu calmo de referência) até a previsão final $f(Y)$ daquela imagem:
     $$f(Y) = E[f(Y)] + \sum_{i=1}^{320} \phi_i$$
   - O valor $\phi_i$ (valor SHAP da feature $i$) mede o impacto médio de incluir aquela feature em todas as coalizões possíveis de features. Se $\phi_i > 0$, aquela feature empurrou a decisão para **EPB (vermelho)**; se $\phi_i < 0$, empurrou para **Sem EPB (azul)**.

2. **Como você trouxe os 320 valores SHAP do espaço do 2DPCA de volta para os $64 \times 64$ pixels da imagem (`get_shap_importance`, linhas 1128–1158)?**
   - Essa é uma das partes mais inteligentes do seu código! O classificador não vê pixels; ele vê a matriz projetada e normalizada $Y_{\text{scaled}}$.
   - No seu código, você calcula o peso relativo de atribuição no espaço projetado:
     $$W = \frac{\phi}{\Delta Y_{\text{scaled}}}$$
   - Desfaz o escalonamento (`StandardScaler`), remodela para $(64 \times 5)$, **retro-projeta pela transposta da matriz de autovetores do 2DPCA** ($P^T$) e multiplica pelo desvio real de intensidade de cada pixel em relação ao fundo médio de céu calmo ($\Delta X = X - E[X_{\text{bg}}]$):
     $$\text{SHAP\_Map}(x,y) = (X - E[X_{\text{bg}}]) \odot \left( \frac{W_{2D}}{\sigma_{\text{scaler}}} P^T \right)$$
   - Dessa forma, a soma espacial preserva a direção do impacto de cada pixel!

3. **Qual a diferença entre o Mapa de Erro de Reconstrução (Slide 5) e o Mapa SHAP (Slide 7)?**
   - No seu código (linha 2799 do `app_v3.ipynb`), você até calcula a correlação de Spearman entre os dois (`corr = 0.346`).
   - **Erro de Reconstrução (Slide 5):** É puramente **não-supervisionado** (só depende do 2DPCA). Mostra *"o que é geometricamente diferente do céu médio"* (qualquer anomalia de contraste tem erro alto).
   - **Mapa SHAP (Slide 7):** É **supervisionado e direcional** (depende das árvores de decisão do classificador). Mostra *"quais dessas anomalias o classificador realmente usou como evidência de EPB (vermelho) e quais ele julgou serem evidência de céu calmo (azul)"*.

---

# SLIDE 8 — Resultados: Operação Contínua (Uma noite inteira, quadro a quadro)

### 🖥️ O que está na tela
- **Sistema interativo de 3 abas genuinamente distintas (`data-tabs="t8"`):**
  1. **Aba 1 (Ativa por padrão): `Noite ativa com EPB (CA)`** (`timeline_ca.png` — Cachoeira Paulista, 22–23/10/2025, das 21:14 às 07:08 UT, 140 imagens). Mostra a probabilidade subindo rapidamente às **21:18:24 UT (81,5%)**, mantendo-se acima de 90% durante toda a passagem das bolhas (pico de **97,4% às 00:33 UT**) com amostras de thumbnails embaixo.
  2. **Aba 2: `Noite dinâmica: formação e dissipação (STR)`** (`timeline_str.png` — 12–13/11/2023, das 22:54 às 08:22 UT, **135 imagens**, 5 transições). Mostra uma evolução super dinâmica: atividade inicial seguida de enfraquecimento/calmaria entre 00:36 e 02:13 UT (faixa verde, probabilidade ~25–35%), **re-intensificação abrupta** quando novas bolhas entram no domo às **02:13 UT** (probabilidade sobe para **96–97%** e se mantém até as 08 UT), e finalmente **dissipação ao amanhecer** às **08:17 UT** (caindo para 36%).
  3. **Aba 3: `Noite calma sem EPB (BJL)`** (`timeline_bjl.png` — São João do Cariri, 22–23/07/2025, das 21:56 às 08:23 UT, **230 imagens consecutivas**). Linha de probabilidade completamente plana perto de zero (**0/230 EPBs, probabilidade média de 2,3%**), barra de status 100% verde durante toda a noite até o amanhecer.

### 🎬 Ação de palco e controles (INTERATIVIDADE!)
- Quando entrar neste slide, a **Aba 1 (`Noite ativa com EPB (CA)`)** estará visível (~20 seg).
- Em seguida, **clique na Aba 2 (`Noite dinâmica: formação e dissipação (STR)`)** para mostrar que o modelo acompanha subidas, descidas e re-intensificações reais (~20 seg).
- Por fim, **clique na Aba 3 (`Noite calma sem EPB (BJL)`)** para mostrar a linha reta em 2,3% com zero falsos positivos (~20 seg).

### 🎙️ O que falar (Roteiro — ~1 minuto)
> *"Passando de imagens isoladas para o cenário real de um observatório: o que acontece quando deixamos o modelo rodando continuamente durante uma noite inteira, quadro a quadro?*
> *(Aponte para a Aba 1 — Noite ativa com EPB em Cachoeira Paulista)*
> *Na primeira aba, temos uma noite de forte atividade em Cachoeira Paulista (140 imagens): logo às 21:18 UT a probabilidade cruza o limiar de 50% e permanece estável acima de 90% durante toda a madrugada.*
> *(Clique na Aba 2 — Noite dinâmica em STR)*
> *Na segunda aba, vemos uma noite muito mais dinâmica, com 135 imagens: o modelo acompanha perfeitamente o enfraquecimento da atividade por volta de 1 hora UT (região verde), detecta a entrada de novas estruturas bem definidas às 02:13 UT — subindo de volta para 97% — e registra a dissipação final ao amanhecer.*
> *(Clique na Aba 3 — Noite calma sem EPB em BJL)*
> *E na terceira aba, o teste de fogo contra alarmes falsos: em São João do Cariri (BJL), ao longo de **230 imagens consecutivas** de céu limpo sem bolha, a probabilidade média ficou cravada em **2,3%**, com **zero falsos positivos** a noite inteira."*

### 🧠 O que você precisa saber a fundo (Backstage deste slide)
- **O que significam as 3 linhas horizontais tracejadas no gráfico de probabilidade?**
  - **Linha Verde (`50%`):** *Threshold* (limiar padrão de decisão binária: acima de 50% é classificado como EPB).
  - **Linha Amarela (`70%`):** Confiança Média (*Medium Confidence*).
  - **Linha Vermelha (`90%`):** Confiança Alta (*High Confidence*).
- **O que significa a linha vertical roxa no finalzinho da noite calma de BJL (por volta de 08:10 UT, ou 05:10 hora local)?**
  - É o **crepúsculo matutino (nascer do Sol)** ou entrada de nebulosidade/claridade ao amanhecer, onde o filtro de Controle de Qualidade do Céu (QC) entra em ação e encerra a janela válida de observação noturna (barra cinza no final da sequência).

---

# SLIDE 9 — Robustez: E quando o céu está nublado?

### 🖥️ O que está na tela
- **Esquerda (`assets/app/app_8_6.png`):** Sequência temporal de uma noite nublada em Cachoeira Paulista (`06/25/2026 • 03:24 – 07:42 UT • 62 images`). Mostra a linha de probabilidade oscilando entre 0% (quando o filtro QC barra a imagem como `NUVEM`, faixas **cinzas** na timeline) e picos de ~97% (faixas **vermelhas** quando o filtro deixa passar uma imagem parcialmente nublada).
- **Direita:**
  - **Card "☁️ Filtro de qualidade do céu":** Antes de classificar, um *"porteiro"* descarta imagens nubladas ou obstruídas, onde fisicamente não é possível ver bolhas.
  - **Alerta "Ainda em evolução":** Em noites de nuvens intermitentes a classificação oscila. É preciso um descarte de nuvens mais robusto e treinado com mais exemplos.

### 🎙️ O que falar (Roteiro — ~50 segundos)
> *"Mas na vida real de observação óptica, nem toda noite tem céu de brigadeiro. O que acontece quando o céu fica nublado?*
> *Se nós passarmos uma imagem totalmente nublada direto pelo CLAHE, o realce adaptativo de contraste tenta amplificar as bordas escuras entre as nuvens na troposfera, e o classificador pode confundir uma fresta escura entre nuvens com uma bolha de plasma a 250 km de altitude.*
> *Para enfrentar isso, na versão mais recente da nossa aplicação nós implementamos uma **arquitetura em cascata com um Porteiro de Qualidade do Céu (Quality Control)**. Antes da imagem ir para o modelo ionosférico, um detector de alta frequência espacial conta os picos pontuais de **estrelas** visíveis no domo. Se as nuvens encobrem as estrelas, o porteiro bloqueia a imagem e marca a timeline em **cinza**, como vemos aqui em 42 das 62 imagens desta noite nublada.*
> *Porém — e aqui entra a nossa transparência científica — vejam as faixas vermelhas entre as cinzas: quando passam nuvens quebradas ou parcialmente transparentes, algumas estrelas aparecem, a imagem passa pelo porteiro e as sombras das próprias nuvens ainda geram falsos positivos. Resolver isso de forma definitiva é um dos nossos principais focos atuais."*

### 🧠 O que você precisa saber a fundo (Como funciona o Porteiro QC no seu código)

Se alguém perguntar **"Como exatamente funciona esse filtro de nuvens atual?"**, você vai dar uma aula explicando as funções `detect_stars` e `assess_sky_quality` do seu sistema:

1. **Princípio Físico (Estrelas como "Velas-Padrão" de Transparência Troposférica):**
   - As nuvens estão na troposfera (~1–12 km), a emissão OI 630 nm está na termosfera/ionosfera (~250 km) e as estrelas estão no infinito óptico.
   - Se o céu está limpo, uma câmera all-sky sensível registra milhares de pixels de fontes pontuais de alta frequência espacial (estrelas).
   - Se há cobertura de nuvens opticamente espessas, a luz estelar sofre espalhamento Mie severo nas gotículas de água/cristais de gelo, apagando os picos pontuais.
2. **Algoritmo do Estágio 1 (`detect_stars` + `assess_sky_quality`):**
   - Passamos um filtro Gaussiano suave (`smooth = gaussian_filter(img, sigma)`) e subtraímos da imagem bruta (`high_freq = img_f - smooth`).
   - Contamos quantos pixels têm pico agudo acima de um limiar (`threshold_diff`) e brilho mínimo (`min_intensity`), obtendo `star_pixels`, além de medir o brilho médio (`raw_mean`) e desvio padrão (`raw_std`).
   - Se `star_pixels` cai abaixo do limiar (`threshold_stars`) ou o brilho médio é incompatível, a imagem recebe o rótulo `CLOUDY` (`is_valid_sky = False`), a probabilidade de EPB é zerada/ignorada e nem o 2DPCA nem o SHAP são executados.
3. **Por que ele falha nas 20 imagens "vermelhas" daquela noite nublada (`app_8_6.png`)?**
   - Porque quando há **nuvens cumulus quebradas ou cirrus esparsas** (ou iluminação forte da Lua nas bordas das nuvens), ainda existem estrelas visíveis nas brechas do céu (ou bordas brilhantes de nuvens iluminadas pela Lua que mimetizam gradientes agudos). O porteiro simples baseado em limiar de estrelas deixa a imagem passar (`CLEAR`/`DEGRADED`), e então as sombras escuras entre as nuvens são realçadas pelo CLAHE e interpretadas pelo 2DPCA como faixas de depleção.
   - **Como resolver no futuro?** Transformando o Estágio 1 (o porteiro) de uma regra de limiar fixo de estrelas para um **classificador dedicado de cobertura de nuvens** (usando textura GLCM, simetria radial, máscara de posição lunar e/ou um classificador multiclasse treinado especificamente com milhares de imagens nubladas de cada estação).

---

# SLIDE 10 — Mensagem para Levar para Casa (Onde estamos — e o que falta)

### 🖥️ O que está na tela
- **Coluna Esquerda (Bento Grid `✔ O que já demonstramos`):**
  1. 🔬 **Aprendizado guiado pela física (SHAP):** O algoritmo foca nas estruturas verticais de depleção em 630 nm, sem depender de artefatos ópticos.
  2. 🎯 **Alto poder discriminativo (~97% de acurácia):** Sensibilidade de 95,7%, especificidade de 97,8% e AUC-ROC de 0,998 na validação cruzada de 5 folds.
  3. ⚡ **Estabilidade noturna e leveza computacional:** Zero falsos positivos em noites calmas inteiras e inferência em milissegundos em CPU padrão (sem GPU).
- **Coluna Direita (Bento Grid `⚠ Ressalvas científicas honestas`):**
  1. 🔗 **Monitoramento em tempo real: "quase lá":** O modelo já tem velocidade de tempo real, mas falta integrar o servidor do EMBRACE a um pipeline automático (ingestão → inferência → alerta).
  2. 🗂️ **Ainda não substitui 100% a análise manual:** Para operação autônoma, é necessário expandir o dataset para mais estações, sazonalidades e fases do ciclo solar.
  3. ☁️ **Filtragem automática de nuvens:** Noites parcialmente nubladas ainda desafiam o porteiro atual; precisamos de um classificador dedicado para descarte de nuvens.
- **Banner de fechamento em toda a largura:**
  - *Hoje: **ferramenta de triagem rápida e apoio ao especialista** → Futuro: **sistema operacional de alerta de clima espacial***

### 🎙️ O que falar (Roteiro — ~45 segundos — O CORAÇÃO DAS SUAS DUAS MENSAGENS!)
> *"Chegando à mensagem principal que eu quero que vocês levem para casa hoje, dividida entre o que já provamos e as nossas ressalvas científicas honestas:*
> *De um lado, nós demonstramos que **sim**, a Inteligência Artificial guiada pela física consegue aprender a morfologia real das bolhas de plasma com quase **97% de acurácia**, de forma leve — rodando em segundos num computador comum sem GPU — e com excelente estabilidade em noites de céu limpo.*
> *Por outro lado, temos duas ressalvas fundamentais: primeiro, sobre **monitoramento em tempo real**: o algoritmo já tem velocidade de tempo real, mas para ser um sistema operacional, ainda precisamos construir a engenharia de integração que conecte automaticamente o servidor onde as câmeras do EMBRACE salvam as imagens à nossa pipeline de inferência e geração de alertas.*
> *Segundo: **o modelo já pode substituir 100% a classificação manual?** Hoje, ainda não de forma autônoma. Para chegar lá, precisamos ampliar o treinamento com muito mais imagens — cobrindo diferentes estações, sazonalidades e fases do ciclo solar — e aperfeiçoar o descarte automático de imagens nubladas. Portanto, hoje o sistema já é uma poderosa **ferramenta de triagem automática e apoio ao pesquisador**, pavimentando o caminho para um alerta operacional."*

### 🧠 O que você precisa saber a fundo (Como explicar a "Integração de Pipeline de ML com o EMBRACE" se perguntarem!)

Você mencionou que ainda não sabia exatamente como explicar tecnicamente o que falta para integrar o sistema do EMBRACE com uma pipeline de ML. Aqui está a explicação clara, simples e profissional para você usar:

- **Como o código funciona hoje (Modo Offline / Batch):**
  Hoje, os arquivos `.png` precisam ser baixados manualmente do portal do EMBRACE para uma pasta local no computador, e o pesquisador roda o notebook (`app_v3.ipynb`) para analisar a pasta.
- **O que é uma "Pipeline de ML em Tempo Real acoplada ao EMBRACE" (Modo Online / Streaming):**
  1. **Captura no Observatório:** O imageador all-sky (ex: em BJL ou CA) tira uma foto de ~15–30 segundos de exposição a cada ~3 minutos e envia o arquivo bruto para o servidor central do EMBRACE/INPE.
  2. **Gatilho de Ingestão (File Watcher / API / Mensageria):** Assim que o arquivo novo cai no diretório do servidor do EMBRACE, um serviço automatizado em segundo plano (um *daemon* ou microsserviço Python) detecta a chegada da imagem instantaneamente.
  3. **Inferência em Cascata (< 50 milissegundos):** Esse serviço carrega o nosso arquivo `epb_recognition_model-v4.0.pkl` (que já tem até verificação de integridade criptográfica SHA-256 pronta no seu código!), roda o Estágio 1 (QC de nuvens) e, se o céu estiver limpo, roda o Estágio 2 (pré-processamento + 2DPCA + Ensemble).
  4. **Publicação / Alerta:** O resultado (probabilidade de EPB + mapa SHAP + flag de nuvem) é gravado em um banco de dados de séries temporais e atualizado ao vivo no painel web do EMBRACE, disparando um alerta se a probabilidade cruzar o limiar por $N$ quadros consecutivos (ex: 2 ou 3 quadros seguidos para evitar spikes isolados).
  - **Detalhe que impressiona:** O seu código `epb_detection_system.py` **já foi programado pensando nisso**! Você já criou a classe `EPBModelData`, serialização compactada via `joblib` e verificação de hash `SHA-256` (`save_model` / `load_model`, linhas 78–120) justamente para garantir segurança quando o modelo for colocado em produção em um servidor!

---

# SLIDE 11 — Próximos Passos (Rumo a um sistema operacional)

### 🖥️ O que está na tela
- **Roadmap em curva / zigue-zague (2 linhas de cards grandes animados `.frag`):**
  - **Linha superior (da esquerda para a direita):**
    - `PASSO 1` 🗂️ **Expansão do dataset:** Treinar com milhares de imagens cobrindo múltiplas latitudes da rede EMBRACE e diferentes fases do ciclo solar.
    - `PASSO 2` ☁️ **Filtro robusto de nuvens:** Aperfeiçoar o estágio de controle de qualidade (QC) para descartar automaticamente céus parcialmente nublados ou com luar intenso.
    - `PASSO 3` 🔗 **Integração ao EMBRACE:** Conectar a chegada de novas imagens nos observatórios a uma pipeline automática de inferência e geração de alertas.
  - **Seta vertical à direita (`▼`) descendo para a linha inferior (da direita para a esquerda):**
    - `PASSO 4` 📡 **Fusão multi-instrumento:** Cruzar as detecções ópticas all-sky com dados de cintilação GNSS (índice $S_4$ / ROTI) e digissondas para validação cruzada sob nuvens.
    - `PASSO 5` 🔁 **Aprendizado contínuo:** Ciclo *human-in-the-loop*: casos limítrofes revisados por especialistas retroalimentam e atualizam os pesos do modelo.

### 🎬 Ação de palco e controles
- Assim como no Slide 4, aperte **`→` 5 vezes** (ou clique sobre os cards) para acender os `PASSOS 1 → 2 → 3 → 4 → 5` seguindo a curva em "S" na tela.

### 🎙️ O que falar (Roteiro — ~20 segundos)
> *(Vá apertando `→` para acender os 5 passos)*
> *"Assim, os nossos próximos passos já estão bem traçados: expandir o banco de treinamento ao longo de diferentes latitudes e fases do ciclo solar; aprimorar o classificador de descarte de nuvens; acoplar o pipeline diretamente ao fluxo de dados do EMBRACE; cruzar as detecções ópticas com índices de cintilação GNSS (como o índice S4) e ionossondas; e implementar um ciclo de aprendizado contínuo onde o especialista valida casos duvidosos para retroalimentar o modelo."*

### 🧠 O que você precisa saber a fundo (Backstage deste slide)
- **Por que o Ciclo Solar importa tanto para detecção óptica de EPBs?**
  - No **Máximo Solar**, o fluxo solar extremo-ultravioleta (F10.7) é alto $\Rightarrow$ a densidade eletrônica da camada F é muito maior $\Rightarrow$ o airglow de fundo de 630 nm é mais brilhante, o Pico Pré-Reversão (PRE) é mais intenso, e as bolhas atingem altitudes de ápice enormes (passando de 1500 km), apresentando alto contraste óptico e múltiplas bifurcações.
  - No **Mínimo Solar**, o airglow de fundo é muito mais fraco (próximo ao ruído de leitura do sensor) e as bolhas são mais rasas e menos frequentes. Um modelo treinado só em máximo solar pode perder sensibilidade no mínimo solar se não tiver exemplos de ambos.
- **Como seria a fusão Multi-instrumento (All-Sky + GNSS $S_4$)?**
  - O imageador óptico depende de **céu limpo e noite sem Lua forte**, mas dá a **morfologia 2D espacial completa** da bolha.
  - Os receptores GNSS (que medem o índice de cintilação de amplitude $S_4$ e o ROTI — *Rate of TEC Index*) **funcionam debaixo de chuva e nuvens**, mas só dão medições pontuais na linha de visada de cada satélite.
  - Juntar os dois resolve o problema das nuvens e permite validar automaticamente as predições ópticas com a cintilação de rádio!

---

# SLIDE 12 — Referências Selecionadas

### 🖥️ O que está na tela
- Lista em duas colunas com as 10 referências bibliográficas do seu artigo (`Githio et al. 2024`, `Okoh et al. 2025`, `Zhong et al. 2025a,b`, `Yacoub et al. 2025`, `Siddiqui 2025`, `Thanakulketsarat et al. 2023`, `Srisamoodkham et al. 2022`, `Sedlak et al. 2023`, `Liu et al. 2022`).
- Rodapé: *Dados: Programa EMBRACE / INPE. Apoio: UFPB, UFCG, CAPES, CNPq.*

### 🎙️ O que falar (Roteiro — ~5 a 10 segundos)
> *"Aqui estão as principais referências da literatura recente que embasaram a nossa arquitetura, e registro o nosso agradecimento ao programa EMBRACE/INPE pela disponibilização dos dados observacionais, e à UFPB, UFCG, CAPES e CNPq pelo apoio institucional."*

---

# SLIDE 13 — Obrigado! / Perguntas?

### 🖥️ O que está na tela
- **Obrigado!** em destaque com gradiente de airglow.
- *Perguntas?*
- Alberto I. C. Vieira · Cosme A. O. B. Figueiredo + Logos (INPE, EMBRACE, UFPB, UFCG).

### 🎙️ O que falar (Roteiro — ~5 segundos)
> *"Muito obrigado pela atenção de todos! Fico à disposição para perguntas e sugestões."*

---

# 🛡️ BANCO COMPLETO DE PERGUNTAS E RESPOSTAS (Q&A DE DEFESA NO SBGEA)

Estude esta seção com carinho. No SBGEA, a plateia terá dois perfis principais fazendo perguntas: **1) Físicos da Ionosfera / Aeronomia (pessoal do INPE, UNIVAP, UEPB, UFCG, etc.)** e **2) Pesquisadores que aplicam IA / Computação ao Clima Espacial**. Abaixo estão todas as perguntas que ambos podem fazer e como responder com segurança total.

---

### Bloco A — Perguntas dos Físicos de Ionosfera e Aeronomia

#### 1. *"As bolhas de plasma no Brasil têm uma forte sazonalidade (ocorrem muito entre setembro e março e quase não ocorrem entre maio e agosto). O seu modelo leva em conta a época do ano ou a hora da noite, ou olha só para a imagem?"*
- **Sua resposta:**
  *"Excelente pergunta. De propósito, o nosso modelo atual olha **exclusivamente para a morfologia espacial da imagem**, sem receber a data ou a hora como entrada. Nós fizemos isso justamente para evitar que o algoritmo desenvolvesse um viés sazonal — por exemplo, se recusar a detectar uma bolha fora de época (como uma bolha gerada por tempestade geomagnética e penetração de campos elétricos em junho/julho). O modelo julga apenas a evidência física óptica presente no quadro."*

#### 2. *"O que acontece quando a Lua está no céu? O brilho da Lua satura metade do domo all-sky. O seu modelo acha que a sombra ou o contraste da Lua é uma bolha?"*
- **Sua resposta:**
  *"Nós tratamos o luar leve a moderado na primeira etapa do pré-processamento físico: como o brilho lunar espalhado cria um gradiente espacial suave, o filtro Gaussiano ($\sigma = 10$) modela esse domo iluminado e o subtrai da imagem antes da extração de features pelo 2DPCA. Agora, quando a Lua cheia está alta no céu e satura completamente os pixels do sensor CCD, não há informação ionosférica recuperável naquela região, e essa condição deve ser filtrada pelo estágio de controle de qualidade (QC), assim como fazemos na presença de cobertura severa de nuvens."*

#### 3. *"Como o modelo lida com a Via Láctea cruzando o campo de visão? A Via Láctea é uma faixa brilhante que cruza o céu de norte a sul, e as bordas escuras ao lado dela às vezes confundem até alunos iniciantes."*
- **Sua resposta:**
  *"Essa é uma das grandes vantagens do pré-processamento combinado com o 2DPCA e validado pelo SHAP. A Via Láctea é uma faixa **positiva (brilhante)** pontilhada por milhares de estrelas de alta frequência espacial, enquanto a bolha de plasma é uma **depleção (faixa escura abrupta)** na emissão difusa de 630 nm. Na noite calma de São João do Cariri (Slide 8, Aba 2), nós vemos claramente nos thumbnails do rodapé a Via Láctea cruzando todo o domo da imagem ao longo de 230 quadros, e a probabilidade de bolha dada pelo nosso modelo ficou em 2,3% a noite inteira, com zero falsos positivos."*

#### 4. *"Vocês conseguem diferenciar uma Bolha de Plasma Equatorial (EPB) de um MSTID (Medium-Scale Traveling Ionospheric Disturbance — Distúrbio Ionosférico Propagante de Média Escala) ou de ondas de gravidade?"*
- **Sua resposta:**
  *"Ótimo ponto. Morfologicamente, as EPBs tendem a se alinhar ao meridiano magnético (Norte–Sul) e derivar para leste, enquanto os MSTIDs eletricamente acoplados no setor brasileiro tipicamente apresentam frentes de fase alinhadas de Noroeste para Sudeste (NW–SE) propagando para Noroeste. No nosso conjunto de treinamento atual, a classificação é binária (presença vs. ausência de depleção de EPB). Para separar automaticamente EPBs de MSTIDs em latitudes baixas/médias como Cachoeira Paulista, o próximo passo natural é expandir o rótulo para multiclasse e incorporar a direção de inclinação e o sentido de deriva entre quadros consecutivos."*

#### 5. *"Além de dizer 'Tem Bolha' ou 'Não Tem Bolha' (classificação da imagem inteira), o sistema consegue medir a velocidade de deriva zonal da bolha ou segmentar o contorno exato dela?"*
- **Sua resposta:**
  *"Hoje o classificador entrega uma probabilidade global por imagem, mas nós já temos dois subprodutos espaciais prontos dentro do pipeline: o **mapa de erro de reconstrução do 2DPCA** (Slide 5) e o **mapa de atribuição SHAP** (Slide 7). Ambos destacam espacialmente onde estão as paredes da bolha na grade $64 \times 64$. A partir desses mapas ao longo de uma sequência temporal (como a do Slide 8), é totalmente viável rastrear o deslocamento oeste-leste do centroide de atenção de um quadro para o outro e estimar a velocidade de deriva zonal após projetar a imagem all-sky na altitude de 250 km (unwarping)."*

#### 6. *"As imagens passaram por 'unwarping' (projeção geográfica plana a 250 km) antes de entrar no modelo, ou estão na geometria original da lente olho-de-peixe?"*
- **Sua resposta:**
  *"Elas entram na geometria circular original da câmera all-sky, apenas redimensionadas para $64 \times 64$ e pré-processadas. Optamos por não exigir a projeção geográfica (unwarping) para manter o pipeline o mais direto e rápido possível sobre as imagens padrões já geradas pelo EMBRACE. E como mostramos nos mapas SHAP (Slide 7), o modelo aprende naturalmente a ignorar a borda circular preta e a distorção de borda da lente olho-de-peixe, concentrando a decisão na região zenital e nos quadrantes superiores."*

---

### Bloco B — Perguntas de Machine Learning, Estatística e Metodologia

#### 7. *"Imagens consecutivas da mesma noite são muito parecidas entre si (alta autocorrelação temporal). Na validação cruzada de 5 folds, se uma imagem das 23:00 cair no treino e a das 23:03 cair no teste, isso não facilita o acerto do modelo?"*
- **Sua resposta (Resposta de altíssimo nível — mostra que você pensou nisso no código!):**
  *"Pergunta fundamental! Em séries temporais geofísicas, quadros vizinhos têm forte correlação temporal. Nós investigamos isso diretamente no nosso código (`epb_detection_system.py`): além do `StratifiedKFold` padrão de 5 folds, nós implementamos uma função específica chamada **`temporal_train_test_split`** (linhas 225–347), que extrai a data (`YYYYMMDD`) do nome de cada arquivo, agrupa todas as imagens por **noite de observação inteira**, e separa noites completas para treino e noites completas (e mais recentes) para teste! Além disso, a prova definitiva de que o modelo generaliza para noites inéditas está nos Slides 8 e 9, onde rodamos o modelo treinado sobre noites inteiras contínuas que nunca participaram do treinamento."*

#### 8. *"No pipeline (Slide 4), vocês aplicam Data Augmentation e SMOTE. Esses métodos foram aplicados antes de dividir em treino e teste, ou só no treino?"*
- **Sua resposta:**
  *"Eles foram aplicados **estritamente dentro do conjunto de treino** (e, na validação cruzada, estritamente dentro da partição de treino de cada um dos 5 folds, usando o `ImbPipeline` da biblioteca `imbalanced-learn`). Nenhuma imagem aumentada ou amostra sintética do SMOTE jamais entra no conjunto de validação ou teste. Nós tomamos esse cuidado justamente para eliminar qualquer possibilidade de vazamento de dados (data leakage)."*

#### 9. *"Por que usar 2DPCA em vez do PCA tradicional ou de um Autoencoder / CNN?"*
- **Sua resposta:**
  *"Três motivos diretos:*
  *1. **Contra o PCA 1D tradicional:** Uma imagem $64 \times 64$ achatada gera um vetor de 4.096 posições e uma matriz de covariância de $4096 \times 4096$ (16,7 milhões de elementos), quebrando a vizinhança espacial entre linhas da imagem. Já o 2DPCA opera direto na matriz $64 \times 64$, gerando uma matriz de covariância pequenininha de $64 \times 64$ (4.096 elementos), que é numericamente muito mais estável de diagonalizar com poucas centenas de amostras.*
  *2. **Contra CNNs / Autoencoders:** O 2DPCA tem solução analítica fechada (decomposição em autovetores), treina em frações de segundo na CPU, não depende de inicialização aleatória de pesos nem de GPUs.*
  *3. **Interpretabilidade física:** Como a transformação do 2DPCA é linear e ortogonal ($Y = X \cdot P$), nós conseguimos inverter a projeção exatamente ($P^T$) para projetar os valores SHAP de volta para os pixels da imagem com rigor matemático."*

#### 10. *"Quanto tempo o modelo leva para treinar e quanto tempo leva para classificar uma imagem nova?"*
- **Sua resposta:**
  *"Em um processador comum (CPU padrão, sem placa de vídeo dedicada), o treinamento completo do sistema — incluindo o cálculo do 2DPCA, normalização, SMOTE e os 3 classificadores do Stacking Ensemble — leva cerca de **2 a 5 segundos**. E a inferência (classificar uma única imagem nova de uma estação do EMBRACE) leva **menos de 20 milissegundos**. Ou seja, para uma noite inteira de 200 imagens, ele processa a noite toda em poucos segundos."*

---

## ✅ Checklist Rápido Antes de Subir no Palco do SBGEA

1. **Logos faltantes (opcional):** Se você tiver os arquivos de imagem das logos do EMBRACE e da UFCG, salve-os como `slides/assets/logo_embrace.png` e `slides/assets/logo_ufcg.png`. O HTML já está programado para detectá-los automaticamente; se não colocar, ele exibe um selo estilizado com as siglas `EMBRACE` e `UFCG`.
2. **Testar os cliques interativos uma vez antes da palestra:**
   - Abra `slides/index.html` no navegador.
   - Aperte `F` (tela cheia).
   - No **Slide 2**, veja a animação da bolha rodando e teste o botão `cor`.
   - No **Slide 4**, aperte `→` 5 vezes para ver os 5 passos do pipeline acendendo um por um.
   - No **Slide 8**, clique na aba **`Noite calma (BJL)`** para alternar entre a noite com bolha e a noite sem bolha.
   - No **Slide 11**, aperte `→` 5 vezes para acender os próximos passos.
3. **Postura nas perguntas sobre limitações:** Sempre que alguém apontar nuvens, tamanho do dataset ou integração em tempo real, sorria e concorde — porque você **já colocou exatamente esses pontos nos Slides 9, 10 e 11**, mostrando maturidade científica e domínio completo do problema operacional!


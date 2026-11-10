# Detecção Automática de Bolhas de Plasma Equatorial em Imagens AirGlow Utilizando Técnicas de Aprendizado de Máquina

*Texto-base narrativo e integrado da apresentação no IX Simpósio Brasileiro de Geofísica Espacial e Aeronomia (SBGEA 2026). Este documento foi escrito como um texto contínuo — o "livro" por trás dos slides — para que a leitura fluida consolide toda a física, a matemática, os resultados, as descobertas diagnósticas e as limitações do trabalho.*

---

## 1. A Física do Céu Noturno e o Fenômeno das Bolhas de Plasma
*(Correspondência narrativa: Slides 1 e 2)*

Logo após o pôr do sol nas regiões equatoriais e de baixa latitude, a ionosfera terrestre atravessa uma transição eletrodinâmica dramática. Durante o dia, a radiação solar extrema no ultravioleta mantém a atmosfera superior intensamente ionizada. Quando o Sol se põe e a fotoionização cessa, a camada E (mais baixa, por volta de 100 a 150 km de altitude) recombina-se rapidamente e desaparece, enquanto a camada F (mais alta, entre 220 e 400 km) permanece densa. Isso cria um gradiente vertical de densidade eletrônica extremamente abrupto na base da camada F: uma camada de plasma pesado e denso apoiada sobre uma região inferior rarefeita.

Ao mesmo tempo, a interação entre o vento termoesférico neutro soprando para leste e o gradiente de condutividade ao longo do terminador solar intensifica o campo elétrico zonal para leste. Esse fenômeno é conhecido como **Pico Pré-Reversão** (*Pre-Reversal Enhancement* — PRE). Como o campo geomagnético terrestre na região equatorial é praticamente horizontal e aponta para o norte, o produto vetorial entre o campo elétrico zonal ($\mathbf{E}$) e o campo magnético ($\mathbf{B}$) gera uma forte deriva eletrodinâmica vertical ($\mathbf{E} \times \mathbf{B}$) que empurra toda a camada F para altitudes elevadas, onde a frequência de colisões entre íons e partículas neutras é menor.

Um fluido denso sustentado contra a gravidade por um fluido rarefeito, sob forte aceleração ascendente, é o cenário clássico da **instabilidade de Rayleigh–Taylor generalizada**. Qualquer perturbação inicial na base da camada F — frequentemente semeada por ondas de gravidade atmosféricas vindas da troposfera — cresce exponencialmente. Grandes colunas ou "bolhas" de plasma rarefeito rompem a base da camada F e ascendem até o topo da ionosfera, estruturando-se ao longo dos tubos de fluxo geomagnético e derivando de oeste para leste ao longo da noite com velocidades típicas entre 50 e 150 m/s.

Embora essas **Bolhas de Plasma Equatoriais (EPBs)** sejam estruturas de plasma invisíveis a olho nu, a química da alta atmosfera permite fotografá-las diretamente do solo. Na base da camada F, em torno de 250 km de altitude, íons de oxigênio ($\text{O}^+$) reagem com moléculas de oxigênio ($\text{O}_2$) por troca de carga ($\text{O}^+ + \text{O}_2 \rightarrow \text{O}_2^+ + \text{O}$). Em seguida, o íon molecular $\text{O}_2^+$ sofre recombinação dissociativa com elétrons livres ($\text{O}_2^+ + e^- \rightarrow \text{O}^*({}^1\text{D}) + \text{O}$), deixando átomos de oxigênio no estado excitado metastável ${}^1\text{D}$. Ao decair de volta ao estado fundamental ${}^3\text{P}$, esse oxigênio emite um fóton vermelho na linha espectral de **630,0 nanômetros** — a aeroluminescência noturna (*nightglow*).

Dentro de uma bolha de plasma, a densidade de elétrons ($n_e$) e de íons $\text{O}^+$ cai de uma a três ordens de grandeza em relação ao plasma vizinho. Sem reagentes suficientes para sustentar a recombinação dissociativa, a emissão óptica em 630,0 nm praticamente se apaga no interior da estrutura. Por isso, quando observamos o céu com um imageador óptico *all-sky* (lente olho-de-peixe de $180^\circ$ de campo de visão, cobrindo cerca de 1.000 km de diâmetro na altitude da camada F), as bolhas de plasma aparecem como **faixas escuras de depleção de emissão**, alongadas na direção norte–sul magnética, frequentemente bifurcadas e movendo-se para leste.

Monitorar essas estruturas sobre o território brasileiro tem importância estratégica. O Brasil situa-se exatamente sob a crista sul da **Anomalia de Ionização Equatorial (EIA)** e possui a maior declinação magnética do planeta. As irregularidades de pequena escala associadas às paredes das bolhas refratam e difratam ondas de rádio transionosféricas, provocando **cintilação severa** em sinais de satélites GNSS (GPS, Galileo) e degradando sistemas de posicionamento de precisão na agricultura, navegação aérea, operações offshore e telecomunicações.

---

## 2. O Gargalo da Triagem Visual e a Armadilha da Autocorrelação Noturna
*(Correspondência narrativa: Slide 3)*

A rede do programa **EMBRACE/INPE** (*Estudo e Monitoramento Brasileiro do Clima Espacial*) mantém imageadores *all-sky* operando continuamente em múltiplos observatórios brasileiros. Cada câmera registra centenas de exposições por noite. Ao longo de meses e anos, isso acumula dezenas de milhares de imagens. Historicamente, a catalogação dessas imagens depende da inspeção visual quadro a quadro por especialistas humanos — um processo lento, exaustivo e sujeito a subjetividade, especialmente quando as depleções estão surgindo ou dissipando na borda do domo, ou quando o céu apresenta poluição luminosa urbana e nebulosidade parcial.

A resposta imediata da visão computacional moderna seria treinar uma rede neural convolucional profunda (CNN) ou um Vision Transformer diretamente sobre os quadros rotulados. Contudo, na geofísica observacional, essa abordagem direta esconde uma armadilha estatística grave: **a unidade estatística verdadeiramente independente não é o quadro isolado, mas a noite de observação**.

Em um imageador *all-sky*, quadros sucessivos são capturados a cada 2 a 5 minutos. Durante uma mesma noite, a transparência atmosférica, a contaminação luminosa urbana, a temperatura do sensor CCD, o nível basal da aeroluminescência solar/geomagnética e a geometria do horizonte local permanecem quase constantes. Se dividirmos 500 ou 1.000 quadros aleatoriamente entre treino e teste (como na validação cruzada *k-fold* convencional no nível de quadro), quadros separados por apenas três minutos da mesma noite cairão simultaneamente no treino e no teste. Qualquer modelo de alta capacidade — especialmente uma rede profunda — simplesmente memorizará a assinatura de fundo daquela noite específica e exibirá uma acurácia aparente superior a 95%, mas falhará drasticamente ao encontrar uma noite inédita no mundo real.

Em nosso conjunto de dados rotulado atual, contamos com **572 imagens em resolução de $512 \times 512$ pixels**, divididas em **255 quadros com EPB** e **255 quadros sem EPB** sob céu limpo (510 quadros para o classificador físico), mais **62 quadros nublados** dedicados à validação do controle de qualidade. Porém, esses 572 quadros provêm de **7 noites de observação distribuídas em 3 estações** da rede EMBRACE: São João do Cariri na Paraíba (`CA`), Cachoeira Paulista em São Paulo (`CP`) e Bom Jesus da Lapa na Bahia (`BJL`). Reconhecer que nossa amostra efetiva no nível superior é de 7 noites moldou todas as escolhas arquiteturais do nosso sistema: precisamos de um pipeline enxuto, fisicamente guiado, linearmente auditável e avaliado estritamente deixando noites inteiras de fora (*Leave-One-Night-Out*).

---

## 3. A Arquitetura em Cascata Guiada pela Física
*(Correspondência narrativa: Slides 4 e 5)*

Em vez de tratar a detecção como uma caixa-preta única que tenta classificar qualquer imagem que chegue do sensor, estruturamos o sistema em uma **arquitetura em cascata de dois estágios**, precedida e guiada pelo conhecimento físico da emissão em 630,0 nm.

### O Estágio 1: O Porteiro de Controle de Qualidade (QC)
Na operação real de um observatório óptico, uma fração substancial das noites é comprometida por cobertura de nuvens, neblina ou contaminação luminosa intensa. Nuvens densas bloqueiam a luz da alta atmosfera e criam manchas escuras irregulares na cúpula que qualquer classificador espacial pode confundir com depleções de plasma. Por isso, o **Estágio 1** funciona como um porteiro determinístico de qualidade do céu antes que o classificador de bolhas seja acionado.

Avaliando a região central útil do domo (raio de $38\%$ do lado da imagem, livre das árvores e estruturas da borda), o porteiro extrai três métricas físicas diretas:
1. **Nitidez estelar (`star_score`):** calculada aplicando o operador Laplaciano $3 \times 3$ ($|\nabla^2 I|$) e contando quantos pixels ultrapassam o limiar de borda aguda de 30 unidades. Quando o céu está limpo, centenas de estrelas pontuais atravessam a imagem e geram alta energia Laplaciana; quando entram nuvens, as estrelas são borradas ou ocultadas e essa contagem despenca.
2. **Brilho médio (`mean_brightness`):** a intensidade média dos pixels dentro do domo.
3. **Contraste global (`contrast_std`):** o desvio-padrão das intensidades no domo, que detecta se a tampa do obturador permaneceu fechada ou se o sensor travou (`mean_brightness < 4` ou `contrast_std < 2`).

Com base nesses indicadores, o Estágio 1 divide os quadros em **três zonas operacionais**:
- **Nublado (`CLOUDY`):** quando a contagem de estrelas cai abaixo de `1.000` ou o brilho médio cai abaixo de `55`. O quadro é imediatamente barrado e recebe o veredito final **Indeterminado**, sem passar ao Estágio 2.
- **Degradado (`DEGRADED`):** uma zona intermediária de alerta, quando o quadro supera o corte mínimo de nuvem, mas apresenta menos de `4.500` pontos de nitidez estelar ou brilho médio abaixo de `66`. O quadro segue para o classificador físico, mas recebe um sinalizador de atenção (**conferir**) para revisão humana.
- **Céu Limpo (`CLEAR`):** quando a nitidez estelar é $\ge 4.500$ e o brilho médio é $\ge 66$, garantindo condições ópticas plenas.

### O Pré-Processamento Físico: Preservando o Sinal Negativo das Depleções ($\delta I/I$)
Quando um quadro é aprovado para análise física (Céu Limpo ou Degradado), o objetivo do pré-processamento é isolar a perturbação ionosférica da geometria instrumental e do fundo astronômico.

Em primeiro lugar, as estrelas pontuais são ruído de alta frequência espacial para a aeroluminescência. Na resolução original de $512 \times 512$ pixels, aplicamos uma **abertura morfológica** (*morphological opening*) com um elemento estruturante elíptico de $7 \times 7$ pixels. Como as estrelas ocupam poucos pixels e as bolhas de plasma têm dezenas de pixels de largura, a abertura morfológica apaga as estrelas preservando intacta a estrutura da camada de 630,0 nm. Em seguida, a imagem sem estrelas é reamostrada por interpolação de área para **$64 \times 64$ pixels** — escala que retém toda a morfologia de meso-escala das bandas de depleção (que medem de 50 a 300 km) enquanto elimina ruído de granulação (*shot noise*) e reduz drasticamente a dimensionalidade.

O passo seguinte é a **estimativa do fundo suave de emissão ($I_\text{fundo}$)**. Em um imageador *all-sky*, mesmo uma noite perfeitamente quieta não tem brilho uniforme: há o efeito de *van Rhijn* (aumento da coluna óptica integrada conforme olhamos do zênite para o horizonte), gradientes suaves da Anomalia de Ionização Equatorial, poluição luminosa urbana próxima ao horizonte (observando que os imageadores da rede EMBRACE não fotografam a Lua) e a vinhetagem natural da lente olho-de-peixe. Além disso, a imagem circular do domo está inscrita em um quadrado cujos cantos externos são pretos (zeros). Se aplicarmos um filtro Gaussiano comum sobre a imagem inteira, os zeros da máscara preta externa puxarão a estimativa de fundo para baixo perto da borda do domo, criando um anel brilhante artificial na periferia. Para evitar esse artefato, definimos a máscara circular estrita do domo útil (raio de $44\%$ do lado da imagem, `fov_radius_frac = 0.44`) e calculamos o fundo suave por **convolução Gaussiana normalizada pela máscara**: dividimos o borramento Gaussiano dos pixels válidos ($\sigma = 10$ na grade de $64 \times 64$) pelo borramento Gaussiano da própria máscara binária. Assim, o fundo suave é estimado exclusivamente a partir de pixels internos ao domo, removendo simultaneamente o efeito *van Rhijn* e a vinhetagem instrumental sem agredir as estruturas de meso-escala.

Tendo a imagem sem estrelas $I$ e o fundo suave $I_\text{fundo}$, calculamos a grandeza física fundamental da aeroluminescência: o **contraste fracionário assinado**,
$$\frac{\delta I}{I} = \frac{I - I_\text{fundo}}{I_\text{fundo}} = \frac{I}{I_\text{fundo}} - 1.$$

Aqui reside uma das decisões físicas mais importantes do nosso trabalho: **jamais truncamos $\delta I/I$ em zero**. Em pipelines genéricos de processamento de imagens médicas ou industriais, é comum zerar desvios negativos antes de aplicar equalização de histograma (como CLAHE). Porém, na física da camada F, uma bolha de plasma é uma **depleção** — uma região onde $I < I_\text{fundo}$ e, portanto, $\delta I/I < 0$. Cortar valores negativos em zero apaga literalmente o interior escuro da bolha, restando apenas ruído nas bordas. Ao manter $\delta I/I$ com seu sinal algébrico completo e normalizá-lo em **z-score dentro da máscara do domo** (média zero e desvio-padrão unitário por quadro, com truncamento suave de *outliers* extremos em $\pm 4\sigma$), as depleções destacam-se como vales negativos nítidos e padronizados, ao mesmo tempo em que removemos o nível de brilho médio global do quadro.

### Redução Matricial via 2DPCA e Classificação Linear
Com o campo $\delta I/I$ normalizado na matriz $X \in \mathbb{R}^{64 \times 64}$, precisamos extrair características discriminativas sem explodir o número de parâmetros. Se achatássemos a imagem para um vetor de $4.096$ dimensões para aplicar o PCA clássico (1DPCA), destruiríamos a vizinhança espacial 2D entre linhas e colunas e precisaríamos estimar uma matriz de covariância de $4.096 \times 4.096$ a partir de poucas centenas de quadros.

Em vez disso, empregamos a **Análise de Componentes Principais Bidimensional (2DPCA)**, proposta por Yang et al. (2004). O 2DPCA opera diretamente sobre as matrizes $64 \times 64$. Calculando a média $\bar{X}$ apenas sobre as imagens de treinamento, formamos a matriz de dispersão espacial das linhas:
$$G_t = \frac{1}{N}\sum_{i=1}^{N} (X_i - \bar{X})^\top (X_i - \bar{X}) \in \mathbb{R}^{64 \times 64}.$$
Notemos a elegância matemática: $G_t$ tem dimensão apenas $64 \times 64$, sendo estimada com alta estabilidade estatística. Extraímos os $k$ autovetores ortonormais $V = [v_1, \dots, v_k] \in \mathbb{R}^{64 \times k}$ associados aos $k$ maiores autovalores de $G_t$. Cada imagem centrada $(X - \bar{X})$ é então projetada linearmente como $Y = (X - \bar{X})V \in \mathbb{R}^{64 \times k}$.

Visualmente, como mostramos no **Slide 5**, quando reconstruímos a imagem a partir do subespaço 2DPCA ($\hat{X} = \bar{X} + (X - \bar{X})VV^\top$) e observamos o mapa de erro absoluto $|X - \hat{X}|$, uma noite quieta (como em Bom Jesus da Lapa) é reconstruída de maneira suave e homogênea, enquanto uma noite com bolha ativa (como em São João do Cariri) concentra energia de reconstrução exatamente ao longo das paredes norte–sul dos canais de depleção.

Por fim, a matriz projetada $Y$ (achatada em um vetor compacto de $64 \times k$ características — $192$ características para $k=3$ selecionado pelo modelo final) é padronizada (`StandardScaler`) e entregue a um classificador linear: uma **Regressão Logística com penalidade $L_2$ e pesos de classe balanceados**. A escolha deliberada de um classificador linear em vez de *ensembles* de árvores (XGBoost, Random Forest) ou kernels não-lineares deve-se a duas razões centrais: primeiro, a forte regularização $L_2$ ($C \in \{0,001;\; 0,01;\; 0,1\}$) impede a memorização de limiares não-lineares espúrios; segundo, a linearidade de ponta a ponta permite **atribuição exata por pixel**, como veremos a seguir.

---

## 4. Avaliação Leave-One-Night-Out, Ablação e o Diagnóstico do Confundidor de Brilho
*(Correspondência narrativa: Slide 6 e suas 3 Abas)*

Para avaliar honestamente a capacidade do sistema de generalizar para uma noite nunca vista, adotamos exclusivamente o protocolo **Leave-One-Night-Out aninhado** (*nested cross-validation*) sobre as 6 noites de céu limpo (510 quadros). Em cada uma das 6 rodadas externas, todos os quadros de uma noite inteira são separados como teste cego. Nas 5 noites restantes, um loop interno *leave-one-night-out* seleciona o número de componentes $k \in \{3, 5, 10\}$ do 2DPCA e a regularização $C \in \{0,001;\; 0,01;\; 0,1\}$ da Regressão Logística maximizando a AUC-ROC (no modelo final ajustado sobre as 6 noites, o procedimento seleciona $k = 3$ e $C = 0,001$). Nenhuma estatística do pré-processamento, da média $\bar{X}$, dos autovetores $V$ ou do escalonador vaza da noite de teste para o treino.

### O Desempenho no Limiar Operacional Calibrado (`t = 0,70`)
Como cada noite de observação possui um nível basal de aeroluminescência, transparência e resposta instrumental próprio dentro de um conjunto enxuto de noites, as probabilidades preditas para uma noite inédita sofrem um leve deslocamento de calibração. No corte padrão de `t = 0,50`, o modelo já entrega **78,6% de acurácia balanceada** (com **96,5% de sensibilidade** e **60,8% de especificidade**). Calibrando o limiar operacional de decisão em **`t = 0,70`** (ponto ótimo da curva de sensibilidade ao limiar demonstrada na **Aba 2 do Slide 6**), equilibramos sensibilidade e especificidade e obtemos nas predições *out-of-night* agregadas:
- **Acurácia Balanceada:** **90,0%** (Acurácia global = 90,0%, índice $J$ de Youden = 0,800; acurácia média por noite de **92,6%** e pior noite com **72,1%**)
- **Sensibilidade (Recall de EPB):** **88,2%** (225 de 255 quadros com EPB detectados corretamente; **F1-score = 0,898**)
- **Especificidade (Céu Limpo sem EPB):** **91,8%** (234 de 255 quadros de céu quieto rejeitados corretamente, com apenas 21 falsos positivos e **Precisão = 91,5%**)
- **AUC-ROC:** **0,950** (e **AUC Precision-Recall = 0,962**, mostradas na **Aba 1 do Slide 6**).

### O que a Escada de Ablação Revelou (`Aba 3 do Slide 6`)
Para entender exatamente quanto cada etapa física contribui para esse resultado e por que certas alternativas foram descartadas, construímos uma escada de ablação rigorosa avaliada no mesmo protocolo *Leave-One-Night-Out*, todas as linhas comparadas no mesmo limiar base `t = 0,50`:
1. **Subtração simples + corte em zero + CLAHE (sem $\delta I < 0$):** a abordagem convencional que trunca os valores negativos em zero e aplica equalização local de histograma fica em acurácia balanceada de **66,3%**, especificidade de 34,5% e AUC-ROC de **0,823** — nossa linha de base.
2. **+1 Contraste com sinal $\delta I/I$ (sem corte em zero, z-score em $64 \times 64$):** simplesmente preservar o sinal negativo de $\delta I/I$ eleva a acurácia balanceada para **68,4%** e salta a AUC-ROC para **0,906**, confirmando matematicamente que cortar os desvios negativos destrói a assinatura física das depleções.
3. **★ +2 Remoção morfológica de estrelas + fundo Gaussiano restrito ao domo (configuração adotada):** ao adicionar a abertura morfológica para remover estrelas em 512 px e estimar o fundo Gaussiano ($\sigma = 10$) exclusivamente dentro da máscara circular do domo, atingimos o **melhor desempenho discriminativo global do estudo**: **78,6% de acurácia balanceada** em `t = 0,50` (chegando a **90,0%** no limiar operacional `t = 0,70`), **96,5% de sensibilidade**, **60,8% de especificidade** e **AUC-ROC de 0,950**.
4. **Alternativa 2b — Achatamento radial em anéis (descartada):** testamos também dividir a imagem pela mediana de brilho em anéis concêntricos após o $\delta I/I$. Essa alternativa **derruba** a acurácia balanceada de 78,6% para **51,2%** e a especificidade de 60,8% para **8,2%** (AUC-ROC cai de 0,950 para **0,791**). Fisicamente, o fundo Gaussiano no domo já remove de forma suave o efeito *van Rhijn* e a vinhetagem da lente; quando uma bolha larga atravessa o zênite (centro do domo, $r \approx 0$), a mediana em anéis interpreta o escurecimento da própria bolha como perfil instrumental e o subtrai, destruindo o contraste da depleção. Por isso, essa alternativa foi descartada.
5. **Alternativa 3 — Processamento integral em $512 \times 512$ (descartada):** rodar toda a cadeia em alta resolução antes de reduzir para $64 \times 64$ no final reduz a acurácia balanceada para **75,3%** (especificidade cai para 54,1%) e a AUC-ROC para **0,936**, além de ser cerca de 64 vezes mais custosa computacionalmente, pois retém ruído fino de fundo que aumenta a variância entre noites — sendo igualmente descartada.

### O Achado Diagnóstico Central: O Confundidor de Brilho Noturno
A maior contribuição científica de uma validação honesta não é esconder fragilidades, mas expô-las. Para testar se as 6 noites de céu limpo apresentavam algum atalho estatístico, treinamos um *baseline* propositalmente "cego" à geometria espacial: uma Regressão Logística alimentada exclusivamente por **5 estatísticas globais de brilho dos pixels brutos dentro do domo** (média, desvio-padrão e percentis 10, 50 e 90 de cada quadro, sem 2DPCA e sem informação de onde estão os pixels).

Para nossa surpresa inicial — e como alerta metodológico fundamental — esse modelo trivial de 5 números atingiu **95,9% de acurácia balanceada e 0,991 de AUC-ROC** no mesmo protocolo *Leave-One-Night-Out*! Isso significa que o brilho global é um detector físico de bolhas de plasma? **De forma alguma.** Significa que, em uma amostra pequena de apenas 6 noites de céu limpo (4 noites com EPB e 2 noites sem EPB), o nível de intensidade global de fundo e as condições sazonais/instrumentais separam quase perfeitamente as noites disponíveis. Um modelo que use o brilho bruto acerta a noite inteira pelo motivo errado (*shortcut learning*). O nosso pipeline espacial 2DPCA normaliza cada imagem em z-score ($\mu = 0, \sigma = 1$ dentro do domo) justamente para bloquear esse atalho de brilho de primeira ordem e forçar a decisão a depender da morfologia espacial — alcançando **0,950 de AUC-ROC e 90,0% de acurácia balanceada** puramente pela forma geométrica das bolhas. O fato de o *baseline* de brilho pontuar 95,9% prova matematicamente por que qualquer trabalho na área precisa auditar confundidores noturnos e expandir o número de noites.

---

## 5. Interpretabilidade Física: Atribuição Linear Exata e Orientação Magnética
*(Correspondência narrativa: Slide 7)*

Como podemos comprovar que o nosso classificador 2DPCA + Regressão Logística está olhando para a morfologia das faixas de depleção e não para artefatos de borda? Em modelos não-lineares complexos, recorre-se a aproximações estocásticas de SHAP ou perturbações locais que sofrem de instabilidade quando os pixels são correlacionados. Em nossa arquitetura, porém, **toda a cadeia após o pré-processamento é estritamente linear**:
1. A projeção 2DPCA é uma transformação linear da imagem pré-processada $X$: $Y = (X - \bar{X})V$.
2. O escalonamento padrão (`StandardScaler`) é uma transformação afim elemento a elemento por $\sigma_s$.
3. O *logit* da Regressão Logística é uma combinação linear das características escalonadas.

Portanto, podemos colapsar os pesos da Regressão Logística ($w$) e os autovetores do 2DPCA ($V$) em um **único mapa exato de pesos no espaço de pixels**,
$$W_\text{pixel} = \tilde{W} V^\top \in \mathbb{R}^{64 \times 64},$$
onde $\tilde{W}$ contém os pesos $w / \sigma_s$ reorganizados na grade $64 \times k$. A contribuição exata de cada pixel $(r, c)$ de uma imagem $X$ em relação ao céu médio de referência $\bar{B}$ é dada simplesmente pelo produto ponto a ponto:
$$\phi(r, c) = W_\text{pixel}(r, c) \cdot \big(X(r, c) - \bar{B}(r, c)\big).$$
Essa é a **atribuição linear intervencional exata**: a soma algébrica de todos os pixels $\sum_{r,c} \phi(r,c)$ somada ao intercepto reproduz o *logit* do modelo com erro numérico zero e custo computacional instantâneo.

Nos mapas apresentados no **Slide 7**, vemos o resultado dessa equação em ação: para um quadro ativo em São João do Cariri (probabilidade predita de **91,7%**), as contribuições positivas em vermelho desenham com precisão os canais alongados norte–sul da bolha de plasma dentro do domo. Em contrapartida, para um quadro de céu quieto em Bom Jesus da Lapa (probabilidade de **7,8%**), o interior do domo exibe contribuições negativas (azuis) e neutras distribuídas, empurrando a decisão com segurança para a ausência de EPB.

### Descartando o Confundidor de Orientação Geomagnética
Além do brilho, investigamos um segundo potencial confundidor físico: como as bolhas alinham-se ao meridiano magnético e a declinação magnética varia entre as estações brasileiras, será que o classificador estaria distinguindo estações pela inclinação angular das faixas?

Para testar isso, calculamos o **tensor de estrutura** a partir dos gradientes direcionais de Sobel ($g_x, g_y$) sobre o campo $\delta I/I$ dentro do domo interno de todos os quadros com EPB, extraindo o ângulo dominante das faixas em relação ao eixo vertical da imagem e sua coerência anisotrópica:
- Em **São João do Cariri (`CA`, 213 quadros com EPB):** o ângulo mediano medido foi de **$+4,5^\circ \pm 20,4^\circ$**, com alta coerência estrutural (**0,749**).
- Em **Cachoeira Paulista (`CP`, 42 quadros com EPB):** o ângulo mediano foi de **$+0,2^\circ \pm 6,6^\circ$**, com coerência de **0,653**.
- (**Bom Jesus da Lapa [`BJL`]** possui apenas noite sem EPB no conjunto atual, portanto 0 quadros com bolha).

Como ambas as estações com EPB apresentam faixas alinhadas muito próximas à vertical (dentro da tolerância de $\pm 25^\circ$), nenhuma rotação corretiva por estação foi necessária (`ROTATION_BY_SITE = {}`), descartando a orientação geomagnética como variável confundidora no dataset atual. Ainda assim, mantemos no slide a ressalva científica rigorosa: mapas de atribuição visualmente coerentes demonstram que o modelo utiliza as feições corretas dentro da imagem, mas não substituem a necessidade de controlar o deslocamento de limiar entre noites.

---

## 6. Monitoramento de Noites Inteiras e o Controle de Nuvens em 3 Vias
*(Correspondência narrativa: Slides 8 e 9)*

Quando aplicamos o pipeline em cascata aos 432 quadros das três sequências temporais completas dedicadas ao monitoramento contínuo, o sistema gera relatórios operacionais quadro a quadro com **três vereditos possíveis** — **EPB** (vermelho), **Sem EPB** (azul) ou **Indeterminado** (cinza, bloqueado pelo Estágio 1) — acompanhados pela faixa amarela de atenção (**conferir**) para céus degradados.

### Relatório Operacional vs. Avaliação Cega Out-of-Night (`Slide 8`)
Nas duas primeiras abas do **Slide 8**, visualizamos o relatório temporal gerado pelo modelo final:
- Na noite com atividade intensa de bolhas em **São João do Cariri (22–23/10/2025, 140 quadros — Aba 1)**, o sistema acompanha toda a evolução física da noite: são **99 quadros classificados como EPB (71,7% dos decididos)** durante as horas de atividade intensa, seguidos por **39 quadros Sem EPB** quando as bolhas se dissipam no fim da madrugada, além de apenas **2 Indeterminados** e **2 marcados como *conferir***, atingindo **98,6% de cobertura útil**.
- Na noite de céu quieto em **Bom Jesus da Lapa (22–23/07/2025, 230 quadros — Aba 2)**, a probabilidade permanece estavelmente próxima de zero durante toda a noite: são **222 quadros Sem EPB (100% dos decididos)**, **0 falsos alarmes de EPB**, **8 Indeterminados** e **6 *conferir*** (**96,5% de cobertura útil**).

Na **Aba 3 do Slide 8**, apresentamos lado a lado a **avaliação cega *Out-of-Night* no limiar operacional `t = 0,70`** dessas mesmas sequências — isto é, o que acontece quando a noite inteira da sequência nunca foi vista durante o treinamento:
- Para a noite ativa de **São João do Cariri (`22–23/10/2025`, 140 quadros)**, o modelo cego mantém excelente aderência física: **62,9% de todos os 140 quadros da noite são classificados como EPB** (88 de 140 quadros, já que os ~50 quadros finais são de dissipação; capturando **92,9% dos 85 quadros rotulados com EPB ativa**).
- Já para a noite limpa de **Bom Jesus da Lapa (`22–23/07/2025`, 230 quadros)**, quando essa noite inteira é retirada do treinamento, o modelo cego em `t = 0,70` apresenta apenas **6,5% de falsos positivos** (15 em 230 quadros, ou **93,5% de rejeição correta** na sequência completa).

Explicar a origem desse deslocamento residual em Bom Jesus da Lapa é um ponto importante da apresentação: nosso dataset possui apenas **duas noites de céu limpo sem EPB**: `BJL 22/07/2025` (inverno) e `CA 07/04/2025` (outono). Quando retiramos Bom Jesus da Lapa para testá-la às cegas, **sobra apenas uma única noite sem bolha em todo o conjunto de treinamento — justamente a noite de outono em Cariri!** O modelo linear treinado sem Bom Jesus da Lapa nunca viu o padrão de fundo instrumental daquele observatório em condições calmas; ainda assim, graças ao pré-processamento físico com fundo Gaussiano no domo e ao limiar `t = 0,70`, **93,5% da sequência inédita** é corretamente rejeitada como céu limpo sem EPB.

### O Desempenho em Noite Nublada e o Custo Atual do QC (`Slide 9`)
No **Slide 9**, examinamos a terceira sequência completa: a noite severamente nublada de inverno em **São João do Cariri (`24–25/06/2026`, 62 quadros)**.
- Se não existisse o Estágio 1 de controle de qualidade e passássemos as imagens de nuvens diretamente ao classificador 2DPCA em `t = 0,70`, **85,5% da noite nublada seria falsamente anunciada como EPB** (pois as manchas escuras de nuvens geram forte contraste negativo $\delta I/I < 0$).
- Com o porteiro de 3 zonas ativo, **34 dos 62 quadros (54,8%) são imediatamente bloqueados como `Indeterminado`** (faixas cinzas) e outros **15 quadros (24,2%) recebem o alerta amarelo `DEGRADED` (*conferir*)**. Com isso, quase **80% da noite nublada é barrada ou sinalizada para revisão humana**, reduzindo os chamados de EPB na cascata total para **45,2%** e para apenas **21,0%** nos quadros que passaram como céu limpo confiável.

Ao mesmo tempo, reportamos com total transparência o **custo operacional atual** desse porteiro: como os limiares (`threshold_stars = 1000` e `min_mean_brightness = 55`) foram calibrados *in-sample* sobre a única noite nublada rotulada disponível, o Estágio 1 atinge precisão de 64,2% para nuvens porque acaba barrando inadvertidamente como `Indeterminado` **19,0% dos quadros legítimos de EPB na noite de 27/10/2025 em Cariri (8 de 42 quadros)** e **12,8% na noite de 18/12/2025 em Cariri (11 de 86 quadros)**. Fisicamente, isso ocorre porque uma bolha de plasma muito larga e profunda que cubra a região central do domo também reduz o brilho médio da imagem abaixo de 55, simulando um escurecimento por nuvem.

---

## 7. A Amostra Efetiva de 7 Noites e o Planejamento Amostral para 20–40 Noites
*(Correspondência narrativa: Slide 10)*

Todos os diagnósticos anteriores convergem para a tabela central apresentada no **Slide 10**: a anatomia das **7 noites rotuladas** que compõem os nossos 572 quadros:

1. **`CA 2025-10-22` (Com EPB):** 85 quadros, Primavera (temporada de EPB), 0% bloqueado pelo QC.
2. **`CA 2025-10-27` (Com EPB):** 42 quadros, Primavera, 19,0% bloqueado pelo QC.
3. **`CA 2025-12-18` (Com EPB):** 86 quadros, Verão, 12,8% bloqueado pelo QC.
4. **`CP 2024-02-09` (Com EPB):** 42 quadros, Verão, 0% bloqueado pelo QC.
5. **`BJL 2025-07-22` (Sem EPB):** 220 quadros, Inverno, 0% bloqueado pelo QC.
6. **`CA 2025-04-07` (Sem EPB):** 35 quadros, Outono (transição · céu limpo), 0% bloqueado pelo QC.
7. **`CA 2026-06-24` (Nublada / QC):** 62 quadros, Inverno, 54,8% bloqueado pelo QC (+24,2% *conferir*).

Ao olhar para essa tabela, ficam evidentes três colinearidades estruturais impostas pelo tamanho amostral reduzido:
- **Colinearidade Estação $\times$ Classe:** Cachoeira Paulista só possui noite com EPB, enquanto Bom Jesus da Lapa só possui noite sem EPB. Apenas São João do Cariri possui ambas as classes, mas com forte assimetria de quadros (213 com EPB contra apenas 35 sem EPB).
- **Colinearidade Sazonalidade $\times$ Classe:** No setor brasileiro, a temporada clássica de bolhas ocorre de outubro a março (primavera/verão). Coincidentemente, todas as 4 noites com EPB do dataset são de outubro a fevereiro, enquanto as 2 noites sem EPB são de abril e julho (outono/inverno, quando a densidade ionosférica basal e o fundo de 630,0 nm são naturalmente diferentes).
- **Colinearidade Instrumental e Iluminação Urbana:** Cada observatório possui óptica, ruído e perfil de poluição luminosa próprios (Cachoeira Paulista situa-se no Vale do Paraíba, com iluminação urbana no horizonte muito distinta do sertão de Bom Jesus da Lapa ou do semiárido de Cariri). Sem noites de ambas as classes em cada sítio, o modelo corre o risco de correlacionar a assinatura de fundo de cada estação à presença ou ausência de bolhas.

Além disso, identificamos um fator técnico importante na preparação primária dos arquivos: as imagens originais dos observatórios são gravadas em formato `.tif` de 16 bits e foram convertidas para `.png` utilizando o comando `magick *.tif -normalize -contrast-stretch 2%x1%`. Como essa operação reescala os percentis de intensidade de cada imagem individualmente entre 0 e 255, um quadro de céu quieto sem pontos extremos tem seu fundo esticado para valores médios mais altos, enquanto um quadro com estrelas brilhantes ou artefatos tem seu fundo comprimido — amplificando artificialmente a diferença de brilho global entre noites.

### Quantas noites são necessárias e quais escolher?
Com apenas 6 noites de céu limpo (4 positivas e 2 negativas), cada noite sem EPB representa **50% de toda a especificidade do estudo**, e o loop interno do *Leave-One-Night-Out* frequentemente encontra divisões de classe única onde a AUC não pode ser calculada, exigindo o uso de hiperparâmetros de segurança (`FALLBACK_PARAMS`).

Estabelecemos, portanto, que a meta estatística para a dissertação é atingir entre **20 e 40 noites independentes**:
- **Mínimo viável (~20 noites, sendo ~10 com EPB e ~10 sem EPB):** reduz o peso individual de cada noite para 10% da sua classe, elimina divisões internas degeneradas e permite congelar de 4 a 6 noites em um conjunto de teste travado (`LOCKED_TEST_NIGHTS`) verdadeiramente intocado.
- **Recomendado para publicação robusta (30 a 40 noites, 15 a 20 por classe):** permite cruzar estação, época do ano e níveis de atividade ionosférica em um desenho fatorial balanceado, reduzindo o erro-padrão das métricas no nível de noite para menos de 8 pontos percentuais.

Mais importante do que adicionar noites aleatórias é selecionar **4 perfis prioritários de noites** que quebrem diretamente os confundidores atuais:
1. **Noites mistas (transição na mesma noite):** noites em que a bolha surge ou se dissipa no meio da madrugada, fornecendo dezenas de quadros `Sem EPB` e dezenas de quadros `Com EPB` sob a mesmíssima câmera, mesma época e mesma atmosfera de fundo.
2. **Noites calmas no pico da temporada de bolhas (outubro a março em CA):** para desacoplar a época do ano e a densidade de fundo da presença de EPB.
3. **Noites com EPB fora de época (outono/inverno) ou com perturbações atípicas:** para desacoplar a ocorrência de depleções da alta densidade ionosférica sazonal de verão.
4. **Ambas as classes em todas as estações (`CP`, `BJL`, `CA` e inclusão de Santa Maria — `STR`):** para garantir invariância instrumental e desacoplar a iluminação urbana local de cada observatório.

---

## 8. Síntese Honesta e Roteiro de Próximos Passos
*(Correspondência narrativa: Slides 11, 12, 13 e 14)*

Chegamos assim à conclusão do trabalho com um balanço científico transparente (**Slide 11**). De um lado, construímos uma **metodologia sólida e auditável**: demonstramos que preservar o sinal negativo do contraste fracionário $\delta I/I$ junto com a remoção morfológica de estrelas e a estimativa de fundo Gaussiano restrita ao domo eleva a separabilidade das depleções para **0,950 de AUC-ROC e 0,962 de PR-AUC**; implementamos um classificador linear enxuto sobre 2DPCA ($k=3$, 192 características) que entrega **90,0% de acurácia balanceada, 88,2% de sensibilidade e 91,8% de especificidade** em validação *Leave-One-Night-Out* estrita no limiar operacional `t = 0,70`; derivamos mapas de **atribuição linear exata por pixel** sem aproximações estocásticas; descartamos o confundidor de inclinação geomagnética via tensor de estrutura; e diagnosticamos quantitativamente o confundidor de brilho noturno. De outro lado, declaramos abertamente as **limitações do estado atual**: o tamanho amostral de $N=7$ noites (apenas 2 sem EPB), a superioridade aparente do *baseline* de brilho (95,9%) causada por esse $N$ reduzido e pela pré-conversão `magick -normalize`, e a calibração *in-sample* do porteiro de nuvens que bloqueia de 13% a 19% de duas noites legítimas com EPB intensa.

Esses achados definem com precisão cirúrgica os **5 próximos passos** da pesquisa (**Slide 12**):
1. **Expansão direcionada para 30 a 40 noites e ativação do Teste Travado (`LOCKED_TEST_NIGHTS`):** priorizando noites mistas, calmas no verão, EPBs fora de época e balanceamento entre `CA`, `CP`, `BJL` e `STR`.
2. **Higiene das bordas e remoção de carimbos na geração das imagens:** remover no script de conversão do EMBRACE os carimbos de data/hora e marcações de orientação (N, S, E, O) próximos à borda do domo, ou aplicar máscara fixa antes da projeção 2DPCA.
3. **Controle explícito do brilho desde a imagem bruta:** substituindo a conversão prévia via ImageMagick (`-normalize -contrast-stretch`) pela leitura direta dos arquivos `.tif` originais em Python, normalizados por percentis calculados exclusivamente dentro do domo sem estrelas, além de testar projeção ortogonal às estatísticas de brilho.
4. **Revalidação cruzada do Estágio 1 (QC):** rotulando múltiplas noites nubladas e parcialmente degradadas para calibrar o porteiro de qualidade fora da amostra e eliminar os falsos bloqueios em noites de EPB intensa.
5. **Fusão multi-instrumento na rede EMBRACE:** cruzando a série temporal de probabilidades ópticas e o ângulo das faixas com dados co-localizados de cintilação GNSS ($S_4$ e ROTI) e perfis de digissondas, transformando o classificador óptico em uma ferramenta integrada para o monitoramento operacional do clima espacial no Brasil.

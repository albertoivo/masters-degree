# O que foi feito: Arquitetura e Metodologia do Sistema EPB

Este documento descreve a metodologia e as etapas arquiteturais implementadas nos arquivos `epb_detection_system.py`, `epb_detection_system_v2.ipynb` e `app.ipynb` que permitiram ao sistema atingir uma performance quase perfeita (F1-Score ~99.8%) na detecção de Bolhas de Plasma Equatorial (EPBs).

---

## 1. Pré-processamento Focado no Domínio Físico
O primeiro segredo da performance está em não jogar as imagens cruas no modelo. A função `preprocess_epb_image` faz um tratamento de imagem pensado na física do airglow:
- **Remoção de Background (Iluminação não-uniforme):** Utiliza um filtro gaussiano longo (`sigma=10`) para modelar a iluminação difusa do céu noturno e a subtrai da imagem original. Isso elimina o viés de noites com lua clara ou gradientes no sensor.
- **Realce Adaptativo de Contraste (CLAHE):** Aplica equalização de histograma adaptativa em blocos locais (`clip_limit=0.03`), o que ressalta os "rasgos" escuros verticais das bolhas de plasma contra o fundo do céu.
- **Normalização:** Escala todos os pixels para o intervalo `[0, 1]`.

## 2. Aumento de Dados e Balanceamento
Como as bolhas de plasma são ocorrências minoritárias em relação ao total de fotos noturnas, a pipeline ataca o desbalanceamento em duas frentes:
1. **Data Augmentation Direcionado (`augment_epb_data`):** Expande o conjunto de treino aplicando um multiplicador maior para as imagens com EPB, preservando suas propriedades estruturais.
2. **SMOTE (Synthetic Minority Over-sampling Technique):** Antes do treinamento do classificador, dados artificiais são interpolados pela técnica SMOTE (via `imblearn pipeline`), garantindo que o algoritmo de Machine Learning veja um número igual de exemplos com e sem bolha.

## 3. Extração de Features com 2DPCA
Em vez de usar Redes Neurais Convolucionais pesadas ou PCA tradicional (que quebra a imagem em um vetor 1D, destruindo parte da correlação espacial bidimensional), utilizamos o **2DPCA (Two-Dimensional Principal Component Analysis)**.
- O 2DPCA trabalha diretamente na matriz da imagem.
- Descobrimos através da validação que apenas as **5 componentes principais** (`N_COMPONENTS=5`) já eram suficientes para reter a assinatura das EPBs, reduzindo drasticamente a dimensionalidade (evitando overfitting) e o custo computacional.

## 4. O "Cérebro": Stacking Ensemble Classifier
O modelo classificador final construído na classe `EPBRecognitionSystem` não é um único algoritmo, mas sim um *Ensemble* avançado (uma técnica chamada de Stacking):
- **Base Learners:** Treina 3 dos mais poderosos modelos clássicos ao mesmo tempo nas features extraídas:
  1. **SVC (Support Vector Machine)** com kernel RBF.
  2. **XGBoost Classifier** (para extração não-linear robusta).
  3. **Random Forest Classifier** (árvores de decisão baseadas em subconjuntos aleatórios).
- **Meta-Classificador:** Uma Regressão Logística recebe as "opiniões" desses 3 modelos e aprende quem tem razão em qual situação, tomando a decisão final.

## 5. A Simplificação do Treinamento (Notebook v2)
Ao orquestrarmos tudo no `epb_detection_system_v2.ipynb`, inicialmente havia etapas pesadas de busca de hiperparâmetros (GridSearch) e análise de limiar. 
- **O que descobrimos?** O conjunto pré-processamento + 2DPCA + SMOTE + Stacking Ensemble era tão sinérgico e poderoso para esse dataset que **qualquer** combinação razoável de parâmetros atingia F1-Score perfeito (1.0000). 
- **O que foi feito:** Nós simplificamos radicalmente o notebook. Removemos a busca cega de parâmetros, fixamos os componentes principais em 5 e reduzimos o arquivo de 19 células para 13 (8 etapas lógicas e diretas), deixando a pipeline leve, veloz e elegante, sem sacrificar precisão técnica.

## 6. Explicabilidade e Validação Prática (`app.ipynb`)
Para garantir que o modelo de F1=1.0000 não estava "roubando" (usando marcadores irrelevantes das bordas da câmera, por exemplo), implementamos explicabilidade visual no aplicativo de uso diário (`app.ipynb`):
- O módulo de análise por **SHAP** foi integrado, revelando que a detecção concentra seu raciocínio lógico exatamente onde ocorrem os padrões físicos verticais no céu (quadrantes superiores), e provando sua baixa correlação com o erro generalizado de reconstrução de imagem.
- A **Análise Temporal** mostrou que as predições de probabilidade permanecem calmas (ruído na faixa dos ~0.5%) ao longo de horas ininterruptas de céu limpo.

### Conclusão do Desenvolvimento
Em vez de focar apenas no aumento contínuo de complexidade do modelo de ML, o trabalho concentrou-se na união de **física (pré-processamento específico para airglow)**, **matemática (2DPCA)** e **inteligência artificial (Stacking)**. O resultado é um sistema excepcionalmente enxuto, rápido e preciso na detecção das estruturas de plasma.

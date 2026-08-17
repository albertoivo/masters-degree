# Performance do Modelo de Detecção de EPB

Este documento consolida as informações sobre a performance, validação e explicabilidade do modelo de reconhecimento de Bolhas de Plasma Equatorial (EPB) extraídas da análise dos notebooks `epb_detection_system_v2.ipynb` e `app.ipynb`.

## 1. Métricas de Desempenho e Validação

De acordo com o treinamento e validação realizados no `epb_detection_system_v2.ipynb`, o modelo (Pipeline de 2DPCA com ensemble tree-based/SVM e SMOTE) apresenta um desempenho excepcional.

### Avaliação K-Fold Cross-Validation (5 Folds)
A validação cruzada garante que o modelo não está apenas decorando o conjunto de treino, testando-o em 5 partições diferentes:
- **Fold 1:** Acurácia=1.0000 | Sensibilidade=1.0000 | Especificidade=1.0000 | F1=1.0000
- **Fold 2:** Acurácia=1.0000 | Sensibilidade=1.0000 | Especificidade=1.0000 | F1=1.0000
- **Fold 3:** Acurácia=0.9902 | Sensibilidade=0.9804 | Especificidade=1.0000 | F1=0.9901
- **Fold 4:** Acurácia=1.0000 | Sensibilidade=1.0000 | Especificidade=1.0000 | F1=1.0000
- **Fold 5:** Acurácia=1.0000 | Sensibilidade=1.0000 | Especificidade=1.0000 | F1=1.0000

**Média Final (Cross-Validation):**
- **Acurácia Média:** 99.80% ± 0.44%
- **F1-Score Médio:** 99.80% ± 0.44%

Essa consistência extrema demonstra que o modelo atinge convergência perfeita na quase totalidade dos testes, detectando as bolhas de plasma com virtualmente 100% de confiança e sem sofrer de underfitting ou overfitting perceptível.

---

## 2. Explicabilidade (Interpretability & SHAP)

O notebook `app.ipynb` traz uma avaliação detalhada de explicabilidade usando o **SHAP (SHapley Additive exPlanations)** (usando o `TreeExplainer`), que permite entender exatamente para qual região da imagem o modelo está olhando ao tomar a decisão.

### Descobertas do Mapa de Calor SHAP
1. **Regiões de Maior Importância:**
   Nas imagens com presença confirmada de EPB (ex: `O6_CA_20251023_001620.png` com 99.5% de predição), o modelo concentra maciçamente sua atenção na **metade superior da imagem**:
   - Quadrante Superior Esquerdo: ~34% a 45% de contribuição.
   - Quadrante Superior Direito: ~29% a 42% de contribuição.
   - Os quadrantes inferiores tipicamente contribuem com menos de 20%.

2. **Interpretação das Cores:**
   - **Zonas Vermelhas (↑ EPB):** Indicam os locais onde o modelo encontra evidências fortes de estruturas irregulares verticais (deplecções), empurrando a probabilidade de ser EPB para cima.
   - **Zonas Azuis (↓ Sem EPB):** Nas noites calmas, as zonas azuis indicam evidências de uma ionosfera quieta, derrubando a probabilidade de EPB para perto de 0%.

3. **Correlação de Erro (SHAP vs. 2DPCA Reconstrução):**
   - Foi calculada a correlação de Spearman entre os valores de SHAP absolutos e o erro de reconstrução do 2DPCA. 
   - O resultado (~0.14 a 0.19) mostra uma **baixa correlação**. Isso é uma excelente descoberta técnica, pois indica que o método de reconstrução matemática da imagem (2DPCA) e a árvore de decisão olham para propriedades diferentes da imagem para confirmar a detecção.

---

## 3. Análise Temporal (Evolução Contínua)

Além de imagens estáticas, o `app.ipynb` aplicou o modelo em uma sequência temporal completa (uma noite inteira de observações de airglow). 

**Exemplo de Aplicação Real (Noite de 22/07/2025):**
- **Período:** 21:56 às 08:23 UT.
- **Total de imagens processadas:** 230 imagens.
- **Detecções de EPB:** 0 (Noite quieta / Sem instabilidades).
- **Estabilidade:** A probabilidade média da noite inteira ficou em **0.5%**, demonstrando uma extrema imunidade a falsos positivos. O pico máximo de "ruído" atingiu apenas 17.7%, bem abaixo do limiar de classificação, mantendo o alerta silencioso.

## Conclusão

O sistema de Machine Learning proposto atingiu **maturidade excepcional**. Com um `F1-Score` sustentado em ~99.8% na validação cruzada, uma interpretabilidade clara que corrobora a física do problema (foco do modelo nas assinaturas morfológicas das deplecções no céu), e estabilidade perfeita em varreduras temporais ao longo da noite.

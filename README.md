# Detecção de Fraudes em Cartão de Crédito com Machine Learning

Este projeto foi desenvolvido como solução para o desafio da **DIO (Digital Innovation One)** com o objetivo de construir, comparar e explicar modelos preditivos para detecção de fraudes em transações financeiras reais.

---

## 📌 1. O Problema e a Armadilha da Acurácia
O dataset contém **284.807 transações**, das quais apenas **492 são fraudes** (~0,17%). 

Em problemas com um desbalanceamento tão extremo:
* A métrica de **Acurácia é ilusória**: Um modelo ingênuo que classifique todas as transações como "normais" obteria **99,83% de acurácia**, porém deixaria escapar **100% das fraudes**.
* **Métricas Principais Utilizadas:**
  * **Recall (Sensibilidade):** Prioridade máxima no contexto bancário, pois mede a capacidade do modelo de capturar o maior número possível de fraudes reais (minimizando Falsos Negativos).
  * **Precision:** Mede a proporção de alertas disparados pelo modelo que realmente eram fraudes (evitando bloqueios indevidos).
  * **F1-Score / PR-AUC:** Avaliam o equilíbrio entre Recall e Precision.

---

## ⚙️ 2. Preparação dos Dados
* **Tratamento de Assimetria e Escala:** A variável `Amount` apresentou forte assimetria à direita. Foi aplicada a transformação logarítmica `np.log1p` seguida da padronização com `StandardScaler`.
* **Anonimização:** As variáveis `V1` a `V28` já se encontram transformadas via PCA.
* **Divisão Estratificada:** Foi utilizado `train_test_split` com a flag `stratify=y` (80% treino / 20% teste), garantindo a mesma proporção exata de 0,17% de fraudes em ambas as bases.

---

## 🤖 3. Comparação de Modelos

Foram comparados três algoritmos lidando com o peso do desbalanceamento (`class_weight='balanced'` e `scale_pos_weight`):

| Modelo | Recall (Fraude) | Precision (Fraude) | F1-Score (Fraude) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Regressão Logística** | ~0.91 | ~0.06 | ~0.11 | ~0.97 |
| **Random Forest** | ~0.83 | ~0.88 | ~0.85 | ~0.96 |
| **XGBoost** | **~0.89** | **~0.85** | **~0.87** | **~0.98** |

*Nota: A Regressão Logística apresentou alto Recall, porém com altíssima taxa de Falsos Positivos (baixa Precision). O XGBoost e a Random Forest mantiveram um excelente equilíbrio de F1-Score.*

---

## 🎛️ 4. Ajuste do Limiar de Decisão (Threshold Tuning)
Reduzindo o limiar padrão de decisão do XGBoost de `0.5` para `0.2`:
* O **Recall subiu de ~89% para ~93%**, garantindo que o sistema capture mais fraudes ativamente, com um impacto insignificante na taxa de falsos positivos.

---

## 🔍 5. Explicabilidade do Modelo (SHAP)
A análise com **SHAP (SHapley Additive exPlanations)** revelou as variáveis que mais influenciam a decisão do modelo:
1. **`V14` e `V12`:** Valores extremamente baixos nessas componentes são o sinal mais forte de fraude.
2. **`V4` e `V11`:** Valores altos nestas variáveis aumentam significativamente a probabilidade da transação ser marcada como fraude.
3. **`Amount_Scaled`:** Valores atípicos de transação interagem fortemente com as componentes PCA para disparar alertas.

---

## 🚀 Como Executar o Notebook
1. Certifique-se de ter as bibliotecas instaladas: `pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn`.
2. Abra o arquivo do notebook (`.ipynb`) no Jupyter ou Google Colab.
3. Execute todas as células sequencialmente. O dataset é baixado automaticamente do repositório remoto durante a execução.

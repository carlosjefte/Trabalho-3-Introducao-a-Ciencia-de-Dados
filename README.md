# Classificador de Crânios Egípcios

O **Classificador de Crânios Egípcios** é um projeto de aprendizado de máquina focado na classificação de crânios de diferentes períodos históricos do Egito Antigo. Utilizando redes neurais e técnicas de aumento de dados, o modelo é capaz de prever a qual era histórica um crânio pertence com base em suas medidas antropométricas.

## Sobre o Projeto

Neste projeto, aplicamos uma **rede neural multicamada** (MLP) para classificar crânios de cinco períodos distintos da história egípcia. O conjunto de dados contém medidas antropométricas de crânios de diferentes eras, e o modelo foi desenvolvido para identificar o período correto a partir dessas informações.

### Funcionalidades

- **Classificação de Crânios:** O modelo classifica crânios em cinco períodos históricos: Pré-dinástico Primitivo, Pré-dinástico Antigo, 12ª e 13ª Dinastias, Período Ptolemaico e Período Romano.
- **Data Augmentation:** O conjunto de dados é ampliado através da adição de ruído aos dados de treinamento para aumentar a robustez do modelo e melhorar a generalização.
- **Treinamento e Avaliação:** O modelo é treinado em um conjunto de dados pré-processado e validado com técnicas como Early Stopping e Redução de Taxa de Aprendizado para evitar overfitting.
- **Salvamento e Carregamento do Modelo:** O modelo treinado pode ser salvo e carregado para uso em previsões futuras.

### Tecnologias Utilizadas

- Python
- Pandas e NumPy para manipulação e processamento de dados
- TensorFlow e Keras para a construção e treinamento da rede neural
- Scikit-learn para pré-processamento dos dados
- Matplotlib e Seaborn para visualização de métricas de desempenho

## Como Usar

1. **Clone o Repositório:**
   ```bash
   git clone https://github.com/carlosjefte/Trabalho-3-Introducao-a-Ciencia-de-Dados.git
   ```
   
2. **Instale as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Treine o Modelo:**
   Utilize o script principal para treinar o modelo com o conjunto de dados de crânios:
   ```bash
   python MLP_Classifier.py
   ```

4. **Previsão de Novos Dados:**
   O modelo treinado pode ser testado para classificar novos crânios:
   ```bash
   python Clasifier_Inference.py
   ```
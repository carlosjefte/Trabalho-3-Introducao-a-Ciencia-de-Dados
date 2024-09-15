# Pokemaos Battle Predictor

**Pokemaos Battle Predictor** é um projeto inspirado no universo de Pokémon, mas com uma reviravolta divertida: em vez de batalhas tradicionais, os monstros de "Pokemaos" se enfrentam jogando pedra, papel ou tesoura! Este repositório contém um modelo de rede neural treinado para prever o vencedor dessas batalhas com base nos atributos dos Pokemaos.

## Sobre o Projeto

Neste projeto, utilizamos um MLPClassifier da biblioteca `scikit-learn` para treinar um modelo capaz de prever qual Pokemao ganhará em uma batalha de pedra, papel ou tesoura. Os dados foram processados e aumentados para melhorar a diversidade do conjunto de treinamento, e o modelo foi avaliado para garantir sua precisão.

### Funcionalidades

- **Previsão de Vencedor:** O modelo é capaz de prever o vencedor entre dois Pokemaos usando seus atributos como entrada.
- **Data Augmentation:** O conjunto de dados é aumentado através da adição de ruído para melhorar a robustez do modelo.
- **Criação de Pares de Batalha:** Gera pares de Pokemaos e define o vencedor com base nas regras do jogo.
- **Salvamento e Carregamento do Modelo:** O modelo treinado e o scaler são salvos para uso futuro em previsões.
- **Simulação de Batalhas Aleatórias:** Permite simular batalhas aleatórias usando Pokemaos do conjunto de teste.

### Tecnologias Utilizadas

- Python
- Pandas e NumPy para manipulação de dados
- Scikit-learn para o modelo de rede neural e pré-processamento
- Joblib para salvar e carregar o modelo treinado

## Como Usar

1. **Clone o Repositório:**
   ```bash
   git clone https://github.com/seu-usuario/pokemaos-battle-predictor.git
   ```
   
2. **Instale as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Treine o Modelo:**
   Use o script principal para treinar o modelo com o conjunto de dados de Pokemaos.

4. **Simule Batalhas:**
   Utilize as funções disponíveis para prever o vencedor entre dois Pokemaos ou para simular batalhas aleatórias.

## Contribuições

Contribuições são bem-vindas! Se você tem ideias para melhorar o modelo ou adicionar novas funcionalidades, sinta-se à vontade para abrir uma issue ou enviar um pull request.

---

Essa descrição fornece uma visão geral do projeto, explica como ele funciona e orienta os usuários sobre como começar a usar o repositório. Ela também abre espaço para contribuições, o que é sempre um bom incentivo para projetos de código aberto no GitHub!

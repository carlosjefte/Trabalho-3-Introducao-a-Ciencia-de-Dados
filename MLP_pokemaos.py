import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Carregar o dataset
df = pd.read_csv('Pokemaos.csv', sep=',', header=None)

# Data augmentation: duplicar e perturbar ligeiramente os dados para aumentar a variedade
def augment_data(data):
    augmented_data = data.copy()
    for _ in range(5):  # Número de vezes para replicar o conjunto original
        # Adicionar ruído normal pequeno para augmentation
        noise = np.random.normal(0, 0.05, data.shape)  # ruído pequeno
        augmented_data = np.vstack((augmented_data, data + noise))
    return augmented_data

# Separar atributos (X) e alvo (y)
X = df.iloc[:, :-1].values  # Atributos
y = df.iloc[:, -1].values   # Probabilidade de vitória

# Aplicar data augmentation
X = augment_data(X)
y = np.tile(y, 6)  # Repetir os alvos para corresponder à dimensão aumentada de X

# Criar pares de batalha (Pokemao1 vs Pokemao2) para treinar o modelo
def create_battle_pairs(X, y):
    pairs = []
    labels = []
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                pairs.append(np.hstack((X[i], X[j])))  # Concatenar os atributos de dois Pokemaos
                labels.append(1 if y[i] > y[j] else 0)  # 1 se Pokemao1 vence, 0 se Pokemao2 vence
    return np.array(pairs), np.array(labels)

# Criar pares de batalhas e os respectivos rótulos
X_pairs, y_pairs = create_battle_pairs(X, y)

# Dividir em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_pairs, y_pairs, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo MLP para classificação de batalhas
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=500, random_state=42)

# Treinar o modelo
mlp.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salvar o modelo treinado
joblib.dump(mlp, 'pokemao_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Salvar o scaler também para normalização futura

# Função para prever o vencedor entre dois Pokemaos
def predict_winner(pokemao1, pokemao2):
    # Carregar o modelo salvo
    model = joblib.load('pokemao_predictor.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Concatenar os atributos dos dois Pokemaos e normalizar
    battle_input = np.hstack((pokemao1, pokemao2)).reshape(1, -1)
    battle_input = scaler.transform(battle_input)
    
    # Prever o vencedor
    prediction = model.predict(battle_input)
    return "Pokemao 1 Vence" if prediction == 1 else "Pokemao 2 Vence"

# Função para selecionar dois Pokemaos aleatórios do conjunto de teste e prever o vencedor
def random_battle_from_test(X_test):
    # Escolher dois índices aleatórios do conjunto de teste
    idx1, idx2 = np.random.choice(len(X_test), 2, replace=False)
    # Extrair atributos dos dois Pokemões
    pokemao1 = X_test[idx1, :X_test.shape[1] // 2]  # Primeira metade dos atributos
    pokemao2 = X_test[idx2, X_test.shape[1] // 2:]  # Segunda metade dos atributos
    # Prever o vencedor usando a função predict_winner
    resultado = predict_winner(pokemao1, pokemao2)
    print(f"Pokemao 1: {pokemao1}, Pokemao 2: {pokemao2}")
    print(f"Resultado da Batalha: {resultado}")

# Exemplo de uso: Prever vencedor entre dois Pokemaos aleatórios do conjunto de teste
random_battle_from_test(X_test)
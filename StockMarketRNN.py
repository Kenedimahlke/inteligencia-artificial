# Bibliotecas
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Coletando dados do yfinance
ticker = 'AAPL'  # Ação Apple 
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
data = data[['Close']]  # preço de fechamento

# Pré-processamento dos Dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Criando Conjuntos de Treinamento e Teste
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 60  
X, y = create_dataset(scaled_data, look_back)

# Dividir em treinamento e teste
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


# Remodelar os dados para [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Definindo a Rede Neural
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(150, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(150))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando a Rede Neural

model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1)

# Testando e Avaliando o Modelo
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))

# Visualizando Gráfico
plt.figure(figsize=(14, 5))
plt.plot(data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='red', label='Valor Real')
plt.plot(data.index[-len(y_test):], predicted_stock_price, color='blue', label='Previsto')
plt.title('Previsão de Preço de Ações - LSTM', fontsize=16)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço da Ação', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Avaliando a Precisão (Erro Quadrático Médio)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predicted_stock_price)
print(f'Mean Squared Error (MSE): {mse}')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluation_metric
import config

# 1. Carregamento dos dados
data = pd.read_csv(f'./{config.DATASET_NAME}')
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.sort_index()

# Selecionamos apenas o preço de fechamento para a persistência
close_series = data['close']

# 2. Definição do índice de corte (Split)
split_idx = config.get_split_index(len(close_series))

# 3. Lógica da Persistência
# Para prever o valor de 'hoje', usamos o valor de 'ontem' (shift de 1)
# O valor real que queremos prever no set de teste:
y_true = close_series.iloc[split_idx:].values

# A previsão (Persistence): pegamos os valores imediatamente anteriores aos do set de teste
# Ou seja, o set de teste começa em split_idx, a previsão começa em split_idx - 1
y_pred = close_series.iloc[split_idx-1 : len(close_series)-1].values

# 4. Ajuste das datas para o gráfico
time = close_series.index[split_idx:]

# 5. Avaliação
print("--- Avaliação do Modelo de Persistência (Benchmark) ---")
evaluation_metric(y_true, y_pred)

# 6. Plotagem
plt.figure(figsize=(10, 6))
plt.plot(time, y_true, label='Real Close Price (t)', color='blue')
plt.plot(time, y_pred, label='Persistence Prediction (t-1)', color='red', linestyle='--')
plt.title('Persistence: Stock market close price prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Salvar resultados para comparação futura
# df_results = pd.DataFrame({'Date': time, 'True': y_true, 'Pred': y_pred})
# df_results.to_csv('./results/persistence_results.csv', index=False)
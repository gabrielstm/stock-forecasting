import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import walk_forward_validation
import config

data = pd.read_csv(f'./data/{config.DATASET_NAME}')
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.loc[:, ['open', 'high', 'low', 'close', 'volume', 'amount']]

residuals = pd.read_csv('./data/ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')

merge_data = pd.merge(data, residuals, left_index=True, right_index=True)

close_col = merge_data.pop('close')
residuos_col = merge_data.pop('0') 

merge_data.insert(len(merge_data.columns), residuos_col.name, residuos_col)

merge_data.insert(len(merge_data.columns), 'close', close_col)

split_idx = config.get_split_index(len(merge_data))
data1 = merge_data.iloc[split_idx:, len(merge_data.columns)-1]

time = pd.Series(data.index[split_idx:])

# n_test should be the length of the test set
n_test = len(data1)
train, test = prepare_data(merge_data, n_test=n_test, n_in=6, n_out=1)

y, yhat = walk_forward_validation(train, test)
evaluation_metric(y, yhat)
plt.figure(figsize=(10, 6))
plt.plot(time, y, label='Real Close Price')
plt.plot(time, yhat, label='Predicted Close Price')
plt.title('XGBoost: Close Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
# modelsemxgb.py
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import *
from model import *

data1 = pd.read_csv("./data/historico_b3_indicadores.csv")

data1.index = pd.to_datetime(data1['trade_date'], format='%Y%m%d')

data1 = data1.loc[:, ['open', 'high', 'low', 'close', 'volume', 'amount']]

TRAIN_LEN = 80
TEST_LEN = 20

split_idx = get_split_index(len(data1))
data = data1.iloc[1:split_idx, :] 
data2 = data1.iloc[split_idx:, :] 

TIME_STEPS = 20

data, normalize = NormalizeMult(data)
print('#', normalize)

yindex = data1.columns.get_loc('close')
pollution_data = data[:, yindex].reshape(len(data), 1)

train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

print(train_X.shape, train_Y.shape)

m = attention_model(INPUT_DIMS=6)
m.summary() 
adam = Adam(learning_rate=0.01)
m.compile(optimizer=adam, loss='mse') 
history = m.fit([train_X], train_Y, epochs=50, batch_size=32, validation_split=0.1)
m.save("./stock_model.h5")
np.save("stock_normalize.npy", normalize)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

class Config:
    def __init__(self):
        self.dimname = 'close'

config = Config()
name = config.dimname

y_hat, y_test = PredictWithData(data2, data1, name, 'stock_model.h5',6)


y_hat = np.array(y_hat, dtype='float64')
y_test = np.array(y_test, dtype='float64')
evaluation_metric(y_test,y_hat)

end_index = len(data1.index) 
start_index_for_plot = end_index - len(y_test) 
time = data1.index[start_index_for_plot : end_index]

plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
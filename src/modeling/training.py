from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
from matplotlib import pyplot
import numpy as np
import pickle
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
data_input_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input.pickle')

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, return_sequences=True, input_shape=(1, X.shape[2])))
	model.add(Dropout(0.2))
	model.add(LSTM(neurons, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(neurons))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def main():
	# load data object from file
	with open(data_input_path, 'rb') as f:
		data = pickle.load(f)
	
	# TEST - retrieve train scaled of one item
	key = ('xbox-one', 'xbox-one-x-1tb-console-project-scorpio-edition')
	scaler = data[key][0]
	train_scaled = data[key][1]
	test_scaled = data[key][2]
	raw_values = data[key][-1]

	# load in model architecture
	lstm_model = fit_lstm(train_scaled, 1, 1000, 1)
	
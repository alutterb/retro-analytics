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


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

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

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

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

	lstm_model = fit_lstm(train_scaled, 1, 1000, 1)
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)

	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
		expected = raw_values[len(train_scaled) + i + 1]
		print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance
	split = len(train_scaled)+1
	test_values = raw_values[split:]
	rmse_lstm = sqrt(mean_squared_error(test_values, predictions))
	niave_prediction = [np.average(test_values)]*len(test_values)
	rmse = sqrt(mean_squared_error(test_values, niave_prediction))
	# compare to simply taking mean and predicting
	print('Test LSTM RMSE: %.3f' % rmse_lstm)
	print('Test Niave RMSE: %.3f' % rmse)

if __name__ == "__main__":
	main()

from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pickle
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, '../../models/LSTM')
data_input_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input.pickle')

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

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

def main():
	# load training data and model from file
	with open(model_path, 'rb') as f:
		data = pickle.load(f)
	
    # load in relevent data objects
	lstm_model = data[0]
	train_scaled = data[1]
	test_scaled = data[2]
	scaler = data[3]
	raw_values = data[4]
	
    # collect predictions
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
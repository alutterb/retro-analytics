
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pickle
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, '../../models/LSTM/models.pickle')
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

def evaluate_model(model, test_scaled, raw_values, train_scaled, scaler):
    predictions = []
    for i in range(len(test_scaled)):
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(model, 1, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        predictions.append(yhat)
        expected = raw_values[len(train_scaled) + i + 1]
        print(f"Month={i+1}, Predicted={yhat:.2f}, Expected={expected:.2f}")

    return predictions

def calculate_rmse(predictions, raw_values, train_scaled):
    split = len(train_scaled) + 1
    test_values = raw_values[split:]
    rmse_lstm = sqrt(mean_squared_error(test_values, predictions))
    naive_prediction = [np.average(test_values)] * len(test_values)
    rmse_naive = sqrt(mean_squared_error(test_values, naive_prediction))
    return rmse_lstm, rmse_naive

def main():
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    print(data.values())
    return 0
    lstm_model, train_scaled, test_scaled, scaler, raw_values = data

    predictions = evaluate_model(lstm_model, test_scaled, raw_values, train_scaled, scaler)
    rmse_lstm, rmse_naive = calculate_rmse(predictions, raw_values, train_scaled)

    print(f'Test LSTM RMSE: {rmse_lstm:.3f}')
    print(f'Test Naive RMSE: {rmse_naive:.3f}')

if __name__ == "__main__":
      main()
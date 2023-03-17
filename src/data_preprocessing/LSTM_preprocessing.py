import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
import os
import pickle

script_dir = os.path.dirname(os.path.realpath(__file__))
historical_prices_path = os.path.join(script_dir, '../../data/preprocessed/scraped_data/historical prices.csv')
data_output_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input.pickle')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def clean_timeseries(timeseries):
	# Check that the input string is not empty
	if len(timeseries.strip()) == 0:
		raise ValueError("Input timeseries string is empty")
	
	# Check that the input string contains at least one comma-separated value
	if "," not in timeseries:
		raise ValueError("Input timeseries string is not properly formatted")
	
	# convert timeseries string to array
	timeseries = timeseries.strip()
	if 'nan' in timeseries:
		return False
	timeseries = eval(timeseries)
	return timeseries

# checks to see if most elements in the array are equivalent
# 1 = all values are unique
# approximately 0 = all values are the same
def check_threshold(arr, threshold):
	if arr:
		unique_vals = set(arr)
		percentage = len(unique_vals) / len(arr)
		return percentage >= threshold
	else:
		return None

def main():
	hist_prices_df = pd.read_csv(historical_prices_path)
	hist_prices_df = hist_prices_df.drop('Unnamed: 0', axis = 1)

	# clean historical prices
	hist_prices_df['historical prices'] = hist_prices_df['historical prices'].apply(clean_timeseries)

	raw_values = hist_prices_df['historical prices']
	names = list(zip(hist_prices_df['console'], hist_prices_df['game']))

	data = {}
	for name, timeseries in zip(names,raw_values):
		# if no missing values
		if timeseries:
			# if enough variation in data, let's preprocess for an LSTM
			if check_threshold(timeseries, 0.4):
				# make series stationary
				diff_vals = difference(timeseries)

				# convert to supervised learning problem
				supervised = timeseries_to_supervised(diff_vals)
				supervised_values = supervised.values

				# split into test and train, then scale
				split = int(len(supervised_values)*0.80)
				train, test = supervised_values[:split], supervised_values[split:]

				# transform the scale of the data
				scaler, train_scaled, test_scaled = scale(train, test)
				data[name] = (scaler, train_scaled, test_scaled, timeseries)

		# if values remain static, no need to preprocess for an LSTM, this will save time 
			else:
				data[name] = None
		else:
			data[name] = False
	
	with open(data_output_path, 'wb') as f:
		pickle.dump(data, f)

if __name__ == "__main__":
	main()
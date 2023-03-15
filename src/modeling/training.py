
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, LearningRateScheduler

import pickle
import os
import math

script_dir = os.path.dirname(os.path.realpath(__file__))
data_input_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input.pickle')
model_output_path = os.path.join(script_dir, '../../models/LSTM/models.pickle')


# Define a learning rate scheduler function for the model training
def step_decay(epoch):
	initial_lrate = 0.001  # Initial learning rate
	drop = 0.5  # Factor by which to decrease the learning rate
	epochs_drop = 50.0  # How often to decrease the learning rate (every 50 epochs)
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

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

	# Define early stopping to stop training if the validation loss doesn't improve for a certain number of epochs
	early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
	
	# Create a learning rate scheduler using the step_decay function
	lrate = LearningRateScheduler(step_decay)
	
	# Define a list of callbacks to use during training
	callbacks_list = [early_stopping, lrate]

 	# Fit the model using the training data, validation split, and the defined callbacks
	model.fit(X, y, validation_split=0.2, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False, callbacks=callbacks_list)
	
	# Reset the model states
	model.reset_states()

def main():
	# load data object from file
	with open(data_input_path, 'rb') as f:
		data = pickle.load(f)
	
	models = {}
	for key in data.keys():
		print("Training LSTM model for %s on %s" % (key[1], key[0]))
		# retrieve training data
		# key = (console, game_name)
		# structure: data[key] = (scaler, train_scaled, test_scaled, timeseries)
		# if data isn't missing
		if data[key]:
			train_scaled = data[key][1]
		else: # else continue to the next example
			continue

		# fit LSTM model
		lstm_model = fit_lstm(train_scaled, 1, 300, 4)

		# save to models object
		models[key] = lstm_model
	
	with open(model_output_path, 'wb') as f:
		pickle.dump(models, f)
	
if __name__ == "__main__":
	main()
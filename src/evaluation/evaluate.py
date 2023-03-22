import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, '../../models/ARIMA/models_ARIMA.pickle')
data_input_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input_ARIMA.pickle')
prediction_output_path = os.path.join(script_dir, '../../data/outputs/predictions.pickle')


def forecast(key_model_tuple, raw_values):
    key, model = key_model_tuple
    if model is None:
        return None
    print(f"Evaluating ARIMA model for {key[1]} on {key[0]}")
    predictions = model.forecast(steps=5)
    return predictions

def load_data(input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_models(model_path):
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models

def save_predictions(predictions, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)

def main():
    arima_models = load_models(model_path)
    input_data = load_data(data_input_path)

    data = {}
    for key, model in arima_models.items():
        raw_values = input_data[key][-1]
        if model is not None:
            predictions = forecast((key, model), raw_values)
            data[key] = predictions
        else:
            data[key] = None
    
    save_predictions(data, prediction_output_path)

if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, '../../models/ARIMA/models_ARIMA.pickle')
prediction_output_path = os.path.join(script_dir, '../../data/outputs/predictions.pickle')


def forecast(key_model_tuple):
    key, model = key_model_tuple
    if model is None:
        return []
    print(f"Evaluating ARIMA model for {key[1]} on {key[0]}")
    predictions = model.forecast(steps=5)
    return predictions


def load_models(model_path):
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models

def save_predictions(predictions, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)

def main():
    arima_models = load_models(model_path)
    data = {}
    for key, model in arima_models.items():
        if model is not None:
            predictions = forecast((key, model))
            data[key] = predictions
        else:
            data[key] = []
    
    save_predictions(data, prediction_output_path)

if __name__ == "__main__":
    main()

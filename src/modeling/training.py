import os
import pickle
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error

import itertools

script_dir = os.path.dirname(os.path.realpath(__file__))
data_input_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input_ARIMA.pickle')
model_output_path = os.path.join(script_dir, '../../models/ARIMA/models_ARIMA.pickle')

warnings.filterwarnings("ignore")  # Ignore ARIMA warnings for better readability

def find_best_order(timeseries, max_p=3, max_d=2, max_q=3):
    best_order = None
    best_aic = float("inf")

    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and d == 0 and q == 0:
            continue
        try:
            model = ARIMA(timeseries, order=(p, d, q))
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, d, q)
        except:
            continue
    return best_order

def train_arima(key_data_tuple):
    key, timeseries = key_data_tuple
    print(f"Training ARIMA model for {key[1]} on {key[0]}")

    if timeseries:
        best_order = find_best_order(timeseries)
        if best_order is not None:
            model = ARIMA(timeseries, order=best_order)
            model_fit = model.fit()
            return key, model_fit
        else:
            return key, None
    else:
        return key, None

def train_models(data, num_processes):
    models = {}
    with Pool(num_processes) as p:
        results = p.map(train_arima, data.items())

    for key, model_fit in results:
        if model_fit is not None:
            models[key] = model_fit
    return models

def load_data(input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_models(models, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(models, f)

def main():
    data = load_data(data_input_path)
    models = train_models(data, num_processes=1)
    save_models(models, model_output_path)

if __name__ == "__main__":
    main()

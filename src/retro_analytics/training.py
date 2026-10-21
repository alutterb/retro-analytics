import pickle
import warnings

from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import Pool

import itertools

from retro_analytics import paths

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

    try:
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
    except Exception as e:
        print(f"Error while training ARIMA model for {key[1]} on {key[0]}: {e}")
        return key, None


# parallel processing on ARIMA model training
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
    data = load_data(paths.arima_input())
    models = train_models(data, num_processes=3)
    save_models(models, paths.arima_models())

if __name__ == "__main__":
    main()

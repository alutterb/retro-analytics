import pickle

from retro_analytics import paths


def forecast(key_model_tuple, steps=5):
    key, model = key_model_tuple
    if model is None:
        return []
    print(f"Evaluating ARIMA model for {key[1]} on {key[0]}")
    predictions = model.forecast(steps=steps)
    return predictions


def forecast_all(models, steps=5):
    """Forecast `steps` points ahead for every fitted model, keyed like the input."""
    return {key: forecast((key, model), steps) for key, model in models.items()}


def load_models(model_path):
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models


def save_predictions(predictions, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)


def main():
    arima_models = load_models(paths.arima_models())
    save_predictions(forecast_all(arima_models), paths.predictions())


if __name__ == "__main__":
    main()

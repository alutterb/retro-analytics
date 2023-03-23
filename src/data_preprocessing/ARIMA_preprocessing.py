import pandas as pd
import os
import pickle

script_dir = os.path.dirname(os.path.realpath(__file__))
historical_prices_path = os.path.join(script_dir, '../../data/preprocessed/scraped_data/historical prices.csv')
data_output_path = os.path.join(script_dir, '../../data/preprocessed/model_input/preprocessed_input_ARIMA.pickle')


def clean_timeseries(timeseries):
    if len(timeseries.strip()) == 0:
        raise ValueError("Input timeseries string is empty")

    if "," not in timeseries:
        raise ValueError("Input timeseries string is not properly formatted")

    timeseries = timeseries.strip()
    if 'nan' in timeseries:
        return False
    timeseries = eval(timeseries)
    return timeseries


def main():
    hist_prices_df = pd.read_csv(historical_prices_path)
    hist_prices_df['historical prices'] = hist_prices_df['historical prices'].apply(clean_timeseries)

    raw_values = hist_prices_df['historical prices']
    names = list(zip(hist_prices_df['console'], hist_prices_df['game']))

    data = {}
    for name, timeseries in zip(names, raw_values):
        if timeseries:
            # In the case of ARIMA, we don't need to make series stationary or scale the data.
            # We'll store the original timeseries directly.
            data[name] = timeseries
        else:
            data[name] = None

    with open(data_output_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()

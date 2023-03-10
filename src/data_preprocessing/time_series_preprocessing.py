import pandas as pd
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
prices_path = os.path.join(script_dir, '../../data/raw/prices.csv')
output_path = os.path.join(script_dir, '../../data/preprocessed/historical prices.csv')


def main():
    prices_df = pd.read_csv(prices_path)
    grouped_df = prices_df.groupby(['console', 'game'])['price', 'date'].agg({'price': list, 'date': list}).reset_index()
    grouped_df = grouped_df.rename(columns={'price': 'historical prices', 'date': 'historical dates'})
    grouped_df.to_csv(output_path)


if __name__ == "__main__":
    main()
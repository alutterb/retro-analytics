import requests
from bs4 import BeautifulSoup
import re
import os
import signal

import csv
import pandas as pd

from datetime import date

class TimeoutException(Exception):
    pass

script_dir = os.path.dirname(os.path.realpath(__file__))
prices_path = os.path.join(script_dir, '../../data/raw/prices.csv')
consoles_path = os.path.join(script_dir, '../../data/raw/consoles.csv')

def timeout_handler(signum, frame):
    raise TimeoutException

def fetch_games(console_URL):
    '''
    Acquires URL links of 50 games for each specified console (NES, Xbox, etc..)

    INPUT: pricecharting url of a specific console, 
    e.g. https://www.pricecharting.com/console/sega-cd for the Sega CD

    OUTPUT: list of game urls
    e.g. https://www.pricecharting.com/game/sega-cd/keio-flying-squadron for Keio flying squadron
    '''
    page = requests.get(console_URL)
    if not page:
        print("NoneType returned - Page not found for %s" % console_URL)
        return None
    soup = BeautifulSoup(page.content, 'html.parser')

    game_urls = []
    table = soup.find('table', attrs={'id': 'games_table'})
    if table:
        for row in table.find_all('tr'):
            # Continue processing the rows in the table you're interested in
            link = row.find('a')
            if link:
                game_urls.append(link['href'])
    game_urls = [r'pricecharting.com' + href for href in game_urls]
    return game_urls

def return_price(game_URL):
    '''
    Acquires price of a particular game

    INPUT: URL to particular gane
    OUTPUT: price of game
    '''
    page = requests.get(game_URL)
    if not page:
        print("NoneType returned - Page not found for %s" % game_URL)
        return None
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="complete_price")
    if results:
        price = results.find("span", class_="price js-price")
        try:
            price_value = float(re.sub(r'[^\.a-zA-Z0-9]','', price.text))
        except ValueError:
            print("NA value returned")
            return None
        return price_value
    else:
        print("results returned NoneType - price not found for %s" % game_URL) 
        return None
    
def update_price_df():
    curr_df = pd.read_csv(prices_path)
    consoles = None
    with open(consoles_path, 'r') as file:
        reader = csv.reader(file)
        consoles = list(reader)[0]
    if consoles:
        signal.signal(signal.SIGALRM, timeout_handler)
        for console in consoles:
            URL = r'http://pricecharting.com/console/' + console
            game_urls = fetch_games(URL)
            for game_url in game_urls:
                game_url = r'http://' + game_url
                name = game_url.split('/')[-1]
                print("Beginning scraping for %s " % name)
                signal.alarm(60)
                try:
                    price = return_price(game_url)
                except (TimeoutException, Exception) as e:
                    print("Game fetch timed out, skipping")
                    continue
                else:
                    signal.alarm(0)
                print("Successfully scraped game")
                today = date.today()
                df_append = pd.DataFrame({"console": [console], "game" : [name], 
                "price" : [price], "date" : [today]})
                curr_df = pd.concat([curr_df, df_append], axis=0, ignore_index=True)
    else:
        print("Console data not found")
        return None
    
    curr_df.to_csv(prices_path, index=False)
    return True # success


def main():
    today = date.today()
    print("Running on:")
    print(today)
    update_price_df()
    
if __name__ == "__main__":
    main()


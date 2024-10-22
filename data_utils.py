import os
import time
from datetime import datetime
import logging
import pandas as pd
import yfinance as yf
from alpha_vantage.techindicators import TechIndicators
from fredapi import Fred
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)

class DataUtils:
    def __init__(self):
        # API Keys
        self.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')

        # init APIs
        if self.alpha_vantage_api_key:
            self.ta = TechIndicators(key=self.alpha_vantage_api_key, output_format='pandas')
        else:
            logging.warning("Alpha Vantage API key not found.")

        if self.fred_api_key:
            self.fred = Fred(api_key=self.fred_api_key)
        else:
            logging.warning("FRED API key not found.")

        # assets
        self.stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'] # Apple, Microsoft, Amazon, Google, Tesla
        self.etfs = ['SPY', 'QQQ', 'XLK', 'XLV', 'XLF'] # SP500, QQQ, Tech, Healthcare, Finance
        self.indices = ['^GSPC', '^IXIC', '^DJI', '^RUT']  # SP500, Nasdaq, Dow Jones, Russell 2000
        self.bonds = ['^TNX', 'IEF', 'TLT']  # 10 Year Treasury Note, iShares Treasury, iShares 20+ Treasury
        self.commodities = ['GC=F', 'CL=F', 'SI=F']  # Gold, Crude Oil, Silver
        self.reits = ['VNQ', 'SCHH', 'IYR']  # Real Estate ETFs (Vanguard, Schwab, iShares)

        # crypto
        self.cryptos = ['BTC-USD', 'ETH-USD', 'LTC-USD']  # Bitcoin, Ethereum, Litecoin

        # economic Indicators
        self.economic_indicators = {
            'GDP': 'GDP',            #Gross Domestic Product
            'CPI': 'CPIAUCSL',       #Consumer Price Index
            'FEDFUNDS': 'FEDFUNDS',  #Federal Interest Rate
            'UNRATE': 'UNRATE',      # Unemployment Rate
            'PPI': 'PPIACO'          # Producer Price Index
        }

        # data directories
        self.raw_data_path = os.path.join('data', 'raw')
        self.processed_data_path = os.path.join('data', 'processed')
        self.create_directories()

    def create_directories(self):
        directories = [
            self.raw_data_path,
            os.path.join(self.raw_data_path, 'yahoo'),
            os.path.join(self.raw_data_path, 'alpha_vantage'),
            os.path.join(self.raw_data_path, 'fred'),
            os.path.join(self.raw_data_path, 'crypto'),
            os.path.join(self.raw_data_path, 'reits'),
            self.processed_data_path
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Directory created or already exists: {directory}")

    def fetch_yahoo_data(self):
        tickers = self.stocks + self.etfs + self.indices + self.bonds + self.commodities + self.reits
        save_path = os.path.join(self.raw_data_path, 'yahoo')
        for ticker in tickers:
            try:
                logging.info(f"Fetching Yahoo Finance data for {ticker}")
                data = yf.download(ticker, start='2010-01-01', end='2023-12-31')
                if not data.empty:
                    file_path = os.path.join(save_path, f"{ticker}.csv")
                    data.to_csv(file_path)
                    logging.info(f"Saved Yahoo Finance data to {file_path}")
                else:
                    logging.warning(f"No Yahoo Finance data found for {ticker}")
            except Exception as e:
                logging.error(f"Error fetching Yahoo Finance data for {ticker}: {e}")

    def fetch_alpha_vantage_indicators(self):
        if not self.alpha_vantage_api_key:
            logging.error("Alpha Vantage API key is missing. Skipping technical indicators fetch.")
            return

        tickers = self.stocks + self.etfs

        indicators = ['SMA', 'EMA', 'RSI', 'ADX']  # Simple + Exponential Moving Averages, Relative Strength Index, Average Directional Index
        save_path = os.path.join(self.raw_data_path, 'alpha_vantage')

        for ticker in tickers:
            for indicator in indicators:
                try:
                    logging.info(f"Fetching {indicator} for {ticker} from Alpha Vantage")
                    if indicator == 'RSI':
                        data, meta_data = self.ta.get_rsi(symbol=ticker, interval='daily')
                    elif indicator == 'ADX':
                        data, meta_data = self.ta.get_adx(symbol=ticker, interval='daily')
                    else:
                        time_period = 20  # default time period
                        if indicator == 'SMA':
                            data, meta_data = self.ta.get_sma(symbol=ticker, interval='daily', time_period=time_period)
                        elif indicator == 'EMA':
                            data, meta_data = self.ta.get_ema(symbol=ticker, interval='daily', time_period=time_period)
                    if not data.empty:
                        file_path = os.path.join(save_path, f"{ticker}_{indicator}.csv")
                        data.to_csv(file_path)
                        logging.info(f"Saved {indicator} data to {file_path}")
                    else:
                        logging.warning(f"No {indicator} data found for {ticker}")
                except Exception as e:
                    logging.error(f"Error fetching {indicator} for {ticker}: {e}")
                finally:
                    time.sleep(12)  # Alpha Vantage rate limits

    def fetch_fred_data(self):
        if not self.fred_api_key:
            logging.error("FRED API key is missing.")
            return

        save_path = os.path.join(self.raw_data_path, 'fred')
        for name, series_id in self.economic_indicators.items():
            try:
                logging.info(f"Fetching {name} ({series_id}) from FRED")
                data = self.fred.get_series(series_id, observation_start='2010-01-01', observation_end='2023-12-31')
                if not data.empty:
                    df = pd.DataFrame(data, columns=[name])
                    df.to_csv(os.path.join(save_path, f"{series_id}.csv"))
                    logging.info(f"Saved FRED data for {name} to {series_id}.csv")
                else:
                    logging.warning(f"No FRED data found for {series_id}")
            except Exception as e:
                logging.error(f"Error fetching FRED data for {series_id}: {e}")

    def fetch_crypto_data(self):
        save_path = os.path.join(self.raw_data_path, 'crypto')
        for crypto in self.cryptos:
            try:
                logging.info(f"Fetching Yahoo Finance data for {crypto}")
                # historical market data from 2010-01-01 to today
                data = yf.download(crypto, start="2010-01-01", end=datetime.now().strftime('%Y-%m-%d'))
                
                if not data.empty:
                    # columns standardization
                    data = data.rename(columns={
                        'Open': 'Price_Open',
                        'High': 'Price_High',
                        'Low': 'Price_Low',
                        'Close': 'Price_Close',
                        'Volume': 'Volume'
                    })
                    
                    data = data[['Price_Open', 'Price_High', 'Price_Low', 'Price_Close', 'Volume']]
                    
                    file_path = os.path.join(save_path, f"{crypto.replace('-USD', '')}.csv")
                    data.to_csv(file_path)
                    logging.info(f"Saved crypto data to {file_path}")
                else:
                    logging.warning(f"No Yahoo Finance data found for {crypto}")
            except Exception as e:
                logging.error(f"Error fetching Yahoo Finance data for {crypto}: {e}")

    def fetch_reits_data(self):
        reits = self.reits
        save_path = os.path.join(self.raw_data_path, 'reits')
        for reit in reits:
            try:
                logging.info(f"Fetching REIT data for {reit} from Yahoo Finance")
                data = yf.download(reit, start='2010-01-01', end='2023-12-31')
                if not data.empty:
                    file_path = os.path.join(save_path, f"{reit}.csv")
                    data.to_csv(file_path)
                    logging.info(f"Saved REIT data to {file_path}")
                else:
                    logging.warning(f"No REIT data found for {reit}")
            except Exception as e:
                logging.error(f"Error fetching REIT data for {reit}: {e}")

    def fetch_all_data(self):
        logging.info("Starting data fetching process.")
        self.fetch_yahoo_data()
        self.fetch_alpha_vantage_indicators()
        self.fetch_fred_data()
        self.fetch_crypto_data()
        self.fetch_reits_data()
        logging.info("Data fetching process completed.")

    def run(self):
        self.fetch_all_data()


if __name__ == "__main__":
    data_utils = DataUtils()
    data_utils.run()
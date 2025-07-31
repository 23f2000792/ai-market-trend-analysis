"""
Data collection module for fetching stock market data.
This module handles downloading historical stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import List, Dict, Optional
from .config import STOCK_SYMBOLS, START_DATE, END_DATE, RAW_DATA_DIR


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    A class to collect and manage stock market data from Yahoo Finance.
    """

    def __init__(self, symbols: List[str] = None, start_date: str = None, end_date: str = None):
        """
        Initialize the data collector.

        Args:
            symbols: List of stock symbols to collect data for
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        self.symbols = symbols or STOCK_SYMBOLS
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE

        # Ensure raw data directory exists
        os.makedirs(RAW_DATA_DIR, exist_ok=True)

    def download_stock_data(self, symbol: str) -> pd.DataFrame:
        try:
            logger.info(f"Downloading data for {symbol}...")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            data.columns = data.columns.str.lower().str.replace(' ', '_')

            # Add symbol column
            data['symbol'] = symbol

            # Reset index to make Date a column
            data.reset_index(inplace=True)
            data.rename(columns={'Date': 'date'}, inplace=True)
            data['date'] = pd.to_datetime(data['date'])

            logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return pd.DataFrame()


    def download_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Download data for all configured stock symbols.

        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        all_data = {}

        for symbol in self.symbols:
            data = self.download_stock_data(symbol)
            if not data.empty:
                all_data[symbol] = data

                # Save individual stock data
                filename = os.path.join(RAW_DATA_DIR, f"{symbol}_raw.csv")
                data.to_csv(filename, index=False)
                logger.info(f"Saved {symbol} data to {filename}")

        return all_data

    def combine_stock_data(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all stock data into a single DataFrame.

        Args:
            stock_data: Dictionary of stock DataFrames

        Returns:
            Combined DataFrame with all stock data
        """
        if not stock_data:
            logger.warning("No stock data to combine")
            return pd.DataFrame()

        # Combine all DataFrames
        combined_data = pd.concat(stock_data.values(), ignore_index=True)

        # Sort by symbol and date
        combined_data = combined_data.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"Combined data shape: {combined_data.shape}")
        return combined_data

    def save_combined_data(self, combined_data: pd.DataFrame) -> str:
        """
        Save combined stock data to CSV file.

        Args:
            combined_data: Combined DataFrame

        Returns:
            Path to saved file
        """
        filename = os.path.join(RAW_DATA_DIR, "combined_stock_data.csv")
        combined_data.to_csv(filename, index=False)
        logger.info(f"Saved combined data to {filename}")
        return filename

    def collect_all_data(self) -> pd.DataFrame:
        """
        Main method to collect all stock data.

        Returns:
            Combined DataFrame with all stock data
        """
        logger.info("Starting data collection process...")

        # Download all stock data
        stock_data = self.download_all_stocks()

        if not stock_data:
            logger.error("No data was successfully downloaded")
            return pd.DataFrame()

        # Combine all data
        combined_data = self.combine_stock_data(stock_data)

        # Save combined data
        self.save_combined_data(combined_data)

        logger.info("Data collection process completed successfully!")
        return combined_data

    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get additional information about a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {str(e)}")
            return {'symbol': symbol, 'error': str(e)}

def main():
    """
    Main function to run data collection.
    """
    collector = StockDataCollector()
    data = collector.collect_all_data()

    if not data.empty:
        print(f"\nData collection completed successfully!")
        print(f"Total records: {len(data)}")
        print(f"Date range: {data['date'].min()} to {data['date'].max()}")
        print(f"Symbols: {data['symbol'].unique().tolist()}")
        print(f"\nFirst few records:")
        print(data.head())
    else:
        print("Data collection failed!")

if __name__ == "__main__":
    main()
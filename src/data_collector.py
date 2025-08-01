import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class StockDataCollector:
    """
    A simple class to collect stock market data.

    Think of this as your data gathering assistant that:
    1. Downloads stock data from Yahoo Finance
    2. Cleans and organizes the data
    3. Saves it in a format ready for analysis
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize the data collector.

        Args:
            symbols: List of stock symbols to track (default: big tech stocks)
        """
        # Default to major tech stocks if none provided
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        self.data = None

        print(f"📊 StockDataCollector initialized with {len(self.symbols)} symbols:")
        print(f"   {', '.join(self.symbols)}")

    def fetch_stock_data(self, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data for all symbols.

        Args:
            period: How far back to get data ('1y', '2y', '5y', 'max')
            interval: Data frequency ('1d', '1wk', '1mo')

        Returns:
            Combined DataFrame with all stock data
        """
        print(f"\n🔄 Fetching {period} of {interval} data...")

        all_data = []

        for symbol in self.symbols:
            try:
                print(f"   📈 Getting data for {symbol}...")

                # Download data from Yahoo Finance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)

                if data.empty:
                    print(f"   ⚠️  No data found for {symbol}")
                    continue

                # Add symbol column and reset index
                data = data.reset_index()
                data['Symbol'] = symbol

                # Standardize column names
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'Symbol']

                # Keep only essential columns
                data = data[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]

                # Add some basic calculations
                data['Daily_Return'] = data['Close'].pct_change()
                data['Price_Range'] = data['High'] - data['Low']
                data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()

                all_data.append(data)
                print(f"   ✅ Got {len(data)} days of data for {symbol}")

            except Exception as e:
                print(f"   ❌ Error fetching {symbol}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data was successfully fetched for any symbol!")

        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)

        # Sort by date and symbol
        self.data = self.data.sort_values(['Date', 'Symbol']).reset_index(drop=True)

        print(f"\n✅ Successfully fetched data:")
        print(f"   📅 Date range: {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
        print(f"   📊 Total rows: {len(self.data):,}")
        print(f"   🏢 Symbols: {self.data['Symbol'].nunique()}")

        return self.data

    def get_latest_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get the most recent data for quick analysis.

        Args:
            days: Number of recent days to return

        Returns:
            DataFrame with recent data
        """
        if self.data is None:
            print("⚠️  No data available. Run fetch_stock_data() first!")
            return pd.DataFrame()

        cutoff_date = self.data['Date'].max() - timedelta(days=days)
        recent_data = self.data[self.data['Date'] >= cutoff_date].copy()

        print(f"📊 Retrieved last {days} days of data ({len(recent_data)} rows)")
        return recent_data

    def save_data(self, filepath: str = "data/raw/stock_data.csv") -> None:
        """
        Save the collected data to a CSV file.

        Args:
            filepath: Where to save the file
        """
        if self.data is None:
            print("⚠️  No data to save. Run fetch_stock_data() first!")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to CSV
        self.data.to_csv(filepath, index=False)

        print(f"\n💾 Data saved successfully!")
        print(f"   📁 Location: {filepath}")
        print(f"   📊 Rows: {len(self.data):,}")
        print(f"   💽 File size: {os.path.getsize(filepath)/1024:.1f} KB")

    def get_data_summary(self) -> None:
        """
        Print a helpful summary of the collected data.
        """
        if self.data is None:
            print("⚠️  No data available. Run fetch_stock_data() first!")
            return

        print("\n" + "="*60)
        print("📊 DATA SUMMARY")
        print("="*60)

        print(f"📅 Date Range: {self.data['Date'].min().date()} to {self.data['Date'].max().date()}")
        print(f"📈 Symbols: {', '.join(sorted(self.data['Symbol'].unique()))}")
        print(f"📊 Total Records: {len(self.data):,}")
        print(f"🗓️  Trading Days: {self.data['Date'].nunique():,}")

        print("\n📈 Price Statistics (Last Close Price):")
        latest_prices = self.data.groupby('Symbol')['Close'].last().sort_values(ascending=False)
        for symbol, price in latest_prices.items():
            print(f"   {symbol}: ${price:.2f}")

        print("\n📊 Data Quality Check:")
        missing_data = self.data.isnull().sum()
        if missing_data.sum() == 0:
            print("   ✅ No missing values found!")
        else:
            print("   ⚠️  Missing values detected:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"      {col}: {count} missing")


# Example usage and testing function
def main():
    """
    Example of how to use the StockDataCollector.
    Run this to test the data collection!
    """
    print("🚀 AI Market Trend Analysis - Data Collection Demo")
    print("=" * 55)

    # Initialize collector with default stocks
    collector = StockDataCollector()

    # Fetch 2 years of daily data (faster for testing)
    data = collector.fetch_stock_data(period="5y", interval="1d")

    # Show summary
    collector.get_data_summary()

    # Save the data
    collector.save_data("data/raw/stock_data.csv")

    # Show a preview of the data
    print("\n📋 Data Preview (First 10 rows):")
    print(data.head(10))

    print("\n🎉 Data collection complete! Ready for feature engineering.")


if __name__ == "__main__":
    main()
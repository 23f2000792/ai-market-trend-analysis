"""
Feature engineering module for creating technical indicators and market features.
This module transforms raw OHLCV data into meaningful features for machine learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .config import TECHNICAL_INDICATORS, PRICE_CHANGE_THRESHOLD

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    A class to engineer features from raw stock market data.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the feature engineer.

        Args:
            config: Configuration dictionary for technical indicators
        """
        self.config = config or TECHNICAL_INDICATORS

    def calculate_sma(self, data: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.

        Args:
            data: DataFrame with stock data
            windows: List of window sizes for SMA calculation

        Returns:
            DataFrame with SMA columns added
        """
        windows = windows or self.config.get("SMA_WINDOWS", [20, 50, 200])

        for window in windows:
            data[f'sma_{window}'] = data['close'].rolling(window=window).mean()

        return data

    def calculate_ema(self, data: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.

        Args:
            data: DataFrame with stock data  
            windows: List of window sizes for EMA calculation

        Returns:
            DataFrame with EMA columns added
        """
        windows = windows or self.config.get("EMA_WINDOWS", [12, 26])

        for window in windows:
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()

        return data

    def calculate_rsi(self, data: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.

        Args:
            data: DataFrame with stock data
            window: Window size for RSI calculation

        Returns:
            DataFrame with RSI column added
        """
        window = window or self.config.get("RSI_WINDOW", 14)

        # Calculate price changes
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        return data

    def calculate_macd(self, data: pd.DataFrame, 
                      fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        """
        Calculate MACD indicators.

        Args:
            data: DataFrame with stock data
            fast: Fast EMA period
            slow: Slow EMA period  
            signal: Signal line EMA period

        Returns:
            DataFrame with MACD columns added
        """
        fast = fast or self.config.get("MACD_FAST", 12)
        slow = slow or self.config.get("MACD_SLOW", 26)
        signal = signal or self.config.get("MACD_SIGNAL", 9)

        # Calculate MACD line
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        data['macd'] = ema_fast - ema_slow

        # Calculate signal line
        data['macd_signal'] = data['macd'].ewm(span=signal).mean()

        # Calculate histogram
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        return data

    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = None, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            data: DataFrame with stock data
            window: Window size for calculation
            std_dev: Number of standard deviations

        Returns:
            DataFrame with Bollinger Bands columns added
        """
        window = window or self.config.get("BOLLINGER_WINDOW", 20)

        # Middle band (SMA)
        data['bb_middle'] = data['close'].rolling(window=window).mean()

        # Calculate standard deviation
        std = data['close'].rolling(window=window).std()

        # Upper and lower bands
        data['bb_upper'] = data['bb_middle'] + (std * std_dev)
        data['bb_lower'] = data['bb_middle'] - (std * std_dev)

        # Bollinger Band Width and %B
        data['bb_width'] = data['bb_upper'] - data['bb_lower']
        data['bb_percent'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

        return data

    def calculate_atr(self, data: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """
        Calculate Average True Range.

        Args:
            data: DataFrame with stock data
            window: Window size for ATR calculation

        Returns:
            DataFrame with ATR column added
        """
        window = window or self.config.get("ATR_WINDOW", 14)

        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['atr'] = true_range.rolling(window=window).mean()

        return data

    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators.

        Args:
            data: DataFrame with stock data

        Returns:
            DataFrame with volume indicators added
        """
        # Volume Moving Average
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()

        # Volume Rate of Change
        data['volume_roc'] = data['volume'].pct_change(periods=10)

        # On-Balance Volume
        obv = []
        obv.append(0)

        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        data['obv'] = obv

        return data

    def calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.

        Args:
            data: DataFrame with stock data

        Returns:
            DataFrame with price features added
        """
        # Price changes
        data['price_change'] = data['close'].diff()
        data['price_change_pct'] = data['close'].pct_change()

        # High-Low spread
        data['hl_spread'] = data['high'] - data['low']
        data['hl_spread_pct'] = (data['high'] - data['low']) / data['close']

        # Opening gap
        data['open_gap'] = data['open'] - data['close'].shift()
        data['open_gap_pct'] = data['open_gap'] / data['close'].shift()

        # Intraday return
        data['intraday_return'] = (data['close'] - data['open']) / data['open']

        return data

    def create_target_variable(self, data: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Create target variable for classification.

        Args:
            data: DataFrame with stock data
            threshold: Threshold for Up/Down classification

        Returns:
            DataFrame with target variable added
        """
        threshold = threshold or PRICE_CHANGE_THRESHOLD

        # Next day price change
        data['next_day_return'] = data['close'].shift(-1) / data['close'] - 1

        # Classification target
        conditions = [
            data['next_day_return'] > threshold,
            data['next_day_return'] < -threshold
        ]
        choices = ['Up', 'Down']

        data['price_direction'] = np.select(conditions, choices, default='Stable')

        return data

    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering functions.

        Args:
            data: DataFrame with raw stock data

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering process...")

        # Sort data by symbol and date
        data = data.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Process each symbol separately to avoid mixing indicators
        processed_data = []

        for symbol in data['symbol'].unique():
            logger.info(f"Engineering features for {symbol}...")

            symbol_data = data[data['symbol'] == symbol].copy()

            # Calculate all technical indicators
            symbol_data = self.calculate_sma(symbol_data)
            symbol_data = self.calculate_ema(symbol_data)
            symbol_data = self.calculate_rsi(symbol_data)
            symbol_data = self.calculate_macd(symbol_data)
            symbol_data = self.calculate_bollinger_bands(symbol_data)
            symbol_data = self.calculate_atr(symbol_data)
            symbol_data = self.calculate_volume_indicators(symbol_data)
            symbol_data = self.calculate_price_features(symbol_data)
            symbol_data = self.create_target_variable(symbol_data)

            processed_data.append(symbol_data)

        # Combine all processed data
        final_data = pd.concat(processed_data, ignore_index=True)

        logger.info(f"Feature engineering completed. Final shape: {final_data.shape}")
        logger.info(f"Features created: {len([col for col in final_data.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']])}")

        return final_data

    def get_feature_list(self) -> List[str]:
        """
        Get list of all engineered features.

        Returns:
            List of feature names
        """
        base_features = ['open', 'high', 'low', 'close', 'volume']

        engineered_features = []

        # SMA features
        for window in self.config.get("SMA_WINDOWS", [20, 50, 200]):
            engineered_features.append(f'sma_{window}')

        # EMA features  
        for window in self.config.get("EMA_WINDOWS", [12, 26]):
            engineered_features.append(f'ema_{window}')

        # Other technical indicators
        engineered_features.extend([
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_percent',
            'atr', 'volume_sma_20', 'volume_roc', 'obv',
            'price_change', 'price_change_pct', 'hl_spread', 'hl_spread_pct',
            'open_gap', 'open_gap_pct', 'intraday_return'
        ])

        return base_features + engineered_features

def main():
    """
    Main function to test feature engineering.
    """
    # This would typically load data from the data collector
    print("Feature engineering module loaded successfully!")

    # Create a sample feature engineer
    fe = FeatureEngineer()
    feature_list = fe.get_feature_list()

    print(f"\nTotal features available: {len(feature_list)}")
    print("\nFeature categories:")
    print("- Base OHLCV features: 5")
    print("- Technical indicators: ~20")
    print("- Price features: ~7")
    print("- Volume features: ~3")

if __name__ == "__main__":
    main()
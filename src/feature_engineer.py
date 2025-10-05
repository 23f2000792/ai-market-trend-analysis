import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import technical analysis library
try:
    import pandas_ta as ta
    TA_AVAILABLE = True
    print("📊 Using pandas_ta for technical indicators")
except ImportError:
    TA_AVAILABLE = False
    print("⚠️  pandas_ta not available, using manual calculations")


class FeatureEngineer:
    """
    A comprehensive feature engineering class that transforms stock data
    into machine learning ready features.

    What this does:
    1. Adds moving averages (simple and exponential)
    2. Calculates momentum indicators (RSI, ROC)
    3. Adds volatility measures (Bollinger Bands, ATR)
    4. Creates volume indicators
    5. Generates target labels for prediction
    """

    def __init__(self):
        """Initialize the feature engineer."""
        print("🔧 FeatureEngineer initialized and ready!")
        print("   Will create 40+ technical indicators for each stock")

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add various moving averages to the data.

        Moving averages help identify trends:
        - Short MA > Long MA = Uptrend
        - Short MA < Long MA = Downtrend
        """
        print("   📈 Adding moving averages...")

        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # Exponential Moving Averages (more weight on recent prices)
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()

        # Moving average ratios (trend strength indicators)
        df['SMA_Ratio_5_20'] = df['SMA_5'] / df['SMA_20']
        df['SMA_Ratio_20_50'] = df['SMA_20'] / df['SMA_50']
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators that show the speed of price changes.

        These help identify overbought/oversold conditions.
        """
        print("   🚀 Adding momentum indicators...")

        # RSI (Relative Strength Index) - measures overbought/oversold
        if TA_AVAILABLE:
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
        else:
            df['RSI_14'] = self._calculate_rsi(df['Close'], window=14)

        # Rate of Change (price momentum)
        df['ROC_5'] = df['Close'].pct_change(5) * 100
        df['ROC_10'] = df['Close'].pct_change(10) * 100
        df['ROC_20'] = df['Close'].pct_change(20) * 100

        # MACD (Moving Average Convergence Divergence)
        if TA_AVAILABLE:
            macd_data = ta.macd(df['Close'])
            df['MACD'] = macd_data['MACD_12_26_9']
            df['MACD_Signal'] = macd_data['MACDs_12_26_9']
            df['MACD_Histogram'] = macd_data['MACDh_12_26_9']
        else:
            macd_line = df['EMA_12'] - df['EMA_26']
            df['MACD'] = macd_line
            df['MACD_Signal'] = macd_line.ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Stochastic Oscillator
        if TA_AVAILABLE:
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']
        else:
            df['Stoch_K'] = self._calculate_stochastic(df, window=14)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators that measure price variability.

        High volatility = risky, Low volatility = stable
        """
        print("   📊 Adding volatility indicators...")

        # Bollinger Bands (price channels based on volatility)
        if TA_AVAILABLE:
            bb = ta.bbands(df['Close'], length=20, std=2.0)
            # Handle possible column name variations across versions
            possible_cols = list(bb.columns)

            upper = next((c for c in possible_cols if "BBU" in c), None)
            middle = next((c for c in possible_cols if "BBM" in c), None)
            lower = next((c for c in possible_cols if "BBL" in c), None)

            df["BB_Upper"] = bb[upper]
            df["BB_Middle"] = bb[middle]
            df["BB_Lower"] = bb[lower]

        else:
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Middle'] = sma_20
            df['BB_Lower'] = sma_20 - (std_20 * 2)

        # Bollinger Band indicators
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Average True Range (volatility measure)
        if TA_AVAILABLE:
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        else:
            df['ATR'] = self._calculate_atr(df, window=14)

        # Historical volatility
        df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std() * np.sqrt(252) * 100
        df['Volatility_30'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252) * 100

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.

        Volume confirms price movements:
        - High volume + price up = strong uptrend
        - Low volume + price up = weak uptrend
        """
        print("   📦 Adding volume indicators...")

        # Volume moving averages
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()

        # Volume ratios
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Volume_Rate_Change'] = df['Volume'].pct_change(5)

        # On Balance Volume (cumulative volume based on price direction)
        if TA_AVAILABLE:
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
        else:
            df['OBV'] = self._calculate_obv(df['Close'], df['Volume'])

        # Volume Price Trend
        if TA_AVAILABLE:
            try:
                if hasattr(ta, "volume_price_trend"):
                    df["VPT"] = ta.volume_price_trend(df["Close"], df["Volume"])
                else:
                    df["VPT"] = (df["Volume"] * (df["Close"].pct_change())).cumsum()
            except Exception as e:
                print("Warning: VPT not available, computing manually.")
                df["VPT"] = (df["Volume"] * (df["Close"].pct_change())).cumsum()

        else:
            df['VPT'] = (df['Close'].pct_change() * df['Volume']).cumsum()

        # Price Volume Trend
        df['PVT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()

        return df

    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price action and pattern features.
        """
        print("   💹 Adding price action features...")

        # Daily price changes
        df['Daily_Return'] = df['Close'].pct_change()
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Price position within daily range
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

        # Consecutive up/down days
        df['Up_Down'] = np.where(df['Daily_Return'] > 0, 1, -1)
        df['Consecutive_Days'] = df.groupby((df['Up_Down'] != df['Up_Down'].shift()).cumsum())['Up_Down'].cumsum()

        # Support and resistance levels (simplified)
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['Distance_to_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_to_Low'] = (df['Close'] - df['Low_20']) / df['Close']

        return df

    def create_target_variable(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Create the target variable for prediction.

        Args:
            threshold: Minimum change to consider as Up/Down (default 1%)

        Creates three classes:
        - 0: Down (price drops > threshold)
        - 1: Stable (price change within threshold)
        - 2: Up (price rises > threshold)
        """
        print(f"   🎯 Creating target variable (threshold: {threshold*100:.1f}%)...")

        # Calculate next day's return
        df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)

        # Create target classes
        df['Target'] = np.where(
            df['Next_Day_Return'] > threshold, 2,  # Up
            np.where(df['Next_Day_Return'] < -threshold, 0, 1)  # Down or Stable
        )

        # Create binary target too (for simpler models)
        df['Target_Binary'] = np.where(df['Next_Day_Return'] > 0, 1, 0)

        # Show target distribution
        target_dist = df['Target'].value_counts().sort_index()
        print(f"      Target distribution:")
        print(f"      📉 Down (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"      📊 Stable (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
        print(f"      📈 Up (2): {target_dist.get(2, 0):,} ({target_dist.get(2, 0)/len(df)*100:.1f}%)")

        return df

    def process_stock_data(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Args:
            df: Raw stock data from data_collector
            threshold: Target classification threshold

        Returns:
            Fully featured dataset ready for machine learning
        """
        print("\n🔧 Starting feature engineering pipeline...")
        print("=" * 50)

        # Work with a copy to avoid modifying original data
        df_processed = df.copy()

        # Sort by symbol and date to ensure proper time series order
        df_processed = df_processed.sort_values(['Symbol', 'Date']).reset_index(drop=True)

        print(f"📊 Processing {len(df_processed):,} rows for {df_processed['Symbol'].nunique()} symbols...")

        # Process each symbol separately to maintain time series integrity
        processed_dfs = []

        for symbol in df_processed['Symbol'].unique():
            print(f"\n🏢 Processing {symbol}...")
            symbol_df = df_processed[df_processed['Symbol'] == symbol].copy()

            # Check if there’s enough data for feature engineering (e.g. SMA_200 needs 200 days)
            if len(symbol_df) < 50:
                print(f"⚠️ Skipping {symbol} due to very limited data ({len(symbol_df)} rows). Need at least 50.")
                continue


            # Add all features
            symbol_df = self.add_moving_averages(symbol_df)
            symbol_df = self.add_momentum_indicators(symbol_df)
            symbol_df = self.add_volatility_indicators(symbol_df)
            symbol_df = self.add_volume_indicators(symbol_df)
            symbol_df = self.add_price_action_features(symbol_df)
            symbol_df = self.create_target_variable(symbol_df, threshold)

            processed_dfs.append(symbol_df)

        # Combine all processed data
        final_df = pd.concat(processed_dfs, ignore_index=True)

        if not processed_dfs:
            raise ValueError("❌ No data was processed. Try lowering the data length requirement or check input.")


        # Remove rows with missing values (mainly from initial periods)
        initial_rows = len(final_df)
        final_df = final_df.dropna(thresh=int(0.9 * df.shape[1]))
        final_rows = len(final_df)

        print(f"\n✅ Feature engineering complete!")
        print(f"   📊 Final dataset: {final_rows:,} rows ({initial_rows - final_rows:,} removed due to NaN)")
        print(f"   🔧 Features created: {len(final_df.columns) - len(df.columns)} new features")
        print(f"   📈 Total features: {len(final_df.columns)}")

        return final_df

    # Helper methods for manual calculations (when pandas_ta is not available)
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index manually."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Stochastic %K manually."""
        lowest_low = df['Low'].rolling(window=window).min()
        highest_high = df['High'].rolling(window=window).max()
        return 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range manually."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=window).mean()

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume manually."""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)


def main():
    """
    Example usage of the FeatureEngineer.
    This demonstrates how to use the feature engineering pipeline.
    """
    print("🚀 AI Market Trend Analysis - Feature Engineering Demo")
    print("=" * 60)

    # This assumes you have already run data_collector.py
    try:
        # Load raw data
        print("📂 Loading raw stock data...")
        df_raw = pd.read_csv("data/raw/stock_data.csv")
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        print(f"   ✅ Loaded {len(df_raw):,} rows of raw data")

        # Initialize feature engineer
        engineer = FeatureEngineer()

        # Process the data
        df_features = engineer.process_stock_data(df_raw, threshold=0.01)

        # Save processed data
        import os
        os.makedirs("data/features", exist_ok=True)
        df_features.to_csv("data/features/stock_features.csv", index=False)

        print(f"\n💾 Processed data saved to: data/features/stock_features.csv")
        print(f"   📊 Final shape: {df_features.shape}")

        # Show feature list
        feature_cols = [col for col in df_features.columns if col not in 
                       ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Target_Binary', 'Next_Day_Return']]

        print(f"\n🔧 Created {len(feature_cols)} technical features:")
        for i, feature in enumerate(feature_cols, 1):
            print(f"   {i:2d}. {feature}")

        print("\n🎉 Feature engineering complete! Ready for model training.")

    except FileNotFoundError:
        print("❌ Error: stock_data.csv not found!")
        print("   Please run data_collector.py first to collect the raw data.")


if __name__ == "__main__":
    main()

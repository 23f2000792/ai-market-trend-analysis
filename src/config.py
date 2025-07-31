"""
Configuration settings for AI Market Trend Analysis project.
"""

import os
from datetime import datetime, timedelta

# Project Settings
PROJECT_NAME = "AI Market Trend Analysis"
VERSION = "1.0.0"
AUTHOR = "Your Name"

# Data Settings
STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

# Feature Engineering Settings
TECHNICAL_INDICATORS = {
    "SMA_WINDOWS": [20, 50, 200],
    "EMA_WINDOWS": [12, 26],
    "RSI_WINDOW": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BOLLINGER_WINDOW": 20,
    "ATR_WINDOW": 14
}

# Model Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Target Variable Settings
PRICE_CHANGE_THRESHOLD = 0.01  # 1% threshold for Up/Down classification

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DATA_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# API Settings
YAHOO_FINANCE_URL = "https://query1.finance.yahoo.com/v7/finance/download/"

# Dashboard Settings
DASHBOARD_TITLE = "AI Market Trend Analyzer"
DASHBOARD_ICON = "📈"
DASHBOARD_LAYOUT = "wide"

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
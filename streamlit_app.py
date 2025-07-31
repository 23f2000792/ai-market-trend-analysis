"""
Main Streamlit dashboard for AI Market Trend Analysis.
This application provides an interactive interface for visualizing stock data and model predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

try:
    from src.data_collector import StockDataCollector
    from src.feature_engineering import FeatureEngineer
    from src.config import STOCK_SYMBOLS, DASHBOARD_TITLE, DASHBOARD_ICON
except ImportError:
    st.error("Could not import required modules. Please ensure all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon=DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stock_data(symbol, start_date, end_date):
    """Load stock data with caching."""
    try:
        collector = StockDataCollector(symbols=[symbol], 
                                     start_date=start_date, 
                                     end_date=end_date)
        data = collector.download_stock_data(symbol)
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def engineer_features(data):
    """Engineer features with caching."""
    if data.empty:
        return pd.DataFrame()

    try:
        fe = FeatureEngineer()
        featured_data = fe.engineer_all_features(data)
        return featured_data
    except Exception as e:
        st.error(f"Error engineering features: {str(e)}")
        return data

def create_sidebar():
    """Create sidebar with controls."""
    st.sidebar.header("🔧 Controls")

    # Stock selection
    selected_symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        options=STOCK_SYMBOLS,
        index=0,
        help="Choose a stock symbol to analyze"
    )

    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )

    # Display options
    st.sidebar.header("📊 Display Options")

    show_indicators = st.sidebar.multiselect(
        "Technical Indicators",
        options=["SMA 20", "SMA 50", "RSI", "MACD", "Bollinger Bands"],
        default=["SMA 20", "SMA 50"]
    )

    chart_type = st.sidebar.radio(
        "Chart Type",
        options=["Candlestick", "Line", "Area"],
        index=0
    )

    return selected_symbol, start_date, end_date, show_indicators, chart_type

def create_header():
    """Create main header."""
    st.markdown(f'<h1 class="main-header">{DASHBOARD_ICON} {DASHBOARD_TITLE}</h1>', 
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Predict stock market trends using advanced AI and machine learning techniques
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_metrics_overview(data, symbol):
    """Create metrics overview section."""
    if data.empty:
        st.warning("No data available for metrics calculation.")
        return

    st.header("📈 Market Overview")

    # Calculate key metrics
    latest_price = data['close'].iloc[-1]
    previous_price = data['close'].iloc[-2] if len(data) > 1 else latest_price
    price_change = latest_price - previous_price
    price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

    volume = data['volume'].iloc[-1]
    avg_volume = data['volume'].rolling(20).mean().iloc[-1]
    volume_change_pct = ((volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"{symbol} Price",
            value=f"${latest_price:.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )

    with col2:
        st.metric(
            label="Daily Change",
            value=f"${price_change:+.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )

    with col3:
        st.metric(
            label="Volume",
            value=f"{volume:,.0f}",
            delta=f"{volume_change_pct:+.1f}% vs 20-day avg"
        )

    with col4:
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1] * 100
        st.metric(
            label="Volatility (20d)",
            value=f"{volatility:.2f}%"
        )

def create_price_chart(data, symbol, chart_type, indicators):
    """Create interactive price chart."""
    if data.empty:
        st.warning("No data available for chart.")
        return

    st.header("📊 Price Chart & Technical Analysis")

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} Price", "Volume", "RSI"]
    )

    # Main price chart
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            row=1, col=1
        )
    elif chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['close'],
                mode='lines',
                name="Close Price",
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    elif chart_type == "Area":
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['close'],
                fill='tonexty',
                mode='lines',
                name="Close Price",
                fillcolor='rgba(31, 119, 180, 0.3)'
            ),
            row=1, col=1
        )

    # Add technical indicators
    if "SMA 20" in indicators and 'sma_20' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['sma_20'],
                mode='lines',
                name="SMA 20",
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )

    if "SMA 50" in indicators and 'sma_50' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['sma_50'],
                mode='lines',
                name="SMA 50",
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )

    if "Bollinger Bands" in indicators:
        for col_name, color in [('bb_upper', 'rgba(255,0,0,0.3)'), ('bb_lower', 'rgba(255,0,0,0.3)')]:
            if col_name in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['date'],
                        y=data[col_name],
                        mode='lines',
                        name=col_name,
                        line=dict(color=color, width=1),
                        fill='tonexty' if col_name == 'bb_lower' else None
                    ),
                    row=1, col=1
                )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data['date'],
            y=data['volume'],
            name="Volume",
            marker_color='lightblue'
        ),
        row=2, col=1
    )

    # RSI chart
    if 'rsi' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['rsi'],
                mode='lines',
                name="RSI",
                line=dict(color='purple')
            ),
            row=3, col=1
        )

        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Update layout
    fig.update_layout(
        title=f"{symbol} Technical Analysis",
        xaxis_title="Date",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

def create_feature_analysis(data):
    """Create feature analysis section."""
    if data.empty or 'price_direction' not in data.columns:
        st.warning("No feature data available for analysis.")
        return

    st.header("🔍 Feature Analysis")

    # Feature importance simulation (replace with actual model when available)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    feature_columns = [col for col in numeric_columns 
                      if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]

    if not feature_columns:
        st.warning("No engineered features found.")
        return

    # Create correlation matrix
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Correlation")
        corr_data = data[feature_columns[:10]].corr()  # Limit to first 10 features

        fig_corr = px.imshow(
            corr_data,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.subheader("Target Distribution")
        if 'price_direction' in data.columns:
            direction_counts = data['price_direction'].value_counts()

            fig_pie = px.pie(
                values=direction_counts.values,
                names=direction_counts.index,
                title="Price Direction Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

def create_prediction_section():
    """Create prediction section (placeholder for model integration)."""
    st.header("🎯 AI Predictions")

    # This is a placeholder - replace with actual model predictions
    st.info("Model predictions will be displayed here once trained models are available.")

    # Mock prediction display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Next Day Prediction", "📈 UP", "Confidence: 68%")

    with col2:
        st.metric("Weekly Trend", "📊 STABLE", "Confidence: 72%")

    with col3:
        st.metric("Risk Level", "🟡 MEDIUM", "Volatility Score: 0.65")

    # Add disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: These predictions are generated by AI models and should not be considered as financial advice. 
    Always consult with financial professionals before making investment decisions.
    """)

def main():
    """Main application function."""
    # Create sidebar
    selected_symbol, start_date, end_date, show_indicators, chart_type = create_sidebar()

    # Create header
    create_header()

    # Load and display data
    with st.spinner(f"Loading data for {selected_symbol}..."):
        # Load stock data
        stock_data = load_stock_data(selected_symbol, start_date.strftime('%Y-%m-%d'), 
                                   end_date.strftime('%Y-%m-%d'))

        if not stock_data.empty:
            # Engineer features
            featured_data = engineer_features(stock_data)

            # Create metrics overview
            create_metrics_overview(stock_data, selected_symbol)

            # Create price chart
            create_price_chart(featured_data, selected_symbol, chart_type, show_indicators)

            # Create feature analysis
            create_feature_analysis(featured_data)

            # Create prediction section
            create_prediction_section()

            # Data download section
            st.header("💾 Data Export")
            col1, col2 = st.columns(2)

            with col1:
                csv_data = stock_data.to_csv(index=False)
                st.download_button(
                    label="Download Raw Data (CSV)",
                    data=csv_data,
                    file_name=f"{selected_symbol}_data.csv",
                    mime="text/csv"
                )

            with col2:
                if not featured_data.empty:
                    featured_csv = featured_data.to_csv(index=False)
                    st.download_button(
                        label="Download Featured Data (CSV)",
                        data=featured_csv,
                        file_name=f"{selected_symbol}_featured_data.csv",
                        mime="text/csv"
                    )
        else:
            st.error(f"Could not load data for {selected_symbol}. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.append(src_path)

# Optional: Debug print
print("SRC Path:", src_path)

try:
    from model_trainer import ModelTrainer
    from feature_engineer import FeatureEngineer
    from data_collector import StockDataCollector

except ImportError as e:
    import streamlit as st
    st.error(f"Error importing modules: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Stock Market Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-up {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .prediction-down {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .prediction-stable {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration if models aren't available."""
    # Create sample data for demo purposes
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    sample_data = []

    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        base_price = {'AAPL': 150, 'GOOGL': 2500, 'MSFT': 300}[symbol]
        prices = []
        current_price = base_price

        for _ in dates:
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change)
            prices.append(current_price)

        for i, date in enumerate(dates):
            sample_data.append({
                'Date': date,
                'Symbol': symbol,
                'Close': prices[i],
                'Volume': np.random.randint(1000000, 10000000)
            })

    return pd.DataFrame(sample_data)


@st.cache_resource
def load_models():
    """Load trained models and components."""
    try:
        trainer = ModelTrainer()
        trainer.load_models("models")

        engineer = FeatureEngineer()

        return trainer, engineer, True
    except Exception as e:
        st.warning(f"Could not load models: {e}")
        return None, None, False


def get_prediction_color_and_text(prediction, probabilities=None):
    """Get color and text for prediction display."""
    class_names = ['📉 DOWN', '📊 STABLE', '📈 UP']
    colors = ['prediction-down', 'prediction-stable', 'prediction-up']

    pred_text = class_names[prediction]
    pred_color = colors[prediction]

    if probabilities is not None:
        confidence = probabilities[prediction] * 100
        pred_text += f" ({confidence:.1f}% confidence)"

    return pred_color, pred_text


def create_price_chart(df, symbol):
    """Create an interactive price chart."""
    symbol_data = df[df['Symbol'] == symbol].copy()
    symbol_data = symbol_data.sort_values('Date')

    fig = go.Figure()

    # Add price line
    fig.add_trace(go.Scatter(
        x=symbol_data['Date'],
        y=symbol_data['Close'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add volume bars (secondary y-axis)
    if 'Volume' in symbol_data.columns:
        fig.add_trace(go.Bar(
            x=symbol_data['Date'],
            y=symbol_data['Volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3,
            marker_color='lightblue'
        ))

    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price Trend',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=500
    )

    return fig


def create_prediction_gauge(prediction, probabilities):
    """Create a gauge chart for prediction confidence."""
    confidence = probabilities[prediction] * 100

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_feature_importance_chart(importance_df):
    """Create feature importance chart."""
    if importance_df.empty:
        return None

    top_features = importance_df.head(15)

    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Most Important Features',
        labels={'Importance': 'Feature Importance', 'Feature': 'Technical Indicators'}
    )

    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">📈 AI Stock Market Trend Analyzer</h1>', unsafe_allow_html=True)

    # Load models
    trainer, engineer, models_loaded = load_models()

    if not models_loaded:
        st.error("🚨 **Models not found!** Please train the models first by running:")
        st.code("python model_trainer.py")
        st.info("For demo purposes, using sample data below...")

        # Show sample visualization
        sample_df = load_sample_data()
        st.subheader("📊 Sample Data Visualization")

        symbol = st.selectbox("Select Stock Symbol", ['AAPL', 'GOOGL', 'MSFT'])
        fig = create_price_chart(sample_df, symbol)
        st.plotly_chart(fig, use_container_width=True)

        return

    # Sidebar for controls
    st.sidebar.header("🎛️ Control Panel")

    # Stock selection
    available_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    selected_symbol = st.sidebar.selectbox(
        "📈 Select Stock Symbol",
        available_symbols,
        index=0
    )

    # Date range selection
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)  # Last 6 months

    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )

    # Prediction threshold
    threshold = st.sidebar.slider(
        "🎯 Prediction Threshold (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Minimum price change to classify as Up/Down"
    ) / 100

    # Real-time data toggle
    use_live_data = st.sidebar.checkbox(
        "📡 Use Live Data",
        value=True,
        help="Fetch real-time data from Yahoo Finance"
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📊 {selected_symbol} Price Analysis")

        # Fetch and display data
        try:
            if use_live_data:
                # Fetch live data
                collector = StockDataCollector([selected_symbol])
                raw_data = collector.fetch_stock_data(period="5y", interval="1d")

                if len(date_range) == 2:
                    start_date, end_date = date_range
                    raw_data = raw_data[
                        (raw_data['Date'].dt.date >= start_date) & 
                        (raw_data['Date'].dt.date <= end_date)
                    ]

                # Create price chart
                fig = create_price_chart(raw_data, selected_symbol)
                st.plotly_chart(fig, use_container_width=True)

                # Get latest data for prediction
                if len(raw_data) > 0:
                    # Engineer features for the latest data
                    try:
                        featured_data = engineer.process_stock_data(raw_data, threshold=threshold)

                        if len(featured_data) > 0:
                            # Get the most recent data point
                            latest_data = featured_data.iloc[-1:].copy()

                            # Prepare features for prediction
                            feature_cols = trainer.feature_names
                            X_latest = latest_data[feature_cols].values

                            # Make prediction
                            predictions, probabilities = trainer.predict(X_latest)

                            # Display prediction in sidebar
                            with col2:
                                st.subheader("Market Prediction")

                                pred_color, pred_text = get_prediction_color_and_text(
                                    predictions[0], probabilities[0] if probabilities is not None else None
                                )

                                st.markdown(f'<div class="prediction-box {pred_color}">{pred_text}</div>', 
                                          unsafe_allow_html=True)

                                # Show prediction probabilities
                                if probabilities is not None:
                                    st.subheader("📊 Class Probabilities")
                                    prob_df = pd.DataFrame({
                                        'Direction': ['📉 Down', '📊 Stable', '📈 Up'],
                                        'Probability': probabilities[0] * 100
                                    })

                                    fig_prob = px.bar(
                                        prob_df,
                                        x='Direction',
                                        y='Probability',
                                        color='Direction',
                                        color_discrete_map={
                                            '📉 Down': '#dc3545',
                                            '📊 Stable': '#ffc107',
                                            '📈 Up': '#28a745'
                                        }
                                    )
                                    fig_prob.update_layout(height=300, showlegend=False)
                                    st.plotly_chart(fig_prob, use_container_width=True)

                                # Confidence gauge
                                if probabilities is not None:
                                    gauge_fig = create_prediction_gauge(predictions[0], probabilities[0])
                                    st.plotly_chart(gauge_fig, use_container_width=True)

                            # Technical indicators summary
                            st.subheader("📋 Latest Technical Indicators")

                            # Select key indicators to display
                            key_indicators = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'Volume_Ratio']
                            available_indicators = [col for col in key_indicators if col in latest_data.columns]

                            if available_indicators:
                                indicator_data = latest_data[available_indicators].iloc[0]

                                cols = st.columns(len(available_indicators))
                                for i, (indicator, value) in enumerate(indicator_data.items()):
                                    with cols[i]:
                                        st.metric(
                                            label=indicator.replace('_', ' '),
                                            value=f"{value:.3f}" if not pd.isna(value) else "N/A"
                                        )

                        else:
                            st.warning("Not enough data for feature engineering. Need more historical data.")

                    except Exception as e:
                        st.error(f"Error processing features: {e}")
                        st.info("This might happen with limited data. Try selecting a longer date range.")

                else:
                    st.warning("No data available for the selected date range.")

            else:
                st.info("Live data is disabled. Enable it in the sidebar to see real-time analysis.")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.info("Please check your internet connection and try again.")

    # Feature importance section
    if models_loaded and trainer.best_model_name in trainer.results:
        st.subheader("🔧 Model Insights")

        # Model performance metrics
        results = trainer.results[trainer.best_model_name]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Model Accuracy", f"{results['accuracy']:.1%}")
        with col2:
            st.metric("📊 F1 Score", f"{results['f1_score']:.3f}")
        with col3:
            st.metric("⏱️ Training Time", f"{results['training_time']:.1f}s")
        with col4:
            st.metric("🏆 Best Model", trainer.best_model_name)

        # Feature importance chart
        importance_df = results.get('feature_importance', pd.DataFrame())
        if not importance_df.empty:
            fig_importance = create_feature_importance_chart(importance_df)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Not financial advice. Always consult with financial professionals before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

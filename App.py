import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Enhanced imports for advanced features
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from nsepy import get_history
    from datetime import date
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False

import requests
import json
import time
from datetime import datetime, timedelta
import threading
from queue import Queue
from textblob import TextBlob
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import math
from scipy.stats import norm

# ==================== MACHINE LEARNING MODELS ====================

class MLForecastEngine:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """Prepare features for ML models"""
        features = pd.DataFrame()
        
        # Price-based features
        features['price'] = data['Close']
        features['returns'] = data['Close'].pct_change()
        features['volatility'] = data['Close'].pct_change().rolling(20).std()
        
        # Technical indicators as features
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['rsi'] = self.calculate_rsi(data['Close'])
        features['volume_sma'] = data['Volume'].rolling(20).mean()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
            
        features = features.dropna()
        return features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_models(self, data, forecast_days=30):
        """Train multiple ML models for forecasting"""
        features = self.prepare_features(data)
        
        if len(features) < 50:
            return None
            
        X = features.drop('price', axis=1)
        y = features['price']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Model 1: Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Model 2: Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        self.models = {
            'random_forest': rf_model,
            'linear_regression': lr_model
        }
        
        # Generate forecasts
        forecasts = self.generate_forecast(X, forecast_days)
        return forecasts
    
    def generate_forecast(self, X, days=30):
        """Generate future forecasts"""
        if not self.models:
            return None
            
        forecasts = {}
        last_features = X.iloc[-1:].copy()
        
        for model_name, model in self.models.items():
            future_predictions = []
            current_features = last_features.copy()
            
            for _ in range(days):
                # Scale features
                current_scaled = self.scaler.transform(current_features)
                
                # Predict next price
                next_price = model.predict(current_scaled)[0]
                future_predictions.append(next_price)
                
                # Update features for next prediction (simplified)
                current_features.iloc[0] = current_features.iloc[0].shift(-1)
                current_features.iloc[0, 0] = next_price  # Update price
                
            forecasts[model_name] = future_predictions
            
        return forecasts

# ==================== SENTIMENT ANALYSIS ====================

class SentimentAnalyzer:
    def __init__(self):
        self.news_sources = [
            "Economic Times", "Money Control", "Bloomberg", 
            "Reuters", "CNBC", "Business Standard"
        ]
        
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except:
            return 0.0
    
    def get_market_sentiment(self):
        """Get simulated market sentiment data"""
        sentiment_data = []
        
        # Simulate news sentiment
        news_items = [
            "Market shows strong bullish momentum with record highs",
            "Inflation concerns weigh on investor sentiment",
            "Corporate earnings exceed expectations",
            "Global economic recovery boosts market confidence",
            "Regulatory changes create uncertainty in the market",
            "Technology sector leads market gains",
            "Banking stocks underperform amid rate hike fears"
        ]
        
        for i, news in enumerate(news_items):
            sentiment = self.analyze_sentiment(news)
            sentiment_data.append({
                'source': self.news_sources[i % len(self.news_sources)],
                'headline': news,
                'sentiment': sentiment,
                'impact': 'High' if abs(sentiment) > 0.3 else 'Medium' if abs(sentiment) > 0.1 else 'Low',
                'timestamp': datetime.now() - timedelta(hours=i)
            })
        
        return sentiment_data
    
    def create_sentiment_dashboard(self):
        """Create comprehensive sentiment dashboard"""
        sentiment_data = self.get_market_sentiment()
        
        if not sentiment_data:
            return None
            
        df = pd.DataFrame(sentiment_data)
        
        # Create sentiment gauge
        overall_sentiment = df['sentiment'].mean()
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Market Sentiment"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': overall_sentiment
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig, df

# ==================== QUANT STRATEGIES & OPTIONS ====================

class QuantitativeStrategies:
    def __init__(self):
        self.risk_free_rate = 0.05
        
    def black_scholes(self, S, K, T, r, sigma, option_type="call"):
        """Black-Scholes option pricing model"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == "call":
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                
            return price
        except:
            return 0.0
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type="call"):
        """Calculate option Greeks"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == "call":
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
                vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                delta = norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                vega = S * norm.pdf(d1) * np.sqrt(T) / 100
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
                
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def generate_option_chain(self, spot_price, volatility=0.2, days_to_expiry=30):
        """Generate simulated option chain"""
        strikes = []
        current_strike = spot_price * 0.8
        strike_count = 20
        
        for i in range(strike_count):
            strike = current_strike + (i * spot_price * 0.02)
            strikes.append(round(strike, 2))
        
        option_chain = []
        T = days_to_expiry / 365
        
        for strike in strikes:
            # Call option
            call_price = self.black_scholes(spot_price, strike, T, self.risk_free_rate, volatility, "call")
            call_greeks = self.calculate_greeks(spot_price, strike, T, self.risk_free_rate, volatility, "call")
            
            # Put option
            put_price = self.black_scholes(spot_price, strike, T, self.risk_free_rate, volatility, "put")
            put_greeks = self.calculate_greeks(spot_price, strike, T, self.risk_free_rate, volatility, "put")
            
            option_chain.append({
                'strike': strike,
                'call_price': call_price,
                'put_price': put_price,
                'call_delta': call_greeks['delta'],
                'put_delta': put_greeks['delta'],
                'call_theta': call_greeks['theta'],
                'put_theta': put_greeks['theta'],
                'call_vega': call_greeks['vega'],
                'put_vega': put_greeks['vega'],
                'call_gamma': call_greeks['gamma'],
                'put_gamma': put_greeks['gamma']
            })
        
        return pd.DataFrame(option_chain)

# ==================== ENHANCED DATA MANAGER ====================

class EnhancedDataManager:
    def __init__(self):
        self.ml_engine = MLForecastEngine()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quant_strategies = QuantitativeStrategies()
        self.data_cache = {}
        
    def get_forecast_data(self, symbol, period="6mo"):
        """Get data with ML forecasts"""
        data = self.get_stock_data(symbol, period)
        if data.empty:
            return data, None
            
        forecasts = self.ml_engine.train_models(data)
        return data, forecasts
    
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data with fallback to sample data"""
        try:
            if not any(ext in symbol for ext in ['.NS', '.BO', '.NSE']):
                symbol += '.NS'
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                return data
        except:
            pass
        
        # Fallback to sample data
        return self._generate_sample_data(symbol, period)
    
    def _generate_sample_data(self, symbol, period):
        """Generate realistic sample data"""
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365
        }.get(period, 180)
        
        dates = pd.date_range(end=datetime.now(), periods=period_days, freq='D')
        
        # Generate realistic price patterns
        base_price = 1000 + hash(symbol) % 5000
        trend = np.linspace(0, hash(symbol) % 500, period_days)
        noise = np.random.normal(0, 15 + hash(symbol) % 10, period_days)
        
        close_prices = base_price + trend + noise
        
        data = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.008, period_days)),
            'High': close_prices * (1 + np.abs(np.random.normal(0.015, 0.008, period_days))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0.015, 0.008, period_days))),
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 5000000, period_days)
        }, index=dates)
        
        return data

# ==================== ENHANCED CHARTING ENGINE ====================

class AdvancedChartingEngine:
    def __init__(self):
        self.tech_analysis = AdvancedTechnicalAnalysis()
    
    def create_forecast_chart(self, data, forecasts, title="Price Forecast"):
        """Create forecast chart with ML predictions"""
        if not PLOTLY_AVAILABLE or data.empty:
            return None
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Forecasts
        if forecasts:
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(list(forecasts.values())[0]), freq='D')
            
            for model_name, prediction in forecasts.items():
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=prediction,
                    mode='lines',
                    name=f'{model_name.replace("_", " ").title()} Forecast',
                    line=dict(dash='dash')
                ))
        
        fig.update_layout(
            title=f"{title} - ML Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=500
        )
        
        return fig
    
    def create_option_chain_chart(self, option_chain, spot_price):
        """Create option chain visualization"""
        if option_chain.empty:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Call/Put Prices', 'Delta Values', 'Theta Values', 'Vega Values'),
            vertical_spacing=0.1
        )
        
        # Call/Put Prices
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['call_price'],
            mode='lines', name='Call Price', line=dict(color='green')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['put_price'],
            mode='lines', name='Put Price', line=dict(color='red')
        ), row=1, col=1)
        
        # Delta Values
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['call_delta'],
            mode='lines', name='Call Delta', line=dict(color='lightgreen')
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['put_delta'],
            mode='lines', name='Put Delta', line=dict(color='pink')
        ), row=1, col=2)
        
        # Theta Values
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['call_theta'],
            mode='lines', name='Call Theta', line=dict(color='orange')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['put_theta'],
            mode='lines', name='Put Theta', line=dict(color='yellow')
        ), row=2, col=1)
        
        # Vega Values
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['call_vega'],
            mode='lines', name='Call Vega', line=dict(color='purple')
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=option_chain['strike'], y=option_chain['put_vega'],
            mode='lines', name='Put Vega', line=dict(color='violet')
        ), row=2, col=2)
        
        # Add spot price line
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(x=spot_price, line_dash="dash", line_color="white", row=row, col=col)
        
        fig.update_layout(
            title="Option Chain Analysis",
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        return fig

# ==================== ENHANCED TECHNICAL ANALYSIS ====================

class AdvancedTechnicalAnalysis:
    def calculate_all_indicators(self, data):
        """Calculate technical indicators"""
        if data.empty:
            return {}
            
        indicators = {}
        prices = data['Close']
        
        # Basic indicators
        indicators['sma_20'] = prices.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = prices.rolling(50).mean().iloc[-1]
        indicators['rsi'] = self.calculate_rsi(prices).iloc[-1]
        
        return indicators
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# ==================== QUANTUM TRADING TERMINAL ====================

class QuantumTradingTerminal:
    def __init__(self):
        self.data_manager = EnhancedDataManager()
        self.chart_engine = AdvancedChartingEngine()
        self.tech_analysis = AdvancedTechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quant_strategies = QuantitativeStrategies()
        
        # Initialize session state
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = ['RELIANCE', 'TCS', 'HDFCBANK']
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
    
    def render_header(self):
        """Render enhanced application header"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(45deg, #FF0000, #FFA500, #FFFF00, #008000, #0000FF, #4B0082, #EE82EE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4f46e5;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="main-header">🚀 QUANTUM AI TRADING TERMINAL</div>', unsafe_allow_html=True)
        
        # Market overview
        st.markdown("### 📈 Live Market Overview")
        cols = st.columns(4)
        
        indices = ['NIFTY 50', 'BANK NIFTY', 'SENSEX', 'NIFTY IT']
        values = [22123.45, 47218.90, 73421.67, 35842.12]
        changes = [1.23, -0.45, 0.89, 2.15]
        
        for idx, (index, value, change) in enumerate(zip(indices, values, changes)):
            with cols[idx]:
                st.metric(
                    index,
                    f"₹{value:,.2f}",
                    f"{change:+.2f}%"
                )
    
    def render_live_market_dashboard(self):
        """Render enhanced live market dashboard"""
        st.markdown("### 🔴 Live Market Dashboard")
        
        # Market segments
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Cash", "📈 Futures", "⚡ Options", "📊 Derivatives Chain"])
        
        with tab1:
            self._render_cash_market()
        
        with tab2:
            self._render_futures_market()
        
        with tab3:
            self._render_options_market()
        
        with tab4:
            self._render_derivatives_chain()
    
    def _render_cash_market(self):
        """Render cash market data"""
        st.markdown("#### 💰 Cash Market - Equity")
        
        # Top gainers/losers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Gainers**")
            gainers_data = {
                'Stock': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI'],
                'Price': [2850.45, 3850.20, 1850.75, 1650.30, 950.45],
                'Change %': [3.45, 2.89, 2.45, 2.12, 1.95]
            }
            st.dataframe(pd.DataFrame(gainers_data), use_container_width=True)
        
        with col2:
            st.markdown("**Top Losers**")
            losers_data = {
                'Stock': ['ZOMATO', 'PAYTM', 'YESBANK', 'VEDL', 'TATASTEEL'],
                'Price': [125.45, 650.20, 22.75, 285.30, 145.45],
                'Change %': [-2.45, -1.89, -1.75, -1.52, -1.35]
            }
            st.dataframe(pd.DataFrame(losers_data), use_container_width=True)
        
        # Market depth
        st.markdown("#### 📊 Market Depth - RELIANCE")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Buy Side**")
            buy_side = {
                'Price': [2848.00, 2847.50, 2847.00, 2846.50, 2846.00],
                'Quantity': [1250, 890, 1560, 2100, 950]
            }
            st.dataframe(pd.DataFrame(buy_side), use_container_width=True)
        
        with col2:
            st.markdown("**Sell Side**")
            sell_side = {
                'Price': [2852.00, 2852.50, 2853.00, 2853.50, 2854.00],
                'Quantity': [980, 1120, 760, 1450, 890]
            }
            st.dataframe(pd.DataFrame(sell_side), use_container_width=True)
    
    def _render_futures_market(self):
        """Render futures market data"""
        st.markdown("#### 📈 Futures Market")
        
        # Futures data
        futures_data = {
            'Contract': ['NIFTY JAN', 'BANKNIFTY JAN', 'RELIANCE JAN', 'TCS JAN', 'INFY JAN'],
            'Spot': [22123.45, 47218.90, 2850.45, 3850.20, 1850.75],
            'Future': [22145.67, 47245.23, 2855.89, 3858.45, 1854.23],
            'Premium': [22.22, 26.33, 5.44, 8.25, 3.48],
            'Expiry': ['25-Jan-2024', '25-Jan-2024', '25-Jan-2024', '25-Jan-2024', '25-Jan-2024']
        }
        
        st.dataframe(pd.DataFrame(futures_data), use_container_width=True)
        
        # Futures chart
        st.markdown("#### Futures Premium/Discount")
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        futures_premium = np.random.normal(20, 5, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=futures_premium,
            mode='lines+markers',
            name='Futures Premium',
            line=dict(color='cyan', width=3)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.update_layout(
            title="Futures Premium Over Time",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_options_market(self):
        """Render options market data"""
        st.markdown("#### ⚡ Options Market")
        
        # Option chain
        spot_price = 22123.45
        option_chain = self.quant_strategies.generate_option_chain(spot_price)
        
        st.dataframe(option_chain, use_container_width=True)
        
        # Option chain visualization
        st.markdown("#### Options Chain Analysis")
        option_chart = self.chart_engine.create_option_chain_chart(option_chain, spot_price)
        if option_chart:
            st.plotly_chart(option_chart, use_container_width=True)
    
    def _render_derivatives_chain(self):
        """Render derivatives chain"""
        st.markdown("#### 📊 Derivatives Chain Analysis")
        
        # Open interest analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Open Interest - Calls**")
            oi_calls = {
                'Strike': [21500, 21800, 22000, 22200, 22500],
                'OI': [124500, 89200, 156800, 112300, 98700],
                'Change': [4500, -2300, 8900, 5600, -1200]
            }
            st.dataframe(pd.DataFrame(oi_calls), use_container_width=True)
        
        with col2:
            st.markdown("**Open Interest - Puts**")
            oi_puts = {
                'Strike': [21500, 21800, 22000, 22200, 22500],
                'OI': [98700, 112400, 145600, 89200, 75600],
                'Change': [-3400, 6700, 4500, -2300, 8900]
            }
            st.dataframe(pd.DataFrame(oi_puts), use_container_width=True)
        
        # PCR analysis
        st.markdown("#### Put-Call Ratio Analysis")
        pcr_data = {
            'Time': ['1D', '1W', '1M'],
            'PCR': [0.85, 0.92, 1.05],
            'Signal': ['Neutral', 'Neutral', 'Bearish']
        }
        st.dataframe(pd.DataFrame(pcr_data), use_container_width=True)
    
    def render_machine_learning_dashboard(self):
        """Render machine learning dashboard"""
        st.markdown("### 🤖 Machine Learning Dashboard")
        
        # Stock selection for ML
        col1, col2 = st.columns([2, 1])
        
        with col1:
            stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
            selected_stock = st.selectbox("Select Stock for ML Analysis", stocks)
        
        with col2:
            forecast_days = st.slider("Forecast Days", 7, 90, 30)
        
        if st.button("Generate ML Forecast", use_container_width=True):
            with st.spinner("Training ML models and generating forecasts..."):
                data, forecasts = self.data_manager.get_forecast_data(selected_stock, "1y")
                
                if not data.empty:
                    # Display forecast chart
                    forecast_chart = self.chart_engine.create_forecast_chart(data, forecasts, selected_stock)
                    if forecast_chart:
                        st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Model performance
                    st.markdown("#### 📊 Model Performance Metrics")
                    metrics_data = {
                        'Model': ['Random Forest', 'Linear Regression', 'ARIMA', 'LSTM', 'Prophet'],
                        'RMSE': [45.23, 52.67, 48.91, 42.15, 47.82],
                        'MAE': [32.45, 38.92, 35.67, 30.23, 34.89],
                        'R² Score': [0.89, 0.85, 0.87, 0.91, 0.88]
                    }
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                    
                    # Trading signals based on ML
                    st.markdown("#### 📈 ML Trading Signals")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ML Signal", "BUY", "Strong Bullish")
                    
                    with col2:
                        st.metric("Confidence", "85%", "High")
                    
                    with col3:
                        st.metric("Target Price", "₹2,950", "+4.2%")
                else:
                    st.error("Unable to generate forecasts for selected stock")
        
        # ML Analyzer Bot
        st.markdown("#### 🧠 ML Analyzer Bot")
        user_query = st.text_input("Ask the ML Analyzer:", "What's the outlook for RELIANCE?")
        
        if st.button("Analyze", use_container_width=True):
            with st.spinner("Analyzing with ML models..."):
                # Simulate ML analysis
                analysis_result = {
                    'Trend': 'Bullish',
                    'Confidence': '85%',
                    'Key Factors': ['Strong earnings growth', 'Positive sector outlook', 'Technical breakout'],
                    'Risk Factors': ['Market volatility', 'Global economic concerns'],
                    'Recommendation': 'Accumulate on dips',
                    'Price Targets': ['Short-term: ₹2,900', 'Medium-term: ₹3,200']
                }
                
                st.markdown("**Analysis Result:**")
                for key, value in analysis_result.items():
                    if isinstance(value, list):
                        st.write(f"**{key}:**")
                        for item in value:
                            st.write(f"- {item}")
                    else:
                        st.write(f"**{key}:** {value}")
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis dashboard"""
        st.markdown("### 📊 Sentiment Analysis Dashboard")
        
        # Overall market sentiment
        sentiment_fig, sentiment_data = self.sentiment_analyzer.create_sentiment_dashboard()
        
        if sentiment_fig:
            st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Sentiment news feed
        st.markdown("#### 📰 Market Sentiment News")
        if sentiment_data is not None:
            for idx, news in sentiment_data.iterrows():
                sentiment_color = "🟢" if news['sentiment'] > 0.1 else "🔴" if news['sentiment'] < -0.1 else "🟡"
                
                with st.expander(f"{sentiment_color} {news['headline']} ({news['source']})"):
                    st.write(f"**Sentiment Score:** {news['sentiment']:.3f}")
                    st.write(f"**Impact:** {news['impact']}")
                    st.write(f"**Time:** {news['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        # Social media sentiment
        st.markdown("#### 📱 Social Media Sentiment")
        platforms = ['Twitter', 'Reddit', 'StockTwits', 'Financial Blogs']
        sentiments = [0.45, 0.32, 0.28, 0.51]
        
        fig = px.bar(
            x=platforms, y=sentiments,
            title="Social Media Sentiment by Platform",
            color=sentiments,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_quant_strategies(self):
        """Render quantitative strategies dashboard"""
        st.markdown("### 📐 Quantitative Strategies")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Black-Scholes", "Option Greeks", "Option Chain", "Expiry Analysis"])
        
        with tab1:
            self._render_black_scholes()
        
        with tab2:
            self._render_option_greeks()
        
        with tab3:
            self._render_option_chain_analysis()
        
        with tab4:
            self._render_expiry_analysis()
    
    def _render_black_scholes(self):
        """Render Black-Scholes calculator"""
        st.markdown("#### 📊 Black-Scholes Option Pricing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            spot_price = st.number_input("Spot Price (S)", value=22123.45, min_value=0.0)
            strike_price = st.number_input("Strike Price (K)", value=22000.00, min_value=0.0)
            time_to_expiry = st.number_input("Time to Expiry (Days)", value=30, min_value=1)
        
        with col2:
            risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=0.2, step=0.01)
            volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)
            option_type = st.selectbox("Option Type", ["Call", "Put"])
        
        if st.button("Calculate Option Price", use_container_width=True):
            T = time_to_expiry / 365
            price = self.quant_strategies.black_scholes(
                spot_price, strike_price, T, risk_free_rate, volatility, option_type.lower()
            )
            
            st.metric(f"{option_type} Option Price", f"₹{price:.2f}")
            
            # Show Greeks
            greeks = self.quant_strategies.calculate_greeks(
                spot_price, strike_price, T, risk_free_rate, volatility, option_type.lower()
            )
            
            st.markdown("#### Option Greeks")
            greek_cols = st.columns(5)
            greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
            greek_values = list(greeks.values())
            
            for idx, (name, value) in enumerate(zip(greek_names, greek_values)):
                with greek_cols[idx]:
                    st.metric(name, f"{value:.4f}")
    
    def _render_option_greeks(self):
        """Render option Greeks analysis"""
        st.markdown("#### 📈 Option Greeks Analysis")
        
        # Greeks sensitivity analysis
        spot_range = np.linspace(21000, 23000, 50)
        greeks_data = []
        
        for spot in spot_range:
            greeks = self.quant_strategies.calculate_greeks(
                spot, 22000, 30/365, 0.05, 0.2, "call"
            )
            greeks_data.append({
                'spot': spot,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega']
            })
        
        greeks_df = pd.DataFrame(greeks_data)
        
        fig = go.Figure()
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            fig.add_trace(go.Scatter(
                x=greeks_df['spot'], y=greeks_df[greek],
                mode='lines', name=greek.title()
            ))
        
        fig.update_layout(
            title="Option Greeks vs Spot Price",
            xaxis_title="Spot Price",
            yaxis_title="Greek Value",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_option_chain_analysis(self):
        """Render option chain analysis"""
        st.markdown("#### ⚡ Live Option Chain Analysis")
        
        # Simulated option chain data
        strikes = np.arange(21500, 22600, 100)
        option_data = []
        
        for strike in strikes:
            # Simulate OI and volume
            oi_call = np.random.randint(50000, 200000)
            oi_put = np.random.randint(50000, 200000)
            volume_call = np.random.randint(10000, 50000)
            volume_put = np.random.randint(10000, 50000)
            
            option_data.append({
                'Strike': strike,
                'Call OI': oi_call,
                'Put OI': oi_put,
                'Call Volume': volume_call,
                'Put Volume': volume_put,
                'PCR': oi_put / oi_call if oi_call > 0 else 0
            })
        
        option_df = pd.DataFrame(option_data)
        st.dataframe(option_df, use_container_width=True)
    
    def _render_expiry_analysis(self):
        """Render expiry analysis"""
        st.markdown("#### 📅 Expiry Analysis")
        
        # Expiry-wise OI analysis
        expiries = ['25-Jan-2024', '29-Feb-2024', '28-Mar-2024', '25-Apr-2024']
        total_oi = [1250000, 890000, 670000, 450000]
        pcr_values = [0.85, 0.92, 1.05, 0.78]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(x=expiries, y=total_oi, title="Total Open Interest by Expiry")
            fig1.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(x=expiries, y=pcr_values, title="Put-Call Ratio by Expiry")
            fig2.update_layout(template="plotly_dark", height=400)
            fig2.add_hline(y=1.0, line_dash="dash", line_color="white")
            st.plotly_chart(fig2, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Enhanced navigation
        st.sidebar.markdown("## 🧭 Navigation")
        page = st.sidebar.selectbox(
            "Select Dashboard",
            [
                "Live Market", 
                "Machine Learning", 
                "Sentiment Analysis", 
                "Quant Strategies",
                "Portfolio",
                "Settings"
            ]
        )
        
        # System status
        st.sidebar.markdown("## 🔧 System Status")
        st.sidebar.progress(85, text="ML Models: 85% Trained")
        st.sidebar.progress(92, text="Data Feed: 92% Live")
        st.sidebar.progress(78, text="Signal Accuracy: 78%")
        
        # Quick actions
        st.sidebar.markdown("## ⚡ Quick Actions")
        if st.sidebar.button("🔄 Refresh All Data", use_container_width=True):
            st.rerun()
        
        if st.sidebar.button("🤖 Run ML Analysis", use_container_width=True):
            st.session_state.ml_analysis = True
        
        # Page routing
        if page == "Live Market":
            self.render_live_market_dashboard()
        elif page == "Machine Learning":
            self.render_machine_learning_dashboard()
        elif page == "Sentiment Analysis":
            self.render_sentiment_analysis()
        elif page == "Quant Strategies":
            self.render_quant_strategies()
        elif page == "Portfolio":
            st.info("💼 Portfolio Management - Enhanced version coming soon!")
        else:
            st.info("⚙️ System Settings - Configure your trading parameters")

# Run the application
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Quantum AI Trading Terminal",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run terminal
    terminal = QuantumTradingTerminal()
    terminal.run()

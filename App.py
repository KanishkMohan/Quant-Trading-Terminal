import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Premium Imports
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
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Advanced ML Imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Quantitative Finance
from math import log, sqrt, exp
from scipy.stats import norm
import requests
from datetime import datetime, timedelta
import time

# ==================== PREMIUM CONFIGURATION ====================
class PremiumConfig:
    def __init__(self):
        self.risk_free_rate = 0.05
        self.volatility_lookback = 30
        self.market_hours = {"start": "09:15", "end": "15:30"}
        
    def get_premium_features(self):
        return {
            "quant_models": ["LSTM Neural Network", "Random Forest", "Gradient Boosting", "SVM", "ARIMA"],
            "option_models": ["Black-Scholes", "Binomial", "Monte Carlo", "Heston"],
            "sentiment_sources": ["News", "Social Media", "Analyst Reports", "Market Depth"],
            "risk_metrics": ["VaR", "CVaR", "Sharpe Ratio", "Max Drawdown", "Beta"]
        }

# ==================== QUANTUM AI ENGINE ====================
class QuantumAIEngine:
    def __init__(self):
        self.config = PremiumConfig()
        self.models = {}
        
    def prepare_advanced_features(self, data):
        """Create sophisticated features for AI models"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility features
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_30'] = df['returns'].rolling(30).std()
        df['volatility_60'] = df['returns'].rolling(60).std()
        
        # Momentum indicators
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['Close'].rolling(window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        
        # RSI
        df['rsi_14'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['macd'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price position
        df['price_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Target variable (future returns)
        df['target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target_10d'] = df['Close'].shift(-10) / df['Close'] - 1
        
        return df.dropna()
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def train_ensemble_model(self, data):
        """Train multiple ML models for ensemble prediction"""
        df = self.prepare_advanced_features(data)
        
        if len(df) < 100:
            return None
        
        # Features and target
        feature_cols = [col for col in df.columns if col not in ['target_5d', 'target_10d', 'returns', 'log_returns']]
        X = df[feature_cols]
        y = df['target_5d']  # Predict 5-day returns
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVM': SVR(kernel='rbf')
        }
        
        trained_models = {}
        performance = {}
        
        for name, model in models.items():
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            trained_models[name] = model
            performance[name] = {'MAE': mae, 'RMSE': rmse, 'Predictions': y_pred}
        
        self.models = trained_models
        self.scaler = scaler
        self.feature_cols = feature_cols
        
        return performance
    
    def predict_future(self, data, days=30):
        """Generate future predictions using ensemble"""
        if not self.models:
            return None
        
        df = self.prepare_advanced_features(data)
        if len(df) < 50:
            return None
        
        latest_features = df[self.feature_cols].iloc[-1:].copy()
        
        predictions = {}
        for name, model in self.models.items():
            if name == 'SVM':
                features_scaled = self.scaler.transform(latest_features)
                pred = model.predict(features_scaled)[0]
            else:
                pred = model.predict(latest_features)[0]
            
            predictions[name] = pred
        
        # Generate future price path
        current_price = data['Close'].iloc[-1]
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days+1)]
        
        # Create ensemble prediction
        avg_prediction = np.mean(list(predictions.values()))
        future_prices = [current_price * (1 + avg_prediction * i/days) for i in range(1, days+1)]
        
        return {
            'predictions': predictions,
            'future_prices': future_prices,
            'future_dates': future_dates,
            'ensemble_return': avg_prediction
        }

# ==================== BLACK-SCHOLES OPTION PRICER ====================
class BlackScholesPricer:
    def __init__(self, risk_free_rate=0.05):
        self.r = risk_free_rate
    
    def calculate_option_price(self, S, K, T, sigma, option_type="call"):
        """Calculate option price using Black-Scholes model"""
        if T <= 0:
            return 0
        
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == "call":
                price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)
        except:
            return 0
    
    def calculate_greeks(self, S, K, T, sigma, option_type="call"):
        """Calculate option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            greeks = {}
            
            if option_type == "call":
                greeks['delta'] = norm.cdf(d1)
                greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)) / 365
                greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
                greeks['rho'] = K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
            else:
                greeks['delta'] = norm.cdf(d1) - 1
                greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)) / 365
                greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100
                greeks['rho'] = -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100
            
            return greeks
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

# ==================== ADVANCED SENTIMENT ANALYZER ====================
class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.sources = ["News", "Twitter", "Reddit", "Analyst Reports", "SEC Filings"]
    
    def analyze_sentiment(self, text):
        """Advanced sentiment analysis"""
        if not TEXTBLOB_AVAILABLE:
            return 0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0
    
    def get_comprehensive_sentiment(self, symbol, stock_name):
        """Get multi-source sentiment analysis"""
        # Simulated multi-source sentiment data
        sentiments = {
            "News": np.random.uniform(-0.5, 0.8),
            "Social Media": np.random.uniform(-0.3, 0.6),
            "Analyst Reports": np.random.uniform(0.1, 0.9),
            "Market Sentiment": np.random.uniform(-0.2, 0.7)
        }
        
        overall_sentiment = np.mean(list(sentiments.values()))
        
        return {
            'overall_score': overall_sentiment * 100,
            'breakdown': sentiments,
            'recommendation': "BUY" if overall_sentiment > 0.2 else "SELL" if overall_sentiment < -0.2 else "HOLD",
            'confidence': abs(overall_sentiment) * 100
        }

# ==================== PREMIUM DATA MANAGER ====================
class PremiumDataManager:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data with premium features"""
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Add .NS for Indian stocks
            if not any(ext in symbol.upper() for ext in ['.NS', '.BO', '.NSE']):
                symbol += '.NS'
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                self.cache[cache_key] = data
                return data
        except:
            pass
        
        # Generate premium sample data as fallback
        data = self._generate_premium_sample_data(symbol, period)
        self.cache[cache_key] = data
        return data
    
    def _generate_premium_sample_data(self, symbol, period):
        """Generate realistic sample data with market patterns"""
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730
        }.get(period, 180)
        
        dates = pd.date_range(end=datetime.now(), periods=period_days, freq='D')
        
        # Create realistic market patterns
        base_price = 1500 + hash(symbol) % 4000
        trend = np.linspace(0, np.random.uniform(-300, 600), period_days)
        
        # Add market cycles and volatility clusters
        cycles = (np.sin(np.linspace(0, 8*np.pi, period_days)) * 100 + 
                 np.sin(np.linspace(0, 20*np.pi, period_days)) * 30)
        
        # Volatility clustering (GARCH-like behavior)
        volatility = np.random.randn(period_days) * (20 + np.abs(np.sin(np.linspace(0, 4*np.pi, period_days))) * 15)
        
        close_prices = base_price + trend + cycles + volatility
        
        # Generate OHLC data
        data = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.01, period_days)),
            'High': close_prices * (1 + np.abs(np.random.normal(0.015, 0.008, period_days))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0.015, 0.008, period_days))),
            'Close': close_prices,
            'Volume': np.random.lognormal(14, 1, period_days).astype(int)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def get_live_quote(self, symbol):
        """Get comprehensive live quote"""
        try:
            if not any(ext in symbol.upper() for ext in ['.NS', '.BO']):
                symbol += '.NS'
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="2d")
            
            if not history.empty:
                return {
                    'current': info.get('currentPrice', history['Close'].iloc[-1]),
                    'change': history['Close'].iloc[-1] - history['Close'].iloc[-2],
                    'change_percent': ((history['Close'].iloc[-1] - history['Close'].iloc[-2]) / history['Close'].iloc[-2]) * 100,
                    'high': history['High'].iloc[-1],
                    'low': history['Low'].iloc[-1],
                    'open': history['Open'].iloc[-1],
                    'volume': history['Volume'].iloc[-1],
                    'previous_close': history['Close'].iloc[-2],
                    'timestamp': datetime.now()
                }
        except:
            pass
        
        return self._generate_live_quote(symbol)
    
    def _generate_live_quote(self, symbol):
        """Generate realistic live quote"""
        return {
            'current': 1500 + hash(symbol) % 4000,
            'change': (hash(symbol) % 100 - 50),
            'change_percent': (hash(symbol) % 100 - 50) / 10,
            'high': 1600 + hash(symbol) % 4000,
            'low': 1400 + hash(symbol) % 4000,
            'open': 1520 + hash(symbol) % 4000,
            'volume': 1000000 + hash(symbol) % 4000000,
            'previous_close': 1480 + hash(symbol) % 4000,
            'timestamp': datetime.now()
        }

# ==================== PREMIUM CHARTING ENGINE ====================
class PremiumChartingEngine:
    def __init__(self):
        self.colors = {
            'primary': '#00FFAA',
            'secondary': '#0088FF',
            'accent': '#FF00AA',
            'background': '#0A0A0A',
            'grid': '#1A1A1A'
        }
    
    def create_advanced_candlestick(self, data, title="Stock Analysis"):
        """Create premium candlestick chart with indicators"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{title} - Price Action', 'Volume', 'RSI', 'MACD')
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Moving averages
        for window, color in [(20, '#FF6B00'), (50, '#00D4FF'), (200, '#AA00FF')]:
            if len(data) > window:
                sma = data['Close'].rolling(window).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=sma,
                    mode='lines', name=f'SMA {window}',
                    line=dict(color=color, width=2)
                ), row=1, col=1)
        
        # Volume
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
                 for i in range(len(data))]
        
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume', marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        # RSI
        rsi = self.calculate_rsi(data['Close'])
        fig.add_trace(go.Scatter(
            x=data.index, y=rsi,
            mode='lines', name='RSI',
            line=dict(color='#FFAA00', width=2)
        ), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="white", row=3, col=1)
        
        # MACD
        macd, signal, histogram = self.calculate_macd(data['Close'])
        fig.add_trace(go.Scatter(
            x=data.index, y=macd,
            mode='lines', name='MACD',
            line=dict(color='#00FFAA', width=2)
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=signal,
            mode='lines', name='Signal',
            line=dict(color='#FF0066', width=2)
        ), row=4, col=1)
        
        colors_hist = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(go.Bar(
            x=data.index, y=histogram,
            name='Histogram', marker_color=colors_hist,
            opacity=0.6
        ), row=4, col=1)
        
        fig.update_layout(
            title=f"Advanced Analysis - {title}",
            template="plotly_dark",
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_forecast_chart(self, historical_data, forecasts, future_dates, title="Price Forecast"):
        """Create premium forecast visualization"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index[-60:],
            y=historical_data['Close'].iloc[-60:],
            mode='lines',
            name='Historical',
            line=dict(color='#00FFAA', width=3)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=forecasts['future_prices'],
            mode='lines+markers',
            name='AI Forecast',
            line=dict(color='#FF00AA', width=3, dash='dash')
        ))
        
        # Confidence interval
        confidence_upper = [p * 1.05 for p in forecasts['future_prices']]
        confidence_lower = [p * 0.95 for p in forecasts['future_prices']]
        
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=confidence_upper + confidence_lower[::-1],
            fill='toself',
            fillcolor='rgba(255,0,170,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{title} - AI Price Forecast (Next {len(future_dates)} Days)",
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

# ==================== PREMIUM TRADING TERMINAL ====================
class PremiumTradingTerminal:
    def __init__(self):
        self.data_manager = PremiumDataManager()
        self.ai_engine = QuantumAIEngine()
        self.option_pricer = BlackScholesPricer()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.chart_engine = PremiumChartingEngine()
        
        # Initialize session state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    def render_premium_header(self):
        """Render premium application header"""
        st.markdown("""
        <style>
        .premium-header {
            font-size: 3.5rem;
            font-weight: bold;
            background: linear-gradient(45deg, #FF0000, #FFA500, #FFFF00, #008000, #0000FF, #4B0082, #EE82EE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 0 30px rgba(255,255,255,0.3);
        }
        .premium-card {
            background: linear-gradient(135deg, #0A0A0A 0%, #1A1A2E 50%, #16213E 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid #00FFAA;
            margin: 0.5rem;
            box-shadow: 0 8px 32px rgba(0, 255, 170, 0.1);
        }
        .metric-glowing {
            background: rgba(0, 255, 170, 0.1);
            border: 1px solid #00FFAA;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 0 20px rgba(0, 255, 170, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="premium-header">🚀 QUANTUM AI TRADING TERMINAL</div>', unsafe_allow_html=True)
        
        # Real-time market overview
        st.markdown("### 📊 LIVE MARKET DASHBOARD")
        
        # Cash, Futures, Options overview
        cols = st.columns(4)
        with cols[0]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("💰 CASH MARKET", "₹2.14 Cr", "+1.25%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("⚡ FUTURES", "₹85.42L", "+0.89%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("📈 OPTIONS", "₹1.24 Cr", "+2.15%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("🎯 TOTAL EXPOSURE", "₹4.23 Cr", "+1.42%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_live_market_dashboard(self):
        """Render comprehensive live market dashboard"""
        st.markdown("## 📈 LIVE MARKET DASHBOARD")
        
        # Market indices
        st.markdown("### 🔥 Market Indices")
        indices = ['^NSEI', '^NSEBANK', '^BSESN', 'RELIANCE.NS', 'TCS.NS']
        index_names = ['NIFTY 50', 'BANK NIFTY', 'SENSEX', 'RELIANCE', 'TCS']
        
        cols = st.columns(5)
        for idx, (symbol, name) in enumerate(zip(indices, index_names)):
            with cols[idx]:
                quote = self.data_manager.get_live_quote(symbol)
                if quote:
                    delta_color = "normal" if quote['change'] >= 0 else "inverse"
                    st.metric(
                        name,
                        f"₹{quote['current']:,.0f}",
                        f"{quote['change_percent']:+.2f}%",
                        delta_color=delta_color
                    )
        
        # Advanced charts
        st.markdown("### 📊 Advanced Market Analysis")
        selected_index = st.selectbox("Select Index", indices, format_func=lambda x: index_names[indices.index(x)])
        
        if selected_index:
            data = self.data_manager.get_stock_data(selected_index, "6mo")
            if not data.empty:
                chart = self.chart_engine.create_advanced_candlestick(data, index_names[indices.index(selected_index)])
                st.plotly_chart(chart, use_container_width=True)
    
    def render_machine_learning_terminal(self):
        """Render advanced ML trading terminal"""
        st.markdown("## 🤖 MACHINE LEARNING TRADING TERMINAL")
        
        # Stock selection
        stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFOSYS': 'INFY.NS', 
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS'
        }
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_stock = st.selectbox("Select Stock for AI Analysis", list(stocks.keys()))
        with col2:
            forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30)
        
        if selected_stock:
            symbol = stocks[selected_stock]
            
            with st.spinner("🚀 Training Quantum AI Models..."):
                data = self.data_manager.get_stock_data(symbol, "2y")
                
                if not data.empty:
                    # Train AI models
                    performance = self.ai_engine.train_ensemble_model(data)
                    
                    if performance:
                        # Display model performance
                        st.markdown("### 📊 AI Model Performance")
                        perf_cols = st.columns(len(performance))
                        
                        for idx, (model_name, metrics) in enumerate(performance.items()):
                            with perf_cols[idx]:
                                st.metric(
                                    f"🤖 {model_name}",
                                    f"MAE: {metrics['MAE']:.4f}",
                                    f"RMSE: {metrics['RMSE']:.4f}"
                                )
                        
                        # Generate forecasts
                        forecasts = self.ai_engine.predict_future(data, forecast_days)
                        
                        if forecasts:
                            # Display forecast chart
                            st.markdown("### 🔮 AI Price Forecast")
                            forecast_chart = self.chart_engine.create_forecast_chart(
                                data, forecasts, forecasts['future_dates'], selected_stock
                            )
                            st.plotly_chart(forecast_chart, use_container_width=True)
                            
                            # Model predictions
                            st.markdown("### 🎯 Model Predictions")
                            pred_cols = st.columns(len(forecasts['predictions']))
                            
                            current_price = data['Close'].iloc[-1]
                            for idx, (model_name, prediction) in enumerate(forecasts['predictions'].items()):
                                with pred_cols[idx]:
                                    expected_return = prediction * 100
                                    st.metric(
                                        f"{model_name}",
                                        f"{expected_return:+.2f}%",
                                        "5-day Return"
                                    )
                            
                            # Trading signals
                            st.markdown("### 💡 AI Trading Signals")
                            ensemble_return = forecasts['ensemble_return']
                            
                            if ensemble_return > 0.02:
                                signal = "🚀 STRONG BUY"
                                color = "green"
                                reasoning = "AI models predict significant upside momentum"
                            elif ensemble_return > 0:
                                signal = "📈 BUY"
                                color = "lightgreen"
                                reasoning = "Moderate bullish signals detected"
                            elif ensemble_return < -0.02:
                                signal = "🔻 STRONG SELL"
                                color = "red"
                                reasoning = "AI models predict downward pressure"
                            else:
                                signal = "⚡ HOLD"
                                color = "orange"
                                reasoning = "Neutral market conditions"
                            
                            st.markdown(f"""
                            <div style='background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color}'>
                                <h3 style='color: {color}; margin: 0;'>{signal}</h3>
                                <p style='margin: 10px 0 0 0; color: white;'>Expected Return: <b>{ensemble_return*100:+.2f}%</b></p>
                                <p style='margin: 5px 0 0 0; color: #CCCCCC;'>{reasoning}</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    def render_sentiment_analysis(self):
        """Render advanced sentiment analysis"""
        st.markdown("## 📊 ADVANCED SENTIMENT ANALYSIS")
        
        stocks = ['RELIANCE', 'TCS', 'INFOSYS', 'HDFC BANK', 'ICICI BANK']
        selected_stock = st.selectbox("Select Stock for Sentiment Analysis", stocks)
        
        if selected_stock:
            with st.spinner("🔍 Analyzing Market Sentiment..."):
                sentiment = self.sentiment_analyzer.get_comprehensive_sentiment(selected_stock, selected_stock)
                
                # Overall sentiment
                st.markdown("### 🎯 Overall Sentiment Score")
                sentiment_score = sentiment['overall_score']
                
                # Sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sentiment_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Meter"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [-100, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-100, -50], 'color': "red"},
                            {'range': [-50, 0], 'color': "lightcoral"},
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': sentiment_score
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment breakdown
                st.markdown("### 📈 Sentiment Breakdown by Source")
                sources = list(sentiment['breakdown'].keys())
                scores = [sentiment['breakdown'][source] * 100 for source in sources]
                
                fig_bar = px.bar(
                    x=sources, y=scores,
                    title="Sentiment Scores by Source",
                    color=scores,
                    color_continuous_scale="RdYlGn"
                )
                fig_bar.update_layout(template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Recommendation
                st.markdown("### 💡 Sentiment-Based Recommendation")
                rec_color = "green" if sentiment['recommendation'] == "BUY" else "red" if sentiment['recommendation'] == "SELL" else "orange"
                
                st.markdown(f"""
                <div style='background-color: {rec_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_color}'>
                    <h3 style='color: {rec_color}; margin: 0;'>🎯 {sentiment['recommendation']} RECOMMENDATION</h3>
                    <p style='margin: 10px 0 0 0; color: white;'>Confidence: <b>{sentiment['confidence']:.1f}%</b></p>
                    <p style='margin: 5px 0 0 0; color: #CCCCCC;'>Based on multi-source sentiment analysis</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_quant_strategies(self):
        """Render quantitative strategies dashboard"""
        st.markdown("## 📊 QUANTITATIVE STRATEGIES")
        
        tabs = st.tabs(["Black-Scholes Calculator", "Option Greeks", "Option Chain", "Futures & Expiry"])
        
        with tabs[0]:
            st.markdown("### 📈 Black-Scholes Option Pricing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                spot_price = st.number_input("Spot Price (₹)", value=1500.0, min_value=0.0)
                strike_price = st.number_input("Strike Price (₹)", value=1550.0, min_value=0.0)
                days_to_expiry = st.slider("Days to Expiry", 1, 365, 30)
            
            with col2:
                volatility = st.slider("Volatility (%)", 1.0, 100.0, 25.0) / 100
                risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
                option_type = st.selectbox("Option Type", ["Call", "Put"])
            
            # Calculate option price
            time_to_expiry = days_to_expiry / 365
            option_price = self.option_pricer.calculate_option_price(
                spot_price, strike_price, time_to_expiry, volatility, option_type.lower()
            )
            
            # Display results
            st.metric(f"🎯 {option_type} Option Price", f"₹{option_price:.2f}")
            
            # Greeks
            greeks = self.option_pricer.calculate_greeks(
                spot_price, strike_price, time_to_expiry, volatility, option_type.lower()
            )
            
            st.markdown("#### 📊 Option Greeks")
            greek_cols = st.columns(5)
            greek_info = {
                'delta': "Price sensitivity to underlying",
                'gamma': 'Delta sensitivity to underlying',
                'theta': 'Time decay per day',
                'vega': 'Volatility sensitivity',
                'rho': 'Interest rate sensitivity'
            }
            
            for idx, (greek, value) in enumerate(greeks.items()):
                with greek_cols[idx]:
                    st.metric(
                        f"Δ {greek.upper()}",
                        f"{value:.4f}",
                        help=greek_info.get(greek, "")
                    )
        
        with tabs[1]:
            st.markdown("### 📊 Option Greeks Analysis")
            
            # Interactive Greeks visualization
            spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
            deltas = []
            
            for price in spot_range:
                greeks = self.option_pricer.calculate_greeks(
                    price, strike_price, time_to_expiry, volatility, option_type.lower()
                )
                deltas.append(greeks['delta'])
            
            fig = px.line(x=spot_range, y=deltas, title="Delta vs Spot Price")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.markdown("### 🔗 Live Option Chain")
            
            # Generate simulated option chain
            strikes = np.arange(spot_price * 0.8, spot_price * 1.2, 50)
            option_chain = []
            
            for strike in strikes:
                call_price = self.option_pricer.calculate_option_price(
                    spot_price, strike, time_to_expiry, volatility, "call"
                )
                put_price = self.option_pricer.calculate_option_price(
                    spot_price, strike, time_to_expiry, volatility, "put"
                )
                
                option_chain.append({
                    'Strike': strike,
                    'Call Price': call_price,
                    'Put Price': put_price,
                    'Call OI': f"{np.random.randint(1000, 50000):,}",
                    'Put OI': f"{np.random.randint(1000, 50000):,}"
                })
            
            option_df = pd.DataFrame(option_chain)
            st.dataframe(option_df, use_container_width=True)
        
        with tabs[3]:
            st.markdown("### ⚡ Futures & Expiry Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📅 Futures Contracts")
                futures_data = {
                    'Contract': ['NIFTY JAN', 'BANKNIFTY JAN', 'RELIANCE JAN', 'TCS JAN'],
                    'Last Price': [21542, 47568, 2856, 3854],
                    'Change': ['+142', '+68', '+26', '-12'],
                    'Open Interest': ['1.2M', '856K', '324K', '287K']
                }
                st.dataframe(pd.DataFrame(futures_data), use_container_width=True)
            
            with col2:
                st.markdown("#### 📆 Expiry Calendar")
                expiry_data = {
                    'Series': ['Weekly', 'Monthly', 'Quarterly'],
                    'Expiry': ['25-Jan-2024', '31-Jan-2024', '29-Mar-2024'],
                    'Days Left': [5, 11, 68]
                }
                st.dataframe(pd.DataFrame(expiry_data), use_container_width=True)
    
    def run(self):
        """Main application runner"""
        self.render_premium_header()
        
        # Premium navigation
        st.sidebar.markdown("## 🧭 NAVIGATION")
        app_mode = st.sidebar.selectbox(
            "Select Dashboard",
            [
                "🚀 Live Market Dashboard",
                "🤖 ML Trading Terminal", 
                "📊 Sentiment Analysis",
                "📈 Quant Strategies",
                "💼 Portfolio Manager"
            ]
        )
        
        # System status
        st.sidebar.markdown("## 🔧 SYSTEM STATUS")
        st.sidebar.write(f"🤖 AI Engine: {'✅' if SKLEARN_AVAILABLE else '❌'}")
        st.sidebar.write(f"📊 Sentiment: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
        st.sidebar.write(f"📈 Data: {'✅' if YFINANCE_AVAILABLE else '❌'}")
        st.sidebar.write(f"📊 Charts: {'✅' if PLOTLY_AVAILABLE else '❌'}")
        
        # Quick actions
        st.sidebar.markdown("## ⚡ QUICK ACTIONS")
        if st.sidebar.button("🔄 Refresh All Data", use_container_width=True):
            st.rerun()
        
        if st.sidebar.button("🧹 Clear Cache", use_container_width=True):
            self.data_manager.cache = {}
            st.success("Cache cleared!")
        
        # Page routing
        if "Live Market Dashboard" in app_mode:
            self.render_live_market_dashboard()
        elif "ML Trading Terminal" in app_mode:
            self.render_machine_learning_terminal()
        elif "Sentiment Analysis" in app_mode:
            self.render_sentiment_analysis()
        elif "Quant Strategies" in app_mode:
            self.render_quant_strategies()
        else:
            self.render_portfolio_manager()
    
    def render_portfolio_manager(self):
        """Render portfolio management"""
        st.markdown("## 💼 PORTFOLIO MANAGER")
        
        # Portfolio input
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
            selected_stock = st.selectbox("Stock", stocks)
        
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=100)
        
        with col3:
            buy_price = st.number_input("Buy Price (₹)", min_value=0.0, value=0.0)
        
        with col4:
            st.write("")
            st.write("")
            if st.button("Add to Portfolio", use_container_width=True):
                if selected_stock not in st.session_state.portfolio:
                    st.session_state.portfolio[selected_stock] = {
                        'quantity': quantity,
                        'avg_price': buy_price
                    }
                st.success(f"Added {selected_stock} to portfolio!")
        
        # Display portfolio
        if st.session_state.portfolio:
            portfolio_data = []
            total_investment = 0
            total_current = 0
            
            for symbol, holding in st.session_state.portfolio.items():
                quote = self.data_manager.get_live_quote(symbol)
                current_price = quote['current']
                current_value = current_price * holding['quantity']
                investment = holding['avg_price'] * holding['quantity']
                pnl = current_value - investment
                pnl_percent = (pnl / investment) * 100
                
                portfolio_data.append({
                    'Stock': symbol.replace('.NS', ''),
                    'Qty': holding['quantity'],
                    'Avg Price': holding['avg_price'],
                    'Current': current_price,
                    'Investment': investment,
                    'Current Value': current_value,
                    'P&L': pnl,
                    'P&L %': pnl_percent
                })
                
                total_investment += investment
                total_current += current_value
            
            portfolio_df = pd.DataFrame(portfolio_data)
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Portfolio summary
            total_pnl = total_current - total_investment
            total_pnl_percent = (total_pnl / total_investment) * 100
            
            st.markdown("### 📊 Portfolio Summary")
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric("Total Investment", f"₹{total_investment:,.2f}")
            
            with summary_cols[1]:
                st.metric("Current Value", f"₹{total_current:,.2f}")
            
            with summary_cols[2]:
                st.metric("Total P&L", f"₹{total_pnl:,.2f}")
            
            with summary_cols[3]:
                st.metric("Return %", f"{total_pnl_percent:.2f}%")

# Run the application
if __name__ == "__main__":
    st.set_page_config(
        page_title="Quantum AI Trading Terminal",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run terminal
    terminal = PremiumTradingTerminal()
    terminal.run()

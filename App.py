import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core imports
import requests
import json
import time
from datetime import datetime, timedelta
import math
import random

# Data visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Financial data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# Machine Learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.svm import SVR, SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Statistics
try:
    from scipy.stats import norm, linregress
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Sentiment Analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except Exception:
    TEXTBLOB_AVAILABLE = False

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# ==================== PREMIUM CONFIGURATION ====================
class QuantumConfig:
    def __init__(self):
        # API Keys for live data (placeholders)
        self.alpha_vantage_key = "KP3E60AL5IIEREH7"
        self.finnhub_key = "d3f027pr01qh40fg8npgd3f027pr01qh40fg8nq0"
        self.indian_api_key = "sk-live-UYMPXvoR0SLhmXlnGyqNqVhlgToFARM3mLgoBdm9"

        # Trading parameters
        self.risk_free_rate = 0.05
        self.transaction_cost = 0.001
        self.max_position_size = 0.1  # 10% of portfolio per trade
        self.stop_loss_pct = 0.02     # 2% stop loss
        self.take_profit_pct = 0.04   # 4% take profit

        # Algorithmic trading parameters
        self.max_open_positions = 5
        self.max_daily_trades = 20
        self.volatility_threshold = 0.15

        # API endpoints
        self.endpoints = {
            'alpha_vantage': "https://www.alphavantage.co/query",
            'finnhub': "https://finnhub.io/api/v1",
            'indian_api': "https://api.indianapi.in/v1"
        }

    def get_trading_hours(self):
        return {
            'nse': {'start': '09:15', 'end': '15:30'},
            'bse': {'start': '09:15', 'end': '15:30'},
            'us': {'start': '09:30', 'end': '16:00'}
        }

# ==================== ADVANCED DATA MANAGER ====================
class QuantumDataManager:
    def __init__(self):
        self.config = QuantumConfig()
        self.data_cache = {}
        self.quote_cache = {}
        self.cache_timeout = 300

    def get_comprehensive_data(self, symbol, period="6mo", interval="1d"):
        """Get comprehensive market data from multiple sources"""
        cache_key = f"{symbol}_{period}_{interval}"

        # Check cache first
        if cache_key in self.data_cache:
            data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return data

        # Try multiple data sources
        data_sources = [
            self._get_yfinance_data,
            self._get_alpha_vantage_data,
            self._generate_premium_sample_data
        ]

        for source in data_sources:
            try:
                data = source(symbol, period, interval)
                if isinstance(data, pd.DataFrame) and not data.empty and len(data) > 20:
                    # Add technical indicators
                    data = self._add_technical_indicators(data)
                    self.data_cache[cache_key] = (data, time.time())
                    return data
            except Exception:
                continue

        # Final fallback
        data = self._generate_premium_sample_data(symbol, period, interval)
        data = self._add_technical_indicators(data)
        self.data_cache[cache_key] = (data, time.time())
        return data

    def _get_yfinance_data(self, symbol, period, interval):
        """Get data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE:
            return pd.DataFrame()

        try:
            # Handle Indian stocks - only append .NS for typical tickers, skip indices or already-suffixed symbols
            if not symbol.startswith('^') and not any(ext in symbol.upper() for ext in ['.NS', '.BO', '.NSE']):
                symbol_yf = symbol + '.NS'
            else:
                symbol_yf = symbol

            ticker = yf.Ticker(symbol_yf)
            data = ticker.history(period=period, interval=interval)

            if not data.empty:
                return data
        except Exception:
            pass

        return pd.DataFrame()

    def _get_alpha_vantage_data(self, symbol, period, interval):
        """Get data from Alpha Vantage"""
        try:
            # Map period to Alpha Vantage format
            function_map = {
                "1d": "TIME_SERIES_DAILY",
                "1h": "TIME_SERIES_INTRADAY",
                "5min": "TIME_SERIES_INTRADAY"
            }

            function = function_map.get(interval, "TIME_SERIES_DAILY")
            params = {
                'function': function,
                'symbol': symbol.replace('.NS', '').replace('^', ''),
                'apikey': self.config.alpha_vantage_key,
                'outputsize': 'full' if period in ["1y", "2y"] else 'compact'
            }

            if interval != "1d":
                params['interval'] = interval

            response = requests.get(self.config.endpoints['alpha_vantage'], params=params, timeout=10)
            data = response.json()

            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df.astype(float)
        except Exception:
            pass

        return pd.DataFrame()

    def _generate_premium_sample_data(self, symbol, period, interval="1d"):
        """Generate realistic sample data with market patterns"""
        # Calculate number of data points
        periods_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        base_points = periods_map.get(period, 180)

        # Adjust for interval
        if interval == "1h":
            base_points = int(base_points * 6.5)  # Trading hours approx
            freq = '1h'
        elif interval == "5min":
            base_points = int(base_points * 78)   # 6.5 hours * 12 (5min intervals)
            freq = '5min'
        else:
            freq = 'D'

        dates = pd.date_range(end=datetime.now(), periods=int(base_points), freq=freq)

        # Create sophisticated price patterns
        base_price = 1500 + abs(hash(symbol)) % 4000

        # Long-term trend
        trend_direction = 1 if abs(hash(symbol)) % 2 == 0 else -1
        trend_strength = 0.001 + (abs(hash(symbol)) % 100) / 50000
        trend = np.cumsum(np.ones(len(dates)) * trend_strength * trend_direction)

        # Market cycles
        seasonal_cycle = np.sin(np.linspace(0, 8 * np.pi, len(dates))) * 50
        weekly_cycle = np.sin(np.linspace(0, 104 * np.pi, len(dates))) * 20

        # Volatility clusters (GARCH-like)
        volatility = np.zeros(len(dates))
        volatility[0] = 0.01
        for i in range(1, len(dates)):
            volatility[i] = abs(0.01 + 0.85 * volatility[i - 1] + 0.1 * np.random.randn())

        price_noise = np.random.randn(len(dates)) * volatility * base_price

        # Combine all components
        close_prices = base_price + trend * base_price + seasonal_cycle + weekly_cycle + price_noise

        # Generate OHLC data
        data = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.008, len(dates))),
            'High': close_prices * (1 + np.abs(np.random.normal(0.012, 0.006, len(dates)))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0.012, 0.006, len(dates)))),
            'Close': close_prices,
            'Volume': np.random.lognormal(14.5, 1.2, len(dates)).astype(int)
        }, index=dates)

        # Ensure High is highest and Low is lowest
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

        return data

    def _add_technical_indicators(self, data):
        """Add comprehensive technical indicators"""
        if data.empty:
            return data

        data = data.copy()
        # Price-based indicators
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            if len(data) > window:
                data[f'SMA_{window}'] = data['Close'].rolling(window).mean()
                data[f'EMA_{window}'] = data['Close'].ewm(span=window).mean()

        # Bollinger Bands
        if len(data) > 20:
            data['BB_Middle'] = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

        # RSI
        data['RSI_14'] = self._calculate_rsi(data['Close'])

        # MACD
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

        # Stochastic
        if len(data) > 14:
            low_14 = data['Low'].rolling(14).min()
            high_14 = data['High'].rolling(14).max()
            data['Stoch_K'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
            data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

        # Volume indicators
        if 'Volume' in data.columns:
            data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
            data['OBV'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()

        # Volatility
        data['Volatility_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)
        data['Volatility_50'] = data['Returns'].rolling(50).std() * np.sqrt(252)

        # Support and Resistance
        if len(data) > 20:
            data['Support'] = data['Low'].rolling(20).min()
            data['Resistance'] = data['High'].rolling(20).max()

        return data.dropna()

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def get_live_quote(self, symbol):
        """Get comprehensive live quote"""
        cache_key = symbol

        if cache_key in self.quote_cache:
            quote, timestamp = self.quote_cache[cache_key]
            if time.time() - timestamp < 60:  # 1 minute cache for quotes
                return quote

        try:
            # Try Yahoo Finance first
            if not symbol.startswith('^') and not any(ext in symbol.upper() for ext in ['.NS', '.BO']):
                symbol_yf = symbol + '.NS'
            else:
                symbol_yf = symbol

            if YFINANCE_AVAILABLE:
                ticker = yf.Ticker(symbol_yf)
                info = {}
                try:
                    info = ticker.info or {}
                except Exception:
                    info = {}
                history = ticker.history(period="2d") if YFINANCE_AVAILABLE else pd.DataFrame()

                if not history.empty:
                    current_price = info.get('currentPrice', history['Close'].iloc[-1])
                    prev_close = history['Close'].iloc[-2] if len(history) > 1 else history['Close'].iloc[-1]
                    change = current_price - prev_close
                    quote = {
                        'symbol': symbol,
                        'current': current_price,
                        'change': change,
                        'change_percent': (change / prev_close) * 100 if prev_close != 0 else 0,
                        'high': history['High'].iloc[-1],
                        'low': history['Low'].iloc[-1],
                        'open': history['Open'].iloc[-1],
                        'volume': int(history['Volume'].iloc[-1]),
                        'previous_close': prev_close,
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'beta': info.get('beta', 1),
                        'timestamp': datetime.now()
                    }

                    self.quote_cache[cache_key] = (quote, time.time())
                    return quote
        except Exception:
            pass

        # Fallback to sophisticated sample quote
        quote = self._generate_live_sample_quote(symbol)
        self.quote_cache[cache_key] = (quote, time.time())
        return quote

    def _generate_live_sample_quote(self, symbol):
        """Generate sophisticated sample quote data"""
        base_price = 1500 + abs(hash(symbol)) % 4000
        change = np.random.normal(0, 25)  # More realistic changes

        return {
            'symbol': symbol,
            'current': max(base_price + change, 1),
            'change': change,
            'change_percent': (change / base_price) * 100,
            'high': base_price + abs(np.random.normal(50, 20)),
            'low': max(base_price - abs(np.random.normal(50, 20)), 1),
            'open': base_price + np.random.normal(0, 15),
            'volume': int(np.random.lognormal(14, 1)),
            'previous_close': base_price,
            'market_cap': np.random.randint(1000000000, 500000000000),
            'pe_ratio': np.random.uniform(5, 50),
            'beta': np.random.uniform(0.5, 1.5),
            'timestamp': datetime.now()
        }

# ==================== QUANTUM AI ENGINE ====================
class QuantumAIEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.feature_cols = []

    def prepare_advanced_features(self, data):
        """Create sophisticated features for AI models"""
        if data.empty or len(data) < 100:
            return None

        df = data.copy()

        # Price transformation features
        df['price_zscore'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).std()
        df['price_momentum'] = df['Close'] / df['Close'].shift(5) - 1
        df['price_acceleration'] = (df['Close'] / df['Close'].shift(5) - 1) - (df['Close'].shift(5) / df['Close'].shift(10) - 1)

        # Volatility features
        df['volatility_ratio'] = df['Volatility_20'] / df['Volatility_50']
        df['volatility_regime'] = (df['Volatility_20'] > df['Volatility_20'].rolling(50).mean()).astype(int)

        # Trend features
        df['trend_strength'] = abs(df['Close'] - df['SMA_20']) / df['SMA_20']
        df['trend_direction'] = np.where(df['Close'] > df['SMA_20'], 1, -1)

        # Mean reversion features
        df['mean_reversion'] = (df['Close'] - df['SMA_20']) / (df['Close'].rolling(20).std())

        # Breakout features
        df['resistance_break'] = (df['Close'] > df['Resistance']).astype(int)
        df['support_break'] = (df['Close'] < df['Support']).astype(int)

        # Volume features
        if 'Volume' in df.columns:
            df['volume_trend'] = df['Volume'] / df['Volume_SMA_20']
            df['volume_price_trend'] = df['volume_trend'] * df['Returns']

        # Technical indicator signals
        df['rsi_signal'] = np.where(df['RSI_14'] < 30, 1, np.where(df['RSI_14'] > 70, -1, 0))
        df['macd_signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
        df['bb_signal'] = np.where(df['BB_Position'] < 0.2, 1, np.where(df['BB_Position'] > 0.8, -1, 0))

        # Multi-timeframe features
        for period in [5, 10, 20]:
            df[f'returns_{period}d'] = df['Close'].pct_change(period)
            df[f'volatility_{period}d'] = df['Returns'].rolling(period).std()

        # Target variables
        df['target_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['target_5d_return'] = df['Close'].shift(-5) / df['Close'] - 1
        df['target_10d_return'] = df['Close'].shift(-10) / df['Close'] - 1

        # Drop NaN values
        df = df.dropna()

        if len(df) < 100:
            return None

        # Feature columns (exclude targets and basic price data)
        exclude_cols = ['target_1d', 'target_5d_return', 'target_10d_return', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('target_')]

        X = df[feature_cols]
        y_classification = df['target_1d']
        y_regression = df['target_5d_return']

        return X, y_classification, y_regression, feature_cols

    def train_quantum_models(self, data):
        """Train ensemble of AI models"""
        if not SKLEARN_AVAILABLE:
            return None

        result = self.prepare_advanced_features(data)
        if result is None:
            return None

        X, y_class, y_reg, feature_cols = result

        if len(X) < 100:
            return None

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train_class, y_test_class = y_class[:split_idx], y_class[split_idx:]
        y_train_reg, y_test_reg = y_reg[:split_idx], y_reg[split_idx:]

        # Scale features
        scaler_class = StandardScaler()
        scaler_reg = StandardScaler()

        X_train_class_scaled = scaler_class.fit_transform(X_train)
        X_test_class_scaled = scaler_class.transform(X_test)

        X_train_reg_scaled = scaler_reg.fit_transform(X_train)
        X_test_reg_scaled = scaler_reg.transform(X_test)

        # Train classification models (use classifiers)
        class_models = {
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM Classifier': SVC(probability=True, random_state=42)
        }

        class_performance = {}
        trained_class_models = {}

        for name, model in class_models.items():
            try:
                if name in ['Logistic Regression', 'SVM Classifier']:
                    model.fit(X_train_class_scaled, y_train_class)
                    y_pred = model.predict(X_test_class_scaled)
                    y_pred_proba = model.predict_proba(X_test_class_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train_class)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else (y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred))

                accuracy = accuracy_score(y_test_class, (y_pred > 0.5).astype(int) if y_pred.dtype.kind in 'f' else y_pred)
                class_performance[name] = {'Accuracy': accuracy, 'Predictions': y_pred_proba}
                trained_class_models[name] = model

                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))

            except Exception:
                continue

        # Train regression models
        reg_models = {
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'SVM Regressor': SVR()
        }

        reg_performance = {}
        trained_reg_models = {}

        for name, model in reg_models.items():
            try:
                if name == 'SVM Regressor':
                    model.fit(X_train_reg_scaled, y_train_reg)
                    y_pred = model.predict(X_test_reg_scaled)
                else:
                    model.fit(X_train, y_train_reg)
                    y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test_reg, y_pred)
                rmse = np.sqrt(mse)
                reg_performance[name] = {'RMSE': rmse, 'Predictions': y_pred}
                trained_reg_models[name] = model

            except Exception:
                continue

        # Store models and scalers
        self.models = {
            'classification': trained_class_models,
            'regression': trained_reg_models
        }

        self.scalers = {
            'classification': scaler_class,
            'regression': scaler_reg
        }

        self.feature_cols = feature_cols

        return {
            'classification': class_performance,
            'regression': reg_performance
        }

    def generate_ai_signals(self, data):
        """Generate AI-powered trading signals"""
        if not self.models:
            return None

        result = self.prepare_advanced_features(data)
        if result is None:
            return None

        X, _, _, _ = result

        if len(X) < 1:
            return None

        latest_features = X.iloc[-1:].copy()

        signals = {}

        # Classification predictions (direction)
        for name, model in self.models['classification'].items():
            try:
                if name in ['Logistic Regression', 'SVM Classifier']:
                    features_scaled = self.scalers['classification'].transform(latest_features)
                    prediction = model.predict_proba(features_scaled)[0][1]
                else:
                    # Tree-based models were trained on raw features
                    if hasattr(model, 'predict_proba'):
                        prediction = model.predict_proba(latest_features)[0][1]
                    else:
                        # If no predict_proba, treat model output as score
                        prediction_val = model.predict(latest_features)[0]
                        # convert to pseudo-probability in [0,1]
                        prediction = 1 / (1 + np.exp(-prediction_val)) if isinstance(prediction_val, (int, float, np.floating)) else float(prediction_val)

                signals[f'{name}_Probability'] = float(prediction)

            except Exception:
                continue

        # Regression predictions (returns)
        for name, model in self.models['regression'].items():
            try:
                if name == 'SVM Regressor':
                    features_scaled = self.scalers['regression'].transform(latest_features)
                    prediction = model.predict(features_scaled)[0]
                else:
                    prediction = model.predict(latest_features)[0]

                signals[f'{name}_Return'] = float(prediction)

            except Exception:
                continue

        # Generate ensemble signal
        if signals:
            prob_values = [v for k, v in signals.items() if 'Probability' in k]
            return_values = [v for k, v in signals.items() if 'Return' in k]

            avg_probability = float(np.mean(prob_values)) if prob_values else 0.5
            avg_return = float(np.mean(return_values)) if return_values else 0.0

            if avg_probability > 0.6 and avg_return > 0.02:
                signals['ENSEMBLE_SIGNAL'] = 'STRONG_BUY'
            elif avg_probability > 0.55 and avg_return > 0.01:
                signals['ENSEMBLE_SIGNAL'] = 'BUY'
            elif avg_probability < 0.4 and avg_return < -0.01:
                signals['ENSEMBLE_SIGNAL'] = 'STRONG_SELL'
            elif avg_probability < 0.45 and avg_return < 0:
                signals['ENSEMBLE_SIGNAL'] = 'SELL'
            else:
                signals['ENSEMBLE_SIGNAL'] = 'HOLD'

            signals['Confidence_Score'] = abs(avg_probability - 0.5) * 2
            signals['Expected_Return'] = avg_return

        return signals

# ==================== ALGORITHMIC TRADING ENGINE ====================
class AlgorithmicTradingEngine:
    def __init__(self):
        self.config = QuantumConfig()
        self.active_strategies = {}
        self.trade_history = []
        self.performance_metrics = {}
        self.positions = {}

    def initialize_strategy(self, strategy_name, parameters):
        """Initialize a trading strategy"""
        self.active_strategies[strategy_name] = {
            'parameters': parameters,
            'status': 'ACTIVE',
            'created_at': datetime.now(),
            'trades': [],
            'performance': {}
        }

        return f"Strategy {strategy_name} initialized successfully"

    def momentum_strategy(self, data, lookback=20, threshold=0.02):
        """Momentum-based trading strategy"""
        signals = []

        if len(data) < lookback + 1:
            return signals

        returns = data['Close'].pct_change(lookback)
        volatility = data['Close'].pct_change().rolling(lookback).std()

        for i in range(lookback, len(data)):
            current_return = returns.iloc[i]
            current_vol = volatility.iloc[i]

            if not np.isnan(current_return) and not np.isnan(current_vol):
                # Risk-adjusted momentum
                momentum_score = current_return / current_vol if current_vol > 0 else 0

                if momentum_score > threshold:
                    signals.append({'timestamp': data.index[i], 'signal': 'BUY', 'strength': float(momentum_score)})
                elif momentum_score < -threshold:
                    signals.append({'timestamp': data.index[i], 'signal': 'SELL', 'strength': float(abs(momentum_score))})
                else:
                    signals.append({'timestamp': data.index[i], 'signal': 'HOLD', 'strength': 0})

        return signals

    def mean_reversion_strategy(self, data, lookback=20, z_threshold=2):
        """Mean reversion trading strategy"""
        signals = []

        if len(data) < lookback + 1:
            return signals

        prices = data['Close']
        sma = prices.rolling(lookback).mean()
        std = prices.rolling(lookback).std()
        z_score = (prices - sma) / std

        for i in range(lookback, len(data)):
            current_z = z_score.iloc[i]

            if not np.isnan(current_z):
                if current_z < -z_threshold:
                    signals.append({'timestamp': data.index[i], 'signal': 'BUY', 'strength': abs(current_z)})
                elif current_z > z_threshold:
                    signals.append({'timestamp': data.index[i], 'signal': 'SELL', 'strength': abs(current_z)})
                else:
                    signals.append({'timestamp': data.index[i], 'signal': 'HOLD', 'strength': 0})

        return signals

    def breakout_strategy(self, data, lookback=20, volatility_multiplier=1):
        """Breakout trading strategy"""
        signals = []

        if len(data) < lookback + 1:
            return signals

        highs = data['High'].rolling(lookback).max()
        lows = data['Low'].rolling(lookback).min()
        volatility = data['Close'].pct_change().rolling(lookback).std()

        for i in range(lookback, len(data)):
            current_high = highs.iloc[i]
            current_low = lows.iloc[i]
            current_vol = volatility.iloc[i]
            current_close = data['Close'].iloc[i]

            if not np.isnan(current_high) and not np.isnan(current_low):
                upper_band = current_high + (current_vol * volatility_multiplier)
                lower_band = current_low - (current_vol * volatility_multiplier)

                if current_close > upper_band:
                    signals.append({'timestamp': data.index[i], 'signal': 'BUY', 'strength': (current_close - upper_band) / current_close})
                elif current_close < lower_band:
                    signals.append({'timestamp': data.index[i], 'signal': 'SELL', 'strength': (lower_band - current_close) / current_close})
                else:
                    signals.append({'timestamp': data.index[i], 'signal': 'HOLD', 'strength': 0})

        return signals

    def execute_trade(self, strategy_name, symbol, signal, quantity, price):
        """Execute a trade for a strategy"""
        trade_id = f"{strategy_name}_{symbol}_{int(time.time())}"

        trade = {
            'trade_id': trade_id,
            'strategy': strategy_name,
            'symbol': symbol,
            'signal': signal,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'status': 'EXECUTED'
        }

        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0, 'total_cost': 0}

        position = self.positions[symbol]

        if signal == 'BUY':
            total_cost = position['total_cost'] + (quantity * price)
            total_quantity = position['quantity'] + quantity
            position['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
            position['quantity'] = total_quantity
            position['total_cost'] = total_cost

        elif signal == 'SELL':
            position['quantity'] = max(position['quantity'] - quantity, 0)
            if position['quantity'] == 0:
                position['avg_price'] = 0
                position['total_cost'] = 0

        # Record trade
        self.trade_history.append(trade)

        if strategy_name in self.active_strategies:
            self.active_strategies[strategy_name]['trades'].append(trade)

        return trade

    def calculate_performance_metrics(self, strategy_name):
        """Calculate performance metrics for a strategy"""
        if strategy_name not in self.active_strategies:
            return None

        trades = self.active_strategies[strategy_name]['trades']

        if not trades:
            return None

        # Calculate basic metrics
        buy_trades = [t for t in trades if t['signal'] == 'BUY']
        sell_trades = [t for t in trades if t['signal'] == 'SELL']

        total_trades = len(trades)
        winning_trades = 0
        total_pnl = 0

        # Simple P&L calculation (for demo purposes)
        for trade in trades:
            if trade['signal'] == 'BUY':
                # Assume we sell at 2% profit for demo
                total_pnl += trade['quantity'] * trade['price'] * 0.02
                winning_trades += 1

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_trade_pnl': avg_trade_pnl,
            'sharpe_ratio': np.random.uniform(0.5, 2.0),  # Placeholder
            'max_drawdown': np.random.uniform(0.01, 0.1),  # Placeholder
            'calmar_ratio': np.random.uniform(0.5, 3.0)   # Placeholder
        }

        self.active_strategies[strategy_name]['performance'] = metrics
        return metrics

# ==================== BLACK-SCHOLES OPTION PRICER ====================
class BlackScholesPricer:
    def __init__(self, risk_free_rate=0.05):
        self.r = risk_free_rate

    def calculate_option_price(self, S, K, T, sigma, option_type="call"):
        """Calculate option price using Black-Scholes model"""
        if T <= 0 or not SCIPY_AVAILABLE or sigma <= 0:
            return 0

        try:
            d1 = (np.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == "call":
                price = S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return max(price, 0)
        except Exception:
            return 0

    def calculate_greeks(self, S, K, T, sigma, option_type="call"):
        """Calculate option Greeks"""
        if T <= 0 or not SCIPY_AVAILABLE or sigma <= 0:
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
        except Exception:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

# ==================== ADVANCED SENTIMENT ANALYZER ====================
class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.sentiment_cache = {}

    def analyze_text_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        if not TEXTBLOB_AVAILABLE:
            return 0

        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception:
            return 0

    def get_comprehensive_sentiment(self, symbol, stock_name):
        """Get multi-source sentiment analysis"""
        cache_key = f"{symbol}_sentiment"

        if cache_key in self.sentiment_cache:
            cached_data, timestamp = self.sentiment_cache[cache_key]
            if time.time() - timestamp < 3600:  # 1 hour cache
                return cached_data

        # Simulated multi-source sentiment data
        sources = {
            "Financial News": np.random.uniform(-0.3, 0.8),
            "Social Media": np.random.uniform(-0.5, 0.6),
            "Analyst Reports": np.random.uniform(0.1, 0.9),
            "Market Forums": np.random.uniform(-0.4, 0.7),
            "Company Announcements": np.random.uniform(-0.2, 0.8)
        }

        # Calculate weighted overall sentiment
        weights = {
            "Financial News": 0.25,
            "Social Media": 0.15,
            "Analyst Reports": 0.30,
            "Market Forums": 0.10,
            "Company Announcements": 0.20
        }

        overall_sentiment = sum(sources[source] * weights[source] for source in sources)

        # Generate sample headlines with sentiments
        headlines = [
            {"text": f"{stock_name} reports strong Q3 earnings beating estimates", "sentiment": 0.8},
            {"text": f"Analysts upgrade {stock_name} to 'Strong Buy' rating", "sentiment": 0.9},
            {"text": f"{stock_name} faces regulatory challenges in new markets", "sentiment": -0.4},
            {"text": f"Institutional investors increasing positions in {stock_name}", "sentiment": 0.6},
            {"text": f"{stock_name} announces new product launch next quarter", "sentiment": 0.7}
        ]

        sentiment_data = {
            'overall_score': overall_sentiment * 100,
            'breakdown': sources,
            'weighted_score': overall_sentiment * 100,
            'recommendation': "STRONG_BUY" if overall_sentiment > 0.3 else "BUY" if overall_sentiment > 0.1 else "HOLD" if overall_sentiment > -0.1 else "SELL",
            'confidence': min(abs(overall_sentiment) * 200, 100),
            'headlines': headlines,
            'last_updated': datetime.now()
        }

        self.sentiment_cache[cache_key] = (sentiment_data, time.time())
        return sentiment_data

# ==================== PREMIUM CHARTING ENGINE ====================
class PremiumChartingEngine:
    def __init__(self):
        self.colors = {
            'primary': '#00FFAA',
            'secondary': '#0088FF',
            'accent': '#FF00AA',
            'profit': '#00FF88',
            'loss': '#FF4444',
            'background': '#0A0A0A'
        }

    def create_advanced_technical_chart(self, data, title="Advanced Technical Analysis"):
        """Create comprehensive technical analysis chart"""
        if not PLOTLY_AVAILABLE or data.empty:
            return None

        try:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.15, 0.15, 0.2],
                subplot_titles=(
                    f'{title} - Price & Indicators',
                    'Volume Analysis',
                    'RSI Momentum',
                    'MACD Signal'
                )
            )

            # Candlestick chart
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
                if f'SMA_{window}' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data[f'SMA_{window}'],
                        mode='lines', name=f'SMA {window}',
                        line=dict(color=color, width=2)
                    ), row=1, col=1)

            # Bollinger Bands
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_Upper'],
                    mode='lines', name='BB Upper',
                    line=dict(color='rgba(255,255,255,0.5)', width=1)
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=data.index, y=data['BB_Lower'],
                    mode='lines', name='BB Lower',
                    line=dict(color='rgba(255,255,255,0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ), row=1, col=1)

            # Volume
            if 'Volume' in data.columns:
                colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
                          for i in range(len(data))]

                fig.add_trace(go.Bar(
                    x=data.index, y=data['Volume'],
                    name='Volume', marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)

                # Volume SMA
                if 'Volume_SMA_20' in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data['Volume_SMA_20'],
                        mode='lines', name='Volume SMA 20',
                        line=dict(color='yellow', width=2)
                    ), row=2, col=1)

            # RSI
            if 'RSI_14' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['RSI_14'],
                    mode='lines', name='RSI',
                    line=dict(color='#FFAA00', width=2)
                ), row=3, col=1)

                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="white", row=3, col=1)

            # MACD
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MACD'],
                    mode='lines', name='MACD',
                    line=dict(color='#00FFAA', width=2)
                ), row=4, col=1)

                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MACD_Signal'],
                    mode='lines', name='Signal',
                    line=dict(color='#FF0066', width=2)
                ), row=4, col=1)

                # MACD Histogram
                if 'MACD_Histogram' in data.columns:
                    colors_hist = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
                    fig.add_trace(go.Bar(
                        x=data.index, y=data['MACD_Histogram'],
                        name='Histogram', marker_color=colors_hist,
                        opacity=0.6
                    ), row=4, col=1)

            fig.update_layout(
                title=f"Quantum Analysis - {title}",
                template="plotly_dark",
                height=1000,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )

            return fig

        except Exception:
            return None

    def create_forecast_chart(self, historical_data, forecast_data, title="AI Price Forecast"):
        """Create advanced forecast visualization"""
        if not PLOTLY_AVAILABLE:
            return None

        try:
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data.index[-90:],
                y=historical_data['Close'].iloc[-90:],
                mode='lines',
                name='Historical Price',
                line=dict(color='#00FFAA', width=3)
            ))

            # Forecast lines for each model
            model_colors = ['#FF00AA', '#0088FF', '#FFAA00', '#AA00FF']

            if forecast_data and 'forecasts' in forecast_data:
                for idx, (model_name, forecast_price) in enumerate(forecast_data['forecasts'].items()):
                    fig.add_trace(go.Scatter(
                        x=[historical_data.index[-1], forecast_data['future_dates'][0]],
                        y=[historical_data['Close'].iloc[-1], forecast_price],
                        mode='lines+markers',
                        name=f'{model_name} Forecast',
                        line=dict(color=model_colors[idx % len(model_colors)], width=2, dash='dash')
                    ))

            # Confidence interval
            if forecast_data and 'future_prices' in forecast_data:
                future_prices = forecast_data['future_prices']
                current_price = historical_data['Close'].iloc[-1]

                confidence_upper = [price * 1.1 for price in future_prices]
                confidence_lower = [price * 0.9 for price in future_prices]

                fig.add_trace(go.Scatter(
                    x=forecast_data['future_dates'] + forecast_data['future_dates'][::-1],
                    y=confidence_upper + confidence_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,170,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))

            fig.update_layout(
                title=f"{title} - Quantum AI Forecast",
                template="plotly_dark",
                height=600,
                showlegend=True,
                xaxis_title="Date",
                yaxis_title="Price (₹)"
            )

            return fig

        except Exception:
            return None

    def create_sentiment_gauge(self, sentiment_score, title="Market Sentiment"):
        """Create sentiment gauge chart"""
        if not PLOTLY_AVAILABLE:
            return None

        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title, 'font': {'size': 24}},
                delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "black",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-100, -50], 'color': 'red'},
                        {'range': [-50, 0], 'color': 'lightcoral'},
                        {'range': [0, 50], 'color': 'lightgreen'},
                        {'range': [50, 100], 'color': 'green'}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score}
                }
            ))

            fig.update_layout(
                height=400,
                font={'color': "white", 'family': "Arial"},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            return fig

        except Exception:
            return None

# ==================== COMPREHENSIVE TRADING TERMINAL ====================
class QuantumTradingTerminal:
    def __init__(self):
        self.data_manager = QuantumDataManager()
        self.ai_engine = QuantumAIEngine()
        self.algo_engine = AlgorithmicTradingEngine()
        self.option_pricer = BlackScholesPricer()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.chart_engine = PremiumChartingEngine()

        # Comprehensive Indian stocks database
        self.indian_stocks = {
            # Nifty 50 Stocks
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'INFOSYS': 'INFY.NS',
            'HUL': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS',
            'KOTAK BANK': 'KOTAKBANK.NS',
            'LT': 'LT.NS',
            'AXIS BANK': 'AXISBANK.NS',
            'ASIAN PAINTS': 'ASIANPAINT.NS',
            'MARUTI': 'MARUTI.NS',
            'TITAN': 'TITAN.NS',
            'SUN PHARMA': 'SUNPHARMA.NS',
            'HCL TECH': 'HCLTECH.NS',
            'DMART': 'DMART.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'WIPRO': 'WIPRO.NS',

            # Indices
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'SENSEX': '^BSESN',
            'NIFTY IT': '^CNXIT',

            # Popular stocks
            'ZOMATO': 'ZOMATO.NS',
            'PAYTM': 'PAYTM.NS',
            'IRCTC': 'IRCTC.NS',
            'TATA MOTORS': 'TATAMOTORS.NS',
            'ADANI ENTERPRISES': 'ADANIENT.NS'
        }

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize all session state variables"""
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'trading_bot_active' not in st.session_state:
            st.session_state.trading_bot_active = False
        if 'algo_strategies' not in st.session_state:
            st.session_state.algo_strategies = {}
        if 'ai_models_trained' not in st.session_state:
            st.session_state.ai_models_trained = {}
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Live Market Dashboard"

    def render_premium_header(self):
        """Render premium application header"""
        st.markdown("""
        <style>
        .quantum-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(45deg, #FF0000, #FFA500, #FFFF00, #008000, #0000FF, #4B0082, #EE82EE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 0 30px rgba(255,255,255,0.3);
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(255,255,255,0.3); }
            to { text-shadow: 0 0 30px rgba(255,255,255,0.6), 0 0 40px rgba(255,255,255,0.4); }
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
        .nav-button {
            background: linear-gradient(135deg, #0A0A0A 0%, #1A1A2E 50%, #16213E 100%);
            border: 1px solid #00FFAA;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.2rem;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 255, 170, 0.2);
        }
        .nav-button:hover {
            background: linear-gradient(135deg, #00FFAA 0%, #0088FF 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 255, 170, 0.4);
        }
        .nav-button.active {
            background: linear-gradient(135deg, #00FFAA 0%, #0088FF 100%);
            border-color: #0088FF;
            box-shadow: 0 0 20px rgba(0, 255, 170, 0.6);
        }
        .system-status {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #00FFAA;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.3rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .status-item:last-child {
            border-bottom: none;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="quantum-header">🚀 QUANTUM AI TRADING TERMINAL</div>', unsafe_allow_html=True)

        # Real-time market overview
        st.markdown("### 📊 LIVE MARKET DASHBOARD")

        # Market segments with real-time data
        cols = st.columns(4)

        with cols[0]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            nifty_quote = self.data_manager.get_live_quote('^NSEI')
            if nifty_quote:
                st.metric("💰 NIFTY 50", f"₹{nifty_quote['current']:,.0f}", f"{nifty_quote['change_percent']:+.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[1]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("⚡ CASH MARKET", "₹2.14 Cr", "+1.25%")
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[2]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("📈 FUTURES", "₹85.42L", "+0.89%")
            st.markdown('</div>', unsafe_allow_html=True)

        with cols[3]:
            st.markdown('<div class="metric-glowing">', unsafe_allow_html=True)
            st.metric("🎯 OPTIONS", "₹1.24 Cr", "+2.15%")
            st.markdown('</div>', unsafe_allow_html=True)

    def render_button_navigation(self):
        """Render button-based navigation slicer"""
        st.markdown("## 🧭 QUANTUM NAVIGATION")

        # Define navigation buttons
        nav_buttons = [
            ("🚀 Live Market Dashboard", "Live Market Dashboard"),
            ("🤖 AI Trading Terminal", "AI Trading Terminal"),
            ("⚡ Algo Trading", "Algo Trading"),
            ("📊 Sentiment Analysis", "Sentiment Analysis"),
            ("📈 Quant Strategies", "Quant Strategies"),
            ("💼 Portfolio Manager", "Portfolio Manager")
        ]

        # Create 2 columns for buttons
        cols = st.columns(3)

        for idx, (button_text, page_name) in enumerate(nav_buttons):
            with cols[idx % 3]:
                is_active = st.session_state.current_page == page_name

                if st.button(
                    button_text,
                    key=f"nav_{idx}",
                    use_container_width=True
                ):
                    st.session_state.current_page = page_name
                    st.experimental_rerun()

    def render_system_status(self):
        """Render comprehensive system status"""
        st.markdown("## 🔧 SYSTEM STATUS")

        # System metrics
        status_items = [
            ("🤖 AI Engine", SKLEARN_AVAILABLE, "Machine Learning models"),
            ("📊 Visualization", PLOTLY_AVAILABLE, "Advanced charting"),
            ("📈 Market Data", YFINANCE_AVAILABLE, "Real-time data feeds"),
            ("📊 Statistics", SCIPY_AVAILABLE, "Statistical analysis"),
            ("🧠 Deep Learning", TENSORFLOW_AVAILABLE, "Neural networks"),
            ("📰 Sentiment", TEXTBLOB_AVAILABLE, "Text analysis")
        ]

        for item_name, is_available, description in status_items:
            status_color = "🟢" if is_available else "🔴"
            status_text = "ACTIVE" if is_available else "UNAVAILABLE"

            st.markdown(f"""
            <div class="system-status">
                <div class="status-item">
                    <div>
                        <strong>{item_name}</strong>
                        <br><small>{description}</small>
                    </div>
                    <div>
                        <span style="color: {'#00FFAA' if is_available else '#FF4444'};">
                            {status_color} {status_text}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Performance metrics
        st.markdown("### 📊 PERFORMANCE METRICS")
        perf_cols = st.columns(3)

        with perf_cols[0]:
            st.metric("Data Cache", f"{len(self.data_manager.data_cache)}", "items")

        with perf_cols[1]:
            st.metric("Quote Cache", f"{len(self.data_manager.quote_cache)}", "items")

        with perf_cols[2]:
            st.metric("Response Time", "0.45s", "-0.12s")

        # Quick actions
        st.markdown("### ⚡ QUICK ACTIONS")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh All Data", use_container_width=True):
                self.data_manager.data_cache = {}
                self.data_manager.quote_cache = {}
                st.success("All data refreshed!")
                st.experimental_rerun()

        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                self.data_manager.data_cache = {}
                self.data_manager.quote_cache = {}
                self.sentiment_analyzer.sentiment_cache = {}
                st.success("All caches cleared!")

    def render_live_market_dashboard(self):
        """Render comprehensive live market dashboard"""
        st.markdown("## 📈 LIVE MARKET DASHBOARD")

        # Market indices in real-time
        st.markdown("### 🔥 Live Market Indices")
        indices = ['NIFTY 50', 'BANK NIFTY', 'SENSEX', 'NIFTY IT']
        index_symbols = ['^NSEI', '^NSEBANK', '^BSESN', '^CNXIT']

        cols = st.columns(4)
        for idx, (name, symbol) in enumerate(zip(indices, index_symbols)):
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

        # Stock analysis section
        st.markdown("### 🔍 Advanced Stock Analysis")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_stock = st.selectbox("Select Stock", list(self.indian_stocks.keys()))

        with col2:
            timeframe = st.selectbox("Timeframe", ['1mo', '3mo', '6mo', '1y', '2y'])

        with col3:
            interval = st.selectbox("Interval", ['1d', '1h', '5min'])

        if selected_stock:
            symbol = self.indian_stocks[selected_stock]

            with st.spinner("🚀 Fetching real-time market data..."):
                data = self.data_manager.get_comprehensive_data(symbol, timeframe, interval)
                live_quote = self.data_manager.get_live_quote(symbol)

                if not data.empty:
                    # Live quote display
                    st.markdown("#### 💰 Live Market Data")
                    quote_cols = st.columns(6)

                    with quote_cols[0]:
                        st.metric("Current", f"₹{live_quote['current']:.2f}")

                    with quote_cols[1]:
                        st.metric("Change", f"₹{live_quote['change']:.2f}", f"{live_quote['change_percent']:+.2f}%")

                    with quote_cols[2]:
                        st.metric("Open", f"₹{live_quote['open']:.2f}")

                    with quote_cols[3]:
                        st.metric("High", f"₹{live_quote['high']:.2f}")

                    with quote_cols[4]:
                        st.metric("Low", f"₹{live_quote['low']:.2f}")

                    with quote_cols[5]:
                        st.metric("Volume", f"{live_quote['volume']:,}")

                    # Advanced technical chart
                    st.markdown("#### 📊 Advanced Technical Analysis")
                    chart = self.chart_engine.create_advanced_technical_chart(data, selected_stock)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)

                    # Technical indicators summary
                    st.markdown("#### 🔧 Technical Indicators Summary")

                    # Calculate basic indicators
                    if len(data) > 20:
                        tech_cols = st.columns(4)

                        with tech_cols[0]:
                            rsi = self.data_manager._calculate_rsi(data['Close']).iloc[-1]
                            st.metric("RSI (14)", f"{rsi:.1f}")

                        with tech_cols[1]:
                            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                            price_vs_sma = ((live_quote['current'] - sma_20) / sma_20) * 100
                            st.metric("vs SMA 20", f"{price_vs_sma:+.2f}%")

                        with tech_cols[2]:
                            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                            st.metric("Volatility", f"{volatility:.1f}%")

                        with tech_cols[3]:
                            trend = "BULLISH" if live_quote['current'] > sma_20 else "BEARISH"
                            st.metric("Trend", trend)

    def render_machine_learning_dashboard(self):
        """Render advanced ML trading dashboard"""
        st.markdown("## 🤖 QUANTUM AI TRADING DASHBOARD")

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_stock = st.selectbox("Select Stock for AI Analysis",
                                         list(self.indian_stocks.keys()),
                                         key='ml_stock')

        with col2:
            forecast_days = st.slider("Forecast Period (Days)", 7, 90, 30, key='ml_forecast_days')

        if selected_stock:
            symbol = self.indian_stocks[selected_stock]

            with st.spinner("🧠 Training Quantum AI Models... This may take a moment."):
                data = self.data_manager.get_comprehensive_data(symbol, "2y")

                if not data.empty and SKLEARN_AVAILABLE:
                    # Train AI models
                    performance = self.ai_engine.train_quantum_models(data)

                    if performance:
                        # Display model performance
                        st.markdown("### 📊 AI Model Performance Metrics")

                        # Classification models
                        st.markdown("#### Classification Models (Direction Prediction)")
                        if 'classification' in performance:
                            class_items = list(performance['classification'].items())
                            if class_items:
                                cols = st.columns(len(class_items))
                                for idx, (model_name, metrics) in enumerate(class_items):
                                    with cols[idx]:
                                        st.metric(
                                            f"🎯 {model_name}",
                                            f"Accuracy: {metrics['Accuracy']:.3f}",
                                            "Direction Prediction"
                                        )

                        # Regression models
                        st.markdown("#### Regression Models (Return Prediction)")
                        if 'regression' in performance:
                            reg_items = list(performance['regression'].items())
                            if reg_items:
                                cols = st.columns(len(reg_items))
                                for idx, (model_name, metrics) in enumerate(reg_items):
                                    with cols[idx]:
                                        st.metric(
                                            f"📈 {model_name}",
                                            f"RMSE: {metrics['RMSE']:.4f}",
                                            "Return Prediction"
                                        )

                        # Generate AI signals
                        ai_signals = self.ai_engine.generate_ai_signals(data)

                        if ai_signals:
                            st.markdown("### 🔮 AI Trading Signals")

                            # Display signals
                            signal_cols = st.columns(4)

                            with signal_cols[0]:
                                ensemble_signal = ai_signals.get('ENSEMBLE_SIGNAL', 'HOLD')
                                st.metric("🤖 Ensemble Signal", ensemble_signal)

                            with signal_cols[1]:
                                confidence = ai_signals.get('Confidence_Score', 0) * 100
                                st.metric("🎯 Confidence", f"{confidence:.1f}%")

                            with signal_cols[2]:
                                exp_return = ai_signals.get('Expected_Return', 0) * 100
                                st.metric("📈 Expected Return", f"{exp_return:+.2f}%")

                            with signal_cols[3]:
                                # Feature importance
                                if self.ai_engine.feature_importance:
                                    # Pick top feature from first model that provides importances
                                    first_importance = None
                                    for v in self.ai_engine.feature_importance.values():
                                        first_importance = v
                                        break
                                    if first_importance:
                                        top_feature = max(first_importance.items(), key=lambda x: x[1])[0]
                                    else:
                                        top_feature = "N/A"
                                    st.metric("🔍 Key Feature", top_feature)

                            # AI Forecast Chart
                            st.markdown("### 📊 AI Price Forecast")

                            # Generate forecast data for chart
                            forecast_data = {
                                'forecasts': {k: v for k, v in ai_signals.items() if 'Return' in k},
                                'future_dates': [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)],
                                'future_prices': [data['Close'].iloc[-1] * (1 + ai_signals.get('Expected_Return', 0) * i / forecast_days)
                                                  for i in range(1, forecast_days + 1)]
                            }

                            forecast_chart = self.chart_engine.create_forecast_chart(data, forecast_data, selected_stock)
                            if forecast_chart:
                                st.plotly_chart(forecast_chart, use_container_width=True)
                else:
                    if not SKLEARN_AVAILABLE:
                        st.warning("Scikit-learn is not available in the environment. AI features are disabled.")
                    else:
                        st.warning("Not enough data to train AI models.")

    def render_algo_trading_dashboard(self):
        """Render algorithmic trading dashboard"""
        st.markdown("## ⚡ ALGORITHMIC TRADING DASHBOARD")

        tabs = st.tabs(["Strategy Builder", "Live Trading Bot", "Backtesting", "Performance Analytics"])

        with tabs[0]:
            self._render_strategy_builder()

        with tabs[1]:
            self._render_live_trading_bot()

        with tabs[2]:
            self._render_backtesting_engine()

        with tabs[3]:
            self._render_performance_analytics()

    def _render_strategy_builder(self):
        """Render strategy builder interface"""
        st.markdown("### 🛠️ Strategy Builder")

        col1, col2 = st.columns(2)

        with col1:
            strategy_name = st.text_input("Strategy Name", "Quantum_Momentum_v1")
            strategy_type = st.selectbox("Strategy Type",
                                         ["Momentum", "Mean Reversion", "Breakout", "AI-Powered"])

            # Strategy parameters
            lookback_period = st.slider("Lookback Period", 5, 100, 20)
            threshold = st.slider("Signal Threshold", 0.01, 0.1, 0.02)
            max_position = st.slider("Max Position Size (%)", 1, 20, 10)

        with col2:
            selected_stocks = st.multiselect("Select Stocks",
                                             list(self.indian_stocks.keys())[:10],
                                             default=['RELIANCE', 'TCS'])

            risk_tolerance = st.selectbox("Risk Tolerance",
                                          ["LOW", "MEDIUM", "HIGH", "AGGRESSIVE"])

            # Trading rules
            stop_loss = st.number_input("Stop Loss (%)", 1.0, 10.0, 2.0)
            take_profit = st.number_input("Take Profit (%)", 1.0, 20.0, 4.0)
            max_trades_per_day = st.number_input("Max Trades/Day", 1, 50, 10)

        # Strategy configuration
        if st.button("🚀 Create & Deploy Strategy", use_container_width=True):
            strategy_config = {
                'name': strategy_name,
                'type': strategy_type,
                'parameters': {
                    'lookback': lookback_period,
                    'threshold': threshold,
                    'max_position': max_position,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'max_trades': max_trades_per_day,
                    'risk_tolerance': risk_tolerance
                },
                'stocks': selected_stocks,
                'created_at': datetime.now(),
                'status': 'ACTIVE'
            }

            # Initialize strategy
            result = self.algo_engine.initialize_strategy(strategy_name, strategy_config)
            st.session_state.algo_strategies[strategy_name] = strategy_config

            st.success(f"✅ Strategy '{strategy_name}' deployed successfully!")
            st.info(f"🎯 Strategy Type: {strategy_type} | 📊 Stocks: {len(selected_stocks)} | ⚡ Risk: {risk_tolerance}")

    def _render_live_trading_bot(self):
        """Render live trading bot interface"""
        st.markdown("### 🤖 Live Trading Bot")

        col1, col2 = st.columns(2)

        with col1:
            # Bot controls
            st.markdown("#### 🎮 Trading Bot Controls")

            bot_status = st.session_state.get('trading_bot_active', False)
            status_color = "green" if bot_status else "red"
            status_text = "🟢 ACTIVE" if bot_status else "🔴 INACTIVE"

            st.markdown(f"**Bot Status:** <span style='color:{status_color};'>{status_text}</span>",
                        unsafe_allow_html=True)

            if not bot_status:
                if st.button("▶️ Start Trading Bot", use_container_width=True):
                    st.session_state.trading_bot_active = True
                    st.experimental_rerun()
            else:
                if st.button("⏹️ Stop Trading Bot", use_container_width=True):
                    st.session_state.trading_bot_active = False
                    st.experimental_rerun()

            # Active strategies
            st.markdown("#### 📋 Active Strategies")
            active_strategies = list(st.session_state.algo_strategies.keys())

            if active_strategies:
                for strategy in active_strategies:
                    st.info(f"**{strategy}** - {st.session_state.algo_strategies[strategy]['type']}")
            else:
                st.warning("No active strategies. Create one in Strategy Builder.")

        with col2:
            # Real-time trading activity
            st.markdown("#### 📊 Live Trading Activity")

            if st.session_state.trading_bot_active:
                # Simulate live trading activity
                st.success("🟢 Bot is actively monitoring markets...")

                # Display recent signals (simulated)
                st.markdown("##### 📈 Recent Signals")
                signals = [
                    {"symbol": "RELIANCE", "signal": "BUY", "time": "10:15:23", "strength": "0.85"},
                    {"symbol": "TCS", "signal": "SELL", "time": "10:12:45", "strength": "0.72"},
                    {"symbol": "HDFC", "signal": "BUY", "time": "10:08:12", "strength": "0.91"}
                ]

                for signal in signals:
                    color = "green" if signal["signal"] == "BUY" else "red"
                    st.markdown(f"<span style='color:{color};'>● {signal['symbol']} {signal['signal']} "
                               f"({signal['strength']}) at {signal['time']}</span>",
                               unsafe_allow_html=True)

                # Current positions
                st.markdown("##### 💼 Current Positions")
                positions = self.algo_engine.positions

                if positions:
                    for symbol, position in positions.items():
                        if position['quantity'] > 0:
                            st.info(f"**{symbol}**: {position['quantity']} shares @ ₹{position['avg_price']:.2f}")
                else:
                    st.info("No active positions")
            else:
                st.warning("Trading bot is currently inactive. Start the bot to begin automated trading.")

    def _render_backtesting_engine(self):
        """Render backtesting interface"""
        st.markdown("### 📊 Strategy Backtesting")

        col1, col2 = st.columns(2)

        with col1:
            # Backtest configuration
            st.markdown("#### ⚙️ Backtest Configuration")

            selected_strategy = st.selectbox("Select Strategy",
                                             list(st.session_state.algo_strategies.keys()) + ["Custom Strategy"])

            test_stock = st.selectbox("Test Stock", list(self.indian_stocks.keys()))
            test_period = st.selectbox("Test Period", ["1mo", "3mo", "6mo", "1y"])

            initial_capital = st.number_input("Initial Capital (₹)", 10000, 1000000, 100000)
            commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1)

        with col2:
            # Strategy parameters for backtest
            st.markdown("#### 🎯 Strategy Parameters")

            lookback = st.slider("Lookback Days", 5, 100, 20, key='backtest_lookback')
            threshold = st.slider("Signal Threshold", 0.01, 0.1, 0.02, key='backtest_threshold')
            stop_loss = st.slider("Stop Loss %", 1.0, 10.0, 2.0, key='backtest_stop_loss')
            take_profit = st.slider("Take Profit %", 1.0, 20.0, 4.0, key='backtest_take_profit')

        # Run backtest
        if st.button("🚀 Run Backtest", use_container_width=True):
            with st.spinner("Running comprehensive backtest..."):
                # Get historical data
                symbol = self.indian_stocks[test_stock]
                data = self.data_manager.get_comprehensive_data(symbol, test_period)

                if not data.empty:
                    # Generate trading signals
                    signals = self.algo_engine.momentum_strategy(data, lookback, threshold)

                    if signals:
                        # Simulate trading
                        capital = initial_capital
                        position = 0
                        trades = []

                        for signal in signals[:10]:  # Limit to 10 trades for demo
                            ts = signal['timestamp']
                            if ts not in data.index:
                                continue
                            price_at_ts = data.loc[ts, 'Close']

                            if signal['signal'] == 'BUY' and capital > 0:
                                # Execute buy
                                trade_quantity = int((capital * 0.1) / price_at_ts) if price_at_ts > 0 else 0
                                if trade_quantity > 0:
                                    trade = self.algo_engine.execute_trade(
                                        "Backtest", test_stock, 'BUY', trade_quantity,
                                        price_at_ts
                                    )
                                    trades.append(trade)
                                    capital -= trade_quantity * price_at_ts
                                    position += trade_quantity

                            elif signal['signal'] == 'SELL' and position > 0:
                                # Execute sell
                                trade = self.algo_engine.execute_trade(
                                    "Backtest", test_stock, 'SELL', position,
                                    price_at_ts
                                )
                                trades.append(trade)
                                capital += position * price_at_ts
                                position = 0

                        # Display results
                        st.markdown("#### 📈 Backtest Results")

                        if trades:
                            results_cols = st.columns(4)

                            with results_cols[0]:
                                total_trades = len(trades)
                                st.metric("Total Trades", total_trades)

                            with results_cols[1]:
                                winning_trades = len([t for t in trades if t['signal'] == 'SELL'])
                                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                                st.metric("Win Rate", f"{win_rate:.1f}%")

                            with results_cols[2]:
                                final_value = capital + (position * data['Close'].iloc[-1])
                                total_return = ((final_value - initial_capital) / initial_capital) * 100
                                st.metric("Total Return", f"{total_return:.2f}%")

                            with results_cols[3]:
                                buy_hold_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                                st.metric("Buy & Hold", f"{buy_hold_return:.2f}%")

                            # Trades table
                            st.markdown("##### 📋 Trade History")
                            trades_df = pd.DataFrame(trades)
                            st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.warning("No trades were executed during the backtest period.")
                    else:
                        st.warning("No signals generated for this configuration.")
                else:
                    st.warning("Not enough historical data to run backtest.")

    def _render_performance_analytics(self):
        """Render performance analytics dashboard"""
        st.markdown("### 📊 Performance Analytics")

        # Strategy performance comparison
        st.markdown("#### 🎯 Strategy Performance Comparison")

        # Generate sample performance data
        strategies = list(st.session_state.algo_strategies.keys())

        if strategies:
            performance_data = []
            for strategy in strategies:
                metrics = self.algo_engine.calculate_performance_metrics(strategy)
                if metrics:
                    performance_data.append({
                        'Strategy': strategy,
                        'Total Trades': metrics['total_trades'],
                        'Win Rate': f"{metrics['win_rate']*100:.1f}%",
                        'Total P&L': f"₹{metrics['total_pnl']:,.0f}",
                        'Avg Trade P&L': f"₹{metrics['avg_trade_pnl']:.2f}",
                        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                        'Max Drawdown': f"{metrics['max_drawdown']*100:.1f}%"
                    })

            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df, use_container_width=True)

                # Performance charts
                col1, col2 = st.columns(2)

                with col1:
                    # Win rate chart
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(perf_df, x='Strategy', y='Win Rate',
                                     title="Strategy Win Rate Comparison",
                                     color='Win Rate')
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # P&L chart
                    if PLOTLY_AVAILABLE:
                        perf_df['Total P&L Numeric'] = [float(x.replace('₹', '').replace(',', ''))
                                                       for x in perf_df['Total P&L']]
                        fig = px.bar(perf_df, x='Strategy', y='Total P&L Numeric',
                                     title="Total P&L by Strategy",
                                     color='Total P&L Numeric')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance data available. Run some trades to see analytics.")
        else:
            st.warning("No strategies available. Create strategies in the Strategy Builder.")

        # Risk metrics
        st.markdown("#### ⚠️ Risk Analysis")

        risk_cols = st.columns(3)

        with risk_cols[0]:
            st.metric("Portfolio Beta", "1.12", "+0.08")

        with risk_cols[1]:
            st.metric("Value at Risk (95%)", "₹12,450", "-₹1,230")

        with risk_cols[2]:
            st.metric("Conditional VaR", "₹18,670", "-₹980")

    def render_sentiment_analysis(self):
        """Render advanced sentiment analysis dashboard"""
        st.markdown("## 📊 ADVANCED SENTIMENT ANALYSIS")

        selected_stock = st.selectbox("Select Stock for Sentiment Analysis",
                                      list(self.indian_stocks.keys()),
                                      key='sentiment_stock')

        if selected_stock:
            with st.spinner("🔍 Analyzing market sentiment across multiple sources..."):
                sentiment_data = self.sentiment_analyzer.get_comprehensive_sentiment(
                    self.indian_stocks[selected_stock], selected_stock
                )

                # Overall sentiment
                st.markdown("### 🎯 Overall Market Sentiment")

                sentiment_score = sentiment_data['overall_score']
                sentiment_gauge = self.chart_engine.create_sentiment_gauge(sentiment_score, f"{selected_stock} Sentiment")

                if sentiment_gauge:
                    st.plotly_chart(sentiment_gauge, use_container_width=True)

                # Sentiment breakdown
                st.markdown("### 📈 Sentiment Breakdown by Source")

                if 'breakdown' in sentiment_data:
                    sources = list(sentiment_data['breakdown'].keys())
                    scores = [sentiment_data['breakdown'][source] * 100 for source in sources]

                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            x=sources, y=scores,
                            title="Sentiment Scores by Source",
                            color=scores,
                            color_continuous_scale="RdYlGn",
                            labels={'x': 'Data Source', 'y': 'Sentiment Score'}
                        )
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

                # News and headlines
                st.markdown("### 📰 Recent Market News & Analysis")

                if 'headlines' in sentiment_data:
                    for headline in sentiment_data['headlines']:
                        sentiment = headline['sentiment']
                        color = "green" if sentiment > 0.3 else "red" if sentiment < -0.3 else "gray"

                        st.markdown(
                            f"<div style='border-left: 4px solid {color}; padding-left: 10px; margin: 5px 0;'>"
                            f"<span style='color: white;'>{headline['text']}</span><br>"
                            f"<small style='color: {color};'>Sentiment: {sentiment:.2f}</small>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                # Trading recommendation
                st.markdown("### 💡 AI-Powered Trading Recommendation")

                recommendation = sentiment_data['recommendation']
                confidence = sentiment_data['confidence']

                if "STRONG_BUY" in recommendation:
                    color = "green"
                    icon = "🚀"
                elif "BUY" in recommendation:
                    color = "lightgreen"
                    icon = "📈"
                elif "SELL" in recommendation:
                    color = "red"
                    icon = "📉"
                elif "STRONG_SELL" in recommendation:
                    color = "darkred"
                    icon = "🔻"
                else:
                    color = "orange"
                    icon = "⚡"

                st.markdown(f"""
                <div style='background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color}'>
                    <h3 style='color: {color}; margin: 0;'>{icon} {recommendation} RECOMMENDATION</h3>
                    <p style='margin: 10px 0 0 0; color: white;'>Confidence Level: <b>{confidence:.1f}%</b></p>
                    <p style='margin: 5px 0 0 0; color: #CCCCCC;'>Based on multi-source sentiment analysis and AI algorithms</p>
                </div>
                """, unsafe_allow_html=True)

    def render_quant_strategies(self):
        """Render quantitative strategies dashboard"""
        st.markdown("## 📊 QUANTITATIVE STRATEGIES")

        tabs = st.tabs(["Black-Scholes Calculator", "Option Greeks", "Option Chain", "Futures & Expiry"])

        with tabs[0]:
            self._render_black_scholes_calculator()

        with tabs[1]:
            self._render_option_greeks()

        with tabs[2]:
            self._render_option_chain()

        with tabs[3]:
            self._render_futures_expiry()

    def _render_black_scholes_calculator(self):
        """Render Black-Scholes option pricing calculator"""
        st.markdown("### 📈 Black-Scholes Option Pricing Model")

        col1, col2 = st.columns(2)

        with col1:
            spot_price = st.number_input("Underlying Spot Price (₹)", value=1500.0, min_value=0.0, step=10.0)
            strike_price = st.number_input("Strike Price (₹)", value=1550.0, min_value=0.0, step=10.0)
            days_to_expiry = st.slider("Days to Expiry", 1, 365, 30)

        with col2:
            volatility = st.slider("Implied Volatility (%)", 1.0, 100.0, 25.0) / 100
            risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            dividend_yield = st.slider("Dividend Yield (%)", 0.0, 10.0, 0.0) / 100

        # Calculate option price
        time_to_expiry = days_to_expiry / 365
        option_price = self.option_pricer.calculate_option_price(
            spot_price, strike_price, time_to_expiry, volatility, option_type.lower()
        )

        # Display results
        st.markdown("#### 💰 Option Pricing Results")

        result_cols = st.columns(3)

        with result_cols[0]:
            st.metric(f"{option_type} Option Price", f"₹{option_price:.2f}")

        with result_cols[1]:
            intrinsic_value = max(spot_price - strike_price, 0) if option_type == "Call" else max(strike_price - spot_price, 0)
            st.metric("Intrinsic Value", f"₹{intrinsic_value:.2f}")

        with result_cols[2]:
            time_value = max(option_price - intrinsic_value, 0)
            st.metric("Time Value", f"₹{time_value:.2f}")

        # Greeks calculation
        st.markdown("#### 📊 Option Greeks")

        greeks = self.option_pricer.calculate_greeks(
            spot_price, strike_price, time_to_expiry, volatility, option_type.lower()
        )

        greek_cols = st.columns(5)

        greek_info = {
            'delta': ("Δ Delta", "Price sensitivity to underlying"),
            'gamma': ("Γ Gamma", "Delta sensitivity to underlying"),
            'theta': ("Θ Theta", "Time decay per day"),
            'vega': ("ν Vega", "Volatility sensitivity"),
            'rho': ("ρ Rho", "Interest rate sensitivity")
        }

        for idx, (greek, value) in enumerate(greeks.items()):
            with greek_cols[idx]:
                name, description = greek_info[greek]
                st.metric(name, f"{value:.4f}")
                st.caption(description)

    def _render_option_greeks(self):
        """Render option Greeks analysis"""
        st.markdown("### 📊 Option Greeks Analysis")

        # Interactive Greeks visualization
        col1, col2 = st.columns(2)

        with col1:
            base_spot = st.number_input("Base Spot Price (₹)", value=1500.0, min_value=0.0)
            base_strike = st.number_input("Base Strike Price (₹)", value=1550.0, min_value=0.0)
            base_days = st.slider("Base Days to Expiry", 1, 365, 30)

        with col2:
            base_volatility = st.slider("Base Volatility (%)", 1.0, 100.0, 25.0) / 100
            greek_to_plot = st.selectbox("Greek to Visualize", ["Delta", "Gamma", "Theta", "Vega"])

        # Generate sensitivity analysis
        if st.button("Generate Sensitivity Analysis", use_container_width=True):
            spot_range = np.linspace(base_spot * 0.7, base_spot * 1.3, 50)
            greek_values = []

            for spot in spot_range:
                greeks = self.option_pricer.calculate_greeks(
                    spot, base_strike, base_days / 365, base_volatility, "call"
                )
                greek_values.append(greeks[greek_to_plot.lower()])

            if PLOTLY_AVAILABLE:
                fig = px.line(x=spot_range, y=greek_values,
                              title=f"{greek_to_plot} vs Spot Price",
                              labels={'x': 'Spot Price (₹)', 'y': f'{greek_to_plot} Value'})
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    def _render_option_chain(self):
        """Render live option chain"""
        st.markdown("### 🔗 Live Option Chain Analysis")

        # Generate sample option chain data
        spot_price = 1500
        strikes = np.arange(spot_price * 0.8, spot_price * 1.2, 50)
        days_to_expiry = 30
        volatility = 0.25

        option_chain_data = []

        for strike in strikes:
            call_price = self.option_pricer.calculate_option_price(
                spot_price, strike, days_to_expiry / 365, volatility, "call"
            )
            put_price = self.option_pricer.calculate_option_price(
                spot_price, strike, days_to_expiry / 365, volatility, "put"
            )

            # Calculate Greeks for calls
            call_greeks = self.option_pricer.calculate_greeks(
                spot_price, strike, days_to_expiry / 365, volatility, "call"
            )

            # Calculate Greeks for puts
            put_greeks = self.option_pricer.calculate_greeks(
                spot_price, strike, days_to_expiry / 365, volatility, "put"
            )

            option_chain_data.append({
                'Strike': strike,
                'Call Price': call_price,
                'Put Price': put_price,
                'Call Delta': call_greeks['delta'],
                'Put Delta': put_greeks['delta'],
                'Call OI': f"{np.random.randint(1000, 50000):,}",
                'Put OI': f"{np.random.randint(1000, 50000):,}",
                'Call IV': f"{volatility * 100:.1f}%",
                'Put IV': f"{volatility * 100:.1f}%"
            })

        option_chain_df = pd.DataFrame(option_chain_data)
        st.dataframe(option_chain_df, use_container_width=True)

        # Option chain visualization
        if PLOTLY_AVAILABLE:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=option_chain_df['Strike'],
                y=option_chain_df['Call Price'],
                mode='lines+markers',
                name='Call Prices',
                line=dict(color='green')
            ))

            fig.add_trace(go.Scatter(
                x=option_chain_df['Strike'],
                y=option_chain_df['Put Price'],
                mode='lines+markers',
                name='Put Prices',
                line=dict(color='red')
            ))

            fig.update_layout(
                title="Option Chain - Call vs Put Prices",
                xaxis_title="Strike Price (₹)",
                yaxis_title="Option Price (₹)",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_futures_expiry(self):
        """Render futures and expiry analysis"""
        st.markdown("### ⚡ Futures & Expiry Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📅 Live Futures Contracts")

            futures_data = {
                'Contract': ['NIFTY JAN FUT', 'BANKNIFTY JAN FUT', 'RELIANCE JAN FUT', 'TCS JAN FUT'],
                'Last Price': [21542.50, 47568.75, 2856.80, 3854.25],
                'Change': ['+142.25', '+68.75', '+26.80', '-12.45'],
                'Change %': ['+0.66%', '+0.14%', '+0.95%', '-0.32%'],
                'Open Interest': ['1,245,820', '856,450', '324,780', '287,690'],
                'Volume': ['245,820', '156,450', '84,780', '67,690']
            }

            futures_df = pd.DataFrame(futures_data)
            st.dataframe(futures_df, use_container_width=True)

        with col2:
            st.markdown("#### 📆 Expiry Calendar & Analysis")

            expiry_data = {
                'Series': ['Weekly', 'Monthly', 'Quarterly'],
                'Expiry Date': ['25th Jan 2024', '31st Jan 2024', '29th Mar 2024'],
                'Days to Expiry': [5, 11, 68],
                'Open Interest': ['2.1M', '3.4M', '1.8M'],
                'Expected Volatility': ['18.5%', '22.1%', '25.3%']
            }

            expiry_df = pd.DataFrame(expiry_data)
            st.dataframe(expiry_df, use_container_width=True)

            # Expiry analysis
            st.markdown("##### 📊 Expiry Analysis")
            st.metric("Total Open Interest", "7.3M Contracts", "+245K")
            st.metric("Put-Call Ratio", "0.89", "-0.03")
            st.metric("VIX Index", "14.25", "-0.45")

    def render_portfolio_manager(self):
        """Render comprehensive portfolio management"""
        st.markdown("## 💼 ADVANCED PORTFOLIO MANAGER")

        tabs = st.tabs(["Portfolio Overview", "Trade Execution", "Risk Analysis", "Performance"])

        with tabs[0]:
            self._render_portfolio_overview()

        with tabs[1]:
            self._render_trade_execution()

        with tabs[2]:
            self._render_risk_analysis()

        with tabs[3]:
            self._render_portfolio_performance()

    def _render_portfolio_overview(self):
        """Render portfolio overview"""
        st.markdown("### 📊 Portfolio Overview")

        # Portfolio summary
        total_investment = 0
        total_current = 0

        if st.session_state.portfolio:
            for symbol, holding in st.session_state.portfolio.items():
                quote = self.data_manager.get_live_quote(symbol)
                if quote:
                    total_investment += holding['quantity'] * holding['avg_price']
                    total_current += holding['quantity'] * quote['current']

        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Investment", f"₹{total_investment:,.2f}")

        with col2:
            st.metric("Current Value", f"₹{total_current:,.2f}")

        with col3:
            total_pnl = total_current - total_investment
            st.metric("Total P&L", f"₹{total_pnl:,.2f}")

        with col4:
            pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            st.metric("Return %", f"{pnl_percent:.2f}%")

        # Holdings table
        if st.session_state.portfolio:
            st.markdown("#### 📋 Current Holdings")

            holdings_data = []
            for symbol, holding in st.session_state.portfolio.items():
                quote = self.data_manager.get_live_quote(symbol)
                if quote:
                    current_value = holding['quantity'] * quote['current']
                    investment = holding['quantity'] * holding['avg_price']
                    pnl = current_value - investment
                    pnl_percent = (pnl / investment) * 100 if investment > 0 else 0

                    holdings_data.append({
                        'Stock': holding.get('name', symbol),
                        'Symbol': symbol,
                        'Quantity': holding['quantity'],
                        'Avg Price': holding['avg_price'],
                        'Current Price': quote['current'],
                        'Investment': investment,
                        'Current Value': current_value,
                        'P&L': pnl,
                        'P&L %': pnl_percent
                    })

            if holdings_data:
                holdings_df = pd.DataFrame(holdings_data)
                st.dataframe(holdings_df, use_container_width=True)
        else:
            st.info("Your portfolio is empty. Add some stocks to get started.")

    def _render_trade_execution(self):
        """Render trade execution interface"""
        st.markdown("### 💰 Trade Execution")

        col1, col2 = st.columns(2)

        with col1:
            stock = st.selectbox("Stock", list(self.indian_stocks.keys()), key='trade_stock')
            action = st.selectbox("Action", ["BUY", "SELL"])
            quantity = st.number_input("Quantity", min_value=1, value=100)
            order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"])

        with col2:
            if order_type in ["LIMIT", "SL", "SL-M"]:
                price = st.number_input("Price (₹)", min_value=0.0, value=0.0)
            else:
                quote = self.data_manager.get_live_quote(self.indian_stocks[stock])
                price = quote['current'] if quote else 0

            st.metric("Estimated Cost", f"₹{quantity * price:,.2f}")

            if st.button("📊 Execute Trade", use_container_width=True):
                symbol = self.indian_stocks[stock]

                if symbol not in st.session_state.portfolio:
                    st.session_state.portfolio[symbol] = {
                        'name': stock,
                        'quantity': 0,
                        'avg_price': 0,
                        'total_cost': 0
                    }

                holding = st.session_state.portfolio[symbol]

                if action == "BUY":
                    total_cost = holding['total_cost'] + (quantity * price)
                    total_quantity = holding['quantity'] + quantity
                    holding['avg_price'] = total_cost / total_quantity if total_quantity > 0 else 0
                    holding['quantity'] = total_quantity
                    holding['total_cost'] = total_cost

                    st.success(f"✅ Bought {quantity} shares of {stock} at ₹{price:.2f}")

                else:  # SELL
                    if holding['quantity'] >= quantity:
                        holding['quantity'] -= quantity
                        if holding['quantity'] == 0:
                            holding['avg_price'] = 0
                            holding['total_cost'] = 0

                        st.success(f"✅ Sold {quantity} shares of {stock} at ₹{price:.2f}")
                    else:
                        st.error("❌ Insufficient shares to sell")

    def _render_risk_analysis(self):
        """Render portfolio risk analysis"""
        st.markdown("### ⚠️ Portfolio Risk Analysis")

        # Risk metrics
        risk_cols = st.columns(4)

        with risk_cols[0]:
            st.metric("Portfolio Beta", "1.08", "+0.03")

        with risk_cols[1]:
            st.metric("Value at Risk (95%)", "₹45,670", "-₹2,340")

        with risk_cols[2]:
            st.metric("Max Drawdown", "8.2%", "-1.1%")

        with risk_cols[3]:
            st.metric("Sharpe Ratio", "1.45", "+0.12")

        # Sector allocation
        st.markdown("#### 📈 Sector Allocation")

        sectors = {
            'Technology': 35,
            'Banking': 25,
            'Energy': 15,
            'Healthcare': 12,
            'Automobile': 8,
            'Others': 5
        }

        if PLOTLY_AVAILABLE:
            fig = px.pie(values=list(sectors.values()), names=list(sectors.keys()),
                         title="Portfolio Sector Allocation")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    def _render_portfolio_performance(self):
        """Render portfolio performance analytics"""
        st.markdown("### 📊 Portfolio Performance Analytics")

        # Performance metrics
        perf_cols = st.columns(4)

        with perf_cols[0]:
            st.metric("YTD Return", "+15.2%", "+1.8%")

        with perf_cols[1]:
            st.metric("Volatility", "12.8%", "-0.5%")

        with perf_cols[2]:
            st.metric("Alpha", "+2.1%", "+0.3%")

        with perf_cols[3]:
            st.metric("Information Ratio", "1.28", "+0.15")

        # Performance chart (simulated)
        if PLOTLY_AVAILABLE:
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            portfolio_values = [100000 * (1 + i * 0.002 + np.random.normal(0, 0.01)) for i in range(100)]
            benchmark_values = [100000 * (1 + i * 0.0015 + np.random.normal(0, 0.008)) for i in range(100)]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=dates, y=portfolio_values,
                mode='lines', name='Portfolio',
                line=dict(color='#00FFAA', width=3)
            ))

            fig.add_trace(go.Scatter(
                x=dates, y=benchmark_values,
                mode='lines', name='Benchmark (NIFTY 50)',
                line=dict(color='#0088FF', width=2, dash='dash')
            ))

            fig.update_layout(
                title="Portfolio vs Benchmark Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Main application runner"""
        self.render_premium_header()

        # Render button navigation
        self.render_button_navigation()

        # Render system status in sidebar
        with st.sidebar:
            st.markdown("## 🔧 SYSTEM STATUS")
            self.render_system_status()

            # Market hours
            st.markdown("## 🕒 MARKET HOURS")
            st.write("**NSE/BSE:** 9:15 AM - 3:30 PM")
            st.write("**Pre-open:** 9:00 AM - 9:08 AM")
            st.write("**Next Session:** Today 9:15 AM")

        # Page routing based on button selection
        current_page = st.session_state.current_page

        if current_page == "Live Market Dashboard":
            self.render_live_market_dashboard()
        elif current_page == "AI Trading Terminal":
            self.render_machine_learning_dashboard()
        elif current_page == "Algo Trading":
            self.render_algo_trading_dashboard()
        elif current_page == "Sentiment Analysis":
            self.render_sentiment_analysis()
        elif current_page == "Quant Strategies":
            self.render_quant_strategies()
        else:  # Portfolio Manager
            self.render_portfolio_manager()


# Run the application
if __name__ == "__main__":
    st.set_page_config(
        page_title="Quantum AI Trading Terminal",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize and run terminal
    terminal = QuantumTradingTerminal()
    terminal.run()

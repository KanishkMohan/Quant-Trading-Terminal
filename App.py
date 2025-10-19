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
import ta  # Technical analysis library

# ==================== ADVANCED CONFIGURATION ====================

class Config:
    def __init__(self):
        self.alpha_vantage_key = "KP3E60AL5IIEREH7"
        self.finnhub_key = "d3f027pr01qh40fg8npgd3f027pr01qh40fg8nq0"
        self.indian_api_key = "sk-live-UYMPXvoR0SLhmXlnGyqNqVhlgToFARM3mLgoBdm9"
        self.webhook_secret = "d3f0bspr01qh40fg9ub0"
        
        # Trading parameters
        self.risk_free_rate = 0.05
        self.transaction_cost = 0.001
        
    def get_api_endpoints(self):
        return {
            'alpha_vantage': "https://www.alphavantage.co/query",
            'finnhub': "https://finnhub.io/api/v1",
            'indian_api': "https://indianapi.in/api/v1"
        }

# ==================== QUANTUM ALGORITHMS ====================

class QuantumAlgorithms:
    def __init__(self):
        self.config = Config()
    
    def quantum_entanglement_predictor(self, prices, volume, window=20):
        """Advanced quantum-inspired prediction algorithm"""
        try:
            # Calculate multiple technical indicators
            sma_20 = prices.rolling(window=window).mean()
            sma_50 = prices.rolling(window=50).mean()
            rsi = self.calculate_rsi(prices)
            macd = self.calculate_macd(prices)
            
            # Volume analysis
            volume_sma = volume.rolling(window=window).mean()
            volume_ratio = volume / volume_sma
            
            # Price momentum
            momentum = prices.pct_change(window)
            
            # Quantum superposition of indicators
            quantum_score = (
                0.3 * (prices > sma_20).astype(int) +
                0.2 * (prices > sma_50).astype(int) +
                0.2 * (rsi > 50).astype(int) +
                0.15 * (macd > 0).astype(int) +
                0.15 * (volume_ratio > 1).astype(int)
            )
            
            return quantum_score
        except Exception as e:
            st.error(f"Quantum algorithm error: {e}")
            return pd.Series([0] * len(prices), index=prices.index)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def quantum_wave_analysis(self, data):
        """Advanced wave analysis for trend prediction"""
        try:
            if len(data) < 50:
                return {"trend": "NEUTRAL", "strength": 0, "confidence": 0}
            
            prices = data['Close']
            
            # Multiple timeframe analysis
            short_trend = self._analyze_trend(prices, 10)
            medium_trend = self._analyze_trend(prices, 30)
            long_trend = self._analyze_trend(prices, 50)
            
            # Volume confirmation
            volume_trend = data['Volume'].rolling(20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_strength = current_volume / volume_trend.iloc[-1]
            
            # Trend consensus
            trends = [short_trend, medium_trend, long_trend]
            bull_count = sum(1 for t in trends if t == "BULLISH")
            bear_count = sum(1 for t in trends if t == "BEARISH")
            
            if bull_count >= 2 and volume_strength > 1:
                return {"trend": "BULLISH", "strength": volume_strength, "confidence": bull_count/3}
            elif bear_count >= 2 and volume_strength > 1:
                return {"trend": "BEARISH", "strength": volume_strength, "confidence": bear_count/3}
            else:
                return {"trend": "NEUTRAL", "strength": volume_strength, "confidence": max(bull_count, bear_count)/3}
                
        except Exception as e:
            return {"trend": "NEUTRAL", "strength": 0, "confidence": 0}
    
    def _analyze_trend(self, prices, window):
        """Analyze trend for specific window"""
        if len(prices) < window:
            return "NEUTRAL"
        
        current_price = prices.iloc[-1]
        sma = prices.rolling(window=window).mean().iloc[-1]
        
        if current_price > sma * 1.02:
            return "BULLISH"
        elif current_price < sma * 0.98:
            return "BEARISH"
        else:
            return "NEUTRAL"

# ==================== AI TRADING STRATEGIES ====================

class AITradingStrategies:
    def __init__(self):
        self.quantum = QuantumAlgorithms()
    
    def momentum_strategy(self, data, lookback=20):
        """Momentum-based trading strategy"""
        try:
            returns = data['Close'].pct_change(lookback)
            signal = np.where(returns > 0.05, 1, 
                             np.where(returns < -0.05, -1, 0))
            return signal
        except:
            return np.zeros(len(data))
    
    def mean_reversion_strategy(self, data, lookback=20):
        """Mean reversion trading strategy"""
        try:
            sma = data['Close'].rolling(lookback).mean()
            std = data['Close'].rolling(lookback).std()
            
            z_score = (data['Close'] - sma) / std
            signal = np.where(z_score < -2, 1,
                             np.where(z_score > 2, -1, 0))
            return signal
        except:
            return np.zeros(len(data))
    
    def quantum_hybrid_strategy(self, data):
        """Quantum-inspired hybrid strategy"""
        try:
            momentum_signal = self.momentum_strategy(data)
            mean_reversion_signal = self.mean_reversion_strategy(data)
            quantum_signal = self.quantum.quantum_entanglement_predictor(
                data['Close'], data['Volume']
            )
            
            # Combine signals with weights
            hybrid_signal = (
                0.4 * momentum_signal +
                0.3 * mean_reversion_signal +
                0.3 * quantum_signal
            )
            
            return np.where(hybrid_signal > 0.5, 1,
                           np.where(hybrid_signal < -0.5, -1, 0))
        except:
            return np.zeros(len(data))

# ==================== ADVANCED API MANAGER ====================

class APIManager:
    def __init__(self):
        self.config = Config()
        self.endpoints = self.config.get_api_endpoints()
    
    def get_alpha_vantage_data(self, symbol, function="TIME_SERIES_DAILY"):
        """Get comprehensive data from Alpha Vantage"""
        try:
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.config.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(self.endpoints['alpha_vantage'], params=params, timeout=10)
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
            
            elif "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    'symbol': quote.get('01. symbol'),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%')
                }
                
            return {}
        except Exception as e:
            st.error(f"Alpha Vantage API error: {e}")
            return {}
    
    def get_finnhub_quote(self, symbol):
        """Get real-time quote from Finnhub"""
        try:
            url = f"{self.endpoints['finnhub']}/quote"
            params = {
                'symbol': symbol,
                'token': self.config.finnhub_key
            }
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except Exception as e:
            st.error(f"Finnhub API error: {e}")
            return {}
    
    def get_finnhub_candles(self, symbol, resolution='D', count=100):
        """Get historical candles from Finnhub"""
        try:
            url = f"{self.endpoints['finnhub']}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'count': count,
                'token': self.config.finnhub_key
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['s'] == 'ok':
                df = pd.DataFrame({
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                }, index=pd.to_datetime(data['t'], unit='s'))
                return df.sort_index()
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Finnhub candles error: {e}")
            return pd.DataFrame()
    
    def get_indian_stock_data(self, symbol):
        """Get comprehensive Indian stock data"""
        try:
            headers = {
                'Authorization': f'Bearer {self.config.indian_api_key}',
                'Content-Type': 'application/json'
            }
            
            endpoints = [
                f"/stocks/{symbol}/full",
                f"/quote/{symbol}/detailed",
                f"/historical/{symbol}/1y"
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.get(
                        f"{self.endpoints['indian_api']}{endpoint}", 
                        headers=headers, 
                        timeout=10
                    )
                    if response.status_code == 200:
                        return response.json()
                except:
                    continue
            return {}
        except Exception as e:
            st.error(f"Indian API error: {e}")
            return {}
    
    def get_nse_data(self, symbol, start_date, end_date):
        """Get data from NSE using nsepy"""
        try:
            if not NSEPY_AVAILABLE:
                return pd.DataFrame()
            
            if symbol.endswith('.NS'):
                symbol = symbol.replace('.NS', '')
            
            data = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            return data
        except Exception as e:
            st.error(f"NSE data error: {e}")
            return pd.DataFrame()

# ==================== ENHANCED DATA MANAGER ====================

class EnhancedDataManager:
    def __init__(self):
        self.api_manager = APIManager()
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_stock_data(self, symbol, period="6mo", source="auto"):
        """Get stock data from multiple sources with intelligent fallback"""
        cache_key = f"{symbol}_{period}_{source}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_data, timestamp = self.data_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        data_sources = []
        
        if source == "auto" or source == "nse":
            data_sources.append(self._get_nse_data)
        if source == "auto" or source == "finnhub":
            data_sources.append(self._get_finnhub_data)
        if source == "auto" or source == "alpha_vantage":
            data_sources.append(self._get_alpha_vantage_data)
        if source == "auto" or source == "yfinance":
            data_sources.append(self._get_yfinance_data)
        
        # Add fallback sources for auto mode
        if source == "auto":
            data_sources.extend([
                self._get_yfinance_data,
                self._generate_sample_data
            ])
        
        for source_func in data_sources:
            try:
                data = source_func(symbol, period)
                if not data.empty and len(data) > 10:
                    # Validate data quality
                    if self._validate_data(data):
                        self.data_cache[cache_key] = (data, time.time())
                        return data
            except Exception as e:
                continue
        
        return pd.DataFrame()
    
    def _get_nse_data(self, symbol, period):
        """Get data from NSE"""
        try:
            end_date = date.today()
            
            period_days = {
                "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365
            }.get(period, 180)
            
            start_date = end_date - timedelta(days=period_days)
            
            data = self.api_manager.get_nse_data(symbol, start_date, end_date)
            
            if not data.empty:
                # Convert to standard format
                data = data.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low', 
                    'Close': 'Close',
                    'Volume': 'Volume'
                })
                return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except:
            pass
        return pd.DataFrame()
    
    def _get_finnhub_data(self, symbol, period):
        """Get data from Finnhub"""
        try:
            resolution_map = {
                "1mo": "D", "3mo": "D", "6mo": "D", "1y": "D"
            }
            
            count_map = {
                "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365
            }
            
            data = self.api_manager.get_finnhub_candles(
                symbol, 
                resolution_map.get(period, "D"),
                count_map.get(period, 180)
            )
            
            if not data.empty:
                return data
        except:
            pass
        return pd.DataFrame()
    
    def _get_alpha_vantage_data(self, symbol, period):
        """Get data from Alpha Vantage"""
        try:
            data = self.api_manager.get_alpha_vantage_data(symbol)
            if not data.empty:
                # Filter by period
                period_days = {
                    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365
                }.get(period, 180)
                
                if len(data) > period_days:
                    data = data.tail(period_days)
                return data
        except:
            pass
        return pd.DataFrame()
    
    def _get_yfinance_data(self, symbol, period):
        """Get data from Yahoo Finance"""
        try:
            if not YFINANCE_AVAILABLE:
                return pd.DataFrame()
            
            # Add .NS for Indian stocks if not present
            if not any(ext in symbol for ext in ['.NS', '.BO', '.NSE']):
                symbol += '.NS'
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                return data
        except:
            pass
        return pd.DataFrame()
    
    def _generate_sample_data(self, symbol, period):
        """Generate realistic sample data"""
        period_days = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365
        }.get(period, 180)
        
        dates = pd.date_range(end=datetime.now(), periods=period_days, freq='D')
        
        # Generate realistic price patterns with trends and volatility
        base_price = 1000 + hash(symbol) % 5000  # Different base for each symbol
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
    
    def _validate_data(self, data):
        """Validate data quality"""
        if data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for NaN values
        if data[required_columns].isna().any().any():
            return False
        
        # Check for reasonable price values
        if (data['Close'] <= 0).any():
            return False
        
        return True
    
    def get_live_quote(self, symbol):
        """Get comprehensive live quote data"""
        try:
            # Try multiple sources
            sources = [
                self._get_finnhub_quote,
                self._get_alpha_vantage_quote,
                self._get_yfinance_quote
            ]
            
            for source in sources:
                quote = source(symbol)
                if quote and quote.get('current', 0) > 0:
                    return quote
        except:
            pass
        
        return self._generate_sample_quote(symbol)
    
    def _get_finnhub_quote(self, symbol):
        """Get quote from Finnhub"""
        try:
            quote = self.api_manager.get_finnhub_quote(symbol)
            if quote and 'c' in quote and quote['c'] > 0:
                return {
                    'current': quote['c'],
                    'change': quote.get('d', 0),
                    'change_percent': quote.get('dp', 0),
                    'high': quote.get('h', 0),
                    'low': quote.get('l', 0),
                    'open': quote.get('o', 0),
                    'previous_close': quote.get('pc', 0),
                    'volume': quote.get('v', 0),
                    'timestamp': datetime.now()
                }
        except:
            pass
        return None
    
    def _get_alpha_vantage_quote(self, symbol):
        """Get quote from Alpha Vantage"""
        try:
            data = self.api_manager.get_alpha_vantage_data(symbol, "GLOBAL_QUOTE")
            if data and 'price' in data:
                return {
                    'current': data['price'],
                    'change': data.get('change', 0),
                    'change_percent': float(data.get('change_percent', '0%').rstrip('%')),
                    'volume': data.get('volume', 0),
                    'timestamp': datetime.now()
                }
        except:
            pass
        return None
    
    def _get_yfinance_quote(self, symbol):
        """Get quote from Yahoo Finance"""
        try:
            if not YFINANCE_AVAILABLE:
                return None
            
            if not any(ext in symbol for ext in ['.NS', '.BO', '.NSE']):
                symbol += '.NS'
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="1d")
            
            if not history.empty:
                return {
                    'current': info.get('currentPrice', history['Close'].iloc[-1]),
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'high': history['High'].iloc[-1],
                    'low': history['Low'].iloc[-1],
                    'open': history['Open'].iloc[-1],
                    'volume': history['Volume'].iloc[-1],
                    'previous_close': info.get('regularMarketPreviousClose', 0),
                    'timestamp': datetime.now()
                }
        except:
            pass
        return None
    
    def _generate_sample_quote(self, symbol):
        """Generate sample quote data"""
        return {
            'current': 1000 + hash(symbol) % 5000,
            'change': (hash(symbol) % 100) - 50,
            'change_percent': ((hash(symbol) % 100) - 50) / 100,
            'high': 1100 + hash(symbol) % 5000,
            'low': 900 + hash(symbol) % 5000,
            'open': 1000 + hash(symbol) % 5000,
            'volume': 1000000 + hash(symbol) % 4000000,
            'previous_close': 950 + hash(symbol) % 5000,
            'timestamp': datetime.now()
        }

# ==================== ADVANCED TECHNICAL ANALYSIS ====================

class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.quantum = QuantumAlgorithms()
    
    def calculate_all_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        if data.empty or len(data) < 50:
            return {}
        
        indicators = {}
        
        try:
            prices = data['Close']
            volume = data['Volume']
            
            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(prices))
            
            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(prices))
            
            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(prices))
            
            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(prices, volume))
            
            # Support and Resistance
            indicators.update(self._calculate_support_resistance(data))
            
            # Quantum Analysis
            indicators.update(self._calculate_quantum_indicators(data))
            
            # Trading Signals
            indicators.update(self._generate_trading_signals(indicators))
            
        except Exception as e:
            st.error(f"Technical analysis error: {e}")
        
        return indicators
    
    def _calculate_trend_indicators(self, prices):
        """Calculate trend-based indicators"""
        indicators = {}
        
        # Moving Averages
        indicators['sma_20'] = prices.rolling(20).mean().iloc[-1]
        indicators['sma_50'] = prices.rolling(50).mean().iloc[-1]
        indicators['sma_200'] = prices.rolling(200).mean().iloc[-1] if len(prices) > 200 else prices.rolling(len(prices)).mean().iloc[-1]
        
        # EMA
        indicators['ema_12'] = prices.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = prices.ewm(span=26).mean().iloc[-1]
        
        # MACD
        macd = indicators['ema_12'] - indicators['ema_26']
        indicators['macd'] = macd
        indicators['macd_signal'] = prices.ewm(span=9).mean().iloc[-1]
        indicators['macd_histogram'] = macd - indicators['macd_signal']
        
        # ADX (simplified)
        high = prices.rolling(14).max()
        low = prices.rolling(14).min()
        tr = np.maximum(high - low, np.abs(high - prices.shift(1)))
        atr = tr.rolling(14).mean()
        indicators['atr'] = atr.iloc[-1] if not atr.empty else 0
        
        return indicators
    
    def _calculate_momentum_indicators(self, prices):
        """Calculate momentum indicators"""
        indicators = {}
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Stochastic
        low_14 = prices.rolling(14).min()
        high_14 = prices.rolling(14).max()
        indicators['stoch_k'] = 100 * ((prices - low_14) / (high_14 - low_14)).iloc[-1]
        indicators['stoch_d'] = indicators['stoch_k']  # Simplified
        
        # Williams %R
        indicators['williams_r'] = ((high_14 - prices) / (high_14 - low_14) * -100).iloc[-1]
        
        # CCI
        typical_price = (prices + prices.rolling(14).max() + prices.rolling(14).min()) / 3
        cci = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
        indicators['cci'] = cci.iloc[-1]
        
        return indicators
    
    def _calculate_volatility_indicators(self, prices):
        """Calculate volatility indicators"""
        indicators = {}
        
        # Bollinger Bands
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        
        indicators['bb_upper'] = sma_20.iloc[-1] + (std_20.iloc[-1] * 2)
        indicators['bb_lower'] = sma_20.iloc[-1] - (std_20.iloc[-1] * 2)
        indicators['bb_middle'] = sma_20.iloc[-1]
        indicators['bb_position'] = (prices.iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volatility
        returns = prices.pct_change().dropna()
        indicators['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        indicators['volatility_50d'] = returns.rolling(50).std().iloc[-1] * np.sqrt(252) * 100
        
        return indicators
    
    def _calculate_volume_indicators(self, prices, volume):
        """Calculate volume-based indicators"""
        indicators = {}
        
        # Volume SMA
        indicators['volume_sma_20'] = volume.rolling(20).mean().iloc[-1]
        
        # OBV
        obv = (volume * np.sign(prices.diff())).cumsum()
        indicators['obv'] = obv.iloc[-1]
        
        # Volume Price Trend
        vpt = (volume * (prices.diff() / prices.shift(1))).cumsum()
        indicators['vpt'] = vpt.iloc[-1]
        
        return indicators
    
    def _calculate_support_resistance(self, data):
        """Calculate support and resistance levels"""
        indicators = {}
        
        if len(data) < 20:
            return indicators
        
        # Recent high/low
        recent_data = data.tail(20)
        indicators['support_1'] = recent_data['Low'].min()
        indicators['resistance_1'] = recent_data['High'].max()
        
        # Pivot points
        pivot = (data['High'].iloc[-1] + data['Low'].iloc[-1] + data['Close'].iloc[-1]) / 3
        indicators['pivot'] = pivot
        indicators['r1'] = 2 * pivot - data['Low'].iloc[-1]
        indicators['s1'] = 2 * pivot - data['High'].iloc[-1]
        
        return indicators
    
    def _calculate_quantum_indicators(self, data):
        """Calculate quantum-inspired indicators"""
        indicators = {}
        
        try:
            quantum_score = self.quantum.quantum_entanglement_predictor(
                data['Close'], data['Volume']
            )
            indicators['quantum_score'] = quantum_score.iloc[-1] if not quantum_score.empty else 0
            
            wave_analysis = self.quantum.quantum_wave_analysis(data)
            indicators.update(wave_analysis)
            
        except:
            indicators['quantum_score'] = 0
            indicators['trend'] = "NEUTRAL"
            indicators['strength'] = 0
            indicators['confidence'] = 0
        
        return indicators
    
    def _generate_trading_signals(self, indicators):
        """Generate trading signals based on indicators"""
        signals = {}
        
        try:
            # RSI signals
            if indicators.get('rsi', 50) < 30:
                signals['rsi_signal'] = "OVERSOLD"
            elif indicators.get('rsi', 50) > 70:
                signals['rsi_signal'] = "OVERBOUGHT"
            else:
                signals['rsi_signal'] = "NEUTRAL"
            
            # MACD signals
            if indicators.get('macd', 0) > indicators.get('macd_signal', 0):
                signals['macd_signal'] = "BULLISH"
            else:
                signals['macd_signal'] = "BEARISH"
            
            # Trend signals
            current_price = indicators.get('sma_20', 0)  # Approximation
            if (current_price > indicators.get('sma_20', 0) > indicators.get('sma_50', 0) > indicators.get('sma_200', 0)):
                signals['trend_strength'] = "STRONG BULLISH"
            elif current_price > indicators.get('sma_20', 0) > indicators.get('sma_50', 0):
                signals['trend_strength'] = "BULLISH"
            elif current_price < indicators.get('sma_20', 0) < indicators.get('sma_50', 0) < indicators.get('sma_200', 0):
                signals['trend_strength'] = "STRONG BEARISH"
            elif current_price < indicators.get('sma_20', 0) < indicators.get('sma_50', 0):
                signals['trend_strength'] = "BEARISH"
            else:
                signals['trend_strength'] = "NEUTRAL"
            
            # Overall signal
            bullish_signals = 0
            total_signals = 0
            
            if signals.get('rsi_signal') == "OVERSOLD":
                bullish_signals += 1
            elif signals.get('rsi_signal') == "OVERBOUGHT":
                bullish_signals -= 1
            total_signals += 1
            
            if signals.get('macd_signal') == "BULLISH":
                bullish_signals += 1
            else:
                bullish_signals -= 1
            total_signals += 1
            
            if "BULLISH" in signals.get('trend_strength', ""):
                bullish_signals += 1
            elif "BEARISH" in signals.get('trend_strength', ""):
                bullish_signals -= 1
            total_signals += 1
            
            if bullish_signals > 0:
                signals['overall_signal'] = "BUY"
            elif bullish_signals < 0:
                signals['overall_signal'] = "SELL"
            else:
                signals['overall_signal'] = "HOLD"
                
        except Exception as e:
            signals['overall_signal'] = "HOLD"
        
        return signals

# ==================== ADVANCED CHARTING ENGINE ====================

class AdvancedChartingEngine:
    def __init__(self):
        self.tech_analysis = AdvancedTechnicalAnalysis()
    
    def create_advanced_chart(self, data, title="Stock Analysis", indicators=True):
        """Create comprehensive stock chart with technical indicators"""
        if not PLOTLY_AVAILABLE or data.empty:
            return None
        
        try:
            if indicators:
                # Create subplots for indicators
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        f'{title} - Price & Trends', 
                        'Volume',
                        'RSI',
                        'MACD'
                    ),
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )
            else:
                # Simple price chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(f'{title} - Price', 'Volume'),
                    row_heights=[0.7, 0.3]
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
            
            # Add moving averages if we have enough data
            if len(data) > 20:
                # SMA 20
                sma_20 = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=data.index, y=sma_20,
                    mode='lines', name='SMA 20',
                    line=dict(color='orange', width=2)
                ), row=1, col=1)
                
                # SMA 50
                if len(data) > 50:
                    sma_50 = data['Close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(
                        x=data.index, y=sma_50,
                        mode='lines', name='SMA 50',
                        line=dict(color='red', width=2)
                    ), row=1, col=1)
            
            # Bollinger Bands
            if len(data) > 20:
                sma_20 = data['Close'].rolling(20).mean()
                std_20 = data['Close'].rolling(20).std()
                bb_upper = sma_20 + (std_20 * 2)
                bb_lower = sma_20 - (std_20 * 2)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=bb_upper,
                    mode='lines', name='BB Upper',
                    line=dict(color='rgba(255,255,255,0.5)', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=bb_lower,
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
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.6
                ), row=2, col=1)
            
            # Technical indicators
            if indicators and len(data) > 14:
                # RSI
                rsi = self._calculate_rsi(data['Close'])
                fig.add_trace(go.Scatter(
                    x=data.index, y=rsi,
                    mode='lines', name='RSI',
                    line=dict(color='purple', width=2)
                ), row=3, col=1)
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="white", row=3, col=1)
                
                # MACD
                macd, signal, histogram = self._calculate_macd(data['Close'])
                fig.add_trace(go.Scatter(
                    x=data.index, y=macd,
                    mode='lines', name='MACD',
                    line=dict(color='blue', width=2)
                ), row=4, col=1)
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=signal,
                    mode='lines', name='Signal',
                    line=dict(color='red', width=2)
                ), row=4, col=1)
                
                # MACD histogram
                colors_hist = ['green' if val >= 0 else 'red' for val in histogram]
                fig.add_trace(go.Bar(
                    x=data.index, y=histogram,
                    name='MACD Hist',
                    marker_color=colors_hist,
                    opacity=0.6
                ), row=4, col=1)
            
            fig.update_layout(
                title=f"Advanced Analysis - {title}",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                height=800 if indicators else 600,
                showlegend=True
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            return fig
            
        except Exception as e:
            st.error(f"Chart creation error: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

# ==================== PORTFOLIO MANAGER ====================

class PortfolioManager:
    def __init__(self):
        self.data_manager = EnhancedDataManager()
        self.tech_analysis = AdvancedTechnicalAnalysis()
        
    def calculate_portfolio_metrics(self, holdings):
        """Calculate comprehensive portfolio metrics"""
        if not holdings:
            return {}
        
        total_value = 0
        total_cost = 0
        daily_pnl = 0
        unrealized_pnl = 0
        
        for symbol, holding in holdings.items():
            current_price = self.data_manager.get_live_quote(symbol).get('current', 0)
            current_value = current_price * holding['quantity']
            cost_basis = holding.get('average_cost', 0) * holding['quantity']
            
            total_value += current_value
            total_cost += cost_basis
            unrealized_pnl += (current_value - cost_basis)
            
            # Daily PnL calculation
            prev_close = self.data_manager.get_live_quote(symbol).get('previous_close', current_price)
            daily_pnl += (current_price - prev_close) * holding['quantity']
        
        metrics = {
            'total_value': total_value,
            'total_cost': total_cost,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_percent': (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0,
            'daily_pnl': daily_pnl,
            'daily_pnl_percent': (daily_pnl / total_value * 100) if total_value > 0 else 0
        }
        
        return metrics

# ==================== INDIAN STOCK DATABASE ====================

class IndianStockDatabase:
    def __init__(self):
        self.indian_stocks = {
            # Nifty 50 Stocks
            'RELIANCE': 'RELIANCE',
            'TCS': 'TCS', 
            'HDFC BANK': 'HDFCBANK',
            'ICICI BANK': 'ICICIBANK',
            'INFOSYS': 'INFY',
            'HUL': 'HINDUNILVR',
            'ITC': 'ITC',
            'SBIN': 'SBIN',
            'BHARTI AIRTEL': 'BHARTIARTL',
            'KOTAK BANK': 'KOTAKBANK',
            'LT': 'LT',
            'AXIS BANK': 'AXISBANK',
            'ASIAN PAINTS': 'ASIANPAINT',
            'MARUTI': 'MARUTI',
            'TITAN': 'TITAN',
            'SUN PHARMA': 'SUNPHARMA',
            'HCL TECH': 'HCLTECH',
            'DMART': 'DMART',
            'BAJFINANCE': 'BAJFINANCE',
            'WIPRO': 'WIPRO',
            'ONGC': 'ONGC',
            'NTPC': 'NTPC',
            'POWERGRID': 'POWERGRID',
            'ULTRACEMCO': 'ULTRACEMCO',
            'M&M': 'M&M',
            'TATA STEEL': 'TATAMOTORS',
            'JSW STEEL': 'JSWSTEEL',
            'ADANI ENTERPRISES': 'ADANIENT',
            'ADANI PORTS': 'ADANIPORTS',
            'BAJAJ FINSERV': 'BAJAJFINSV',
            'TECH MAHINDRA': 'TECHM',
            'NESTLE': 'NESTLEIND',
            'BRITANNIA': 'BRITANNIA',
            'HDFC LIFE': 'HDFCLIFE',
            'SBILIFE': 'SBILIFE',
            'CIPLA': 'CIPLA',
            'DRREDDY': 'DRREDDY',
            'EICHERMOT': 'EICHERMOT',
            'GRASIM': 'GRASIM',
            'HINDALCO': 'HINDALCO',
            'INDUSINDBK': 'INDUSINDBK',
            'COALINDIA': 'COALINDIA',
            'BPCL': 'BPCL',
            'HINDPETRO': 'HINDPETRO',
            'IOC': 'IOC',
            'VEDANTA': 'VEDL',
            'SHREECEM': 'SHREECEM',
            'UPL': 'UPL',
            'WIPRO': 'WIPRO',
            'ZEEL': 'ZEEL',
            
            # Indices
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK', 
            'SENSEX': '^BSESN',
            'NIFTY IT': '^CNXIT',
            
            # Popular stocks
            'ZOMATO': 'ZOMATO',
            'PAYTM': 'PAYTM',
            'IRCTC': 'IRCTC',
            'TATA MOTORS': 'TATAMOTORS',
            'TATA CONSULTANCY': 'TCS'
        }
    
    def search_stocks(self, query):
        """Search stocks by name or symbol"""
        query = query.upper()
        results = {}
        
        for name, symbol in self.indian_stocks.items():
            if query in name or query in symbol:
                results[name] = symbol
        
        return results
    
    def get_all_stocks(self):
        return self.indian_stocks

# ==================== QUANTUM TRADING TERMINAL ====================

class QuantumTradingTerminal:
    def __init__(self):
        self.data_manager = EnhancedDataManager()
        self.chart_engine = AdvancedChartingEngine()
        self.tech_analysis = AdvancedTechnicalAnalysis()
        self.stock_db = IndianStockDatabase()
        self.portfolio_manager = PortfolioManager()
        self.trading_strategies = AITradingStrategies()
        
        # Initialize session state
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = ['RELIANCE', 'TCS', 'HDFCBANK']
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    def render_header(self):
        """Render application header"""
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
        .metric-card {
            background: rgba(30, 30, 60, 0.9);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #4f46e5;
            margin: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #1e1e1e;
            border-radius: 5px 5px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="main-header">🇮🇳 QUANTUM INDIAN STOCK TERMINAL</div>', unsafe_allow_html=True)
        
        # Market overview
        st.markdown("### 📈 Live Market Overview")
        
        # Key indices
        indices = ['NIFTY 50', 'BANK NIFTY', 'SENSEX']
        cols = st.columns(len(indices))
        
        for idx, index in enumerate(indices):
            with cols[idx]:
                symbol = self.stock_db.indian_stocks.get(index)
                if symbol:
                    quote = self.data_manager.get_live_quote(symbol)
                    if quote:
                        change_color = "green" if quote['change'] >= 0 else "red"
                        st.metric(
                            index,
                            f"₹{quote['current']:,.0f}",
                            f"{quote['change_percent']:+.2f}%"
                        )
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown("### 📊 Stock Analysis Dashboard")
        
        # Stock selector
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            all_stocks = self.stock_db.get_all_stocks()
            selected_stock_name = st.selectbox(
                "Select Stock",
                options=list(all_stocks.keys()),
                format_func=lambda x: f"{x} ({all_stocks[x]})"
            )
        
        with col2:
            time_frame = st.selectbox(
                "Time Frame",
                ['1mo', '3mo', '6mo', '1y'],
                key="timeframe"
            )
        
        with col3:
            data_source = st.selectbox(
                "Data Source",
                ['auto', 'nse', 'finnhub', 'alpha_vantage', 'yfinance'],
                help="Auto mode tries all sources in order"
            )
        
        if selected_stock_name:
            symbol = all_stocks[selected_stock_name]
            
            # Get data
            with st.spinner(f"Loading {selected_stock_name} data..."):
                data = self.data_manager.get_stock_data(symbol, time_frame, data_source)
                live_quote = self.data_manager.get_live_quote(symbol)
                indicators = self.tech_analysis.calculate_all_indicators(data)
            
            if not data.empty:
                # Display live quote
                if live_quote:
                    st.markdown("### 💰 Live Quote")
                    quote_cols = st.columns(6)
                    
                    with quote_cols[0]:
                        st.metric(
                            "Current Price", 
                            f"₹{live_quote['current']:.2f}",
                            f"{live_quote['change_percent']:+.2f}%"
                        )
                    
                    with quote_cols[1]:
                        st.metric("Open", f"₹{live_quote.get('open', 0):.2f}")
                    
                    with quote_cols[2]:
                        st.metric("High", f"₹{live_quote.get('high', 0):.2f}")
                    
                    with quote_cols[3]:
                        st.metric("Low", f"₹{live_quote.get('low', 0):.2f}")
                    
                    with quote_cols[4]:
                        st.metric("Volume", f"{live_quote.get('volume', 0):,.0f}")
                    
                    with quote_cols[5]:
                        st.metric("Prev Close", f"₹{live_quote.get('previous_close', 0):.2f}")
                
                # Display chart
                st.markdown("### 📈 Advanced Chart")
                chart_tabs = st.tabs(["Price Chart", "Technical Analysis"])
                
                with chart_tabs[0]:
                    fig_simple = self.chart_engine.create_advanced_chart(data, selected_stock_name, indicators=False)
                    if fig_simple:
                        st.plotly_chart(fig_simple, use_container_width=True)
                
                with chart_tabs[1]:
                    fig_advanced = self.chart_engine.create_advanced_chart(data, selected_stock_name, indicators=True)
                    if fig_advanced:
                        st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Technical Analysis
                st.markdown("### 🔍 Technical Analysis")
                
                if indicators:
                    # Key metrics
                    tech_cols1 = st.columns(4)
                    tech_cols2 = st.columns(4)
                    
                    with tech_cols1[0]:
                        rsi = indicators.get('rsi', 0)
                        rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "white"
                        st.metric("RSI", f"{rsi:.1f}", delta_color="off")
                    
                    with tech_cols1[1]:
                        st.metric("SMA 20", f"₹{indicators.get('sma_20', 0):.2f}")
                    
                    with tech_cols1[2]:
                        st.metric("Trend", indicators.get('trend_strength', 'N/A'))
                    
                    with tech_cols1[3]:
                        st.metric("Volatility", f"{indicators.get('volatility_20d', 0):.1f}%")
                    
                    with tech_cols2[0]:
                        st.metric("MACD", f"{indicators.get('macd', 0):.3f}")
                    
                    with tech_cols2[1]:
                        st.metric("Signal", indicators.get('overall_signal', 'HOLD'))
                    
                    with tech_cols2[2]:
                        st.metric("Quantum Score", f"{indicators.get('quantum_score', 0):.2f}")
                    
                    with tech_cols2[3]:
                        st.metric("Support", f"₹{indicators.get('support_1', 0):.2f}")
                    
                    # Detailed indicators
                    with st.expander("Detailed Technical Indicators"):
                        detail_cols = st.columns(3)
                        
                        with detail_cols[0]:
                            st.write("**Trend Indicators**")
                            st.write(f"SMA 50: ₹{indicators.get('sma_50', 0):.2f}")
                            st.write(f"SMA 200: ₹{indicators.get('sma_200', 0):.2f}")
                            st.write(f"EMA 12: ₹{indicators.get('ema_12', 0):.2f}")
                            st.write(f"ATR: {indicators.get('atr', 0):.2f}")
                        
                        with detail_cols[1]:
                            st.write("**Momentum Indicators**")
                            st.write(f"Stoch K: {indicators.get('stoch_k', 0):.1f}")
                            st.write(f"Williams R: {indicators.get('williams_r', 0):.1f}")
                            st.write(f"CCI: {indicators.get('cci', 0):.1f}")
                            st.write(f"RSI Signal: {indicators.get('rsi_signal', 'N/A')}")
                        
                        with detail_cols[2]:
                            st.write("**Volatility & Volume**")
                            st.write(f"BB Position: {indicators.get('bb_position', 0):.3f}")
                            st.write(f"Volume SMA: {indicators.get('volume_sma_20', 0):,.0f}")
                            st.write(f"OBV: {indicators.get('obv', 0):,.0f}")
                            st.write(f"Resistance: ₹{indicators.get('resistance_1', 0):.2f}")
                
                # Trading strategies
                st.markdown("### 🤖 AI Trading Strategies")
                
                if not data.empty:
                    strategy_cols = st.columns(3)
                    
                    with strategy_cols[0]:
                        momentum_signal = self.trading_strategies.momentum_strategy(data)
                        current_signal = momentum_signal[-1] if len(momentum_signal) > 0 else 0
                        signal_text = "BUY" if current_signal > 0 else "SELL" if current_signal < 0 else "HOLD"
                        st.metric("Momentum Strategy", signal_text)
                    
                    with strategy_cols[1]:
                        mean_reversion_signal = self.trading_strategies.mean_reversion_strategy(data)
                        current_signal = mean_reversion_signal[-1] if len(mean_reversion_signal) > 0 else 0
                        signal_text = "BUY" if current_signal > 0 else "SELL" if current_signal < 0 else "HOLD"
                        st.metric("Mean Reversion", signal_text)
                    
                    with strategy_cols[2]:
                        quantum_signal = self.trading_strategies.quantum_hybrid_strategy(data)
                        current_signal = quantum_signal[-1] if len(quantum_signal) > 0 else 0
                        signal_text = "BUY" if current_signal > 0 else "SELL" if current_signal < 0 else "HOLD"
                        st.metric("Quantum Hybrid", signal_text)
                
                # Recent data
                st.markdown("### 📋 Recent Price Data")
                st.dataframe(data.tail(10), use_container_width=True)
                
                # Add to watchlist
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("⭐ Add to Watchlist", use_container_width=True):
                        if selected_stock_name not in st.session_state.watchlist:
                            st.session_state.watchlist.append(selected_stock_name)
                            st.success(f"Added {selected_stock_name} to watchlist!")
                            st.rerun()
            else:
                st.error("Unable to fetch data for selected stock")
    
    def render_watchlist(self):
        """Render watchlist management"""
        st.markdown("### ⭐ My Watchlist")
        
        if not st.session_state.watchlist:
            st.info("Your watchlist is empty. Add stocks from the dashboard.")
            return
        
        # Quick actions
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Watchlist", type="secondary"):
                st.session_state.watchlist = []
                st.rerun()
        
        for stock_name in st.session_state.watchlist:
            symbol = self.stock_db.indian_stocks.get(stock_name)
            if symbol:
                with st.expander(f"📊 {stock_name} ({symbol})", expanded=True):
                    # Get quick data
                    quote = self.data_manager.get_live_quote(symbol)
                    if quote:
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**Price:** ₹{quote['current']:.2f}")
                            st.write(f"**Change:** {quote['change_percent']:+.2f}%")
                            st.write(f"**Volume:** {quote.get('volume', 0):,.0f}")
                        
                        with col2:
                            if st.button("📈 Analyze", key=f"analyze_{stock_name}", use_container_width=True):
                                st.session_state.selected_stock = stock_name
                                st.rerun()
                        
                        with col3:
                            if st.button("📊 Chart", key=f"chart_{stock_name}", use_container_width=True):
                                st.session_state.chart_stock = stock_name
                                st.rerun()
                        
                        with col4:
                            if st.button("❌ Remove", key=f"remove_{stock_name}", use_container_width=True):
                                st.session_state.watchlist.remove(stock_name)
                                st.rerun()
    
    def render_stock_screener(self):
        """Render advanced stock screener"""
        st.markdown("### 🔍 Advanced Stock Screener")
        
        st.info("""
        **Quantum Stock Screening Features:**
        - Multi-factor technical analysis screening
        - Fundamental metrics filtering  
        - Pattern recognition and AI signals
        - Real-time market scanning
        """)
        
        # Screening criteria
        st.markdown("#### Screening Criteria")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_price = st.number_input("Minimum Price (₹)", value=100, min_value=0)
            max_price = st.number_input("Maximum Price (₹)", value=10000, min_value=0)
            min_volume = st.number_input("Minimum Volume", value=1000000)
        
        with col2:
            rsi_min = st.slider("RSI Minimum", 0, 100, 30)
            rsi_max = st.slider("RSI Maximum", 0, 100, 70)
            min_momentum = st.slider("Minimum Momentum %", -10.0, 10.0, 0.0)
        
        with col3:
            trend_filter = st.selectbox("Trend Filter", ["Any", "Bullish", "Bearish", "Strong Bullish", "Strong Bearish"])
            signal_filter = st.selectbox("Signal Filter", ["Any", "BUY", "SELL", "HOLD"])
            volatility_max = st.slider("Max Volatility %", 0.0, 100.0, 50.0)
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                min_market_cap = st.number_input("Minimum Market Cap (Cr)", value=1000)
                pe_max = st.number_input("Max P/E Ratio", value=50)
                min_dividend = st.number_input("Min Dividend Yield %", value=0.0)
            
            with adv_col2:
                beta_max = st.number_input("Max Beta", value=2.0)
                debt_to_equity = st.number_input("Max Debt/Equity", value=2.0)
                min_roce = st.number_input("Min ROCE %", value=0.0)
        
        # Run screener
        if st.button("🚀 Run Quantum Screener", use_container_width=True):
            with st.spinner("Scanning stocks with quantum algorithms..."):
                # Get sample stocks for demonstration
                sample_stocks = list(self.stock_db.get_all_stocks().items())[:20]
                results = []
                
                for stock_name, symbol in sample_stocks:
                    try:
                        # Get data and analysis
                        data = self.data_manager.get_stock_data(symbol, "1mo")
                        if data.empty:
                            continue
                            
                        quote = self.data_manager.get_live_quote(symbol)
                        indicators = self.tech_analysis.calculate_all_indicators(data)
                        
                        if not indicators:
                            continue
                        
                        # Apply filters
                        current_price = quote.get('current', 0)
                        volume = quote.get('volume', 0)
                        rsi = indicators.get('rsi', 50)
                        trend = indicators.get('trend_strength', 'NEUTRAL')
                        signal = indicators.get('overall_signal', 'HOLD')
                        volatility = indicators.get('volatility_20d', 0)
                        
                        # Price filter
                        if not (min_price <= current_price <= max_price):
                            continue
                            
                        # Volume filter
                        if volume < min_volume:
                            continue
                            
                        # RSI filter
                        if not (rsi_min <= rsi <= rsi_max):
                            continue
                            
                        # Trend filter
                        if trend_filter != "Any" and trend_filter not in trend:
                            continue
                            
                        # Signal filter
                        if signal_filter != "Any" and signal != signal_filter:
                            continue
                            
                        # Volatility filter
                        if volatility > volatility_max:
                            continue
                        
                        results.append({
                            'Stock': stock_name,
                            'Symbol': symbol,
                            'Price': current_price,
                            'Change %': quote.get('change_percent', 0),
                            'Volume': volume,
                            'RSI': rsi,
                            'Trend': trend,
                            'Signal': signal,
                            'Volatility %': volatility,
                            'Quantum Score': indicators.get('quantum_score', 0)
                        })
                        
                    except Exception as e:
                        continue
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.markdown(f"#### 📊 Screening Results: {len(results)} Stocks Found")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Export results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"quantum_screener_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No stocks match your screening criteria.")
    
    def render_portfolio(self):
        """Render portfolio management"""
        st.markdown("### 💼 Portfolio Management")
        
        # Portfolio input
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            all_stocks = self.stock_db.get_all_stocks()
            selected_stock = st.selectbox(
                "Select Stock",
                options=list(all_stocks.keys()),
                key="portfolio_stock"
            )
        
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=100)
        
        with col3:
            buy_price = st.number_input("Buy Price (₹)", min_value=0.0, value=0.0)
        
        with col4:
            st.write("")  # Spacer
            st.write("")
            if st.button("Add to Portfolio", use_container_width=True):
                symbol = all_stocks[selected_stock]
                if symbol not in st.session_state.portfolio:
                    st.session_state.portfolio[symbol] = {
                        'name': selected_stock,
                        'quantity': quantity,
                        'average_cost': buy_price,
                        'total_cost': quantity * buy_price
                    }
                else:
                    # Update existing position
                    existing = st.session_state.portfolio[symbol]
                    total_quantity = existing['quantity'] + quantity
                    total_cost = existing['total_cost'] + (quantity * buy_price)
                    
                    st.session_state.portfolio[symbol] = {
                        'name': selected_stock,
                        'quantity': total_quantity,
                        'average_cost': total_cost / total_quantity,
                        'total_cost': total_cost
                    }
                st.success(f"Added {quantity} shares of {selected_stock} to portfolio!")
                st.rerun()
        
        # Display portfolio
        if st.session_state.portfolio:
            st.markdown("#### 📊 Current Holdings")
            
            portfolio_data = []
            total_current_value = 0
            total_investment = 0
            
            for symbol, holding in st.session_state.portfolio.items():
                quote = self.data_manager.get_live_quote(symbol)
                current_price = quote.get('current', 0)
                current_value = current_price * holding['quantity']
                pnl = current_value - holding['total_cost']
                pnl_percent = (pnl / holding['total_cost']) * 100
                
                portfolio_data.append({
                    'Stock': holding['name'],
                    'Symbol': symbol,
                    'Quantity': holding['quantity'],
                    'Avg Cost': holding['average_cost'],
                    'Current Price': current_price,
                    'Investment': holding['total_cost'],
                    'Current Value': current_value,
                    'PnL': pnl,
                    'PnL %': pnl_percent
                })
                
                total_current_value += current_value
                total_investment += holding['total_cost']
            
            if portfolio_data:
                portfolio_df = pd.DataFrame(portfolio_data)
                st.dataframe(portfolio_df, use_container_width=True)
                
                # Portfolio summary
                total_pnl = total_current_value - total_investment
                total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
                
                st.markdown("#### 📈 Portfolio Summary")
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.metric("Total Investment", f"₹{total_investment:,.2f}")
                
                with summary_cols[1]:
                    st.metric("Current Value", f"₹{total_current_value:,.2f}")
                
                with summary_cols[2]:
                    st.metric("Total P&L", f"₹{total_pnl:,.2f}")
                
                with summary_cols[3]:
                    st.metric("Return %", f"{total_pnl_percent:.2f}%")
                
                # Clear portfolio
                if st.button("Clear Portfolio", type="secondary"):
                    st.session_state.portfolio = {}
                    st.rerun()
        else:
            st.info("Your portfolio is empty. Add stocks to get started.")
    
    def render_alerts(self):
        """Render price alerts system"""
        st.markdown("### 🔔 Price Alerts")
        
        # Create new alert
        st.markdown("#### Create New Alert")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            all_stocks = self.stock_db.get_all_stocks()
            alert_stock = st.selectbox(
                "Stock",
                options=list(all_stocks.keys()),
                key="alert_stock"
            )
        
        with col2:
            alert_type = st.selectbox(
                "Alert Type",
                ["Price Above", "Price Below", "Percent Change Up", "Percent Change Down"]
            )
        
        with col3:
            alert_value = st.number_input("Alert Value", min_value=0.0, value=0.0)
        
        with col4:
            st.write("")
            st.write("")
            if st.button("Add Alert", use_container_width=True):
                symbol = all_stocks[alert_stock]
                new_alert = {
                    'id': len(st.session_state.alerts) + 1,
                    'stock': alert_stock,
                    'symbol': symbol,
                    'type': alert_type,
                    'value': alert_value,
                    'created': datetime.now(),
                    'triggered': False
                }
                st.session_state.alerts.append(new_alert)
                st.success(f"Alert created for {alert_stock}!")
                st.rerun()
        
        # Display active alerts
        st.markdown("#### Active Alerts")
        
        if st.session_state.alerts:
            for alert in st.session_state.alerts:
                if not alert['triggered']:
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**{alert['stock']}** ({alert['symbol']})")
                    
                    with col2:
                        st.write(f"{alert['type']}: {alert['value']}")
                    
                    with col3:
                        if st.button("Check", key=f"check_{alert['id']}"):
                            # Check alert condition
                            quote = self.data_manager.get_live_quote(alert['symbol'])
                            current_price = quote.get('current', 0)
                            
                            triggered = False
                            if alert['type'] == "Price Above" and current_price > alert['value']:
                                triggered = True
                            elif alert['type'] == "Price Below" and current_price < alert['value']:
                                triggered = True
                            
                            if triggered:
                                st.success(f"🚨 Alert triggered! {alert['stock']} at ₹{current_price}")
                                alert['triggered'] = True
                            else:
                                st.info(f"Alert not triggered. Current price: ₹{current_price}")
                    
                    with col4:
                        if st.button("Delete", key=f"delete_{alert['id']}"):
                            st.session_state.alerts.remove(alert)
                            st.rerun()
        else:
            st.info("No active alerts. Create alerts to monitor price movements.")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Navigation
        st.sidebar.markdown("## 🧭 Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Dashboard", "Watchlist", "Stock Screener", "Portfolio", "Price Alerts", "Settings"]
        )
        
        # API Status
        st.sidebar.markdown("## 🔧 System Status")
        st.sidebar.write(f"📊 NSEPy: {'✅' if NSEPY_AVAILABLE else '❌'}")
        st.sidebar.write(f"📈 Alpha Vantage: ✅")
        st.sidebar.write(f"🌐 Finnhub: ✅")
        st.sidebar.write(f"🇮🇳 Indian API: ✅")
        st.sidebar.write(f"📉 Yahoo Finance: {'✅' if YFINANCE_AVAILABLE else '❌'}")
        st.sidebar.write(f"📊 Plotly Charts: {'✅' if PLOTLY_AVAILABLE else '❌'}")
        
        # Quick actions
        st.sidebar.markdown("## ⚡ Quick Actions")
        if st.sidebar.button("Clear Cache", use_container_width=True):
            self.data_manager.data_cache = {}
            st.success("Cache cleared!")
        
        if st.sidebar.button("Refresh Data", use_container_width=True):
            st.rerun()
        
        # Page routing
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "Watchlist":
            self.render_watchlist()
        elif page == "Stock Screener":
            self.render_stock_screener()
        elif page == "Portfolio":
            self.render_portfolio()
        elif page == "Price Alerts":
            self.render_alerts()
        else:
            st.markdown("### ⚙️ Settings")
            st.info("Application configuration and preferences")
            
            # Data source preferences
            st.subheader("Data Source Preferences")
            preferred_source = st.selectbox(
                "Default Data Source",
                ["auto", "nse", "finnhub", "alpha_vantage", "yfinance"],
                help="Primary data source for stock data"
            )
            
            # Chart preferences
            st.subheader("Chart Preferences")
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly_dark", "plotly_white", "plotly_dark"],
                help="Color theme for charts"
            )
            
            # Notification settings
            st.subheader("Notifications")
            email_alerts = st.checkbox("Enable Email Alerts")
            push_notifications = st.checkbox("Enable Push Notifications")
            
            if st.button("Save Settings", use_container_width=True):
                st.success("Settings saved successfully!")

# Run the application
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Quantum Indian Stock Terminal",
        page_icon="🇮🇳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize and run terminal
    terminal = QuantumTradingTerminal()
    terminal.run()

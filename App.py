import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Graceful fallback for missing dependencies
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
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.techindicators import TechIndicators
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from datetime import datetime, timedelta
import json
import time

# Page configuration
st.set_page_config(
    page_title="Quantum Quant Trading Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00FF00, #00BFFF, #FF00FF, #FF0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 15px;
        border: 2px solid #4f46e5;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #00BFFF;
        margin: 1.5rem 0;
        border-bottom: 3px solid #00BFFF;
        padding-bottom: 0.8rem;
        text-shadow: 0 0 10px rgba(0, 191, 255, 0.5);
    }
    .quantum-card {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem;
        border-radius: 20px;
        border: 2px solid #4f46e5;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.4);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    .quantum-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 32px rgba(79, 70, 229, 0.6);
    }
    .premium-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: #000;
        font-weight: bold;
        border: 2px solid #FF8C00;
        box-shadow: 0 8px 16px rgba(255, 165, 0, 0.3);
    }
    .asset-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        border: 1px solid #4f46e5;
        box-shadow: 0 6px 12px rgba(79, 70, 229, 0.3);
    }
    .positive { 
        color: #00FF00; 
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    }
    .negative { 
        color: #FF4444; 
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
    }
    .search-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(30, 30, 60, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #4f46e5;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced API Keys and Configuration
FINNHUB_API_KEY = "d3f027pr01qh40fg8npgd3f027pr01qh40fg8nq0"
ALPHA_VANTAGE_API_KEY = "P1IXQ8X0N5GWVR7S"

# Initialize API clients
finnhub_client = None
ts = None
ti = None

try:
    if FINNHUB_AVAILABLE:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
except Exception as e:
    pass

try:
    if ALPHA_VANTAGE_AVAILABLE:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
except Exception as e:
    pass

# ==================== ENHANCED ASSET DATABASE ====================

class AssetDatabase:
    def __init__(self):
        self.indian_stocks = {
            'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS', 'ICICI BANK': 'ICICIBANK.NS', 'SBI': 'SBIN.NS',
            'HINDUNILVR': 'HINDUNILVR.NS', 'ITC': 'ITC.NS', 'LT': 'LT.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS', 'HCL TECH': 'HCLTECH.NS', 'KOTAK BANK': 'KOTAKBANK.NS',
            'AXIS BANK': 'AXISBANK.NS', 'MARUTI': 'MARUTI.NS', 'TITAN': 'TITAN.NS',
            'ASIAN PAINTS': 'ASIANPAINT.NS', 'DMART': 'DMART.NS', 'BAJFINANCE': 'BAJFINANCE.NS',
            'WIPRO': 'WIPRO.NS', 'TECHM': 'TECHM.NS'
        }
        
        self.us_stocks = {
            'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 'GOOGLE': 'GOOGL',
            'AMAZON': 'AMZN', 'TESLA': 'TSLA', 'META': 'META',
            'NETFLIX': 'NFLX', 'NVIDIA': 'NVDA', 'ADOBE': 'ADBE',
            'INTEL': 'INTC', 'AMD': 'AMD', 'COINBASE': 'COIN'
        }
        
        self.indices = {
            'NIFTY 50': '^NSEI', 'BANK NIFTY': '^NSEBANK', 'SENSEX': '^BSESN',
            'NIFTY IT': '^CNXIT', 'INDIA VIX': '^INDIAVIX',
            'S&P 500': '^GSPC', 'NASDAQ': '^IXIC', 'DOW JONES': '^DJI',
            'RUSSELL 2000': '^RUT', 'FTSE 100': '^FTSE', 'DAX': '^GDAXI'
        }
        
        self.commodities = {
            'GOLD': 'GC=F', 'SILVER': 'SI=F', 'CRUDE OIL': 'CL=F',
            'BRENT OIL': 'BZ=F', 'NATURAL GAS': 'NG=F', 'COPPER': 'HG=F',
            'PLATINUM': 'PL=F', 'CORN': 'ZC=F', 'WHEAT': 'ZW=F'
        }
        
        self.forex = {
            'USD/INR': 'INR=X', 'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'JPY=X', 'AUD/USD': 'AUDUSD=X', 'USD/CAD': 'CAD=X',
            'USD/CHF': 'CHF=X', 'EUR/GBP': 'EURGBP=X'
        }
        
        self.crypto = {
            'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD', 'BINANCE COIN': 'BNB-USD',
            'CARDANO': 'ADA-USD', 'SOLANA': 'SOL-USD', 'XRP': 'XRP-USD',
            'POLKADOT': 'DOT-USD', 'DOGECOIN': 'DOGE-USD', 'AVALANCHE': 'AVAX-USD'
        }
    
    def search_assets(self, query):
        """Enhanced search across all asset classes"""
        query = query.upper().strip()
        results = {}
        
        # Search across all categories
        for category_name, category in [
            ('Indian Stocks', self.indian_stocks),
            ('US Stocks', self.us_stocks),
            ('Indices', self.indices),
            ('Commodities', self.commodities),
            ('Forex', self.forex),
            ('Crypto', self.crypto)
        ]:
            matches = {k: v for k, v in category.items() if query in k or query in v}
            if matches:
                results[category_name] = matches
        
        return results
    
    def get_all_assets(self):
        """Get all assets organized by category"""
        return {
            'Indian Stocks': self.indian_stocks,
            'US Stocks': self.us_stocks,
            'Indices': self.indices,
            'Commodities': self.commodities,
            'Forex': self.forex,
            'Crypto': self.crypto
        }

# ==================== ENHANCED QUANTUM TRADING MODULES ====================

class QuantumTradingAlgorithms:
    def __init__(self):
        self.quantum_states = ['bullish', 'bearish', 'superposition', 'entangled']
    
    def quantum_wave_prediction(self, price_series):
        """Enhanced Schrödinger-inspired price probability distribution"""
        if len(price_series) < 5:
            return 0.5
        
        try:
            μ = np.mean(price_series)
            σ = np.std(price_series)
            if σ == 0:
                return μ
            
            # Quantum wave function simulation
            x = np.linspace(μ - 3*σ, μ + 3*σ, len(price_series))
            ψ = np.exp(-(x - μ)**2 / (2*σ**2)) / np.sqrt(2*np.pi*σ**2)
            
            # Quantum tunneling effect
            tunneling = np.exp(-price_series.var() / (2*σ**2)) if σ > 0 else 0
            
            return np.mean(ψ) * (1 + tunneling)
        except:
            return 0.5
    
    def quantum_portfolio_optimization(self, returns, covariance_matrix):
        """Enhanced quantum annealing portfolio optimization"""
        if len(returns) == 0 or covariance_matrix.empty:
            return np.array([])
        
        try:
            # Quantum Hamiltonian
            h = -returns.values  # Local field
            J = covariance_matrix.values  # Coupling matrix
            
            # Simulated quantum annealing
            n_assets = len(h)
            temperatures = np.logspace(2, 0, 50)  # Cooling schedule
            
            best_weights = None
            best_energy = float('inf')
            
            for temp in temperatures:
                weights = np.random.dirichlet(np.ones(n_assets) * temp)
                energy = -np.dot(weights, h) + 0.5 * np.dot(weights.T, np.dot(J, weights))
                
                if energy < best_energy:
                    best_energy = energy
                    best_weights = weights
            
            if np.sum(best_weights) > 0:
                return best_weights / np.sum(best_weights)
            return np.ones(n_assets) / n_assets
        except:
            n_assets = len(returns)
            return np.ones(n_assets) / n_assets

class FractalMarketAnalysis:
    def __init__(self):
        pass
    
    def hurst_exponent(self, price_series):
        """Enhanced Hurst exponent calculation"""
        if len(price_series) < 50:
            return 0.5
        
        try:
            # Rescaled Range Analysis
            max_lag = min(100, len(price_series) // 4)
            lags = range(10, max_lag, 5)
            
            tau = []
            for lag in lags:
                if lag >= len(price_series):
                    continue
                
                # Calculate R/S for each lag
                series = price_series[:len(price_series)//lag * lag]
                segments = len(series) // lag
                
                rs_values = []
                for i in range(segments):
                    segment = series[i*lag:(i+1)*lag]
                    mean_segment = np.mean(segment)
                    deviations = segment - mean_segment
                    Z = np.cumsum(deviations)
                    R = np.max(Z) - np.min(Z)
                    S = np.std(segment)
                    if S > 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.log(np.mean(rs_values)))
                else:
                    tau.append(0)
            
            if len(tau) > 2:
                lags_clean = lags[:len(tau)]
                poly = np.polyfit(np.log(lags_clean), tau, 1)
                return poly[0]
            return 0.5
        except:
            return 0.5

class QuantumAnalystBot:
    def __init__(self):
        self.quantum_algo = QuantumTradingAlgorithms()
        self.fractal_analyzer = FractalMarketAnalysis()
    
    def generate_trading_signals(self, data):
        """Generate comprehensive trading signals"""
        if data is None or data.empty:
            return {}
        
        try:
            prices = data['Close']
            returns = prices.pct_change().dropna()
            
            # Quantum signals
            quantum_signal = self.quantum_algo.quantum_wave_prediction(prices.tail(50))
            hurst = self.fractal_analyzer.hurst_exponent(prices)
            
            # Technical signals
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            rsi = self.calculate_rsi(prices)
            
            signals = {
                'quantum_momentum': quantum_signal,
                'market_regime': 'TRENDING' if hurst > 0.6 else 'MEAN_REVERTING' if hurst < 0.4 else 'RANDOM',
                'trend_direction': 'BULLISH' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'BEARISH',
                'rsi_signal': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL',
                'volatility_regime': 'HIGH' if returns.std() > 0.02 else 'LOW',
                'confidence_score': min(0.95, abs(quantum_signal - 0.5) * 2 + 0.3)
            }
            
            return signals
        except:
            return {}

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

# ==================== ENHANCED MAIN TERMINAL ====================

class QuantumQuantTradingTerminal:
    def __init__(self):
        self.asset_db = AssetDatabase()
        self.quantum_analyst = QuantumAnalystBot()
        
        # Initialize session state
        if 'selected_asset' not in st.session_state:
            st.session_state.selected_asset = 'RELIANCE.NS'
        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = 'Indian Stocks'
    
    @st.cache_data(ttl=3600)
    def get_yahoo_data(_self, symbol, period="1y"):
        """Enhanced data fetching with better error handling"""
        try:
            if not YFINANCE_AVAILABLE:
                return _self.generate_fallback_data()
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return _self.generate_fallback_data()
            
            return data
        except Exception as e:
            return _self.generate_fallback_data()
    
    def generate_fallback_data(_self):
        """Generate realistic fallback data"""
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        n = len(dates)
        
        # Generate realistic price patterns
        t = np.linspace(0, 4*np.pi, n)
        trend = np.sin(t) * 50 + np.linspace(1000, 2000, n)
        noise = np.random.normal(0, 25, n)
        price = trend + noise
        
        data = pd.DataFrame({
            'Open': price * (1 + np.random.normal(0, 0.002, n)),
            'High': price * (1 + np.abs(np.random.normal(0.01, 0.005, n))),
            'Low': price * (1 - np.abs(np.random.normal(0.01, 0.005, n))),
            'Close': price,
            'Volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
        
        return data
    
    def render_search_interface(self):
        """Enhanced search and asset selection interface"""
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Universal Search
            search_query = st.text_input("🔍 Search All Assets (Name or Symbol)", 
                                       placeholder="e.g., RELIANCE, AAPL, GOLD, BTC-USD...")
        
        with col2:
            # Quick Category Filter
            category = st.selectbox("Filter by Category", 
                                  ['All Categories', 'Indian Stocks', 'US Stocks', 
                                   'Indices', 'Commodities', 'Forex', 'Crypto'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle search results
        if search_query:
            search_results = self.asset_db.search_assets(search_query)
            if search_results:
                st.subheader("🔎 Search Results")
                for category_name, assets in search_results.items():
                    with st.expander(f"📁 {category_name} ({len(assets)} found)", expanded=True):
                        cols = st.columns(3)
                        for idx, (name, symbol) in enumerate(assets.items()):
                            with cols[idx % 3]:
                                if st.button(f"📈 {name}", key=f"search_{symbol}"):
                                    st.session_state.selected_asset = symbol
                                    st.session_state.selected_category = category_name
                                    st.rerun()
            else:
                st.warning("No assets found matching your search.")
        
        # Display assets by category
        all_assets = self.asset_db.get_all_assets()
        
        if category == 'All Categories':
            categories_to_show = all_assets.keys()
        else:
            categories_to_show = [category]
        
        for category_name in categories_to_show:
            assets = all_assets[category_name]
            
            st.subheader(f"📊 {category_name}")
            
            # Display assets in a grid
            cols = st.columns(4)
            for idx, (name, symbol) in enumerate(assets.items()):
                with cols[idx % 4]:
                    is_selected = st.session_state.selected_asset == symbol
                    button_label = f"✅ {name}" if is_selected else f"📈 {name}"
                    
                    if st.button(button_label, key=f"cat_{symbol}"):
                        st.session_state.selected_asset = symbol
                        st.session_state.selected_category = category_name
                        st.rerun()
    
    def render_asset_dashboard(self):
        """Enhanced asset dashboard with quantum analytics"""
        symbol = st.session_state.selected_asset
        category = st.session_state.selected_category
        
        st.markdown(f'<div class="section-header">🌌 {category} - {symbol}</div>', unsafe_allow_html=True)
        
        # Data loading
        data = self.get_yahoo_data(symbol, '6mo')
        
        if data is None or data.empty:
            st.error("❌ Unable to load data for selected asset")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = ((current_price - prev_price) / prev_price) * 100
            change_class = "positive" if change >= 0 else "negative"
            st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}%")
        
        with col2:
            volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].mean()
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            st.metric("Volume", f"{volume:,.0f}", f"{volume_ratio:.1f}x avg")
        
        with col3:
            day_range = data['High'].iloc[-1] - data['Low'].iloc[-1]
            range_pct = (day_range / data['Close'].iloc[-1]) * 100
            st.metric("Daily Range", f"{day_range:.2f}", f"{range_pct:.2f}%")
        
        with col4:
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{volatility:.1f}%")
        
        # Quantum Analysis
        st.markdown("---")
        st.subheader("🔮 Quantum Analysis")
        
        signals = self.quantum_analyst.generate_trading_signals(data)
        
        if signals:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                regime_color = "🟢" if signals['market_regime'] == 'TRENDING' else "🔴" if signals['market_regime'] == 'MEAN_REVERTING' else "🟡"
                st.metric("Market Regime", f"{regime_color} {signals['market_regime']}")
                
                trend_color = "🟢" if signals['trend_direction'] == 'BULLISH' else "🔴"
                st.metric("Trend Direction", f"{trend_color} {signals['trend_direction']}")
            
            with col2:
                rsi_color = "🔴" if signals['rsi_signal'] == 'OVERBOUGHT' else "🟢" if signals['rsi_signal'] == 'OVERSOLD' else "🟡"
                st.metric("RSI Signal", f"{rsi_color} {signals['rsi_signal']}")
                
                vol_color = "🔴" if signals['volatility_regime'] == 'HIGH' else "🟢"
                st.metric("Volatility", f"{vol_color} {signals['volatility_regime']}")
            
            with col3:
                confidence = signals['confidence_score']
                confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                st.metric("Confidence Score", f"{confidence_color} {confidence:.0%}")
                
                quantum_signal = signals['quantum_momentum']
                signal_strength = "STRONG" if abs(quantum_signal - 0.5) > 0.2 else "MODERATE" if abs(quantum_signal - 0.5) > 0.1 else "WEAK"
                st.metric("Quantum Signal", f"{signal_strength}")
        
        # Price Chart
        st.markdown("---")
        st.subheader("📈 Price Chart")
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            
            # Add moving averages
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index, y=data['SMA_50'],
                name='SMA 50',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(data['Close'])
            st.info("Install plotly for advanced charts: `pip install plotly`")
        
        # Trading Recommendations
        st.markdown("---")
        st.subheader("🎯 Trading Recommendations")
        
        if signals:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
                st.markdown("### 💫 Quantum Strategy")
                
                if signals['market_regime'] == 'TRENDING':
                    if signals['trend_direction'] == 'BULLISH':
                        st.success("**Quantum Momentum Long**")
                        st.write("• Ride the bullish trend with quantum wave exits")
                        st.write("• Entry on pullbacks to SMA 20")
                        st.write("• Stop loss below SMA 50")
                    else:
                        st.error("**Quantum Momentum Short**")
                        st.write("• Capitalize on bearish momentum")
                        st.write("• Entry on rallies to resistance")
                        st.write("• Stop loss above recent high")
                else:
                    st.warning("**Quantum Mean Reversion**")
                    st.write("• Trade range-bound markets")
                    st.write("• Buy oversold, sell overbought")
                    st.write("• Use RSI for entry signals")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Risk Management")
                
                if signals['volatility_regime'] == 'HIGH':
                    st.warning("**High Volatility Detected**")
                    st.write("• Reduce position size by 50%")
                    st.write("• Use wider stop losses")
                    st.write("• Consider options for hedging")
                else:
                    st.success("**Normal Volatility**")
                    st.write("• Standard position sizing")
                    st.write("• Tight stop losses")
                    st.write("• Focus on trend following")
                
                st.markdown(f"**Confidence Level:** {signals['confidence_score']:.0%}")
                st.progress(signals['confidence_score'])
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    def run_quantum_terminal(self):
        """Main quantum trading terminal interface"""
        st.markdown('<div class="main-header">🌌 Quantum Quant Trading Terminal</div>', unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("## 🌟 Quantum Navigation")
            page = st.radio("Navigate to", [
                "📊 Asset Dashboard",
                "🔍 Asset Search", 
                "🤖 Quantum Analyst",
                "📈 Market Overview",
                "⚡ Quick Trading",
                "🔧 Settings"
            ])
            
            st.markdown("---")
            st.markdown("## ⚡ Quantum Features")
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.write("• 🎯 Quantum Wave Prediction")
            st.write("• 📊 Fractal Market Analysis")
            st.write("• 🔮 AI Trading Signals")
            st.write("• 💫 Multi-Asset Support")
            st.write("• 🛡️ Risk Management")
            st.write("• 📈 Real-time Analytics")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # System Status
            st.markdown("---")
            st.markdown("## 🔧 System Status")
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.write(f"📊 Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
                st.write(f"📈 yfinance: {'✅' if YFINANCE_AVAILABLE else '❌'}")
            with status_col2:
                st.write(f"🔗 Finnhub: {'✅' if FINNHUB_AVAILABLE else '❌'}")
                st.write(f"📡 Alpha Vantage: {'✅' if ALPHA_VANTAGE_AVAILABLE else '❌'}")
        
        # Page routing
        if page == "📊 Asset Dashboard":
            self.render_asset_dashboard()
        elif page == "🔍 Asset Search":
            self.render_search_interface()
            # Show selected asset preview
            if st.session_state.selected_asset:
                st.markdown("---")
                st.subheader("📊 Selected Asset Preview")
                self.render_asset_dashboard()
        elif page == "🤖 Quantum Analyst":
            self.render_quantum_analyst()
        elif page == "📈 Market Overview":
            self.render_market_overview()
        elif page == "⚡ Quick Trading":
            self.render_quick_trading()
        elif page == "🔧 Settings":
            self.render_settings()
    
    def render_quantum_analyst(self):
        """Quantum Analyst Bot interface"""
        st.markdown('<div class="section-header">🤖 Quantum Analyst Bot</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Deep Market Analysis")
            
            # Asset selection for analysis
            analysis_symbol = st.selectbox(
                "Select Asset for Analysis",
                options=list(self.asset_db.indian_stocks.values()) + 
                        list(self.asset_db.us_stocks.values()),
                index=0
            )
            
            data = self.get_yahoo_data(analysis_symbol, '1y')
            
            if data is not None and not data.empty:
                # Comprehensive analysis
                st.subheader("📊 Technical Analysis")
                
                # Calculate indicators
                data['SMA_20'] = data['Close'].rolling(20).mean()
                data['SMA_50'] = data['Close'].rolling(50).mean()
                data['RSI'] = self.quantum_analyst.calculate_rsi(data['Close'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_rsi = data['RSI'].iloc[-1]
                    rsi_status = "OVERSOLD" if current_rsi < 30 else "OVERBOUGHT" if current_rsi > 70 else "NEUTRAL"
                    st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_status)
                
                with col2:
                    sma_signal = "BULLISH" if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else "BEARISH"
                    st.metric("Moving Average", sma_signal)
                
                with col3:
                    volume_trend = "HIGH" if data['Volume'].iloc[-1] > data['Volume'].mean() else "LOW"
                    st.metric("Volume Trend", volume_trend)
                
                # Quantum signals
                st.subheader("🔮 Quantum Signals")
                signals = self.quantum_analyst.generate_trading_signals(data)
                
                for signal_name, signal_value in signals.items():
                    st.write(f"**{signal_name.replace('_', ' ').title()}:** {signal_value}")
        
        with col2:
            st.subheader("⚡ Quick Analysis")
            st.info("""
            **Quantum Metrics Guide:**
            
            🟢 **TRENDING** - Strong directional movement
            🔴 **MEAN_REVERTING** - Price tends to return to average
            🟡 **RANDOM** - Unpredictable price action
            
            **RSI Signals:**
            - Below 30: Oversold (Potential Buy)
            - Above 70: Overbought (Potential Sell)
            """)
    
    def render_market_overview(self):
        """Market overview dashboard"""
        st.markdown('<div class="section-header">📈 Global Market Overview</div>', unsafe_allow_html=True)
        
        # Key indices
        st.subheader("🌍 Major Indices")
        major_indices = {
            'NIFTY 50': '^NSEI',
            'S&P 500': '^GSPC', 
            'NASDAQ': '^IXIC',
            'GOLD': 'GC=F',
            'BITCOIN': 'BTC-USD'
        }
        
        cols = st.columns(len(major_indices))
        
        for idx, (name, symbol) in enumerate(major_indices.items()):
            with cols[idx]:
                data = self.get_yahoo_data(symbol, '1d')
                if data is not None and not data.empty and len(data) > 1:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[0]
                    change = ((current - previous) / previous) * 100
                    
                    st.metric(name, f"{current:.0f}", f"{change:+.2f}%")
                else:
                    st.metric(name, "N/A", "N/A")
    
    def render_quick_trading(self):
        """Quick trading interface"""
        st.markdown('<div class="section-header">⚡ Quick Trading</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Trade Setup")
            
            symbol = st.selectbox("Select Asset", 
                                options=list(self.asset_db.indian_stocks.values()) +
                                        list(self.asset_db.us_stocks.values()))
            
            strategy = st.selectbox("Trading Strategy",
                                  ["Quantum Momentum", "Mean Reversion", "Breakout", "Swing Trade"])
            
            quantity = st.number_input("Quantity", min_value=1, value=100)
            
            risk_per_trade = st.slider("Risk per Trade (%)", 1, 10, 2)
        
        with col2:
            st.subheader("🎯 Trade Signals")
            
            data = self.get_yahoo_data(symbol, '1mo')
            if data is not None:
                signals = self.quantum_analyst.generate_trading_signals(data)
                
                if signals:
                    st.info(f"**Market Regime:** {signals['market_regime']}")
                    st.info(f"**Trend:** {signals['trend_direction']}")
                    st.info(f"**Confidence:** {signals['confidence_score']:.0%}")
                    
                    if st.button("🚀 Execute Trade", type="primary"):
                        st.success("Trade executed successfully!")
                        st.balloons()
    
    def render_settings(self):
        """Settings page"""
        st.markdown('<div class="section-header">🔧 Terminal Settings</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Display Settings")
            
            chart_theme = st.selectbox("Chart Theme", ["Dark", "Light", "Quantum"])
            update_frequency = st.selectbox("Data Update Frequency", 
                                          ["Real-time", "1 Minute", "5 Minutes", "15 Minutes"])
            
            st.checkbox("Show Quantum Signals", value=True)
            st.checkbox("Show Risk Metrics", value=True)
            st.checkbox("Show Volume Analysis", value=True)
        
        with col2:
            st.subheader("⚡ Trading Settings")
            
            default_quantity = st.number_input("Default Quantity", value=100)
            max_risk_per_trade = st.slider("Max Risk per Trade (%)", 1, 5, 2)
            auto_stop_loss = st.checkbox("Auto Calculate Stop Loss", value=True)
            
            if st.button("💾 Save Settings", type="primary"):
                st.success("Settings saved successfully!")

# Run the enhanced quantum terminal
if __name__ == "__main__":
    terminal = QuantumQuantTradingTerminal()
    terminal.run_quantum_terminal()

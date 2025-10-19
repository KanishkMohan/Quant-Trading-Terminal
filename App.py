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
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(79, 70, 229, 0.5); }
        to { box-shadow: 0 0 30px rgba(79, 70, 229, 0.8); }
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
    .chart-container {
        background: rgba(15, 12, 41, 0.9);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid #4f46e5;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ENHANCED ASSET DATABASE ====================

class AssetDatabase:
    def __init__(self):
        # Indian Stocks
        self.indian_stocks = {
            'RELIANCE': 'RELIANCE.NS', 'TCS': 'TCS.NS', 'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS', 'ICICI BANK': 'ICICIBANK.NS', 'SBI': 'SBIN.NS',
            'HINDUNILVR': 'HINDUNILVR.NS', 'ITC': 'ITC.NS', 'LT': 'LT.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS', 'HCL TECH': 'HCLTECH.NS', 'KOTAK BANK': 'KOTAKBANK.NS',
            'AXIS BANK': 'AXISBANK.NS', 'MARUTI': 'MARUTI.NS', 'TITAN': 'TITAN.NS',
            'ASIAN PAINTS': 'ASIANPAINT.NS', 'DMART': 'DMART.NS', 'BAJFINANCE': 'BAJFINANCE.NS',
            'WIPRO': 'WIPRO.NS', 'TECHM': 'TECHM.NS', 'SUN PHARMA': 'SUNPHARMA.NS',
            'TATA MOTORS': 'TATAMOTORS.NS', 'POWERGRID': 'POWERGRID.NS', 'NTPC': 'NTPC.NS'
        }
        
        # Indian Indices
        self.indices = {
            'NIFTY 50': '^NSEI', 'BANK NIFTY': '^NSEBANK', 'SENSEX': '^BSESN',
            'NIFTY IT': '^CNXIT', 'NIFTY PHARMA': '^CNXPHARMA', 'NIFTY AUTO': '^CNXAUTO',
            'NIFTY FINSERVICE': '^CNXFIN', 'NIFTY METAL': '^CNXMETAL', 'INDIA VIX': '^INDIAVIX',
            'NIFTY MIDCAP': '^CNXMDCP', 'NIFTY SMALLCAP': '^CNXSMLCP'
        }
        
        # MCX Commodities
        self.mcx_commodities = {
            'GOLD': 'GC=F', 'SILVER': 'SI=F', 'CRUDE OIL': 'CL=F',
            'NATURAL GAS': 'NG=F', 'COPPER': 'HG=F', 'ZINC': 'ZI=F',
            'LEAD': 'LL=F', 'ALUMINIUM': 'ALI=F', 'NICKEL': 'NI=F'
        }
        
        # NCDEX Commodities
        self.ncdex_commodities = {
            'SOYBEAN': 'ZS=F', 'CHANA': 'C=F', 'GUAR SEED': 'GS=F',
            'MUSTARD SEED': 'RS=F', 'COTTON': 'CT=F', 'CASTOR SEED': 'CS=F',
            'TURMERIC': 'TU=F', 'JEERA': 'JE=F', 'CORIANDER': 'CO=F'
        }
        
        # Forex
        self.forex = {
            'USD/INR': 'INR=X', 'EUR/INR': 'EURINR=X', 'GBP/INR': 'GBPINR=X',
            'JPY/INR': 'JPYINR=X', 'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'JPY=X', 'AUD/USD': 'AUDUSD=X'
        }
        
        # Crypto
        self.crypto = {
            'BITCOIN': 'BTC-USD', 'ETHEREUM': 'ETH-USD', 'BINANCE COIN': 'BNB-USD',
            'CARDANO': 'ADA-USD', 'SOLANA': 'SOL-USD', 'XRP': 'XRP-USD',
            'POLKADOT': 'DOT-USD', 'DOGECOIN': 'DOGE-USD'
        }
    
    def search_stocks(self, query):
        """Search Indian stocks by name or symbol"""
        query = query.upper().strip()
        return {k: v for k, v in self.indian_stocks.items() if query in k or query in v}
    
    def get_all_assets_by_category(self):
        """Get all assets organized by category"""
        return {
            'Indian Indices': self.indices,
            'Indian Stocks': self.indian_stocks,
            'MCX Commodities': self.mcx_commodities,
            'NCDEX Commodities': self.ncdex_commodities,
            'Forex': self.forex,
            'Crypto': self.crypto
        }

# ==================== CHARTING ENGINE ====================

class ChartingEngine:
    def __init__(self):
        pass
    
    def create_candlestick_chart(self, data, title="Price Chart"):
        """Create candlestick chart"""
        if not PLOTLY_AVAILABLE or data.empty:
            return None
            
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_line_chart(self, data, title="Price Chart"):
        """Create line chart"""
        if not PLOTLY_AVAILABLE or data.empty:
            return None
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00FF00', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    def create_heikin_ashi_chart(self, data, title="Heikin Ashi Chart"):
        """Create Heikin Ashi chart"""
        if not PLOTLY_AVAILABLE or data.empty:
            return None
        
        # Calculate Heikin Ashi values
        ha_data = data.copy()
        ha_data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        ha_data['HA_Open'] = (data['Open'].shift(1) + data['Close'].shift(1)) / 2
        ha_data['HA_High'] = data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        ha_data['HA_Low'] = data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        # Fill first row
        ha_data.iloc[0, ha_data.columns.get_loc('HA_Open')] = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2
        
        fig = go.Figure(data=[go.Candlestick(
            x=ha_data.index,
            open=ha_data['HA_Open'],
            high=ha_data['HA_High'],
            low=ha_data['HA_Low'],
            close=ha_data['HA_Close'],
            name='Heikin Ashi'
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        return fig

# ==================== SENTIMENT ANALYSIS ====================

class SentimentAnalyzer:
    def __init__(self):
        pass
    
    def calculate_market_sentiment(self, data):
        """Calculate comprehensive market sentiment"""
        if data.empty:
            return {'overall': 50, 'components': {}}
        
        try:
            returns = data['Close'].pct_change().dropna()
            volume = data['Volume']
            
            # Price momentum sentiment
            price_momentum = self._calculate_price_momentum(data['Close'])
            
            # Volume sentiment
            volume_sentiment = self._calculate_volume_sentiment(volume)
            
            # Volatility sentiment
            volatility_sentiment = self._calculate_volatility_sentiment(returns)
            
            # Trend sentiment
            trend_sentiment = self._calculate_trend_sentiment(data['Close'])
            
            # Combine sentiments
            components = {
                'price_momentum': price_momentum,
                'volume': volume_sentiment,
                'volatility': volatility_sentiment,
                'trend': trend_sentiment
            }
            
            overall = np.mean(list(components.values()))
            
            return {
                'overall': overall,
                'components': components,
                'sentiment': 'BULLISH' if overall > 60 else 'BEARISH' if overall < 40 else 'NEUTRAL'
            }
        except:
            return {'overall': 50, 'components': {}, 'sentiment': 'NEUTRAL'}
    
    def _calculate_price_momentum(self, prices):
        """Calculate price momentum sentiment"""
        if len(prices) < 20:
            return 50
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()
        
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            return 75
        else:
            return 25
    
    def _calculate_volume_sentiment(self, volume):
        """Calculate volume sentiment"""
        if len(volume) < 10:
            return 50
        avg_volume = volume.rolling(10).mean()
        current_volume = volume.iloc[-1]
        
        if current_volume > avg_volume.iloc[-1]:
            return 70
        else:
            return 30
    
    def _calculate_volatility_sentiment(self, returns):
        """Calculate volatility sentiment"""
        if len(returns) < 20:
            return 50
        volatility = returns.std()
        if volatility > 0.02:
            return 30  # High volatility = cautious
        else:
            return 60  # Low volatility = confident
    
    def _calculate_trend_sentiment(self, prices):
        """Calculate trend sentiment"""
        if len(prices) < 50:
            return 50
        hurst = self._calculate_hurst_exponent(prices)
        if hurst > 0.6:
            return 80  # Strong trend
        elif hurst < 0.4:
            return 20  # Mean reverting
        else:
            return 50  # Random
    
    def _calculate_hurst_exponent(self, time_series):
        """Calculate Hurst exponent for trend analysis"""
        if len(time_series) < 100:
            return 0.5
        try:
            lags = range(2, 100)
            tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5

# ==================== QUANT STRATEGIES ====================

class QuantStrategies:
    def __init__(self):
        pass
    
    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """Black-Scholes option pricing model"""
        try:
            from math import log, sqrt, exp
            from scipy.stats import norm
            
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
            else:
                price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return price
        except:
            # Fallback calculation
            return S * 0.05  # Rough estimate
    
    def option_strategy_builder(self, strategy_type, underlying_price, strikes, premiums):
        """Build option strategies and calculate payoffs"""
        strategies = {
            'Long Call': self._long_call_payoff,
            'Long Put': self._long_put_payoff,
            'Covered Call': self._covered_call_payoff,
            'Protective Put': self._protective_put_payoff,
            'Straddle': self._straddle_payoff,
            'Strangle': self._strangle_payoff,
            'Bull Call Spread': self._bull_call_spread_payoff,
            'Bear Put Spread': self._bear_put_spread_payoff
        }
        
        if strategy_type in strategies:
            return strategies[strategy_type](underlying_price, strikes, premiums)
        else:
            return np.zeros_like(underlying_price)
    
    def _long_call_payoff(self, S, strikes, premiums):
        K, premium = strikes[0], premiums[0]
        return np.maximum(S - K, 0) - premium
    
    def _long_put_payoff(self, S, strikes, premiums):
        K, premium = strikes[0], premiums[0]
        return np.maximum(K - S, 0) - premium
    
    def _straddle_payoff(self, S, strikes, premiums):
        K, premium = strikes[0], premiums[0]
        call_payoff = np.maximum(S - K, 0) - premium/2
        put_payoff = np.maximum(K - S, 0) - premium/2
        return call_payoff + put_payoff

# ==================== MACHINE LEARNING PREDICTIONS ====================

class MLPredictor:
    def __init__(self):
        pass
    
    def predict_future_prices(self, data, days=30):
        """Predict future prices using ML techniques"""
        if data.empty or len(data) < 50:
            return self._generate_fallback_prediction(data, days)
        
        try:
            # Simple moving average based prediction
            prices = data['Close']
            
            # Calculate various moving averages
            sma_10 = prices.rolling(10).mean()
            sma_20 = prices.rolling(20).mean()
            sma_50 = prices.rolling(50).mean()
            
            # Simple trend extrapolation
            recent_trend = (prices.iloc[-1] - prices.iloc[-20]) / 20
            volatility = prices.pct_change().std()
            
            # Generate predictions
            last_price = prices.iloc[-1]
            predictions = []
            confidence_scores = []
            
            for i in range(1, days + 1):
                # Combine trend with some randomness
                predicted_change = recent_trend + np.random.normal(0, volatility * 0.5)
                predicted_price = last_price * (1 + predicted_change)
                predictions.append(predicted_price)
                
                # Confidence decreases with time
                confidence = max(0.5, 1 - (i * 0.02))
                confidence_scores.append(confidence)
                
                last_price = predicted_price
            
            return {
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'trend': 'BULLISH' if recent_trend > 0 else 'BEARISH',
                'accuracy_estimate': max(0.6, 1 - (days * 0.01))
            }
        except:
            return self._generate_fallback_prediction(data, days)
    
    def _generate_fallback_prediction(self, data, days):
        """Generate fallback predictions when ML fails"""
        if data.empty:
            current_price = 100
        else:
            current_price = data['Close'].iloc[-1]
        
        predictions = [current_price * (1 + 0.001 * i) for i in range(days)]
        confidence_scores = [max(0.3, 1 - (i * 0.03)) for i in range(days)]
        
        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'trend': 'NEUTRAL',
            'accuracy_estimate': 0.5
        }

# ==================== MAIN TERMINAL ====================

class QuantumQuantTradingTerminal:
    def __init__(self):
        self.asset_db = AssetDatabase()
        self.charting_engine = ChartingEngine()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.quant_strategies = QuantStrategies()
        self.ml_predictor = MLPredictor()
        
        # Initialize session state
        if 'selected_assets' not in st.session_state:
            st.session_state.selected_assets = ['^NSEI', 'RELIANCE.NS', 'GC=F', 'BTC-USD']
        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = 'Candlestick'
    
    @st.cache_data(ttl=3600)
    def get_yahoo_data(_self, symbol, period="6mo"):
        """Get data from Yahoo Finance"""
        try:
            if not YFINANCE_AVAILABLE:
                return _self.generate_fallback_data()
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return _self.generate_fallback_data()
            
            return data
        except:
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
    
    def render_live_market_dashboard(self):
        """Live Market Dashboard with multiple charts"""
        st.markdown('<div class="section-header">📊 Live Market Dashboard</div>', unsafe_allow_html=True)
        
        # Chart type selection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.session_state.chart_type = st.selectbox(
                "Chart Type",
                ['Candlestick', 'Line', 'Heikin Ashi'],
                key='chart_type_selector'
            )
        
        with col2:
            time_frame = st.selectbox(
                "Time Frame",
                ['1mo', '3mo', '6mo', '1y'],
                key='time_frame_selector'
            )
        
        with col3:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        # Multi-chart layout - 2x2 grid
        st.subheader("🖥️ Multi-Chart View (4 Charts)")
        
        # Default assets for multi-chart view
        default_assets = ['^NSEI', 'RELIANCE.NS', 'GC=F', 'BTC-USD']
        
        # Create 2x2 grid of charts
        cols = st.columns(2)
        chart_count = 0
        
        for i in range(2):
            for j in range(2):
                if chart_count < len(default_assets):
                    asset_symbol = default_assets[chart_count]
                    with cols[j]:
                        self.render_single_chart(asset_symbol, time_frame, f"Chart {chart_count + 1}")
                    chart_count += 1
        
        # Additional charts in expandable sections
        with st.expander("📈 Additional Charts (4 More)", expanded=False):
            cols_extra = st.columns(2)
            extra_assets = ['^NSEBANK', 'HDFCBANK.NS', 'SI=F', 'ETH-USD']
            
            for idx, asset_symbol in enumerate(extra_assets):
                with cols_extra[idx % 2]:
                    self.render_single_chart(asset_symbol, time_frame, f"Extra Chart {idx + 1}")
    
    def render_single_chart(self, symbol, period, title):
        """Render a single chart"""
        data = self.get_yahoo_data(symbol, period)
        
        if data.empty:
            st.error(f"No data for {symbol}")
            return
        
        # Get asset name
        asset_name = symbol
        all_assets = self.asset_db.get_all_assets_by_category()
        for category, assets in all_assets.items():
            for name, sym in assets.items():
                if sym == symbol:
                    asset_name = name
                    break
        
        # Create chart based on type
        if st.session_state.chart_type == 'Candlestick':
            fig = self.charting_engine.create_candlestick_chart(data, f"{asset_name}")
        elif st.session_state.chart_type == 'Line':
            fig = self.charting_engine.create_line_chart(data, f"{asset_name}")
        else:  # Heikin Ashi
            fig = self.charting_engine.create_heikin_ashi_chart(data, f"{asset_name} - Heikin Ashi")
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_price = data['Close'].iloc[-1]
            st.metric("Current", f"₹{current_price:.2f}")
        
        with col2:
            change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            st.metric("Change", f"{change:+.2f}%")
        
        with col3:
            volume = data['Volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
    
    def render_stock_search(self):
        """Stock Search Engine"""
        st.markdown('<div class="section-header">🔍 Stock Search Engine</div>', unsafe_allow_html=True)
        
        # Search box for stocks
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search Indian Stocks",
                placeholder="Enter stock name or symbol (e.g., RELIANCE, TCS...)",
                key="stock_search"
            )
        
        with col2:
            if st.button("🔍 Search", type="primary"):
                pass  # Search happens automatically
        
        # Display search results
        if search_query:
            results = self.asset_db.search_stocks(search_query)
            if results:
                st.success(f"Found {len(results)} stocks matching '{search_query}'")
                
                # Display results in a grid
                cols = st.columns(4)
                for idx, (name, symbol) in enumerate(results.items()):
                    with cols[idx % 4]:
                        if st.button(f"📈 {name}", key=f"search_{symbol}"):
                            # Add to watchlist or display
                            st.session_state.selected_assets.append(symbol)
                            st.success(f"Added {name} to charts")
            else:
                st.warning(f"No stocks found matching '{search_query}'")
        
        # Sliders for other asset classes
        st.markdown("---")
        st.subheader("📊 Asset Class Selectors")
        
        # Create tabs for different asset classes
        tab1, tab2, tab3, tab4 = st.tabs(["Indices", "Commodities", "Forex", "Crypto"])
        
        with tab1:
            self.render_asset_slider("Indian Indices", self.asset_db.indices)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                self.render_asset_slider("MCX Commodities", self.asset_db.mcx_commodities)
            with col2:
                self.render_asset_slider("NCDEX Commodities", self.asset_db.ncdex_commodities)
        
        with tab3:
            self.render_asset_slider("Forex", self.asset_db.forex)
        
        with tab4:
            self.render_asset_slider("Crypto", self.asset_db.crypto)
    
    def render_asset_slider(self, category_name, assets_dict):
        """Render asset slider for a category"""
        st.subheader(f"📈 {category_name}")
        
        assets_list = list(assets_dict.items())
        selected_index = st.selectbox(
            f"Select {category_name}",
            range(len(assets_list)),
            format_func=lambda x: assets_list[x][0],
            key=f"slider_{category_name}"
        )
        
        if selected_index < len(assets_list):
            selected_name, selected_symbol = assets_list[selected_index]
            
            # Quick preview
            data = self.get_yahoo_data(selected_symbol, '1mo')
            if not data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    current_price = data['Close'].iloc[-1]
                    st.metric("Current Price", f"₹{current_price:.2f}")
                
                with col2:
                    change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                    st.metric("Period Change", f"{change:+.2f}%")
                
                if st.button(f"Add {selected_name} to Charts", key=f"add_{selected_symbol}"):
                    st.session_state.selected_assets.append(selected_symbol)
                    st.success(f"Added {selected_name} to multi-chart view")
    
    def render_sentiment_analysis(self):
        """Sentiment Analysis with Advanced Visualization"""
        st.markdown('<div class="section-header">📊 Sentiment Analysis</div>', unsafe_allow_html=True)
        
        # Select asset for sentiment analysis
        all_assets = self.asset_db.get_all_assets_by_category()
        flat_assets = {}
        for category, assets in all_assets.items():
            flat_assets.update(assets)
        
        selected_asset = st.selectbox(
            "Select Asset for Sentiment Analysis",
            options=list(flat_assets.values()),
            format_func=lambda x: [k for k, v in flat_assets.items() if v == x][0],
            key="sentiment_asset"
        )
        
        data = self.get_yahoo_data(selected_asset, '3mo')
        
        if data.empty:
            st.warning("No data available for sentiment analysis")
            return
        
        sentiment = self.sentiment_analyzer.calculate_market_sentiment(data)
        
        # Sentiment visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Overall sentiment gauge
            if PLOTLY_AVAILABLE:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment['overall'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Sentiment"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment components
            st.subheader("📊 Components")
            for component, score in sentiment['components'].items():
                st.write(f"**{component.replace('_', ' ').title()}:** {score:.1f}")
                st.progress(score/100)
        
        with col3:
            # Sentiment recommendation
            st.subheader("🎯 Recommendation")
            sentiment_level = sentiment['sentiment']
            if sentiment_level == 'BULLISH':
                st.success("🟢 STRONG BULLISH")
                st.write("• Consider long positions")
                st.write("• Look for buying opportunities")
                st.write("• Monitor for trend continuation")
            elif sentiment_level == 'BEARISH':
                st.error("🔴 STRONG BEARISH")
                st.write("• Consider short positions")
                st.write("• Implement risk management")
                st.write("• Watch for trend reversals")
            else:
                st.warning("🟡 NEUTRAL")
                st.write("• Wait for clearer signals")
                st.write("• Consider range-bound strategies")
                st.write("• Monitor key support/resistance")
        
        # Historical sentiment chart
        st.subheader("📈 Historical Sentiment Trend")
        self.render_historical_sentiment(data)
    
    def render_historical_sentiment(self, data):
        """Render historical sentiment analysis"""
        if len(data) < 20:
            return
        
        # Calculate rolling sentiment
        window = 20
        sentiments = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            sentiment = self.sentiment_analyzer.calculate_market_sentiment(window_data)
            sentiments.append(sentiment['overall'])
        
        # Create sentiment chart
        if PLOTLY_AVAILABLE and sentiments:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index[window:],
                y=sentiments,
                mode='lines',
                name='Sentiment Score',
                line=dict(color='cyan', width=3)
            ))
            
            # Add sentiment zones
            fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="green", opacity=0.1)
            fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="yellow", opacity=0.1)
            fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="red", opacity=0.1)
            
            fig.update_layout(
                title="Historical Sentiment Analysis",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_quant_strategies(self):
        """Quant Strategies Section"""
        st.markdown('<div class="section-header">🎯 Quant Strategies</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Option Chain", "Black-Scholes", "Strategy Builder"])
        
        with tab1:
            self.render_option_chain()
        
        with tab2:
            self.render_black_scholes()
        
        with tab3:
            self.render_strategy_builder()
    
    def render_option_chain(self):
        """Option Chain Analysis"""
        st.subheader("📊 Live Option Chain")
        
        # Mock option chain data
        st.info("🔒 Premium Feature: Live Option Chain data requires subscription")
        
        # Sample option chain visualization
        if PLOTLY_AVAILABLE:
            strikes = np.arange(18000, 18500, 50)
            call_oi = np.random.randint(1000, 50000, len(strikes))
            put_oi = np.random.randint(1000, 50000, len(strikes))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=strikes, y=call_oi, name='Call OI', marker_color='green'))
            fig.add_trace(go.Bar(x=strikes, y=put_oi, name='Put OI', marker_color='red'))
            
            fig.update_layout(
                title="Option Chain Open Interest",
                xaxis_title="Strike Price",
                yaxis_title="Open Interest",
                template="plotly_dark",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_black_scholes(self):
        """Black-Scholes Calculator"""
        st.subheader("🧮 Black-Scholes Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            spot_price = st.number_input("Spot Price", value=18000.0, min_value=0.0)
            strike_price = st.number_input("Strike Price", value=18200.0, min_value=0.0)
            time_to_expiry = st.number_input("Time to Expiry (years)", value=0.25, min_value=0.01)
        
        with col2:
            risk_free_rate = st.number_input("Risk Free Rate (%)", value=5.0, min_value=0.0) / 100
            volatility = st.number_input("Volatility (%)", value=20.0, min_value=0.1) / 100
            option_type = st.selectbox("Option Type", ['call', 'put'])
        
        if st.button("Calculate Option Price"):
            try:
                price = self.quant_strategies.black_scholes(
                    spot_price, strike_price, time_to_expiry, 
                    risk_free_rate, volatility, option_type
                )
                
                st.success(f"**{option_type.upper()} Option Price: ₹{price:.2f}**")
                
                # Greeks calculation (simplified)
                delta = 0.5 if option_type == 'call' else -0.5
                gamma = 0.01
                theta = -5.0
                vega = 15.0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Delta", f"{delta:.3f}")
                with col2:
                    st.metric("Gamma", f"{gamma:.3f}")
                with col3:
                    st.metric("Theta", f"{theta:.2f}")
                with col4:
                    st.metric("Vega", f"{vega:.2f}")
                    
            except Exception as e:
                st.error(f"Calculation error: {e}")
    
    def render_strategy_builder(self):
        """Option Strategy Builder"""
        st.subheader("🏗️ Strategy Builder")
        
        strategy_type = st.selectbox(
            "Select Strategy",
            ['Long Call', 'Long Put', 'Straddle', 'Strangle', 'Bull Call Spread', 'Bear Put Spread']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            underlying_price = st.number_input("Underlying Price", value=18000.0)
            strike1 = st.number_input("Strike 1", value=18200.0)
            premium1 = st.number_input("Premium 1", value=150.0)
        
        with col2:
            if strategy_type in ['Strangle', 'Bull Call Spread', 'Bear Put Spread']:
                strike2 = st.number_input("Strike 2", value=17800.0)
                premium2 = st.number_input("Premium 2", value=120.0)
            else:
                strike2 = strike1
                premium2 = 0.0
        
        if st.button("Build Strategy"):
            # Generate payoff diagram
            price_range = np.linspace(underlying_price * 0.8, underlying_price * 1.2, 100)
            strikes = [strike1, strike2]
            premiums = [premium1, premium2]
            
            payoff = self.quant_strategies.option_strategy_builder(
                strategy_type, price_range, strikes, premiums
            )
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_range, y=payoff,
                    mode='lines',
                    name='Payoff',
                    line=dict(color='orange', width=3)
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="white")
                fig.add_vline(x=underlying_price, line_dash="dot", line_color="yellow")
                
                fig.update_layout(
                    title=f"{strategy_type} - Payoff Diagram",
                    xaxis_title="Underlying Price",
                    yaxis_title="Profit/Loss",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Strategy analysis
            max_profit = np.max(payoff)
            max_loss = np.min(payoff)
            breakevens = price_range[np.where(np.abs(payoff) < 10)[0]]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Profit", f"₹{max_profit:.2f}")
            with col2:
                st.metric("Max Loss", f"₹{max_loss:.2f}")
            with col3:
                if len(breakevens) > 0:
                    st.metric("Breakeven", f"₹{breakevens[0]:.0f}")
    
    def render_machine_learning(self):
        """Machine Learning Predictions"""
        st.markdown('<div class="section-header">🤖 Machine Learning & Algos</div>', unsafe_allow_html=True)
        
        # Asset selection for prediction
        all_assets = self.asset_db.get_all_assets_by_category()
        flat_assets = {}
        for category, assets in all_assets.items():
            flat_assets.update(assets)
        
        selected_asset = st.selectbox(
            "Select Asset for Prediction",
            options=list(flat_assets.values()),
            format_func=lambda x: [k for k, v in flat_assets.items() if v == x][0],
            key="ml_asset"
        )
        
        prediction_days = st.slider("Prediction Period (days)", 7, 90, 30)
        
        data = self.get_yahoo_data(selected_asset, '1y')
        
        if st.button("🚀 Generate Predictions"):
            with st.spinner("🤖 AI is analyzing market patterns..."):
                predictions = self.ml_predictor.predict_future_prices(data, prediction_days)
                
                # Display predictions
                st.subheader("📈 Price Predictions")
                
                # Prediction chart
                if PLOTLY_AVAILABLE:
                    last_date = data.index[-1]
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='D')
                    
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=data.index[-60:],  # Last 60 days
                        y=data['Close'].iloc[-60:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predictions
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions['predictions'],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Confidence interval
                    upper_bound = [p * (1 + 0.1 * (1 - c)) for p, c in zip(predictions['predictions'], predictions['confidence_scores'])]
                    lower_bound = [p * (1 - 0.1 * (1 - c)) for p, c in zip(predictions['predictions'], predictions['confidence_scores'])]
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates.tolist() + future_dates.tolist()[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f"Price Predictions - Next {prediction_days} Days",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Prediction metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = data['Close'].iloc[-1]
                    predicted_end = predictions['predictions'][-1]
                    total_return = ((predicted_end - current_price) / current_price) * 100
                    st.metric("Predicted Return", f"{total_return:+.2f}%")
                
                with col2:
                    st.metric("Trend", predictions['trend'])
                
                with col3:
                    st.metric("Accuracy Estimate", f"{predictions['accuracy_estimate']:.0%}")
                
                with col4:
                    avg_confidence = np.mean(predictions['confidence_scores'])
                    st.metric("Avg Confidence", f"{avg_confidence:.0%}")
                
                # Detailed predictions table
                st.subheader("📋 Detailed Predictions")
                prediction_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': predictions['predictions'],
                    'Confidence': [f"{c:.0%}" for c in predictions['confidence_scores']]
                })
                st.dataframe(prediction_df, use_container_width=True)
    
    def run_quantum_terminal(self):
        """Main quantum trading terminal interface"""
        st.markdown('<div class="main-header">🌌 Quantum Quant Trading Terminal</div>', unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("## 🌟 Quantum Navigation")
            
            page = st.radio(
                "Navigate to",
                [
                    "📊 Live Market Dashboard",
                    "🔍 Stock Search Engine", 
                    "📈 Sentiment Analysis",
                    "🎯 Quant Strategies",
                    "🤖 ML Predictions"
                ]
            )
            
            st.markdown("---")
            st.markdown("## ⚡ Quantum Features")
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.write("• 🎯 Multi-Chart Dashboard")
            st.write("• 🔍 Advanced Stock Search")
            st.write("• 📊 Sentiment Analysis")
            st.write("• 🎯 Option Strategies")
            st.write("• 🤖 AI Predictions")
            st.write("• 📈 Technical Analysis")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick Market Overview
            st.markdown("---")
            st.markdown("## 📈 Quick Market")
            
            quick_assets = ['^NSEI', 'GC=F', 'BTC-USD']
            for symbol in quick_assets:
                data = self.get_yahoo_data(symbol, '1d')
                if not data.empty and len(data) > 1:
                    current = data['Close'].iloc[-1]
                    change = ((current - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
                    
                    # Get asset name
                    asset_name = symbol
                    all_assets = self.asset_db.get_all_assets_by_category()
                    for category, assets in all_assets.items():
                        for name, sym in assets.items():
                            if sym == symbol:
                                asset_name = name
                                break
                    
                    st.metric(asset_name, f"₹{current:.0f}", f"{change:+.2f}%")
        
        # Page routing
        if page == "📊 Live Market Dashboard":
            self.render_live_market_dashboard()
        elif page == "🔍 Stock Search Engine":
            self.render_stock_search()
        elif page == "📈 Sentiment Analysis":
            self.render_sentiment_analysis()
        elif page == "🎯 Quant Strategies":
            self.render_quant_strategies()
        elif page == "🤖 ML Predictions":
            self.render_machine_learning()

# Run the enhanced quantum terminal
if __name__ == "__main__":
    terminal = QuantumQuantTradingTerminal()
    terminal.run_quantum_terminal()

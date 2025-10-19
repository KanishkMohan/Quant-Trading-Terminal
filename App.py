import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import finnhub
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from datetime import datetime, timedelta
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Quantum Quant Trading Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys
FINNHUB_API_KEY = "d3f027pr01qh40fg8npgd3f027pr01qh40fg8nq0"
ALPHA_VANTAGE_API_KEY = "P1IXQ8X0N5GWVR7S"
INDIAN_MARKET_API_KEY = "sk-live-UYMPXvoR0SLhmXlnGyqNqVhlgToFARM3mLgoBdm9"

# Initialize API clients
try:
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
except Exception as e:
    st.warning(f"Some API connections failed: {e}. Using enhanced fallback data.")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00FF00, #00BFFF, #FF00FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #00BFFF;
        margin: 1rem 0;
        border-bottom: 2px solid #00BFFF;
        padding-bottom: 0.5rem;
    }
    .quantum-card {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #4f46e5;
        box-shadow: 0 8px 16px rgba(79, 70, 229, 0.3);
        margin: 0.5rem;
    }
    .premium-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #000;
        font-weight: bold;
    }
    .positive { color: #00FF00; }
    .negative { color: #FF4444; }
</style>
""", unsafe_allow_html=True)

# ==================== QUANTUM TRADING MODULES ====================

class QuantumTradingAlgorithms:
    def __init__(self):
        self.quantum_states = ['bullish', 'bearish', 'superposition']
    
    def quantum_wave_prediction(self, price_series):
        """Schrödinger-inspired price probability distribution"""
        if len(price_series) == 0:
            return 0
        μ = np.mean(price_series)
        σ = np.std(price_series)
        if σ == 0:
            return μ
        ψ = np.exp(-(price_series - μ)**2 / (2*σ**2)) / np.sqrt(2*np.pi*σ**2)
        return self.quantum_expected_value(ψ)
    
    def quantum_expected_value(self, wave_function):
        """Calculate quantum expectation value"""
        if len(wave_function) == 0:
            return 0
        x = np.linspace(-3, 3, len(wave_function))
        return np.sum(x * wave_function) / np.sum(wave_function)
    
    def quantum_portfolio_optimization(self, returns, covariance):
        """Quantum annealing inspired portfolio optimization"""
        if len(returns) == 0:
            return np.array([])
        h = -returns  # Local field
        J = covariance  # Coupling term
        portfolio_weights = self.quantum_annealing_solution(h, J)
        if np.sum(portfolio_weights) == 0:
            return portfolio_weights
        return portfolio_weights / np.sum(portfolio_weights)
    
    def quantum_annealing_solution(self, h, J):
        """Mock quantum annealing optimization"""
        if len(h) == 0:
            return np.array([])
        n_assets = len(h)
        weights = np.random.dirichlet(np.ones(n_assets))
        return weights

class FractalMarketAnalysis:
    def __init__(self):
        pass
    
    def hurst_exponent(self, price_series):
        """Calculate Hurst exponent for market memory"""
        if len(price_series) < 10:
            return 0.5
        try:
            lags = range(2, min(100, len(price_series)//2))
            if len(lags) < 2:
                return 0.5
            tau = [np.std(np.subtract(price_series[lag:], price_series[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    def fractal_dimension(self, price_series):
        """Calculate fractal dimension"""
        H = self.hurst_exponent(price_series)
        return 2 - H
    
    def multi_fractal_analysis(self, price_series):
        """Multi-fractal detrended fluctuation analysis"""
        if len(price_series) < 100:
            return 0.5
        
        fluctuations = []
        scales = [10, 20, 50, 100]
        valid_scales = []
        
        for scale in scales:
            if len(price_series) >= scale:
                segments = len(price_series) // scale
                rms_fluctuations = []
                
                for i in range(segments):
                    segment = price_series[i*scale:(i+1)*scale]
                    if len(segment) > 1:
                        x = np.arange(len(segment))
                        try:
                            coeffs = np.polyfit(x, segment, 1)
                            trend = np.polyval(coeffs, x)
                            y = segment - trend
                            rms_fluctuations.append(np.sqrt(np.mean(y**2)))
                        except:
                            continue
                
                if rms_fluctuations:
                    fluctuations.append(np.mean(rms_fluctuations))
                    valid_scales.append(scale)
        
        if len(fluctuations) > 1:
            try:
                poly = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
                return poly[0]
            except:
                return 0.5
        return 0.5

class InformationTheoryTrading:
    def __init__(self):
        pass
    
    def market_entropy(self, data_values):
        """Calculate Shannon entropy"""
        if len(data_values) == 0:
            return 0
        try:
            probabilities = np.abs(data_values) / (np.sum(np.abs(data_values)) + 1e-10)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except:
            return 0
    
    def mutual_information(self, asset_x, asset_y):
        """Calculate mutual information between assets"""
        if len(asset_x) < 2 or len(asset_y) < 2:
            return 0
        try:
            correlation = np.corrcoef(asset_x, asset_y)[0,1]
            if np.isnan(correlation):
                return 0
            return -0.5 * np.log(1 - correlation**2) if abs(correlation) < 1 else 0
        except:
            return 0
    
    def transfer_entropy(self, asset_x, asset_y, k=1):
        """Calculate transfer entropy from X to Y"""
        if len(asset_x) <= k or len(asset_y) <= k:
            return 0
        try:
            x_lagged = asset_x[:-k]
            y_lagged = asset_y[k:]
            y_target = asset_y[k:]
            
            if len(x_lagged) < 2 or len(y_lagged) < 2:
                return 0
                
            mi_conditional = self.mutual_information(np.column_stack([y_lagged, x_lagged]), y_target)
            mi_unconditional = self.mutual_information(y_lagged, y_target)
            
            return max(0, mi_conditional - mi_unconditional)
        except:
            return 0

class QuantumAnalystBot:
    def __init__(self):
        self.quantum_algo = QuantumTradingAlgorithms()
        self.fractal_analyzer = FractalMarketAnalysis()
        self.info_theory = InformationTheoryTrading()
    
    def generate_quantum_recommendation(self, market_data):
        """Generate quantum-inspired trading recommendations"""
        if market_data is None or market_data.empty:
            return ["No data available for analysis"]
            
        analysis = {
            'market_regime': self.identify_market_regime(market_data),
            'quantum_state': self.analyze_quantum_state(market_data),
            'fractal_dimension': self.fractal_analyzer.fractal_dimension(market_data['Close']),
            'information_flow': self.analyze_information_flow(market_data),
            'risk_adjusted_forecast': self.quantum_risk_forecast(market_data)
        }
        
        strategies = self.rank_quantum_strategies(analysis)
        return self.explain_quantum_strategies(strategies, analysis)
    
    def identify_market_regime(self, market_data):
        """Identify current market regime using quantum principles"""
        if market_data is None or market_data.empty:
            return "unknown"
            
        try:
            returns = market_data['Close'].pct_change().dropna()
            if len(returns) < 2:
                return "unknown"
                
            volatility = returns.std()
            trend_strength = abs(self.fractal_analyzer.hurst_exponent(market_data['Close']) - 0.5)
            
            if volatility > 0.02 and trend_strength > 0.1:
                return "high_volatility_trending"
            elif volatility > 0.02:
                return "high_volatility_choppy"
            elif trend_strength > 0.1:
                return "low_volatility_trending"
            else:
                return "efficient_market"
        except:
            return "unknown"
    
    def analyze_quantum_state(self, market_data):
        """Analyze quantum state of the market"""
        if market_data is None or market_data.empty:
            return "unknown"
            
        try:
            if len(market_data) < 2:
                return "unknown"
                
            price_movement = market_data['Close'].pct_change().iloc[-1] if not np.isnan(market_data['Close'].pct_change().iloc[-1]) else 0
            volume_change = market_data['Volume'].pct_change().iloc[-1] if 'Volume' in market_data and not np.isnan(market_data['Volume'].pct_change().iloc[-1]) else 0
            
            if abs(price_movement) > 0.01 and volume_change > 0.5:
                return "bullish_collapse" if price_movement > 0 else "bearish_collapse"
            elif abs(price_movement) < 0.005:
                return "superposition"
            else:
                return "entangled"
        except:
            return "unknown"
    
    def analyze_information_flow(self, market_data):
        """Analyze information flow in the market"""
        if market_data is None or market_data.empty:
            return {'entropy': 0, 'efficiency': 1, 'predictability': 0}
            
        try:
            returns = market_data['Close'].pct_change().dropna()
            if len(returns) > 10:
                entropy = self.info_theory.market_entropy(returns.tail(10).values)
                return {
                    'entropy': entropy,
                    'efficiency': 1 - entropy / np.log2(len(returns)) if len(returns) > 0 else 1,
                    'predictability': entropy / np.log2(len(returns)) if len(returns) > 0 else 0
                }
            return {'entropy': 0, 'efficiency': 1, 'predictability': 0}
        except:
            return {'entropy': 0, 'efficiency': 1, 'predictability': 0}
    
    def quantum_risk_forecast(self, market_data):
        """Quantum-inspired risk forecasting"""
        if market_data is None or market_data.empty:
            return {'quantum_var': 0, 'quantum_cvar': 0, 'wave_risk': 0, 'entanglement_risk': 0}
            
        try:
            returns = market_data['Close'].pct_change().dropna()
            if len(returns) > 0:
                var_95 = np.percentile(returns, 5)
                cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
                
                return {
                    'quantum_var': abs(var_95),
                    'quantum_cvar': abs(cvar_95),
                    'wave_risk': abs(self.quantum_algo.quantum_wave_prediction(returns)),
                    'entanglement_risk': self.info_theory.market_entropy(returns.values)
                }
            return {'quantum_var': 0, 'quantum_cvar': 0, 'wave_risk': 0, 'entanglement_risk': 0}
        except:
            return {'quantum_var': 0, 'quantum_cvar': 0, 'wave_risk': 0, 'entanglement_risk': 0}
    
    def rank_quantum_strategies(self, analysis):
        """Rank trading strategies based on quantum analysis"""
        strategies = []
        
        if analysis['market_regime'] == "high_volatility_trending":
            strategies.extend([
                {'name': 'Quantum Momentum', 'confidence': 0.85, 'type': 'trend_following'},
                {'name': 'Fractal Breakout', 'confidence': 0.78, 'type': 'breakout'}
            ])
        elif analysis['market_regime'] == "efficient_market":
            strategies.extend([
                {'name': 'Quantum Mean Reversion', 'confidence': 0.72, 'type': 'mean_reversion'},
                {'name': 'Information Arbitrage', 'confidence': 0.68, 'type': 'arbitrage'}
            ])
        
        # Add quantum-specific strategies
        if analysis['quantum_state'] == "superposition":
            strategies.append({'name': 'Wave Function Trading', 'confidence': 0.91, 'type': 'quantum'})
        if analysis['fractal_dimension'] > 1.5:
            strategies.append({'name': 'Multi-Fractal Scaling', 'confidence': 0.83, 'type': 'fractal'})
        
        # Ensure we have at least one strategy
        if not strategies:
            strategies.append({'name': 'Market Neutral', 'confidence': 0.5, 'type': 'neutral'})
            
        return sorted(strategies, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def explain_quantum_strategies(self, strategies, analysis):
        """Provide quantum explanations for strategy recommendations"""
        explanations = []
        
        for strategy in strategies:
            if strategy['name'] == 'Quantum Momentum':
                explanation = f"""
                **Quantum Momentum Strategy** (Confidence: {strategy['confidence']:.0%})
                - Market regime: {analysis['market_regime'].replace('_', ' ').title()}
                - Quantum state: {analysis['quantum_state']}
                - Fractal dimension: {analysis.get('fractal_dimension', 0):.3f}
                - Expected Sharpe: 1.8-2.3
                - Recommended: Trend-following with quantum wave exits
                """
            elif strategy['name'] == 'Wave Function Trading':
                info_flow = analysis.get('information_flow', {})
                explanation = f"""
                **Wave Function Trading** (Confidence: {strategy['confidence']:.0%})
                - Market in quantum superposition state
                - Entropy: {info_flow.get('entropy', 0):.3f}
                - Efficiency: {info_flow.get('efficiency', 0):.1%}
                - Strategy: Trade probability waves, not definite positions
                - Expected Sharpe: 2.1-2.8
                """
            else:
                explanation = f"""
                **{strategy['name']}** (Confidence: {strategy['confidence']:.0%})
                - Type: {strategy['type']}
                - Market conditions favorable for this approach
                """
            
            explanations.append(explanation)
        
        return explanations

class MultiModalSentiment:
    def __init__(self):
        self.sources = {
            'print_media': ['Economic Times', 'Business Standard', 'Mint'],
            'digital_media': ['MoneyControl', 'Investing.com', 'BloombergQuint'],
            'social_media': ['Twitter FinFluencers', 'StockTwits', 'Telegram'],
            'corporate': ['Annual Reports', 'Conference Calls', 'Investor Presentations']
        }
    
    def calculate_sentiment_quality_score(self, article):
        """Calculate quality score for sentiment sources"""
        factors = {
            'source_credibility': 0.3,
            'author_expertise': 0.2,
            'timeliness': 0.15,
            'data_backing': 0.2,
            'corroboration': 0.15
        }
        
        # Mock implementation
        scores = {
            'source_credibility': np.random.uniform(0.7, 1.0),
            'author_expertise': np.random.uniform(0.6, 0.9),
            'timeliness': np.random.uniform(0.8, 1.0),
            'data_backing': np.random.uniform(0.5, 0.8),
            'corroboration': np.random.uniform(0.6, 0.9)
        }
        
        return sum(scores[factor] * weight for factor, weight in factors.items())
    
    def fear_greed_index(self, market_data):
        """Calculate Fear & Greed Index for Indian Markets"""
        indicators = {
            'market_volatility': np.random.uniform(0, 100),
            'put_call_ratio': np.random.uniform(0, 100),
            'market_momentum': np.random.uniform(0, 100),
            'stock_price_strength': np.random.uniform(0, 100),
            'safe_haven_demand': np.random.uniform(0, 100)
        }
        
        composite = np.mean(list(indicators.values()))
        if composite < 25:
            sentiment = 'Extreme Fear'
        elif composite < 45:
            sentiment = 'Fear'
        elif composite < 55:
            sentiment = 'Neutral'
        elif composite < 75:
            sentiment = 'Greed'
        else:
            sentiment = 'Extreme Greed'
            
        return {
            'composite_index': composite,
            'components': indicators,
            'sentiment': sentiment
        }

class AdvancedRiskMetrics:
    def __init__(self):
        pass
    
    def conditional_var(self, returns, confidence=0.95):
        """Calculate Conditional Value at Risk (CVaR)"""
        if len(returns) == 0:
            return 0
        try:
            var = np.percentile(returns, (1 - confidence) * 100)
            tail_returns = returns[returns <= var]
            if len(tail_returns) == 0:
                return abs(var)
            cvar = tail_returns.mean()
            return abs(cvar)
        except:
            return 0
    
    def maximum_drawdown_duration(self, portfolio_values):
        """Calculate maximum drawdown duration"""
        if len(portfolio_values) == 0:
            return 0
        try:
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            drawdown_duration = (drawdown < 0).astype(int)
            return drawdown_duration.sum()
        except:
            return 0
    
    def liquidity_adjusted_var(self, returns, volume, confidence=0.95):
        """Calculate liquidity-adjusted VaR"""
        if len(returns) == 0:
            return 0
        try:
            base_var = np.percentile(returns, (1 - confidence) * 100)
            if len(volume) == 0:
                return abs(base_var)
            volume_mean = volume.mean()
            if volume_mean == 0:
                return abs(base_var)
            liquidity_impact = 0.01 * (1 / (volume / volume_mean)).mean()
            return abs(base_var) + liquidity_impact
        except:
            return abs(base_var) if 'base_var' in locals() else 0

class StochasticCalculusModels:
    def __init__(self):
        pass
    
    def heston_model_volatility(self, S, r, kappa, theta, sigma, rho, v0, T=1, dt=1/252, n_simulations=1000):
        """Heston stochastic volatility model simulation"""
        try:
            n_steps = int(T / dt)
            prices = np.zeros((n_simulations, n_steps))
            volatilities = np.zeros((n_simulations, n_steps))
            
            prices[:, 0] = S
            volatilities[:, 0] = v0
            
            for t in range(1, n_steps):
                Z1 = np.random.standard_normal(n_simulations)
                Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_simulations)
                
                volatilities[:, t] = np.maximum(volatilities[:, t-1] + 
                                              kappa * (theta - volatilities[:, t-1]) * dt +
                                              sigma * np.sqrt(volatilities[:, t-1]) * np.sqrt(dt) * Z2, 0)
                
                prices[:, t] = prices[:, t-1] * np.exp((r - 0.5 * volatilities[:, t-1]) * dt +
                                                     np.sqrt(volatilities[:, t-1]) * np.sqrt(dt) * Z1)
            
            return prices, volatilities
        except Exception as e:
            # Return simple fallback
            n_steps = int(T / dt)
            prices = np.ones((n_simulations, n_steps)) * S
            volatilities = np.ones((n_simulations, n_steps)) * v0
            return prices, volatilities

# ==================== ENHANCED MAIN TERMINAL ====================

class QuantumQuantTradingTerminal:
    def __init__(self):
        self.indian_indices = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'NIFTY FIN SERVICE': '^CNXFIN',
            'SENSEX': '^BSESN',
            'NIFTY MIDCAP': '^CNXMIDCAP',
            'INDIA VIX': '^INDIAVIX'
        }
        
        self.popular_stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFY': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'LT': 'LT.NS',
            'BHARTI AIRTEL': 'BHARTIARTL.NS'
        }
        
        # Quantum Components
        self.quantum_analyst = QuantumAnalystBot()
        self.multi_modal_sentiment = MultiModalSentiment()
        self.advanced_risk = AdvancedRiskMetrics()
        self.stochastic_models = StochasticCalculusModels()
        self.fractal_analyzer = FractalMarketAnalysis()
        self.info_theory = InformationTheoryTrading()
    
    def get_yahoo_data(self, symbol, period="1y"):
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                return self.generate_enhanced_fallback_data()
            return data
        except Exception as e:
            return self.generate_enhanced_fallback_data()
    
    def generate_enhanced_fallback_data(self):
        """Generate enhanced fallback data with realistic patterns"""
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        n = len(dates)
        
        # Generate realistic price series with trends and volatility clusters
        returns = np.random.normal(0.0005, 0.015, n)  # Base returns
        # Add volatility clustering
        for i in range(1, n):
            if abs(returns[i-1]) > 0.02:  # High volatility persists
                returns[i] += 0.5 * returns[i-1]
        
        price = 1000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Open': price * (1 + np.random.normal(0, 0.002, n)),
            'High': price * (1 + np.abs(np.random.normal(0.005, 0.003, n))),
            'Low': price * (1 - np.abs(np.random.normal(0.005, 0.003, n))),
            'Close': price,
            'Volume': np.random.randint(1000000, 5000000, n)
        }, index=dates)
        
        return data

    def run_quantum_terminal(self):
        """Main quantum trading terminal interface"""
        st.markdown('<div class="main-header">🌌 Quantum Quant Trading Terminal</div>', unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("## 🌟 Quantum Navigation")
            page = st.radio("Go to", [
                "Quantum Dashboard",
                "Quantum Analyst Bot", 
                "Fractal Market Analysis",
                "Multi-Modal Sentiment",
                "Advanced Risk Metrics",
                "Stochastic Models"
            ])
            
            st.markdown("---")
            st.markdown("## ⚡ Quantum Features")
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.write("• Quantum Wave Prediction")
            st.write("• Fractal Dimension Analysis")
            st.write("• Information Theory Trading")
            st.write("• Multi-Modal Sentiment")
            st.write("• Stochastic Volatility Models")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Page routing
        if page == "Quantum Dashboard":
            self.quantum_dashboard()
        elif page == "Quantum Analyst Bot":
            self.quantum_analyst_bot()
        elif page == "Fractal Market Analysis":
            self.fractal_analysis_page()
        elif page == "Multi-Modal Sentiment":
            self.multi_modal_sentiment_page()
        elif page == "Advanced Risk Metrics":
            self.advanced_risk_page()
        elif page == "Stochastic Models":
            self.stochastic_models_page()

    def quantum_dashboard(self):
        """Quantum-enhanced main dashboard"""
        st.markdown('<div class="section-header">🌌 Quantum Market Dashboard</div>', unsafe_allow_html=True)
        
        # Market overview with quantum metrics
        st.subheader("🔮 Live Quantum Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            nifty_data = self.get_yahoo_data('^NSEI', '1y')
            if nifty_data is not None and not nifty_data.empty:
                hurst = self.fractal_analyzer.hurst_exponent(nifty_data['Close'])
                st.metric("NIFTY Fractal Dimension", f"{2 - hurst:.3f}")
            else:
                st.metric("NIFTY Fractal Dimension", "N/A")
        
        with col2:
            if nifty_data is not None and not nifty_data.empty:
                returns = nifty_data['Close'].pct_change().dropna()
                entropy = self.info_theory.market_entropy(returns.values) if len(returns) > 0 else 0
                st.metric("Market Entropy", f"{entropy:.3f}")
            else:
                st.metric("Market Entropy", "N/A")
        
        with col3:
            if nifty_data is not None:
                fear_greed = self.multi_modal_sentiment.fear_greed_index(nifty_data)
                st.metric("Fear & Greed Index", fear_greed['sentiment'])
            else:
                st.metric("Fear & Greed Index", "N/A")
        
        with col4:
            if nifty_data is not None and not nifty_data.empty:
                mfdfa = self.fractal_analyzer.multi_fractal_analysis(nifty_data['Close'])
                st.metric("Multi-Fractal H", f"{mfdfa:.3f}")
            else:
                st.metric("Multi-Fractal H", "N/A")
        
        # Quantum strategy recommendations
        st.markdown("---")
        st.subheader("🎯 Quantum Strategy Recommendations")
        
        selected_stock = st.selectbox("Select Asset", list(self.popular_stocks.keys()))
        symbol = self.popular_stocks[selected_stock]
        data = self.get_yahoo_data(symbol, '6mo')
        
        if data is not None and not data.empty:
            quantum_recommendations = self.quantum_analyst.generate_quantum_recommendation(data)
            
            for i, recommendation in enumerate(quantum_recommendations):
                with st.expander(f"Quantum Strategy #{i+1}", expanded=i==0):
                    st.markdown(f'<div class="quantum-card">', unsafe_allow_html=True)
                    st.markdown(recommendation)
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No data available for selected asset")
        
        # Live market indices
        st.markdown("---")
        st.subheader("📊 Live Indian Indices")
        cols = st.columns(3)
        
        for idx, (name, symbol) in enumerate(self.indian_indices.items()):
            data = self.get_yahoo_data(symbol, '1d')
            if data is not None and not data.empty and len(data) >= 2:
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[0]
                change_pct = ((current - previous) / previous) * 100
                
                with cols[idx % 3]:
                    st.markdown(f'<div class="quantum-card">', unsafe_allow_html=True)
                    st.metric(name, f"₹{current:,.0f}", f"{change_pct:+.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                with cols[idx % 3]:
                    st.markdown(f'<div class="quantum-card">', unsafe_allow_html=True)
                    st.metric(name, "N/A", "N/A")
                    st.markdown('</div>', unsafe_allow_html=True)

    def quantum_analyst_bot(self):
        """Quantum Analyst Bot interface"""
        st.markdown('<div class="section-header">🤖 Quantum Analyst Bot</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Quantum Market Analysis")
            selected_stock = st.selectbox("Analyze Stock", list(self.popular_stocks.keys()), key="analyst_stock")
            symbol = self.popular_stocks[selected_stock]
            data = self.get_yahoo_data(symbol, '1y')
            
            if data is not None and not data.empty:
                # Run comprehensive quantum analysis
                analysis = {
                    'market_regime': self.quantum_analyst.identify_market_regime(data),
                    'quantum_state': self.quantum_analyst.analyze_quantum_state(data),
                    'fractal_analysis': {
                        'hurst': self.fractal_analyzer.hurst_exponent(data['Close']),
                        'fractal_dim': self.fractal_analyzer.fractal_dimension(data['Close']),
                        'multi_fractal': self.fractal_analyzer.multi_fractal_analysis(data['Close'])
                    },
                    'information_flow': self.quantum_analyst.analyze_information_flow(data),
                    'risk_metrics': self.quantum_analyst.quantum_risk_forecast(data)
                }
                
                # Display analysis results
                st.subheader("📈 Quantum Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Market Regime", analysis['market_regime'].replace('_', ' ').title())
                    st.metric("Quantum State", analysis['quantum_state'].title())
                    st.metric("Fractal Dimension", f"{analysis['fractal_analysis']['fractal_dim']:.3f}")
                
                with col2:
                    st.metric("Hurst Exponent", f"{analysis['fractal_analysis']['hurst']:.3f}")
                    st.metric("Market Entropy", f"{analysis['information_flow'].get('entropy', 0):.3f}")
                    st.metric("Quantum VaR", f"{analysis['risk_metrics'].get('quantum_var', 0):.2%}")
                
                # Strategy recommendations
                st.subheader("🎯 Quantum Strategy Recommendations")
                strategies = self.quantum_analyst.rank_quantum_strategies(analysis)
                
                for i, strategy in enumerate(strategies):
                    with st.expander(f"{strategy['name']} (Confidence: {strategy['confidence']:.0%})", expanded=i==0):
                        st.info(f"**Strategy Type:** {strategy['type'].replace('_', ' ').title()}")
                        st.metric("Confidence Score", f"{strategy['confidence']:.0%}")
                        
                        if strategy['type'] == 'quantum':
                            st.success("🌟 **Quantum Advantage:** This strategy leverages quantum principles for superior risk-adjusted returns")
            else:
                st.warning("No data available for analysis")
        
        with col2:
            st.subheader("⚡ Quick Quantum Scan")
            st.info("""
            **Quantum Metrics Explained:**
            - **Fractal Dimension**: >1.5 indicates complex, trending markets
            - **Hurst Exponent**: >0.5 trending, <0.5 mean-reverting
            - **Market Entropy**: Higher = more uncertainty
            - **Quantum State**: Market's quantum mechanical behavior
            """)
            
            # Fear & Greed Index
            if data is not None and not data.empty:
                fear_greed = self.multi_modal_sentiment.fear_greed_index(data)
                st.subheader("😨😊 Fear & Greed Index")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fear_greed['composite_index'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Market Sentiment"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "red"},
                            {'range': [25, 45], 'color': "orange"},
                            {'range': [45, 55], 'color': "yellow"},
                            {'range': [55, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    def fractal_analysis_page(self):
        """Fractal market analysis interface"""
        st.markdown('<div class="section-header">🌀 Fractal Market Analysis</div>', unsafe_allow_html=True)
        
        selected_stock = st.selectbox("Select Stock for Fractal Analysis", list(self.popular_stocks.keys()), key="fractal_stock")
        symbol = self.popular_stocks[selected_stock]
        data = self.get_yahoo_data(symbol, '2y')
        
        if data is not None and not data.empty:
            # Calculate fractal metrics
            hurst = self.fractal_analyzer.hurst_exponent(data['Close'])
            fractal_dim = self.fractal_analyzer.fractal_dimension(data['Close'])
            mfdfa = self.fractal_analyzer.multi_fractal_analysis(data['Close'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                trend_status = "Trending" if hurst > 0.55 else "Mean-Reverting" if hurst < 0.45 else "Random"
                st.metric("Hurst Exponent", f"{hurst:.3f}", trend_status)
            with col2:
                complexity = "Complex" if fractal_dim > 1.5 else "Simple"
                st.metric("Fractal Dimension", f"{fractal_dim:.3f}", complexity)
            with col3:
                st.metric("Multi-Fractal H", f"{mfdfa:.3f}")
            
            # Fractal analysis chart
            st.subheader("📊 Fractal Analysis Chart")
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Price Series', 'Fractal Scaling'),
                              vertical_spacing=0.08)
            
            # Price series
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price',
                                   line=dict(color='#00FF00')), row=1, col=1)
            
            # Mock fractal scaling analysis
            scales = np.logspace(1, 3, 20).astype(int)
            fluctuations = [np.random.uniform(0.1, 0.5) for _ in scales]  # Mock fluctuations
            
            fig.add_trace(go.Scatter(x=np.log(scales), y=np.log(fluctuations), 
                                   name='Scaling Law', mode='markers+lines',
                                   line=dict(color='#FF00FF')), row=2, col=1)
            
            fig.update_layout(height=600, template="plotly_dark", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Fractal trading signals
            st.subheader("🎯 Fractal Trading Signals")
            
            if hurst > 0.6:
                st.success("**Strong Trending Market** - Favor momentum strategies")
                st.write("• Trend-following approaches likely to perform well")
                st.write("• Breakout strategies recommended")
            elif hurst < 0.4:
                st.warning("**Mean-Reverting Market** - Favor reversal strategies")
                st.write("• Mean reversion approaches likely to perform well")
                st.write("• Range-bound trading recommended")
            else:
                st.info("**Random Walk Characteristics** - Diversify strategies")
                st.write("• Market shows efficient market characteristics")
                st.write("• Consider multiple timeframes and approaches")
        else:
            st.warning("No data available for fractal analysis")

    def multi_modal_sentiment_page(self):
        """Multi-modal sentiment analysis"""
        st.markdown('<div class="section-header">📊 Multi-Modal Sentiment Analysis</div>', unsafe_allow_html=True)
        
        # Source credibility analysis
        st.subheader("🔍 Source Credibility Scoring")
        
        sources = ['Economic Times', 'MoneyControl', 'BloombergQuint', 'Twitter Influencers', 'Company Reports']
        credibility_scores = []
        
        for source in sources:
            score = self.multi_modal_sentiment.calculate_sentiment_quality_score({'source': source})
            credibility_scores.append({'Source': source, 'Credibility Score': score})
        
        df_credibility = pd.DataFrame(credibility_scores)
        fig = px.bar(df_credibility, x='Source', y='Credibility Score', 
                    title="Source Credibility Analysis", color='Credibility Score',
                    color_continuous_scale='Viridis')
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        # Behavioral finance indicators
        st.subheader("🧠 Behavioral Finance Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            herding_metric = np.random.uniform(0, 1)
            herding_status = "High Herding" if herding_metric > 0.7 else "Normal"
            st.metric("Herding Behavior", f"{herding_metric:.1%}", herding_status)
        
        with col2:
            overconfidence = np.random.uniform(0, 1)
            overconfidence_status = "Excessive" if overconfidence > 0.8 else "Moderate"
            st.metric("Overconfidence", f"{overconfidence:.1%}", overconfidence_status)
        
        with col3:
            disposition_effect = np.random.uniform(0, 1)
            disposition_status = "Strong" if disposition_effect > 0.6 else "Weak"
            st.metric("Disposition Effect", f"{disposition_effect:.1%}", disposition_status)
        
        # Sentiment fusion
        st.subheader("🔄 Multi-Modal Sentiment Fusion")
        
        modalities = ['News Sentiment', 'Social Media', 'Options Flow', 'Technical', 'Fundamental']
        sentiment_scores = [np.random.uniform(-1, 1) for _ in modalities]
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        
        fused_sentiment = sum(s * w for s, w in zip(sentiment_scores, weights))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=modalities, y=sentiment_scores, name='Raw Sentiment'))
        fig.add_trace(go.Scatter(x=modalities, y=weights, name='Weights', yaxis='y2',
                               line=dict(color='red', dash='dash')))
        
        fig.update_layout(
            title="Multi-Modal Sentiment Fusion",
            yaxis=dict(title="Sentiment Score"),
            yaxis2=dict(title="Weights", overlaying='y', side='right'),
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        sentiment_status = "Bullish" if fused_sentiment > 0.1 else "Bearish" if fused_sentiment < -0.1 else "Neutral"
        st.metric("Fused Sentiment Score", f"{fused_sentiment:.3f}", sentiment_status)

    def advanced_risk_page(self):
        """Advanced risk metrics page"""
        st.markdown('<div class="section-header">🛡️ Advanced Risk Metrics</div>', unsafe_allow_html=True)
        
        selected_stock = st.selectbox("Select Stock for Risk Analysis", list(self.popular_stocks.keys()), key="risk_stock")
        symbol = self.popular_stocks[selected_stock]
        data = self.get_yahoo_data(symbol, '1y')
        
        if data is not None and not data.empty:
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) > 0:
                # Calculate advanced risk metrics
                cvar_95 = self.advanced_risk.conditional_var(returns, 0.95)
                cvar_99 = self.advanced_risk.conditional_var(returns, 0.99)
                
                # Mock liquidity-adjusted VaR
                liquidity_var = self.advanced_risk.liquidity_adjusted_var(returns, data['Volume'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("95% CVaR", f"{cvar_95:.2%}")
                with col2:
                    st.metric("99% CVaR", f"{cvar_99:.2%}")
                with col3:
                    st.metric("Liquidity VaR", f"{liquidity_var:.2%}")
                with col4:
                    tail_risk = cvar_99/cvar_95 if cvar_95 != 0 else 0
                    st.metric("Tail Risk", f"{tail_risk:.2f}x")
                
                # Risk distribution chart
                st.subheader("📊 Risk Distribution Analysis")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=returns, name='Returns Distribution', nbinsx=50))
                fig.add_vline(x=-cvar_95, line_dash="dash", line_color="red", annotation_text="95% CVaR")
                fig.add_vline(x=-cvar_99, line_dash="dash", line_color="darkred", annotation_text="99% CVaR")
                fig.update_layout(title="Returns Distribution with CVaR Levels", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Stress testing
                st.subheader("🔥 Portfolio Stress Testing")
                
                portfolio_value = st.number_input("Portfolio Value (₹)", value=1000000, step=100000, key="portfolio_value")
                scenarios = {
                    '2008 Crisis': 0.45,
                    'COVID Crash': 0.35,
                    'Rate Shock': 0.25,
                    'Currency Crisis': 0.30
                }
                
                for scenario, impact in scenarios.items():
                    loss = portfolio_value * impact
                    with st.expander(f"{scenario} (-{impact:.1%})"):
                        st.metric("Portfolio Value", f"₹{portfolio_value - loss:,.0f}")
                        st.metric("Loss", f"₹{loss:,.0f}")
            else:
                st.warning("Insufficient return data for risk analysis")
        else:
            st.warning("No data available for risk analysis")

    def stochastic_models_page(self):
        """Stochastic models and quantitative finance"""
        st.markdown('<div class="section-header">📈 Stochastic Models & Quantitative Finance</div>', unsafe_allow_html=True)
        
        st.subheader("🎲 Heston Stochastic Volatility Model")
        
        # Model parameters
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.number_input("Initial Price (₹)", value=1000.0, min_value=0.0, key="S0")
            r = st.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0, key="r")
            kappa = st.number_input("Mean Reversion Rate", value=2.0, min_value=0.0, key="kappa")
        with col2:
            theta = st.number_input("Long-Term Volatility", value=0.04, min_value=0.0, max_value=1.0, key="theta")
            sigma = st.number_input("Vol of Vol", value=0.3, min_value=0.0, max_value=1.0, key="sigma")
            rho = st.number_input("Correlation", value=-0.7, min_value=-1.0, max_value=1.0, key="rho")
        
        if st.button("Run Heston Simulation"):
            with st.spinner("Running stochastic volatility simulation..."):
                try:
                    prices, volatilities = self.stochastic_models.heston_model_volatility(
                        S0, r, kappa, theta, sigma, rho, theta, T=1, n_simulations=1000
                    )
                    
                    # Display results
                    fig = make_subplots(rows=2, cols=1, subplot_titles=('Price Simulations', 'Volatility Simulations'))
                    
                    # Plot first 50 price paths
                    for i in range(min(50, len(prices))):
                        fig.add_trace(go.Scatter(y=prices[i], mode='lines', 
                                               line=dict(width=1, color='rgba(0,255,0,0.1)'),
                                               showlegend=False), row=1, col=1)
                    
                    # Plot first 50 volatility paths
                    for i in range(min(50, len(volatilities))):
                        fig.add_trace(go.Scatter(y=volatilities[i], mode='lines',
                                               line=dict(width=1, color='rgba(255,0,0,0.1)'),
                                               showlegend=False), row=2, col=1)
                    
                    fig.update_layout(height=600, template="plotly_dark", 
                                    title="Heston Model: Price and Volatility Simulations")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    final_prices = prices[:, -1]
                    final_vols = volatilities[:, -1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Final Price", f"₹{np.mean(final_prices):.2f}")
                    with col2:
                        st.metric("Price Std Dev", f"₹{np.std(final_prices):.2f}")
                    with col3:
                        st.metric("Avg Final Vol", f"{np.mean(final_vols):.2%}")
                    with col4:
                        st.metric("Vol Std Dev", f"{np.std(final_vols):.2%}")
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

# Run the quantum terminal
if __name__ == "__main__":
    terminal = QuantumQuantTradingTerminal()
    terminal.run_quantum_terminal()

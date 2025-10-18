# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import ta
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Quant Trading Terminal Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00FF00, #00BFFF);
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
    .metric-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .premium-card {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #000;
    }
    .positive {
        color: #00FF00;
    }
    .negative {
        color: #FF4444;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1E1E1E 0%, #2D2D2D 100%);
    }
</style>
""", unsafe_allow_html=True)

class BlackBoxRiskManager:
    def __init__(self):
        self.risk_limits = {
            'max_portfolio_var': 5.0,  # 5% VAR
            'max_drawdown': 10.0,      # 10% max drawdown
            'max_position_concentration': 25.0,  # 25% per position
            'max_sector_exposure': 40.0  # 40% per sector
        }
    
    def calculate_var(self, portfolio_returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        return np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    def stress_test(self, portfolio, scenarios):
        """Stress test portfolio against various scenarios"""
        results = {}
        for scenario, impact in scenarios.items():
            results[scenario] = portfolio * (1 - impact)
        return results
    
    def check_circuit_breakers(self, market_data):
        """Check for market-wide circuit breakers"""
        # Mock implementation
        return {
            'market_wide': False,
            'stock_specific': [],
            'index_level': False
        }

class BlackBoxAlgoEngine:
    def __init__(self):
        self.encrypted_strategies = {}
        self.performance_tracking = {}
        self.risk_manager = BlackBoxRiskManager()
        self.strategy_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.strategy_key)
    
    def encrypt_strategy(self, strategy_code):
        """Encrypt strategy code"""
        encrypted_code = self.cipher_suite.encrypt(strategy_code.encode())
        return encrypted_code
    
    def deploy_strategy(self, strategy_id, strategy_name, encrypted_code, parameters, capital):
        """Deploy encrypted strategy"""
        strategy_data = {
            'id': strategy_id,
            'name': strategy_name,
            'encrypted_code': encrypted_code,
            'parameters': parameters,
            'capital': capital,
            'deployment_time': datetime.now(),
            'status': 'active',
            'performance': {
                'total_pnl': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        }
        self.encrypted_strategies[strategy_id] = strategy_data
        return strategy_id
    
    def monitor_performance(self, strategy_id):
        """Monitor strategy performance in real-time"""
        if strategy_id in self.encrypted_strategies:
            strategy = self.encrypted_strategies[strategy_id]
            # Mock performance updates
            strategy['performance']['total_pnl'] += np.random.uniform(-1000, 3000)
            strategy['performance']['sharpe_ratio'] = np.random.uniform(1.0, 3.0)
            strategy['performance']['win_rate'] = np.random.uniform(55, 85)
            return strategy['performance']
        return None
    
    def get_active_strategies(self):
        """Get all active strategies"""
        return [s for s in self.encrypted_strategies.values() if s['status'] == 'active']

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.news_sources = {
            'economic_times': {'weight': 0.3, 'color': '#FF6B6B'},
            'money_control': {'weight': 0.25, 'color': '#4ECDC4'},
            'business_standard': {'weight': 0.2, 'color': '#45B7D1'},
            'livemint': {'weight': 0.15, 'color': '#96CEB4'},
            'reuters_india': {'weight': 0.1, 'color': '#FFEAA7'}
        }
        self.verified_sources = ['economic_times', 'money_control', 'reuters_india']
    
    def get_authentic_news(self):
        """Fetch mock authentic news data"""
        news_items = []
        topics = ['Earnings Report', 'Government Policy', 'Market Analysis', 'Company News', 'Economic Data']
        sentiments = ['positive', 'negative', 'neutral']
        
        for source, config in self.news_sources.items():
            for i in range(3):  # 3 news items per source
                news_items.append({
                    'source': source.replace('_', ' ').title(),
                    'headline': f"{np.random.choice(topics)}: Major development in Indian markets",
                    'sentiment': np.random.choice(sentiments),
                    'impact_score': np.random.uniform(0.1, 1.0),
                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                    'verified': source in self.verified_sources,
                    'color': config['color']
                })
        
        return sorted(news_items, key=lambda x: x['impact_score'], reverse=True)
    
    def calculate_market_sentiment(self):
        """Calculate weighted market sentiment"""
        news_items = self.get_authentic_news()
        weighted_sentiment = 0
        total_weight = 0
        
        for news in news_items:
            source_weight = self.news_sources[news['source'].lower().replace(' ', '_')]['weight']
            sentiment_score = 1 if news['sentiment'] == 'positive' else -1 if news['sentiment'] == 'negative' else 0
            weighted_sentiment += sentiment_score * source_weight * news['impact_score']
            total_weight += source_weight * news['impact_score']
        
        overall_sentiment = (weighted_sentiment / total_weight) if total_weight > 0 else 0
        return {
            'score': overall_sentiment,
            'trend': 'Bullish' if overall_sentiment > 0.1 else 'Bearish' if overall_sentiment < -0.1 else 'Neutral',
            'confidence': min(abs(overall_sentiment) * 2, 1.0),
            'news_count': len(news_items),
            'breakdown': {source: np.random.uniform(-1, 1) for source in self.news_sources.keys()}
        }

class AIStrategyGenerator:
    def __init__(self):
        self.strategy_templates = {
            'momentum': self.generate_momentum_strategy,
            'mean_reversion': self.generate_mean_reversion_strategy,
            'breakout': self.generate_breakout_strategy,
            'arbitrage': self.generate_arbitrage_strategy,
            'market_making': self.generate_market_making_strategy
        }
    
    def generate_from_natural_language(self, text_input):
        """Generate strategy from natural language input"""
        # Mock implementation - in production, use NLP models
        strategy_type = self.classify_strategy_type(text_input)
        return self.strategy_templates[strategy_type](text_input)
    
    def classify_strategy_type(self, text):
        """Classify strategy type from text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['momentum', 'trend', 'follow']):
            return 'momentum'
        elif any(word in text_lower for word in ['reversion', 'bounce', 'oversold']):
            return 'mean_reversion'
        elif any(word in text_lower for word in ['breakout', 'break', 'resistance']):
            return 'breakout'
        else:
            return 'momentum'  # default
    
    def generate_momentum_strategy(self, params):
        return {
            'type': 'momentum',
            'name': 'AI Momentum Strategy',
            'parameters': {
                'lookback_period': 20,
                'entry_threshold': 0.02,
                'exit_threshold': -0.01,
                'stop_loss': 0.03,
                'take_profit': 0.06
            },
            'performance': {
                'expected_return': 15.5,
                'max_drawdown': 12.3,
                'win_rate': 62.7
            }
        }

class InstitutionalRiskManager:
    def __init__(self):
        self.scenarios = {
            '2008_crisis': 0.45,  # 45% portfolio decline
            'covid_crash': 0.35,  # 35% decline
            'interest_rate_shock': 0.25,  # 25% decline
            'currency_crisis': 0.30  # 30% decline
        }
    
    def calculate_portfolio_var(self, portfolio_value, returns_series, confidence=0.95):
        """Calculate portfolio Value at Risk"""
        var = np.percentile(returns_series, (1 - confidence) * 100)
        return abs(var * portfolio_value)
    
    def stress_test_portfolio(self, portfolio_value):
        """Stress test portfolio against historical scenarios"""
        results = {}
        for scenario, impact in self.scenarios.items():
            results[scenario] = {
                'portfolio_value': portfolio_value * (1 - impact),
                'loss': portfolio_value * impact,
                'impact_percentage': impact * 100
            }
        return results
    
    def correlation_analysis(self, assets_data):
        """Analyze correlation between assets"""
        returns = pd.DataFrame(assets_data).pct_change().dropna()
        correlation_matrix = returns.corr()
        return correlation_matrix

class SmartOrderRouter:
    def __init__(self):
        self.venues = ['NSE', 'BSE', 'Dark Pool', 'Smart Pool']
        self.venue_weights = {
            'NSE': 0.4,
            'BSE': 0.3,
            'Dark Pool': 0.2,
            'Smart Pool': 0.1
        }
    
    def route_order(self, symbol, quantity, order_type):
        """Smart order routing"""
        # Mock implementation - in production, use real market data
        venue = np.random.choice(list(self.venue_weights.keys()), p=list(self.venue_weights.values()))
        return {
            'venue': venue,
            'executed_price': np.random.uniform(19500, 19600) if 'NIFTY' in symbol else np.random.uniform(42000, 43000),
            'quantity': quantity,
            'timestamp': datetime.now(),
            'slippage': np.random.uniform(0.01, 0.1)
        }
    
    def twap_execution(self, symbol, total_quantity, duration_hours):
        """Time-Weighted Average Price execution"""
        intervals = duration_hours * 12  # 5-minute intervals
        orders = []
        for i in range(intervals):
            orders.append({
                'interval': i + 1,
                'quantity': total_quantity / intervals,
                'time': datetime.now() + timedelta(minutes=5 * i)
            })
        return orders

class SocialTradingEngine:
    def __init__(self):
        self.trader_rankings = self.generate_trader_rankings()
        self.community_sentiment = {}
    
    def generate_trader_rankings(self):
        """Generate mock trader rankings"""
        traders = []
        for i in range(20):
            traders.append({
                'rank': i + 1,
                'name': f'Trader_{1000 + i}',
                'performance': np.random.uniform(15, 85),
                'win_rate': np.random.uniform(55, 90),
                'followers': np.random.randint(100, 10000),
                'strategy_type': np.random.choice(['Momentum', 'Swing', 'Scalping', 'Options']),
                'verified': np.random.choice([True, False], p=[0.7, 0.3])
            })
        return sorted(traders, key=lambda x: x['performance'], reverse=True)
    
    def get_sentiment_heatmap(self, symbols):
        """Generate community sentiment heatmap"""
        heatmap = {}
        for symbol in symbols:
            heatmap[symbol] = {
                'sentiment': np.random.choice(['Bullish', 'Bearish', 'Neutral']),
                'strength': np.random.uniform(0.5, 1.0),
                'traders_count': np.random.randint(50, 500)
            }
        return heatmap

class RegulatoryComplianceSuite:
    def __init__(self):
        self.pdt_threshold = 4  # Pattern Day Trader threshold
    
    def generate_pnl_report(self, trades, start_date, end_date):
        """Generate P&L report for tax purposes"""
        # Mock implementation
        return {
            'total_profit': sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0),
            'total_loss': abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0)),
            'net_pnl': sum(trade.get('profit', 0) for trade in trades),
            'taxable_amount': max(0, sum(trade.get('profit', 0) for trade in trades))
        }
    
    def check_pdt_rule(self, user_trades):
        """Check Pattern Day Trader rule compliance"""
        day_trades = len([t for t in user_trades if t.get('day_trade', False)])
        return {
            'is_pdt': day_trades >= self.pdt_threshold,
            'day_trades_count': day_trades,
            'remaining_trades': max(0, self.pdt_threshold - day_trades)
        }
    
    def generate_audit_trail(self, user_activities):
        """Generate regulatory audit trail"""
        return {
            'user_id': 'USER_123',
            'period': 'Q3 2024',
            'total_activities': len(user_activities),
            'trades_executed': len([a for a in user_activities if a['type'] == 'trade']),
            'compliance_score': np.random.uniform(85, 100)
        }

# Enhanced existing classes with new features
class EnhancedQuantTradingTerminal:
    def __init__(self):
        self.indian_indices = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'NIFTY FINANCIAL SERVICES': '^CNXFIN',
            'SENSEX': '^BSESN',
            'NIFTY MIDCAP 100': '^CNXMIDCAP',
            'INDIA VIX': '^INDIAVIX'
        }
        
        # Enhanced components
        self.black_box_engine = BlackBoxAlgoEngine()
        self.advanced_sentiment = AdvancedSentimentAnalyzer()
        self.ai_strategy_generator = AIStrategyGenerator()
        self.institutional_risk = InstitutionalRiskManager()
        self.smart_router = SmartOrderRouter()
        self.social_trading = SocialTradingEngine()
        self.regulatory_suite = RegulatoryComplianceSuite()

    def create_black_box_trading_dashboard(self):
        st.markdown('<div class="section-header">🔒 Black Box Algo Trading</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Strategy Vault", "Live Monitoring", "Risk Management", "Performance Analytics"])
        
        with tab1:
            self.create_strategy_vault()
        
        with tab2:
            self.create_live_monitoring()
        
        with tab3:
            self.create_institutional_risk_dashboard()
        
        with tab4:
            self.create_performance_analytics()

    def create_strategy_vault(self):
        st.subheader("🔐 Encrypted Strategy Vault")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Deploy New Strategy")
            strategy_name = st.text_input("Strategy Name", "Alpha Momentum Pro")
            strategy_type = st.selectbox("Strategy Type", ["Momentum", "Mean Reversion", "Arbitrage", "Market Making"])
            capital = st.number_input("Allocated Capital (₹)", value=100000, min_value=10000, step=10000)
            
            # Strategy parameters
            st.subheader("Strategy Parameters")
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                lookback = st.slider("Lookback Period", 5, 50, 20)
                stop_loss = st.number_input("Stop Loss (%)", value=2.0, min_value=0.5, max_value=10.0, step=0.5)
            with param_col2:
                take_profit = st.number_input("Take Profit (%)", value=4.0, min_value=1.0, max_value=20.0, step=0.5)
                max_positions = st.slider("Max Positions", 1, 10, 5)
            
            if st.button("🚀 Deploy Encrypted Strategy", type="primary"):
                strategy_id = f"STRAT_{np.random.randint(10000, 99999)}"
                # Mock encryption
                strategy_code = f"""
                # Encrypted Strategy: {strategy_name}
                # Type: {strategy_type}
                # Parameters: lookback={lookback}, sl={stop_loss}%, tp={take_profit}%
                """
                encrypted_code = self.black_box_engine.encrypt_strategy(strategy_code)
                
                parameters = {
                    'lookback_period': lookback,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'max_positions': max_positions
                }
                
                self.black_box_engine.deploy_strategy(
                    strategy_id, strategy_name, encrypted_code, parameters, capital
                )
                st.success(f"✅ Strategy '{strategy_name}' deployed successfully!")
                st.info(f"Strategy ID: {strategy_id} | Capital: ₹{capital:,}")
        
        with col2:
            st.markdown("### Active Strategies")
            active_strategies = self.black_box_engine.get_active_strategies()
            
            for strategy in active_strategies[:3]:  # Show top 3
                with st.container():
                    st.markdown(f'<div class="premium-card">', unsafe_allow_html=True)
                    st.write(f"**{strategy['name']}**")
                    st.write(f"ID: {strategy['id']}")
                    st.write(f"Capital: ₹{strategy['capital']:,}")
                    st.write(f"Deployed: {strategy['deployment_time'].strftime('%Y-%m-%d %H:%M')}")
                    
                    # Performance metrics
                    perf = strategy['performance']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("P&L", f"₹{perf['total_pnl']:,.0f}")
                    with col2:
                        st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

    def create_live_monitoring(self):
        st.subheader("📊 Real-time Strategy Monitoring")
        
        # Live performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Strategies", len(self.black_box_engine.get_active_strategies()))
        with col2:
            st.metric("Total Capital", "₹2,450,000")
        with col3:
            st.metric("Daily P&L", "₹45,230", "+2.1%")
        with col4:
            st.metric("Success Rate", "78.5%")
        
        # Live strategy performance chart
        st.subheader("Live Performance Dashboard")
        strategies = self.black_box_engine.get_active_strategies()
        
        if strategies:
            # Create performance chart
            fig = go.Figure()
            
            for strategy in strategies:
                # Mock live performance data
                timestamps = pd.date_range(start='2024-09-18 09:15', end='2024-09-18 15:30', freq='5min')
                performance = np.cumsum(np.random.normal(0, 1000, len(timestamps)))
                
                fig.add_trace(go.Scatter(
                    x=timestamps, y=performance,
                    name=strategy['name'],
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Live Strategy Performance",
                xaxis_title="Time",
                yaxis_title="P&L (₹)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection alerts
        st.subheader("🚨 Anomaly Detection")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.error("High Drawdown Alert: STRAT_456")
        with col2:
            st.warning("Volume Spike: BANKNIFTY")
        with col3:
            st.info("Correlation Break: NIFTY-USDINR")

    def create_advanced_sentiment_dashboard(self):
        st.markdown('<div class="section-header">📰 Advanced Sentiment Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Real-time sentiment from authentic sources
            st.subheader("📊 Live Market Sentiment")
            sentiment_data = self.advanced_sentiment.calculate_market_sentiment()
            
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sentiment_data['score'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Market Sentiment Score"},
                delta = {'reference': 0},
                gauge = {
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-100, -20], 'color': "red"},
                        {'range': [-20, 20], 'color': "yellow"},
                        {'range': [20, 100], 'color': "green"}],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_data['score'] * 100}}
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # News feed
            st.subheader("📰 Live News Feed")
            news_items = self.advanced_sentiment.get_authentic_news()
            
            for news in news_items[:5]:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{news['headline']}**")
                        st.caption(f"Source: {news['source']} • {news['timestamp'].strftime('%H:%M')}")
                    with col2:
                        sentiment_color = "🟢" if news['sentiment'] == 'positive' else "🔴" if news['sentiment'] == 'negative' else "🟡"
                        st.write(f"{sentiment_color} {news['sentiment'].title()}")
                        if news['verified']:
                            st.write("✅ Verified")
        
        with col2:
            # Source credibility
            st.subheader("🔍 Source Analysis")
            for source, config in self.advanced_sentiment.news_sources.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{source.replace('_', ' ').title()}**")
                with col2:
                    st.write(f"{config['weight']*100:.0f}%")
            
            # Sentiment breakdown
            st.subheader("📈 Sentiment Breakdown")
            breakdown = sentiment_data['breakdown']
            for source, score in breakdown.items():
                st.write(f"{source.replace('_', ' ').title()}: {score:+.2f}")

    def create_institutional_risk_dashboard(self):
        st.markdown('<div class="section-header">🛡️ Institutional Risk Management</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Portfolio VAR", "Stress Testing", "Correlation Analysis", "Circuit Breakers"])
        
        with tab1:
            st.subheader("📊 Value at Risk (VAR) Analysis")
            portfolio_value = st.number_input("Portfolio Value (₹)", value=1000000, step=100000)
            
            # Mock returns data
            returns = np.random.normal(0.001, 0.02, 1000)  # 1000 days of returns
            
            var_95 = self.institutional_risk.calculate_portfolio_var(portfolio_value, returns, 0.95)
            var_99 = self.institutional_risk.calculate_portfolio_var(portfolio_value, returns, 0.99)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("95% VAR (1 Day)", f"₹{var_95:,.0f}")
            with col2:
                st.metric("99% VAR (1 Day)", f"₹{var_99:,.0f}")
            
            # VAR distribution chart
            fig = px.histogram(x=returns * portfolio_value, nbins=50, 
                             title="Portfolio Returns Distribution")
            fig.add_vline(x=-var_95, line_dash="dash", line_color="red", 
                         annotation_text="95% VAR")
            fig.add_vline(x=-var_99, line_dash="dash", line_color="darkred",
                         annotation_text="99% VAR")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🔥 Portfolio Stress Testing")
            portfolio_value = st.number_input("Current Portfolio Value", value=1000000, step=100000, key="stress_test")
            
            stress_results = self.institutional_risk.stress_test_portfolio(portfolio_value)
            
            for scenario, result in stress_results.items():
                with st.expander(f"{scenario.replace('_', ' ').title()} (-{result['impact_percentage']:.1f}%)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Portfolio Value", f"₹{result['portfolio_value']:,.0f}")
                    with col2:
                        st.metric("Loss", f"₹{result['loss']:,.0f}")
                    with col3:
                        st.metric("Impact", f"-{result['impact_percentage']:.1f}%")
        
        with tab3:
            st.subheader("🔗 Correlation Analysis")
            symbols = ['NIFTY 50', 'BANK NIFTY', 'USDINR', 'GOLD', 'RELIANCE', 'TCS']
            
            # Mock correlation matrix
            corr_matrix = self.institutional_risk.correlation_analysis(
                {symbol: np.random.normal(0, 1, 100) for symbol in symbols}
            )
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Asset Correlation Matrix", color_continuous_scale="RdBu_r")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("⚡ Circuit Breaker Status")
            circuit_status = self.black_box_engine.risk_manager.check_circuit_breakers(None)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if circuit_status['market_wide']:
                    st.error("Market-wide Circuit: ACTIVE")
                else:
                    st.success("Market-wide Circuit: INACTIVE")
            
            with col2:
                st.info("Index Level: NORMAL")
            
            with col3:
                st.warning("Stock-specific: 2 ACTIVE")

    def create_social_trading_dashboard(self):
        st.markdown('<div class="section-header">👥 Social Trading & Copy Trading</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Trader Rankings", "Copy Trading", "Community Sentiment"])
        
        with tab1:
            st.subheader("🏆 Top Performing Traders")
            traders = self.social_trading.trader_rankings[:10]
            
            for i, trader in enumerate(traders):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.write(f"**#{trader['rank']} {trader['name']}**")
                        if trader['verified']:
                            st.write("✅ Verified Pro")
                    with col2:
                        st.metric("Performance", f"{trader['performance']:.1f}%")
                    with col3:
                        st.metric("Win Rate", f"{trader['win_rate']:.1f}%")
                    with col4:
                        if st.button("Copy", key=f"copy_{trader['name']}"):
                            st.success(f"Now copying {trader['name']}!")
        
        with tab2:
            st.subheader("🔄 Active Copy Trading")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Your Copied Traders**")
                # Mock copied traders
                copied_traders = [
                    {'name': 'Trader_1001', 'performance': 45.6, 'copied_since': '2024-08-01'},
                    {'name': 'Trader_1005', 'performance': 32.1, 'copied_since': '2024-08-15'}
                ]
                
                for trader in copied_traders:
                    with st.container():
                        st.write(f"**{trader['name']}** - Since {trader['copied_since']}")
                        st.metric("Performance", f"{trader['performance']:.1f}%")
            
            with col2:
                st.subheader("Copy Settings")
                capital = st.number_input("Copy Capital (₹)", value=50000, step=10000)
                risk_level = st.select_slider("Risk Level", ["Low", "Medium", "High"])
                auto_rebalance = st.checkbox("Auto Rebalance", value=True)
                
                if st.button("Apply Settings"):
                    st.success("Copy trading settings updated!")
        
        with tab3:
            st.subheader("🔥 Community Sentiment Heatmap")
            symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HINDUNILVR']
            heatmap = self.social_trading.get_sentiment_heatmap(symbols)
            
            for symbol, data in heatmap.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{symbol}**")
                with col2:
                    sentiment_color = "🟢" if data['sentiment'] == 'Bullish' else "🔴" if data['sentiment'] == 'Bearish' else "🟡"
                    st.write(f"{sentiment_color} {data['sentiment']} ({data['strength']:.0%})")
                with col3:
                    st.write(f"👥 {data['traders_count']}")

    def create_regulatory_compliance_dashboard(self):
        st.markdown('<div class="section-header">⚖️ Regulatory Compliance Suite</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["P&L Reporting", "PDT Monitoring", "Audit Trail"])
        
        with tab1:
            st.subheader("📊 Automated P&L Reporting")
            
            # Mock trade data
            trades = [
                {'symbol': 'RELIANCE', 'quantity': 100, 'profit': 2500, 'date': '2024-09-01'},
                {'symbol': 'TCS', 'quantity': 50, 'profit': -1200, 'date': '2024-09-02'},
                {'symbol': 'INFY', 'quantity': 75, 'profit': 1800, 'date': '2024-09-03'},
                {'symbol': 'HDFC', 'quantity': 60, 'profit': 3200, 'date': '2024-09-04'}
            ]
            
            report = self.regulatory_suite.generate_pnl_report(
                trades, '2024-09-01', '2024-09-18'
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Profit", f"₹{report['total_profit']:,.0f}")
            with col2:
                st.metric("Total Loss", f"₹{report['total_loss']:,.0f}")
            with col3:
                st.metric("Net P&L", f"₹{report['net_pnl']:,.0f}")
            with col4:
                st.metric("Taxable Amount", f"₹{report['taxable_amount']:,.0f}")
            
            if st.button("📄 Generate Tax Report"):
                st.success("Tax report generated successfully!")
                st.download_button("Download PDF", data="Mock PDF Content", file_name="tax_report_2024.pdf")
        
        with tab2:
            st.subheader("⚠️ Pattern Day Trader Monitoring")
            
            # Mock user trades
            user_trades = [
                {'symbol': 'RELIANCE', 'day_trade': True, 'date': '2024-09-18'},
                {'symbol': 'TCS', 'day_trade': True, 'date': '2024-09-18'},
                {'symbol': 'INFY', 'day_trade': False, 'date': '2024-09-18'},
                {'symbol': 'HDFC', 'day_trade': True, 'date': '2024-09-18'}
            ]
            
            pdt_status = self.regulatory_suite.check_pdt_rule(user_trades)
            
            col1, col2 = st.columns(2)
            with col1:
                if pdt_status['is_pdt']:
                    st.error("🚨 PATTERN DAY TRADER")
                else:
                    st.success("✅ PDT Rule Compliant")
                
                st.metric("Day Trades Today", pdt_status['day_trades_count'])
                st.metric("Remaining Trades", pdt_status['remaining_trades'])
            
            with col2:
                st.info("""
                **PDT Rule Information:**
                - 4+ day trades in 5 rolling business days
                - Requires ₹25L+ capital if flagged as PDT
                - Reset period: 5 business days
                """)
        
        with tab3:
            st.subheader("📝 Regulatory Audit Trail")
            
            # Mock user activities
            activities = [
                {'type': 'login', 'timestamp': '2024-09-18 09:15:00'},
                {'type': 'trade', 'symbol': 'RELIANCE', 'timestamp': '2024-09-18 09:30:00'},
                {'type': 'trade', 'symbol': 'TCS', 'timestamp': '2024-09-18 10:15:00'},
                {'type': 'logout', 'timestamp': '2024-09-18 15:30:00'}
            ]
            
            audit_trail = self.regulatory_suite.generate_audit_trail(activities)
            
            st.write(f"**User ID:** {audit_trail['user_id']}")
            st.write(f"**Period:** {audit_trail['period']}")
            st.write(f"**Total Activities:** {audit_trail['total_activities']}")
            st.write(f"**Trades Executed:** {audit_trail['trades_executed']}")
            st.metric("Compliance Score", f"{audit_trail['compliance_score']:.1f}%")
            
            if st.button("Generate Audit Report"):
                st.success("Audit report generated for regulatory submission!")

    def create_ai_strategy_generator(self):
        st.markdown('<div class="section-header">🤖 AI Strategy Generator</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("💬 Natural Language Strategy Creator")
            strategy_prompt = st.text_area(
                "Describe your trading strategy:",
                "Create a momentum strategy for BANKNIFTY with 2% stop loss and 4% take profit that trades on 15-minute timeframe"
            )
            
            if st.button("🚀 Generate AI Strategy", type="primary"):
                with st.spinner("AI is generating your optimized strategy..."):
                    # Simulate AI processing
                    import time
                    time.sleep(2)
                    
                    strategy = self.ai_strategy_generator.generate_from_natural_language(strategy_prompt)
                    
                    st.success("✅ AI Strategy Generated Successfully!")
                    
                    # Display strategy
                    st.subheader("Generated Strategy")
                    st.write(f"**Name:** {strategy['name']}")
                    st.write(f"**Type:** {strategy['type']}")
                    
                    st.subheader("Optimized Parameters")
                    for param, value in strategy['parameters'].items():
                        st.write(f"- {param.replace('_', ' ').title()}: {value}")
                    
                    st.subheader("Expected Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{strategy['performance']['expected_return']:.1f}%")
                    with col2:
                        st.metric("Max Drawdown", f"{strategy['performance']['max_drawdown']:.1f}%")
                    with col3:
                        st.metric("Win Rate", f"{strategy['performance']['win_rate']:.1f}%")
        
        with col2:
            st.subheader("🎯 Strategy Templates")
            template_options = {
                "Momentum": "Trend-following strategies",
                "Mean Reversion": "Oversold/overbought bounce plays",
                "Breakout": "Support/resistance breakouts",
                "Arbitrage": "Multi-leg pricing inefficiencies",
                "Market Making": "Bid-ask spread capture"
            }
            
            selected_template = st.selectbox("Choose Template", list(template_options.keys()))
            st.info(template_options[selected_template])
            
            st.subheader("⚙️ Optimization Settings")
            optimization_type = st.radio("Optimization Method", 
                                       ["Genetic Algorithm", "Grid Search", "Bayesian Optimization"])
            backtest_period = st.slider("Backtest Period (Months)", 1, 24, 12)
            include_stress_test = st.checkbox("Include Stress Testing", value=True)
            
            if st.button("Optimize Parameters"):
                st.info("Running parameter optimization...")

    # Include all previous methods from the enhanced terminal
    def create_market_overview(self):
        st.markdown('<div class="main-header">🚀 AI-Powered Quant Trading Terminal Pro</div>', unsafe_allow_html=True)
        st.markdown("""
        **Institutional-Grade Trading Platform with Black Box Algorithms, Advanced Risk Management, 
        and Regulatory Compliance for Indian Markets**
        """)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Live Strategies", "24", "3 new")
        with col2:
            st.metric("Total Capital", "₹8.2Cr", "1.2%")
        with col3:
            st.metric("Success Rate", "82.3%", "2.1%")
        with col4:
            st.metric("Active Users", "1,247", "15 new")
        
        # Market indices in a grid
        st.markdown("### 📈 Live Market Indices")
        cols = st.columns(4)
        for idx, (name, symbol) in enumerate(self.indian_indices.items()):
            price, change, change_percent = self.get_live_data(symbol)
            if price is not None:
                col = cols[idx % 4]
                with col:
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label=name,
                        value=f"₹{price:,.2f}" if price > 1000 else f"₹{price:.2f}",
                        delta=f"{change:+.2f} ({change_percent:+.2f}%)"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    def get_live_data(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='5m')
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                previous_close = data['Close'].iloc[0]
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
                return current_price, change, change_percent
        except:
            return None, None, None
        return None, None, None

def main():
    terminal = EnhancedQuantTradingTerminal()
    
    # Sidebar navigation with enhanced options
    with st.sidebar:
        st.markdown("# 🚀 Quant Terminal Pro")
        st.markdown("---")
        
        # Premium features badge
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.write("**⚡ PREMIUM FEATURES**")
        st.write("Black Box Algorithms")
        st.write("Institutional Risk Mgmt")
        st.write("AI Strategy Generator")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        selected = st.selectbox(
            "Navigation",
            [
                "Market Overview", 
                "Black Box Trading", 
                "AI Strategy Generator",
                "Advanced Sentiment", 
                "Institutional Risk", 
                "Social Trading",
                "Regulatory Compliance"
            ],
            index=0
        )
    
    # Main content based on selection
    if selected == "Market Overview":
        terminal.create_market_overview()
        
        # Feature highlights
        st.markdown("---")
        st.markdown("## 🎯 Platform Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔒 Black Box Trading")
            st.write("Military-grade encrypted algorithms")
            st.write("Strategy-as-a-Service deployment")
            st.write("Real-time performance monitoring")
        with col2:
            st.markdown("### 🤖 AI-Powered Analytics")
            st.write("Natural language strategy creation")
            st.write("Ensemble machine learning models")
            st.write("Sentiment analysis from verified sources")
        with col3:
            st.markdown("### 🛡️ Institutional Tools")
            st.write("Portfolio VAR calculation")
            st.write("Regulatory compliance suite")
            st.write("Social trading integration")
    
    elif selected == "Black Box Trading":
        terminal.create_black_box_trading_dashboard()
    
    elif selected == "AI Strategy Generator":
        terminal.create_ai_strategy_generator()
    
    elif selected == "Advanced Sentiment":
        terminal.create_advanced_sentiment_dashboard()
    
    elif selected == "Institutional Risk":
        terminal.create_institutional_risk_dashboard()
    
    elif selected == "Social Trading":
        terminal.create_social_trading_dashboard()
    
    elif selected == "Regulatory Compliance":
        terminal.create_regulatory_compliance_dashboard()

if __name__ == "__main__":
    main()
"""
PolyMarket Analyzer - Advanced Trading Analytics Dashboard
Author: Jake Yunkee
Version: 1.0.0

A sophisticated Flask-based web application that provides real-time analytics
and insights for PolyMarket trading data. Built by a Full-Stack Developer
specializing in Python and .NET, with experience from Epsilon Net and 
Satori Analytics. Features include:
- Real-time trade monitoring and analysis
- Advanced trader profiling with performance metrics
- Market correlation and volatility analysis
- Trust score calculation based on multiple factors
- Interactive data visualization

The application demonstrates expertise in:
- Python backend development with Flask
- Asynchronous programming for efficient API calls
- Advanced statistical calculations for trading metrics
- Modern frontend development with responsive design
"""

from flask import Flask, jsonify, render_template, request
import requests
from datetime import datetime
from collections import defaultdict
import os
from dotenv import load_dotenv
import logging
import aiohttp
import asyncio
import nest_asyncio

# Enable nested asyncio support for compatibility with Jupyter-like environments
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Configure logging with appropriate format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

async def fetch_profile_data(session, url, headers):
    """
    Asynchronously fetch data from a specific PolyMarket API endpoint.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session for making requests
        url (str): The API endpoint URL
        headers (dict): Request headers
    
    Returns:
        dict: JSON response from the API or None if request fails
    """
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                logger.error(f"Error response from {url}: {response.status}")
                return None
            return await response.json()
    except Exception as e:
        logger.error(f"Error fetching from {url}: {str(e)}")
        return None

async def fetch_all(address):
    """
    Fetch and aggregate all profile data for a specific wallet address.
    
    This function coordinates multiple async API calls to gather comprehensive
    trader data including:
    - Portfolio value
    - Profit/loss metrics
    - Trading volume
    - Position data
    - Trading activity history
    
    Args:
        address (str): The wallet address to fetch data for
    
    Returns:
        dict: Aggregated trader profile data including advanced metrics
    """
    try:
        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "en-US,en;q=0.9",
            "sec-ch-ua": "\"Google Chrome\";v=\"129\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site"
        }

        urls = [
            f"https://data-api.polymarket.com/value?user={address}",
            f"https://lb-api.polymarket.com/profit?window=all&limit=1&address={address}",
            f"https://lb-api.polymarket.com/volume?window=all&limit=1&address={address}",
            f"https://data-api.polymarket.com/traded?user={address}",
            f"https://data-api.polymarket.com/positions?user={address}&sizeThreshold=.1",
            f"https://data-api.polymarket.com/activity?user={address}&limit=1000&offset=0"
        ]

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_profile_data(session, url, headers) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results with error handling
        value = 0
        profit = 0
        volume = 0
        traded = 0
        positions = []
        activity = []

        if isinstance(results[0], list) and results[0]:
            value = results[0][0].get('value', 0)
        if isinstance(results[1], list) and results[1]:
            profit = results[1][0].get('amount', 0)
        if isinstance(results[2], list) and results[2]:
            volume = results[2][0].get('amount', 0)
        if results[3] and not isinstance(results[3], Exception):
            traded = results[3].get('traded', 0)
        if results[4] and not isinstance(results[4], Exception):
            positions = results[4]
        if results[5] and not isinstance(results[5], Exception):
            activity = results[5]

        # Calculate scores and metrics
        trust_score = calculate_trader_trust_score(
            activity=activity,
            positions=positions,
            value=value,
            profit=profit,
            volume=volume,
            traded=traded
        )

        volatility = calculate_return_volatility(activity)
        market_correlation = calculate_market_correlation(positions)
        trade_consistency = calculate_trade_consistency(defaultdict(int, {
            datetime.fromtimestamp(trade['timestamp']).date(): 1 
            for trade in activity
        }))
        market_timing = calculate_market_timing_score(activity)

        return {
            'value': value,
            'profit': profit,
            'volume': volume,
            'traded': traded,
            'positions': positions,
            'activity': activity,
            'trust_score': trust_score,
            'score_components': {
                'win_rate': calculate_win_rate(activity),
                'roi': calculate_profit_percentage(profit, value),
                'risk_management': calculate_risk_management_score(positions),
                'activity_level': calculate_activity_level(traded),
                'market_reading': market_timing * 100
            },
            'advanced_metrics': {
                'volatility': volatility * 100,
                'market_correlation': market_correlation * 100,
                'trade_consistency': trade_consistency * 100,
                'market_timing': market_timing * 100
            }
        }
    except Exception as e:
        logger.error(f"Error in fetch_all: {str(e)}")
        raise

def calculate_win_rate(activities):
    """
    Calculate the trader's win rate based on trading history.
    
    Args:
        activities (list): List of trading activities
    
    Returns:
        float: Win rate percentage (0-100)
    """
    if not activities:
        return 0
    winning_trades = sum(1 for trade in activities if trade['side'] == 'SELL' and trade['usdcSize'] > trade['size'])
    return (winning_trades / len(activities)) * 100

def calculate_profit_percentage(profit, value):
    """
    Calculate the return on investment as a percentage.
    
    Args:
        profit (float): Total profit in USDC
        value (float): Total portfolio value
    
    Returns:
        float: ROI percentage
    """
    if value == 0:
        return 0
    return (profit / value) * 100

def calculate_activity_level(traded):
    """
    Calculate trader activity level based on total trades.
    Normalized to a 0-100 scale where 1000 trades = 100.
    
    Args:
        traded (int): Total number of trades
    
    Returns:
        float: Activity level score (0-100)
    """
    # Assuming 1000 trades is considered very active
    return min((traded / 1000) * 100, 100)

def calculate_risk_management_score(positions):
    """
    Calculate risk management score based on position sizing and diversification.
    
    Evaluates:
    - Position size distribution
    - Market diversification
    - Risk concentration
    
    Args:
        positions (list): List of trading positions
    
    Returns:
        float: Risk management score (0-100)
    """
    """Calculate risk management score based on position sizing and diversification."""
    if not positions:
        return 0
    
    position_sizes = [pos['size'] for pos in positions]
    avg_size = sum(position_sizes) / len(position_sizes)
    max_size = max(position_sizes)
    size_ratio = avg_size / max_size if max_size > 0 else 0
    
    unique_markets = len(set(pos['title'] for pos in positions))
    diversification = min((unique_markets / 10) * 100, 100)
    
    return ((size_ratio * 50) + (diversification * 50)) / 100

def calculate_market_reading_score(activity):
    """Calculate market reading score based on profitable exits."""
    if not activity:
        return 0
    
    profitable_exits = sum(1 for trade in activity 
                         if trade['side'] == 'SELL' and trade['usdcSize'] > trade['size'])
    total_exits = sum(1 for trade in activity if trade['side'] == 'SELL')
    
    return (profitable_exits / total_exits * 100) if total_exits > 0 else 0

def calculate_trader_trust_score(activity, positions, value, profit, volume, traded):
    """
    Calculate a comprehensive trust score (0-100) for a trader based on multiple factors:
    - Win Rate (30%): Consistency in profitable trades
    - ROI (25%): Risk-adjusted returns
    - Risk Management (20%): Position sizing and diversification
    - Activity Score (15%): Trading frequency and consistency
    - Market Reading (10%): Ability to exit positions profitably
    """
    if not activity or not positions:
        return 0

    # Win Rate (30%) - Enhanced to consider profit magnitude
    wins = 0
    total_profit = 0
    total_trades = len(activity)
    for trade in activity:
        if trade['side'] == 'SELL':
            profit = trade['usdcSize'] - trade['size']
            if profit > 0:
                wins += 1
                total_profit += profit
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    # Weight win rate by profit magnitude
    avg_profit_per_win = total_profit / wins if wins > 0 else 0
    win_score = min((win_rate * (1 + avg_profit_per_win / 100)), 100) * 0.3

    # ROI Score (25%) - Risk-adjusted returns
    roi = (profit / value) * 100 if value > 0 else 0
    volatility = calculate_return_volatility(activity)
    risk_adjusted_roi = roi / volatility if volatility > 0 else roi
    roi_score = min(risk_adjusted_roi * 2, 100) * 0.25

    # Risk Management Score (20%)
    position_sizes = [pos['size'] for pos in positions]
    avg_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
    max_size = max(position_sizes) if position_sizes else 0
    size_ratio = avg_size / max_size if max_size > 0 else 0
    
    unique_markets = len(set(pos['title'] for pos in positions))
    market_correlation = calculate_market_correlation(positions)
    diversification = min((unique_markets / 10) * (1 - market_correlation), 100)
    
    risk_score = ((size_ratio * 50) + (diversification * 50)) / 100 * 0.2

    # Activity Score (15%) - Enhanced for consistency
    daily_trades = defaultdict(int)
    for trade in activity:
        date = datetime.fromtimestamp(trade['timestamp']).date()
        daily_trades[date] += 1
    
    avg_daily_trades = sum(daily_trades.values()) / len(daily_trades) if daily_trades else 0
    trade_consistency = calculate_trade_consistency(daily_trades)
    activity_score = min((avg_daily_trades / 5) * trade_consistency * 100, 100) * 0.15

    # Market Reading Score (10%) - Enhanced with timing analysis
    market_timing_score = calculate_market_timing_score(activity)
    market_score = market_timing_score * 0.10

    # Calculate final score
    final_score = win_score + roi_score + risk_score + activity_score + market_score
    
    return round(final_score, 2)

def calculate_return_volatility(trades):
    """Calculate the volatility of trading returns."""
    returns = []
    for trade in trades:
        if trade['side'] == 'SELL':
            ret = (trade['usdcSize'] - trade['size']) / trade['size']
            returns.append(ret)
    
    if not returns:
        return 1  # Default to 1 if no returns
    
    mean_return = sum(returns) / len(returns)
    squared_diff_sum = sum((r - mean_return) ** 2 for r in returns)
    volatility = (squared_diff_sum / len(returns)) ** 0.5
    
    return max(volatility, 0.01)  # Minimum volatility of 1%

def calculate_market_correlation(positions):
    """Calculate correlation between different market positions."""
    if len(positions) < 2:
        return 0
    
    # Group positions by market
    market_positions = defaultdict(list)
    for pos in positions:
        market_positions[pos['title']].append(pos)
    
    # Calculate correlation between market returns
    correlations = []
    markets = list(market_positions.keys())
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            market1_returns = [
                (pos['currentValue'] - pos['initialValue']) / pos['initialValue']
                for pos in market_positions[markets[i]]
            ]
            market2_returns = [
                (pos['currentValue'] - pos['initialValue']) / pos['initialValue']
                for pos in market_positions[markets[j]]
            ]
            
            if market1_returns and market2_returns:
                correlation = calculate_correlation(market1_returns, market2_returns)
                correlations.append(abs(correlation))
    
    return sum(correlations) / len(correlations) if correlations else 0

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient between two lists."""
    if len(x) != len(y):
        return 0
    
    n = len(x)
    if n == 0:
        return 0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = (sum((val - mean_x) ** 2 for val in x)) ** 0.5
    std_y = (sum((val - mean_y) ** 2 for val in y)) ** 0.5
    
    if std_x == 0 or std_y == 0:
        return 0
    
    return covariance / (std_x * std_y)

def calculate_trade_consistency(daily_trades):
    """Calculate consistency score based on trading frequency."""
    if not daily_trades:
        return 0
    
    trades_per_day = list(daily_trades.values())
    mean_trades = sum(trades_per_day) / len(trades_per_day)
    variance = sum((t - mean_trades) ** 2 for t in trades_per_day) / len(trades_per_day)
    
    # Lower variance means more consistency
    consistency = 1 / (1 + variance)
    return min(consistency, 1)

def calculate_market_timing_score(activity):
    """Calculate market timing score based on entry and exit points."""
    if not activity:
        return 0
    
    timing_scores = []
    market_prices = defaultdict(list)
    
    # Group trades by market and collect prices
    for trade in activity:
        market_prices[trade['title']].append(trade['price'])
    
    for trade in activity:
        if trade['side'] == 'SELL':
            market_range = max(market_prices[trade['title']]) - min(market_prices[trade['title']])
            if market_range > 0:
                # Calculate how close to optimal the exit was
                price_position = (trade['price'] - min(market_prices[trade['title']])) / market_range
                timing_scores.append(price_position)
    
    return sum(timing_scores) / len(timing_scores) if timing_scores else 0

def calculate_analytics(trades):
    """Calculate analytics from trade data."""
    if not trades:
        return {
            'total_volume': 0,
            'average_price': 0,
            'unique_traders': 0,
            'total_trades': 0,
            'buy_sell_ratio': {'buy': 0, 'sell': 0},
            'top_markets': [],
            'top_traders': []
        }

    total_volume = sum(trade['size'] * trade['price'] for trade in trades)
    average_price = sum(trade['price'] for trade in trades) / len(trades)
    unique_traders = len(set(trade['proxy_wallet'] for trade in trades))
    total_trades = len(trades)
    buy_count = sum(1 for trade in trades if trade['side'] == 'BUY')
    sell_count = total_trades - buy_count

    market_counts = {}
    trader_volumes = {}
    for trade in trades:
        market_counts[trade['title']] = market_counts.get(trade['title'], 0) + 1
        volume = trade['size'] * trade['price']
        trader_volumes[trade['pseudonym']] = trader_volumes.get(trade['pseudonym'], 0) + volume

    top_markets = sorted(market_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_traders = sorted(trader_volumes.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'total_volume': round(total_volume, 2),
        'average_price': round(average_price, 2),
        'unique_traders': unique_traders,
        'total_trades': total_trades,
        'buy_sell_ratio': {'buy': buy_count, 'sell': sell_count},
        'top_markets': top_markets,
        'top_traders': top_traders
    }

def fetch_polymarket_data():
    """Fetch trade data from Polymarket API."""
    url = "https://data-api.polymarket.com/trades"
    params = {
        "takerOnly": "true",
        "limit": "50",
        "offset": "0",
        "filterType": "CASH",
        "filterAmount": "1"
    }
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        trades = response.json()

        processed_trades = []
        for trade in trades:
            try:
                transaction = {
                    'proxy_wallet': trade.get('proxyWallet', 'N/A'),
                    'side': trade.get('side', 'N/A'),
                    'asset': trade.get('asset', 'N/A'),
                    'condition_id': trade.get('conditionId', 'N/A'),
                    'size': round(float(trade.get('size', 0)), 2),
                    'price': round(float(trade.get('price', 0)), 2),
                    'amount': round(float(trade.get('amount', 0)), 2),
                    'timestamp': datetime.fromtimestamp(trade.get('timestamp', 0)).strftime('%I:%M:%S %p'),
                    'title': trade.get('title', 'N/A'),
                    'slug': trade.get('slug', 'N/A'),
                    'event_slug': trade.get('eventSlug', 'N/A'),
                    'outcome': trade.get('outcome', 'N/A'),
                    'outcome_index': trade.get('outcomeIndex', 'N/A'),
                    'name': trade.get('name', 'N/A'),
                    'pseudonym': trade.get('pseudonym', 'N/A'),
                    'transaction_hash': trade.get('transactionHash', 'N/A'),
                }
                processed_trades.append(transaction)
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing trade: {str(e)}")
                continue

        analytics = calculate_analytics(processed_trades)
        return {'trades': processed_trades, 'analytics': analytics}

    except requests.RequestException as e:
        logger.error(f"Error fetching data from Polymarket: {str(e)}")
        return {'trades': [], 'analytics': None}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {'trades': [], 'analytics': None}

@app.route('/')
def home():
    """
    Render the main dashboard page with real-time market analytics.
    Implements a modern, responsive UI with interactive charts.
    """
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/profile/<address>')
def profile(address):
    """
    Render the trader profile page with comprehensive performance metrics.
    
    Args:
        address (str): Trader's wallet address
    """
    try:
        return render_template('profile.html', address=address)
    except Exception as e:
        logger.error(f"Error rendering profile template: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/profile/<address>')
def get_profile_data(address):
    """API endpoint to get profile data."""
    try:
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(fetch_all(address))
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in /api/profile endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'status': 500
        }), 500

@app.route('/trades')
def get_trades():
    """API endpoint to get trade data."""
    try:
        data = fetch_polymarket_data()
        analytics = calculate_analytics(data['trades'])
        return jsonify({'trades': data['trades'], 'analytics': analytics})
    except Exception as e:
        logger.error(f"Error in /trades endpoint: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Configure server settings from environment variables
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

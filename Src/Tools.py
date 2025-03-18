import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

# ------------------------------ Compute feature ----------------------------- #
def compute_rsi(series, period=14):
    """
    Calcule manuellement le RSI (Relative Strength Index).
    RSI = 100 - 100 / (1 + RS)
    RS = moyenne_pondérée_exponentielle(gains) / moyenne_pondérée_exponentielle(pertes)
    """
    delta = series.diff(1).dropna()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    avg_gain = gains.ewm(com=period-1, min_periods=period).mean()
    avg_loss = losses.ewm(com=period-1, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    # Ré-alignement
    rsi = rsi.reindex(series.index, method='bfill').fillna(method='bfill')
    return rsi

def compute_ema(series, period=12):
    """Calcule une EMA (Exponential Moving Average)."""
    return series.ewm(span=period, adjust=False).mean()

def compute_volatility(series, window=10):
    """ Calcule la volatilité (écart-type des prix de clôture sur une fenêtre). """
    return series.pct_change().rolling(window=window).std()

# ----------------------------------- Plots ---------------------------------- #
def plot_candlestick(data):
    """
    Affiche un graphique en chandelier avec un style moderne et épuré.
    """
    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='gray')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)

    fig, (ax, ax_volume) = plt.subplots(2, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    mpf.plot(data, type='candle', style=s, ax=ax, volume=ax_volume)
    
    ax.set_title("Graphique en Chandeliers Japonais", fontsize=14, fontweight='bold')
    
    plt.show()


def plot_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Affiche le MACD avec sa ligne de signal et son histogramme.
    """
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
 
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    histogram = macd - signal
 
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index, macd, label="MACD", color='blue', linewidth=2)
    ax.plot(data.index, signal, label="Signal", color='red', linestyle='dashed', linewidth=2)
    ax.bar(data.index, histogram, color=np.where(histogram > 0, 'green', 'red'), alpha=0.5)
    
    ax.axhline(0, color='black', linewidth=1, linestyle='--')  # Ligne horizontale zéro
    ax.set_title("MACD (Moving Average Convergence Divergence)", fontsize=14, fontweight='bold')
    ax.legend()
    plt.show()

def plot_rsi(data, window=14):
    """
    Affiche le RSI (Relative Strength Index) avec une période donnée.
    """
    # Calcul du RSI
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0) 
    loss = np.where(delta < 0, -delta, 0) 

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data.index, rsi, label=f'RSI {window}', color='purple', linewidth=2)
    ax.axhline(70, linestyle='dashed', color='red', label='Surachat (70)')
    ax.axhline(30, linestyle='dashed', color='green', label='Survente (30)')
    ax.axhline(50, linestyle='dashed', color='gray', alpha=0.5) 

    ax.set_title("RSI (Relative Strength Index)", fontsize=14, fontweight='bold')
    ax.legend()
    plt.show()

# ---------------------------------- Metrics --------------------------------- #

# Fonction pour calculer la métrique E(R) (annualised expected trade return)
def annualised_expected_return(profits, num_trades):
    return np.mean(profits) * 252 / num_trades  # 252 jours de trading par an

# Fonction pour calculer la métrique std(R) (annualised standard deviation of trade return)
def annualised_std_return(profits, num_trades):
    return np.std(profits) * np.sqrt(252 / num_trades)  # Annualisation de la volatilité

# Fonction pour calculer la Downside Deviation (DD)
def downside_deviation(profits, num_trades):
    negative_returns = [r for r in profits if r < 0]
    if len(negative_returns) == 0:
        return 0.0
    return np.std(negative_returns) * np.sqrt(252 / num_trades)

# Fonction pour calculer la Sharpe Ratio
def sharpe_ratio(expected_return, std_return):
    return expected_return / std_return if std_return != 0 else 0.0

# Fonction pour calculer la Sortino Ratio
def sortino_ratio(expected_return, downside_deviation):
    return expected_return / downside_deviation if downside_deviation != 0 else 0.0

# Fonction pour calculer le Maximum Drawdown (MDD)
def max_drawdown(profits):
    cumulative_returns = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    return drawdowns.min()  # Le MDD est la perte maximale

# Fonction pour calculer la Calmar Ratio
def calmar_ratio(expected_return, mdd):
    return expected_return / abs(mdd) if mdd != 0 else 0.0

# Fonction pour calculer le pourcentage de retours positifs
def percent_positive_returns(profits):
    positive_returns = [r for r in profits if r > 0]
    return len(positive_returns) / len(profits) * 100

# Fonction pour calculer le ratio entre les retours positifs et négatifs
def ave_p_ave_l(profits):
    positive_returns = [r for r in profits if r > 0]
    negative_returns = [r for r in profits if r < 0]
    
    if len(negative_returns) == 0:
        return np.inf  # Éviter la division par zéro, retourne l'infini si pas de perte
    
    avg_positive = np.mean(positive_returns) if len(positive_returns) > 0 else 0
    avg_negative = np.mean(negative_returns) if len(negative_returns) > 0 else 0
    
    return avg_positive / abs(avg_negative)

# Fonction principale pour calculer toutes les métriques
def calculate_metrics(profits):
    num_trades = len(profits)
    
    expected_return = annualised_expected_return(profits, num_trades)
    std_return = annualised_std_return(profits, num_trades)
    dd = downside_deviation(profits, num_trades)
    sharpe = sharpe_ratio(expected_return, std_return)
    sortino = sortino_ratio(expected_return, dd)
    mdd = max_drawdown(profits)
    calmar = calmar_ratio(expected_return, mdd)
    positive_pct = percent_positive_returns(profits)
    ave_p_ave_l_ratio = ave_p_ave_l(profits)
    
    metrics = {
        "E(R)": expected_return,
        "std(R)": std_return,
        "Downside Deviation (DD)": dd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "MDD": mdd,
        "Calmar Ratio": calmar,
        "% +ve Returns": positive_pct,
        "Ave. P / Ave. L": ave_p_ave_l_ratio
    }
    
    return metrics
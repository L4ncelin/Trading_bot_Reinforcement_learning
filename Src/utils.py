import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm.notebook import tqdm

# ------------------------------ Data treatment ------------------------------ #

def get_commodities_data(tickers, start="2010-01-01", end="2023-01-01", interval="1d"):
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, interval=interval)
        data[ticker] = df
    return data

def add_technical_indicators(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (tp - sma_tp) / (0.015 * mean_dev)

    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df['-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([df['High'] - df['Low'], 
                    np.abs(df['High'] - df['Close'].shift(1)), 
                    np.abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(span=14, adjust=False).mean() / atr)
    df['-DI'] = 100 * (df['-DM'].ewm(span=14, adjust=False).mean() / atr)
    dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = dx.ewm(span=14, adjust=False).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    df['Volume'] = np.log1p(df['Volume'])  # Pour aplatir les grosses valeurs
    df['Volume'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
    
    df.dropna(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX', 'BB_Mid', 'BB_Std', 'BB_Upper', 'BB_Lower']]
    return df


# Fonction pour ajouter les bandes de Bollinger
def add_bollinger_bands(df, window=20):
    df = df.copy()
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    return df.dropna()


def test_agent(env, model, n_tests, visualize=False):
    metrics = {
        'steps': [],  # Les pas de l'agent
        'balances': [],  # Les balances des comptes
        'net_worths': [],  # Le Net Worth à chaque étape
        'shares_held': {ticker: [] for ticker in env.envs[0].commodity_data.keys()}  # Quantité de chaque actif détenu
    }
    
    obs = env.reset()
    for i in range(n_tests):
        metrics['steps'].append(i)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step([action])
        if visualize:
            env.envs[0].render()  # Option pour visualiser à chaque étape
        metrics['balances'].append(env.envs[0].balance)  # Ajouter la balance de l'agent
        metrics['net_worths'].append(env.envs[0].net_worth)  # Ajouter le Net Worth de l'agent
        for ticker in env.envs[0].commodity_data.keys():
            metrics['shares_held'][ticker].append(env.envs[0].shares_held[ticker])  # Quantité des actions détenues
        if done:
            obs = env.reset()  # Réinitialiser l'environnement si l'épisode est terminé
    return metrics



# ----------------------------------- Ploit ---------------------------------- #
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_net_worth(test_metrics, env):
    """
    Affiche l'évolution du Net Worth avec les dates réelles en abscisse.
    """
    # Récupération des dates depuis l'index du DataFrame (on prend le 1er ticker comme référence)
    df_index = env.envs[0].commodity_data[env.envs[0].tickers[0]].index
    
    # On aligne les dates avec la longueur des steps testés
    dates = df_index[:len(test_metrics['net_worths'])]

    # Création du graphique
    plt.figure(figsize=(12,6))
    plt.plot(dates, test_metrics['net_worths'], label="Net Worth", color='blue', linewidth=1.8)
    plt.title("Évolution du Net Worth sur le set de Test")
    plt.xlabel("Date")
    plt.ylabel("Net Worth")
    plt.grid(True)
    plt.legend()

    # Formatage de la date en abscisse
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
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
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CommodityTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, commodity_data, transaction_cost_percent=0.005, initial_balance=10000):
        super(CommodityTradingEnv, self).__init__()
        
        self.commodity_data = {ticker: df for ticker, df in commodity_data.items() if not df.empty}
        self.tickers = list(self.commodity_data.keys())
        if not self.tickers:
            raise ValueError("Aucune donnée disponible pour les commodities.")
        
        sample_df = next(iter(self.commodity_data.values()))
        self.n_features = len(sample_df.columns)

        self.recent_actions = {ticker: [] for ticker in self.tickers} 
        
        # Espace d'actions discret (inchangé)
        self.action_list = [-1, -0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75, 1]
        self.num_actions_per_commodity = len(self.action_list)
        self.action_space = spaces.Discrete(self.num_actions_per_commodity ** len(self.tickers))
        
        # Espace d'observation
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        
        self.initial_balance = initial_balance 
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        
        self.current_step = 0
        self.max_steps = max(0, min(len(df) for df in self.commodity_data.values()) - 1)
        self.transaction_cost_percent = transaction_cost_percent
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        frame = np.zeros(self.obs_shape)
        idx = 0
        for ticker in self.tickers:
            df = self.commodity_data[ticker]
            if self.current_step < len(df):
                frame[idx:idx+self.n_features] = df.iloc[self.current_step].values
            else:
                frame[idx:idx+self.n_features] = df.iloc[-1].values
            idx += self.n_features
        
        # Ajout des informations supplémentaires
        frame[-4-len(self.tickers)] = self.balance
        frame[-3-len(self.tickers):-3] = [self.shares_held[ticker] for ticker in self.tickers]
        frame[-3] = self.net_worth
        frame[-2] = self.max_net_worth
        frame[-1] = self.current_step
        
        return frame
    
    def decode_action(self, action):
        decoded_indices = []
        temp = action
        n = len(self.tickers)
        for _ in range(n):
            decoded_indices.append(temp % self.num_actions_per_commodity)
            temp //= self.num_actions_per_commodity
        decoded_indices.reverse()
        decoded_actions = [self.action_list[idx] for idx in decoded_indices]
        return decoded_actions
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action[0])
        actions = self.decode_action(action)
        
        prev_net_worth = self.net_worth
        self.current_step += 1
        if self.current_step > self.max_steps:
            return self._next_observation(), 0, True, False, {}

        penalty = 0
        current_prices = {}

        for i, ticker in enumerate(self.tickers):
            price = float(self.commodity_data[ticker]['Close'].iloc[self.current_step])
            act = actions[i]
            current_prices[ticker] = price

            # Ajout de l'action actuelle à l'historique
            self.recent_actions[ticker].append(act)
            if len(self.recent_actions[ticker]) > 15:
                self.recent_actions[ticker].pop(0)

            if act < 0 and self.shares_held[ticker] <= 0:
                penalty -= 0.05

            # --- EXÉCUTION DES ACTIONS ---
            if act > 0:  # Achat
                shares_to_buy = int(self.balance * act / price)
                cost = shares_to_buy * price
                transaction_cost = cost * self.transaction_cost_percent
                self.balance -= (cost + transaction_cost)
                self.shares_held[ticker] += shares_to_buy

            elif act < 0 and self.shares_held[ticker] > 0:  # Vente
                shares_to_sell = int(self.shares_held[ticker] * abs(act))
                sale = shares_to_sell * price
                transaction_cost = sale * self.transaction_cost_percent
                self.balance += (sale - transaction_cost)
                self.shares_held[ticker] -= shares_to_sell
                self.total_shares_sold[ticker] += shares_to_sell
                self.total_sales_value[ticker] += sale

        self.net_worth = self.balance + sum(self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        reward = (self.net_worth - prev_net_worth) / prev_net_worth + penalty


        


        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        return self._next_observation(), reward, done, False, {}
    
    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        for ticker in self.tickers:
            print(f"{ticker} Shares held: {self.shares_held[ticker]}")
        print(f"Net worth: {self.net_worth:.2f} | Profit: {profit:.2f}")
    
    def close(self):
        pass
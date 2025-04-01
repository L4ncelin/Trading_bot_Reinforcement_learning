import pickle

class BollingerBaselineAgent:
    def __init__(self, env):
        """
        A simple, rule-based Bollinger baseline agent.
        Assumptions:
         - The observation for each asset contains:
             • The asset’s “Close” price at index (base_idx + 3)
             • A Bollinger upper value at index (base_idx + 12)
             • A Bollinger lower value at index (base_idx + 13)
         - When the current price is below the lower band, the agent issues a full buy signal.
         - When the current price is above the upper band, it issues a full sell signal.
         - Otherwise, it holds (no action).
         - The agent uses the environment’s discrete action space. 
        """
        self.env = env.envs[0] if hasattr(env, "envs") else env
        self.n_features = env.envs[0].n_features


    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        """
        Determines the action based on Bollinger signals.
        For each asset:
          - if Close < bb_lower => full buy (1.0)
          - if Close > bb_upper => full sell (-1.0)
          - else hold (0.0)
        The continuous signals are then mapped to the closest discrete action defined in env.action_list.
        """
        tickers = self.env.tickers
        n = len(tickers)
        actions = []
        obs = obs[0]  # Retire la dimension superflue
        
        for i in range(n):
            base_idx = i * self.n_features
            # Extract the necessary features from the observation.
            # Adjust the indices if your observation structure is different.
            close = obs[base_idx + 3]
            bb_upper = obs[base_idx + 12]
            bb_lower = obs[base_idx + 13]
            
            if close < bb_lower:
                actions.append(1.0)    # Signal full buy
            elif close > bb_upper:
                actions.append(-1.0)   # Signal full sell
            else:
                actions.append(0.0)    # Hold
        
        # Convert continuous signals into the discrete action index.
        action = self.encode_action(actions)
        return action, None

    def encode_action(self, action_vector):
        """
        Maps a vector of continuous action signals to a single discrete action integer.
        Uses the environment's action_list and number of actions per commodity.
        """
        def closest(val):
            return min(range(len(self.env.action_list)), key=lambda i: abs(val - self.env.action_list[i]))
        
        index = 0
        base = self.env.num_actions_per_commodity
        for val in action_vector:
            idx = closest(val)
            index = index * base + idx
        return index

    def save(self, path):
        """Saves the agent via pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Loads a previously saved agent."""
        with open(path, "rb") as f:
            return pickle.load(f)
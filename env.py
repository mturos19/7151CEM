# portfolio_env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt

class PortfolioEnv(gym.Env):
    """
    a custom gym environment for portfolio optimization.
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 50,
        trading_cost: float = 0.001,
        risk_free_rate: float = 0.02,
        cvar_alpha: float = 0.05,
        initial_balance: float = 1000000,
        cache_data: bool = True
    ):        
        super(PortfolioEnv, self).__init__()
        
        # load and preprocess data
        self.raw_data = pd.read_csv(data_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        self.raw_data.set_index('Date', inplace=True)
        
        # ensure data is properly sorted
        self.raw_data.sort_index(inplace=True)
        
        # configuration
        self.window_size = window_size
        self.trading_cost = trading_cost
        self.risk_free_rate = risk_free_rate / 252  # daily risk-free rate
        self.cvar_alpha = cvar_alpha
        self.initial_balance = initial_balance
        
        # define asset universe (only columns ending with '_close')
        self.assets = [col for col in self.raw_data.columns if col.endswith('_Close')]
        
        # add this validation
        if len(self.assets) == 0:
            raise ValueError("No asset columns found ending with '_Close'")
            
        self.num_assets = len(self.assets)
        
        # define macroeconomic features
        self.macro_features = ['GDP', 'CPI', 'FEDFUNDS', 'UNRATE', 'PPI']
        
        # validate macro features
        missing_features = [feat for feat in self.macro_features if feat not in self.raw_data.columns]
        if missing_features:
            raise ValueError(f"Missing macro features in data: {missing_features}")
        
        # precompute portfolio returns
        self.returns_data = self.raw_data[self.assets].pct_change().fillna(0)
        self.asset_cov = self.returns_data.cov()
        
        # process macroeconomic features
        self.processed_features = self._process_features()
        
        # define action and observation spaces
        self.action_space = spaces.Box(
            low=0, 
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        num_features = self.processed_features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, num_features),
            dtype=np.float32
        )
        
        # trading variables
        self.current_step = None
        self.current_weights = None
        self.portfolio_value = None
        self.returns_memory = None
        self.weights_memory = None
        
        # initialize state
        self.reset()
    
    def _process_features(self) -> pd.DataFrame:
        """
        process macroeconomic features into a dataframe using rolling windows for observations.
        """
        features = self.raw_data[self.macro_features].copy()
        
        # fill missing values
        features.fillna(method='ffill', inplace=True)
        features.fillna(method='bfill', inplace=True)
        
        # normalize features
        normalized_features = (features - features.mean()) / (features.std() + 1e-8)
        
        return normalized_features
    
    def _get_observation(self) -> np.ndarray:
        """
        construct the observation (state) from the current position using a rolling window of macro features.
        """
        end_idx = self.current_step + 1  # inclusive of current_step
        start_idx = end_idx - self.window_size
        
        if start_idx < 0:
            # pad with zeros if not enough data
            pad_length = abs(start_idx)
            window_features = self.processed_features.iloc[0:end_idx].values
            padding = np.zeros((pad_length, self.processed_features.shape[1]))
            observation = np.vstack((padding, window_features))
        else:
            window_features = self.processed_features.iloc[start_idx:end_idx].values
            observation = window_features
        
        return observation
    
    def _calculate_portfolio_metrics(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        calculate portfolio return, volatility, and cvar using historical simulation.
        """
        # ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum
            
        portfolio_return = np.sum(returns * weights)
        
        # add small constant to avoid numerical issues
        portfolio_vol = np.sqrt(np.maximum(weights.T @ self.asset_cov @ weights, 1e-10))
        
        # historical simulation for cvar with error handling
        portfolio_returns = self.returns_data.dot(weights)
        if len(portfolio_returns) > 0:
            sorted_returns = np.sort(portfolio_returns)
            index = max(1, int(self.cvar_alpha * len(sorted_returns)))
            cvar = -sorted_returns[:index].mean()
        else:
            cvar = 0.0
            
        return portfolio_return, portfolio_vol, cvar
    
    def _calculate_reward(
        self, 
        old_weights: np.ndarray, 
        new_weights: np.ndarray
    ) -> float:
        """
        calculate the reward (risk-adjusted returns) with market impact-based transaction costs.
        """
        # get returns for the current step
        current_returns = self.returns_data.iloc[self.current_step].values
        
        # calculate reward components
        portfolio_return, portfolio_vol, cvar = self._calculate_portfolio_metrics(
            new_weights, current_returns
        )
        
        # calculate turnover and transaction costs with market impact
        turnover = np.sum(np.abs(new_weights - old_weights))
        market_impact_cost = turnover ** 2 * self.trading_cost  # quadratic cost function
        
        # reward: maximize return, minimize cvar and transaction costs
        reward = portfolio_return - (cvar * 0.5) - market_impact_cost
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        execute one step in the environment with enhanced logging.
        """
        if self.current_step + 1 >= len(self.returns_data):
            raise RuntimeError("Episode has already ended")

        # ensure action is numpy array and handle nan values
        action = np.array(action, dtype=np.float32)
        if np.any(np.isnan(action)):
            action = np.ones_like(action) / len(action)
        
        # clip actions to prevent extreme allocations
        max_weight = 0.3
        action = np.clip(action, 0, max_weight)
        action = action / (action.sum() + 1e-8)  # normalize to sum to 1
        
        # get old weights
        old_weights = self.current_weights
        
        # calculate reward
        reward = self._calculate_reward(old_weights, action)
        
        # update portfolio value
        self.portfolio_value *= (1 + reward)
        
        # update state
        self.current_weights = action
        self.weights_memory.append(action)
        self.returns_memory.append(reward)
        self.current_step += 1
        
        # get new observation
        observation = self._get_observation()
        
        # check if episode is done
        done = self.current_step + 1 >= len(self.returns_data)
        
        # calculate portfolio metrics for logging
        current_returns = self.returns_data.iloc[self.current_step].values
        portfolio_return, portfolio_vol, cvar = self._calculate_portfolio_metrics(
            action, current_returns
        )
        
        # additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': action,
            'step': self.current_step,
            'returns': reward,
            'turnover': np.sum(np.abs(action - old_weights)),
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_vol,
            'cvar': cvar
        }
        
        # early stopping conditions
        target_return = 1.2 * self.initial_balance  # example target
        max_allowed_drawdown = 0.8  # example threshold
        
        # calculate current drawdown
        peak = max(self.portfolio_value, self.initial_balance)
        drawdown = (peak - self.portfolio_value) / peak
        
        if self.portfolio_value >= target_return or drawdown >= max_allowed_drawdown:
            done = True
            info['early_stop'] = {
                'reason': 'target_return_reached' if self.portfolio_value >= target_return else 'max_drawdown_exceeded'
            }
        
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        reset the environment to initial state
        """
        self.current_step = 0
        self.current_weights = np.ones(self.num_assets) / self.num_assets
        self.portfolio_value = self.initial_balance
        self.returns_memory = []
        self.weights_memory = [self.current_weights]
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """
        render the environment with detailed visualizations.
        """
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Portfolio Value: ${self.portfolio_value:,.2f}')
            print('Current Weights:')
            for asset, weight in zip(self.assets, self.current_weights):
                print(f'{asset}: {weight:.4f}')
            
            # plot portfolio value history
            plt.figure(figsize=(10, 5))
            plt.plot(np.cumprod(1 + np.array(self.returns_memory)) * self.initial_balance, label='Portfolio Value')
            plt.title('Portfolio Value History')
            plt.xlabel('Steps')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.show()
            
            # plot weights distribution
            plt.figure(figsize=(10, 5))
            weights_array = np.array(self.weights_memory)
            for i, asset in enumerate(self.assets):
                plt.plot(weights_array[:, i], label=asset)
            plt.title('Asset Weights Over Time')
            plt.xlabel('Steps')
            plt.ylabel('Weight')
            plt.legend()
            plt.show()
    
    def save_state(self, filepath: str):
        """
        save the current state of the environment to a file.
        """
        state = {
            'current_step': self.current_step,
            'current_weights': self.current_weights,
            'portfolio_value': self.portfolio_value,
            'returns_memory': self.returns_memory,
            'weights_memory': self.weights_memory
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """
        load the environment state from a file.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.current_step = state['current_step']
        self.current_weights = state['current_weights']
        self.portfolio_value = state['portfolio_value']
        self.returns_memory = state['returns_memory']
        self.weights_memory = state['weights_memory']
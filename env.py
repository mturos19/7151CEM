# portfolio_env.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, SMAIndicator
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

class PortfolioEnv(gym.Env):
    """
    a custom gym environment for portfolio optimization.
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 365,
        trading_cost: float = 0.001,
        risk_free_rate: float = 0.02,
        cvar_alpha: float = 0.05,
        initial_balance: float = 1000000,
        cache_data: bool = True,
        rebalance_frequency: str = 'monthly',
        additional_features: bool = True
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
        self.rebalance_frequency = rebalance_frequency
        self.steps_per_rebalance = self._get_steps_per_rebalance()
        self.next_rebalance_step = self.steps_per_rebalance
        self.additional_features = additional_features
        
        # define asset universe (only columns ending with '_close')
        self.assets = [col for col in self.raw_data.columns if col.endswith('_Close')]
        
        # add this validation
        if len(self.assets) == 0:
            raise ValueError("No asset columns found ending with '_Close'")
            
        self.num_assets = len(self.assets)
        
        # macroeconomic features
        self.macro_features = ['GDP', 'CPI', 'FEDFUNDS', 'UNRATE', 'PPI']
        
        # validate macro features
        missing_features = [feat for feat in self.macro_features if feat not in self.raw_data.columns]
        if missing_features:
            raise ValueError(f"Missing macro features in data: {missing_features}")
        
        # precompute portfolio returns
        self.returns_data = self.raw_data[self.assets].pct_change().fillna(0)
        self.asset_cov = self.returns_data.cov().values
        
        # process macroeconomic features
        self.processed_features = self._process_features()
        
        # action and observation spaces
        self.action_space = spaces.Box(
            low=0, 
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )
        
        num_features = self.processed_features.shape[1]
        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(self.window_size, self.processed_features.shape[1]),
            dtype=np.float32
        )
        
        # debug obs space
        print(f"Observation Space Shape: {self.observation_space.shape}")
        
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
        process macroeconomic and technical features into a dataframe using rolling windows for observations.
        """
        # extract macroeconomic features
        features = self.raw_data[self.macro_features].copy()
        
        # handle missing values using updated methods
        features = features.ffill().bfill()
        
        # normalize macroeconomic features
        normalized_macro = (features - features.mean()) / (features.std() + 1e-8)
        
        if self.additional_features:
            # create separate dataframes for each asset's indicators
            asset_features = []
            
            for asset in self.assets:
                # calculate all indicators for this asset
                indicators = {}
                
                # momentum indicators
                rsi = RSIIndicator(close=self.raw_data[asset], window=14)
                sma20 = SMAIndicator(close=self.raw_data[asset], window=20)
                sma50 = SMAIndicator(close=self.raw_data[asset], window=50)
                bb = BollingerBands(close=self.raw_data[asset], window=20, window_dev=2)
                macd = MACD(close=self.raw_data[asset])
                
                # store all indicators in dictionary
                indicators.update({
                    f'{asset}_RSI': rsi.rsi(),
                    f'{asset}_SMA_20': sma20.sma_indicator(),
                    f'{asset}_SMA_50': sma50.sma_indicator(),
                    f'{asset}_Bollinger_High': bb.bollinger_hband(),
                    f'{asset}_Bollinger_Low': bb.bollinger_lband(),
                    f'{asset}_MACD': macd.macd(),
                    f'{asset}_MACD_Signal': macd.macd_signal(),
                    f'{asset}_MACD_Diff': macd.macd_diff()
                })
                
                # convert dictionary to dataframe
                asset_df = pd.DataFrame(indicators)
                asset_features.append(asset_df)
                
            
            # combine all technical features at once
            technical_features = pd.concat(asset_features, axis=1)
            
            # handle missing values
            technical_features = technical_features.ffill().bfill()
            
            # normalize technical features
            normalized_technical = (technical_features - technical_features.mean()) / (technical_features.std() + 1e-8)
            
            # combine macroeconomic and technical features
            combined_features = pd.concat([normalized_macro, normalized_technical], axis=1)
        else:
            combined_features = normalized_macro
        
        print("Shape of features df: ",combined_features.shape)
        
        return combined_features
    
    def _get_observation(self) -> np.ndarray:
        """
        construct the observation (state) from the current position using a rolling window of macro features.
        """
        end_idx = self.current_step + 1
        start_idx = end_idx - self.window_size
        
        if start_idx < 0:
            pad_length = abs(start_idx)
            window_features = self.processed_features.iloc[0:end_idx].values
            padding = np.zeros((pad_length, self.processed_features.shape[1]))
            observation = np.vstack((padding, window_features))
        else:
            window_features = self.processed_features.iloc[start_idx:end_idx].values
            observation = window_features
        
        return observation.astype(np.float32)
    
    def _calculate_portfolio_metrics(
        self, 
        weights: np.ndarray, 
        returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        calculate portfolio return, volatility, and cvar using historical simulation.
        """
        # ensure inputs are numpy arrays
        weights = np.array(weights)
        returns = np.array(returns)
        
        # numpy operations
        weights = np.clip(weights, a_min=0, a_max=None)
        weights_sum = weights.sum()
        if weights_sum > 0:
            weights = weights / weights_sum
            
        portfolio_return = (returns * weights).sum()
        portfolio_vol = np.sqrt(np.maximum(
            weights.dot(self.asset_cov).dot(weights),
            1e-10
        ))
        
        # calculate cvar
        portfolio_returns = self.returns_data.values @ weights
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
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        execute one step in the environment.
        """
        # store old weights for turnover calculation
        old_weights = self.current_weights.copy()

        # add debug print
        current_date = self.raw_data.index[self.current_step]
        next_date = self.raw_data.index[self.current_step + 1]
        #print(f"Step from {current_date} to {next_date}")
        
        # only update weights if it's time to rebalance
        if self.current_step >= self.next_rebalance_step:
            # ensure action is valid and normalized
            action = np.clip(action, a_min=0, a_max=None)  # ensure non-negative
            action_sum = action.sum()
            if action_sum > 0:
                self.current_weights = action / action_sum
            else:
                self.current_weights = np.ones(self.num_assets) / self.num_assets
            self.next_rebalance_step = self.current_step + self.steps_per_rebalance
        
        # move forward one step
        self.current_step += 1
        
        # check if we've reached the end of our data
        done = self.current_step >= len(self.raw_data) - 1
        
        # get daily returns for the current step
        daily_returns = self.returns_data.iloc[self.current_step].values
        
        # calculate portfolio return
        portfolio_return = np.sum(self.current_weights * daily_returns)
        
        # update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        # calculate reward
        reward = self._calculate_reward(old_weights, self.current_weights)
        
        # get new observation
        observation = self._get_observation()
        
        # store history
        self.returns_memory.append(portfolio_return)
        self.weights_memory.append(self.current_weights.copy())
        
        # additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'returns': portfolio_return,
            'step': self.current_step,
        }
        
        # early stopping only after sufficient history
        if self.current_step > self.window_size + 30:
            peak = max([self.initial_balance] + 
                      [self.initial_balance * np.prod(1 + np.array(self.returns_memory[:i])) 
                       for i in range(len(self.returns_memory))])
            drawdown = (peak - self.portfolio_value) / peak
            
            if drawdown > 0.5:  # 50% max drawdown
                done = True
                info['early_stop'] = 'max_drawdown'
        
        # convert observation to numpy before returning
        if isinstance(observation, np.ndarray):
            observation = observation.astype(np.float32)
        
        return observation, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """
        reset the environment to initial state.
        """
        super().reset(seed=seed)
        
        # ensure we have enough data for initial window
        self.current_step = self.window_size

        # add debug print
        current_date = self.raw_data.index[self.current_step]
        print(f"Environment reset - Starting at date: {current_date}")

        self.portfolio_value = self.initial_balance
        self.returns_memory = []
        self.weights_memory = []
        
        # initialize with equal weights
        self.current_weights = np.ones(self.num_assets) / self.num_assets
        self.next_rebalance_step = self.current_step + self.steps_per_rebalance
        
        # create info dict
        info = {
            'initial_portfolio_value': self.initial_balance,
            'initial_weights': self.current_weights.copy(),
            'num_assets': self.num_assets
        }
        
        return self._get_observation(), info
    
    def render(self, mode='human'):
        """
        render the environment with detailed visualizations.
        """
        if mode == 'human':
            # create a figure with multiple subplots
            fig = plt.figure(figsize=(20, 12))  # increased figure size
            
            # 1. portfolio value history
            ax1 = plt.subplot(2, 2, 1)
            portfolio_values = np.cumprod(1 + np.array(self.returns_memory)) * self.initial_balance
            ax1.plot(portfolio_values, 'b-', label='Portfolio Value')
            ax1.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Investment')
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # 2. cumulative returns
            ax2 = plt.subplot(2, 2, 2)
            cumulative_returns = (portfolio_values - self.initial_balance) / self.initial_balance * 100
            ax2.plot(cumulative_returns, 'g-', label='Cumulative Returns')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_title('Cumulative Returns (%)')
            ax2.set_xlabel('Trading Days')
            ax2.set_ylabel('Return (%)')
            ax2.legend()
            ax2.grid(True)
            
            # 3. asset allocation over time (modified)
            ax3 = plt.subplot(2, 2, 3)
            weights_array = np.array(self.weights_memory)
            
            # create proper x-axis values
            x_values = np.arange(len(weights_array))
            
            # only plot top 5 assets by final weights for clarity
            final_weights = weights_array[-1] if len(weights_array) > 0 else self.current_weights
            top_indices = np.argsort(final_weights)[-5:]  # get indices of top 5 weights
            
            for idx in top_indices:
                asset_name = self.assets[idx].replace('_Close', '')
                ax3.plot(x_values, weights_array[:, idx], label=asset_name, linewidth=2)
                
            ax3.set_title('Top 5 Asset Weights Over Time')
            ax3.set_xlabel('Trading Steps')
            ax3.set_ylabel('Weight')
            ax3.set_xlim(0, len(weights_array) - 1)  # set proper x-axis limits
            ax3.set_ylim(0, max(weights_array.max(), 0.05))  # set proper y-axis limits with minimum range
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax3.grid(True)
            
            # add text showing number of total assets
            ax3.text(0.02, 0.98, f'showing top 5 of {len(self.assets)} assets', 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top')
            
            # 4. current asset allocation (pie chart) - modified to show top 5
            ax4 = plt.subplot(2, 2, 4)
            asset_names = [self.assets[i].replace('_Close', '') for i in top_indices]
            weights = [self.current_weights[i] for i in top_indices]
            others_weight = 1 - sum(weights)
            
            if others_weight > 0:
                asset_names.append('Others')
                weights.append(others_weight)
            
            ax4.pie(weights, labels=asset_names, autopct='%1.1f%%')
            ax4.set_title('Current Portfolio Allocation (Top 5 Assets)')
            
            # adjust layout with more space for legends
            plt.tight_layout()
            plt.show()
            
            # print detailed weights information
            print("\n=== Current Asset Allocation ===")
            sorted_weights = sorted(zip(self.assets, self.current_weights), 
                                  key=lambda x: x[1], reverse=True)
            for asset, weight in sorted_weights:
                if weight > 0.01:  # only show assets with >1% allocation
                    print(f"{asset.replace('_Close', ''):15s}: {weight*100:6.2f}%")
    
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

    def _get_steps_per_rebalance(self) -> int:
        if self.rebalance_frequency == 'monthly':
            return 30  # approximate number of days in a month
        elif self.rebalance_frequency == 'yearly':
            return 365
        elif self.rebalance_frequency == 'weekly':
            return 7
        else:
            raise ValueError("Invalid rebalance_frequency. Choose 'monthly', 'yearly', or 'weekly'.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env import PortfolioEnv
import gymnasium as gym

def make_env(data_path, window_size, rebalance_frequency, additional_features, rank=0):
    """
    Utility function to create the PortfolioEnv environment.
    """
    def _init():
        env = PortfolioEnv(
            data_path=data_path,
            window_size=window_size,
            rebalance_frequency=rebalance_frequency,
            trading_cost=0.001,
            risk_free_rate=0.02,
            cvar_alpha=0.05,
            initial_balance=1000000,
            cache_data=True,
            additional_features=additional_features
        )
        return env
    return _init

def analyze_model_performance(model_path="models/final_model",
                              vec_normalize_path="models/vec_normalize.pkl",
                              data_path='data/processed/merged_data.csv'):
    """
    Analyze the trained model's performance against the S&P 500.
    """
    # Load the environment
    window_size = 365
    rebalance_frequency = 'monthly'
    additional_features = True
    n_envs = 4

    env = SubprocVecEnv([make_env(data_path, window_size, rebalance_frequency, additional_features, rank=i) for i in range(n_envs)])
    
    # Load the VecNormalize statistics
    vec_normalize = VecNormalize.load(vec_normalize_path, env)
    
    # Do not update the normalization statistics
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    
    # Load the trained model
    model = PPO.load(model_path, env=env)
    
    # Reset the environment
    obs = env.reset()
    
    # Load the merged data
    merged_data = pd.read_csv(data_path, parse_dates=['Date'])
    merged_data.sort_values('Date', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)
    
    portfolio_values = []
    sp500_values = []
    dates = []
    
    for i in range(len(merged_data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Assuming the environment has a 'portfolio_value' attribute
        portfolio_value = env.get_attr('portfolio_value')[0]
        portfolio_values.append(portfolio_value)
        
        # Get the S&P 500 close price
        sp500 = merged_data.loc[i, '^GSPC_Close']
        sp500_values.append(sp500)
        
        # Get the date
        date = merged_data.loc[i, 'Date']
        dates.append(date)
        
        if dones[0]:
            break
    
    # Convert to pandas Series
    portfolio_series = pd.Series(portfolio_values, index=dates)
    sp500_series = pd.Series(sp500_values, index=dates)
    
    # Calculate cumulative returns
    portfolio_returns = portfolio_series.pct_change().fillna(0)
    sp500_returns = sp500_series.pct_change().fillna(0)
    
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_sp500 = (1 + sp500_returns).cumprod()
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_portfolio, label='Portfolio Returns')
    plt.plot(cumulative_sp500, label='S&P 500 Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Portfolio Returns vs S&P 500')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_vs_sp500_returns.png')
    plt.show()

if __name__ == "__main__":
    analyze_model_performance()
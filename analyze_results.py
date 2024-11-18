import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env import PortfolioEnv
import gymnasium as gym
from matplotlib.animation import FuncAnimation

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

    #print("Date range in merged data:", merged_data['Date'].min(), "to", merged_data['Date'].max())
    
    portfolio_values = []
    sp500_values = []
    asset_weights = []
    dates = []
    
    # Add these diagnostic prints
    print("Starting analysis from", merged_data['Date'].min(), "to", merged_data['Date'].max())
    
    done = False
    obs = env.reset()
    
    for i in range(len(merged_data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # Get the portfolio value and other data regardless of done status
        portfolio_value = env.get_attr('portfolio_value')[0]
        portfolio_values.append(portfolio_value)
        
        sp500 = merged_data.loc[i, '^GSPC_Close']
        sp500_values.append(sp500)
        
        weights = env.get_attr('current_weights')[0]
        asset_weights.append(weights)
        
        date = merged_data.loc[i, 'Date']
        dates.append(date)
        
        # If done, reset the environment but continue the loop
        if dones[0]:
            print(f"Episode done at date {date}, resetting environment")
            obs = env.reset()
    
    # Add diagnostic print
    print(f"Collected {len(portfolio_values)} portfolio values from {dates[0]} to {dates[-1]}")
    
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

    # Visualization of asset weights over time
    asset_weights = np.array(asset_weights)
    num_assets = asset_weights.shape[1]
    
    # Grouping assets into classes based on the provided data
    asset_classes = {
        'Equities': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Stock indices
        'Bonds': [9, 10, 11, 12],                  # Bond indices
        'Commodities': [13, 14, 15, 16],           # Commodities like Gold, Oil
        'Real Estate': [17, 18],                   # Real estate indices
        'Currencies': [19, 20, 21, 22],            # Currency pairs
        'Crypto': [23, 24],                        # Bitcoin, ETH (removed LTC)
        # Indices 25-28 are macro indicators (Interest rate, GDP, CPI etc)
    }
    
    # Summing weights by asset class
    class_weights = {class_name: asset_weights[:, indices].sum(axis=1) for class_name, indices in asset_classes.items()}
    
    # Plotting asset class weights
    plt.figure(figsize=(12, 6))
    for class_name, weights in class_weights.items():
        plt.plot(dates, weights, label=f'{class_name} Weight')
    plt.xlabel('Date')
    plt.ylabel('Asset Class Weights')
    plt.title('Asset Class Weights Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('asset_class_weights_over_time.png')
    plt.show()

    # Pie chart animation of asset class weights (monthly)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Convert dates to pandas datetime and get monthly data
    dates_pd = pd.to_datetime(dates)
    monthly_dates = dates_pd.to_period('M').unique()
    print("Asset weights shape:", asset_weights.shape)
    def update(frame):
        ax.clear()
        month_end = monthly_dates[frame].to_timestamp(how='end')
        # Find the closest date using absolute difference
        idx = abs(dates_pd - month_end).argmin()
        weights = [class_weights[class_name][idx] for class_name in class_weights]
        ax.pie(weights, labels=list(class_weights.keys()), autopct='%1.1f%%')
        ax.set_title(f'Asset Class Weights on {month_end.strftime("%Y-%m-%d")}')

    ani = FuncAnimation(fig, update, frames=len(monthly_dates), repeat=False)
    ani.save('asset_class_weights_animation.gif', writer='pillow')
    
    
    
if __name__ == "__main__":
    analyze_model_performance()
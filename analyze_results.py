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

def analyze_model_performance(model_path="models/final_model_100k",
                              vec_normalize_path="models/vec_normalize_100k.pkl",
                              data_path='data/processed/merged_data.csv'):
    """
    Analyze the trained model's performance against the S&P 500.
    """
    # load the environment
    window_size = 365
    rebalance_frequency = 'monthly'
    additional_features = True
    n_envs = 4

    env = SubprocVecEnv([make_env(data_path, window_size, rebalance_frequency, additional_features, rank=i) for i in range(n_envs)])
    
    # load VecNormalize statistics
    vec_normalize = VecNormalize.load(vec_normalize_path, env)
    
    # stopping the update of normalization statistics
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    
    # load model
    model = PPO.load(model_path, env=env)
    
    # environment reset
    obs = env.reset()
    
    # load data
    merged_data = pd.read_csv(data_path, parse_dates=['Date'])
    merged_data.sort_values('Date', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)
   # merged_data = merged_data[merged_data['Date'] <= '2011-01-01']

    #print("Date range in merged data:", merged_data['Date'].min(), "to", merged_data['Date'].max())
    
    portfolio_values = []
    sp500_values = []
    asset_weights = []
    dates = []
    
    # diagnostic prints
    #print("Starting analysis from", merged_data['Date'].min(), "to", merged_data['Date'].max())
    
    done = False
    obs = env.reset()
    
    # scaling factor to maintain continuity
    scale_factor = 1.0
    last_portfolio_value = None
    
    for i in range(len(merged_data)):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        # current portfolio value
        portfolio_value = env.get_attr('portfolio_value')[0]
        
        # handle environment reset
        if dones[0]:
            print(f"Environment reset at index {i}, date {merged_data.loc[i, 'Date']}")
            obs = env.reset()
            
            # updating scale factor to maintain continuity
            if last_portfolio_value is not None:
                scale_factor *= last_portfolio_value / 1000000  # Assuming initial_balance is 1000000
            continue
        
        # scaling the value to maintain continuity
        scaled_portfolio_value = portfolio_value * scale_factor
        last_portfolio_value = scaled_portfolio_value
        
        date = merged_data.loc[i, 'Date']
        dates.append(date)
        portfolio_values.append(scaled_portfolio_value)
        
        sp500 = merged_data.loc[i, '^GSPC_Close']
        sp500_values.append(sp500)
        
        weights = env.get_attr('current_weights')[0]
        asset_weights.append(weights)
    
    # debug print
    #print(f"Collected {len(portfolio_values)} portfolio values from {dates[0]} to {dates[-1]}")
    
    portfolio_series = pd.Series(portfolio_values, index=dates)
    sp500_series = pd.Series(sp500_values, index=dates)
    
    # cumulative returns
    portfolio_returns = portfolio_series.pct_change().fillna(0)
    sp500_returns = sp500_series.pct_change().fillna(0)
    
    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_sp500 = (1 + sp500_returns).cumprod()
    
    # performance metrics
    def calculate_metrics(returns, benchmark_returns, risk_free_rate=0.02):
        # total profit
        total_return = (cumulative_portfolio.iloc[-1] - 1) * 100
        
        # monthly returns for best/worst month calculation
        monthly_returns = returns.resample('ME').agg(lambda x: (1 + x).prod() - 1) * 100
        
        # Win Rate calculation (monthly basis)
        win_rate = (monthly_returns > 0).mean() * 100
        
        # other metrics
        annual_returns = ((1 + returns.mean()) ** 252 - 1) * 100
        annual_vol = returns.std() * np.sqrt(252) * 100
        
        # Sharpe & Sortino
        excess_returns = returns - risk_free_rate/252
        sharpe_ratio = (np.sqrt(252) * excess_returns.mean() / returns.std()) * 100
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = ((annual_returns/100 - risk_free_rate) / downside_std) * 100 if downside_std != 0 else np.nan
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns/rolling_max - 1) * 100
        max_drawdown = drawdowns.min()
        
        return {
            'Total Portfolio Profit': total_return,
            'Annual Return': annual_returns,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Best Month': monthly_returns.max(),
            'Worst Month': monthly_returns.min()
        }
    
    # calculate metrics
    metrics = calculate_metrics(portfolio_returns, sp500_returns)
    
    # financial metrics
    print(f"\nTotal Portfolio Profit: {metrics['Total Portfolio Profit']:.2f}%")
    print("\nML-Enhanced Portfolio Metrics:")
    print("----------")
    print(f"Annual Return       : {metrics['Annual Return']:>10.2f}%")
    print(f"Annual Volatility   : {metrics['Annual Volatility']:>10.2f}%")
    print(f"Sharpe Ratio        : {metrics['Sharpe Ratio']:>10.2f}%")
    print(f"Sortino Ratio       : {metrics['Sortino Ratio']:>10.2f}%")
    print(f"Max Drawdown        : {metrics['Max Drawdown']:>10.2f}%")
    print(f"Win Rate            : {metrics['Win Rate']:>10.2f}%")
    print(f"Best Month          : {metrics['Best Month']:>10.2f}%")
    print(f"Worst Month         : {metrics['Worst Month']:>10.2f}%")
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('portfolio_metrics.csv', index=False)
    
    # return graph 
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_portfolio, label='Portfolio Returns')
    plt.plot(cumulative_sp500, label='S&P 500 Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Portfolio Returns vs S&P 500')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('portfolio_vs_sp500_returns_100k.png')
    plt.show()

    # asset weights over time
    asset_weights = np.array(asset_weights)
    
    asset_classes = {
        'us_large_cap': [4, 5],  # SPY_Close, QQQ_Close
        'us_tech': [0, 1, 2, 3],  # MSFT_Close, AMZN_Close, GOOGL_Close, TSLA_Close
        'commodities': [16, 17, 18],  # GC=F_Close, CL=F_Close, SI=F_Close
        'real_estate': [19, 20, 21],  # VNQ_Close, SCHH_Close, IYR_Close
        'crypto_major': [22, 23],  # BTC_Close, ETH_Close
        'crypto_alt': [24]  # LTC_Close
    }
    
    # summing weights by asset class
    class_weights = {class_name: asset_weights[:, indices].sum(axis=1) for class_name, indices in asset_classes.items()}
    
    # DataFrame for stacked area plot
    df_weights = pd.DataFrame(index=dates)
    for class_name, indices in asset_classes.items():
        df_weights[class_name] = asset_weights[:, indices].sum(axis=1)
    df_weights = df_weights.div(df_weights.sum(axis=1), axis=0)

    
    # stacked area plot
    plt.figure(figsize=(15, 8))
    df_weights.plot(kind='area', stacked=True, colormap='viridis')
    plt.title('Portfolio Asset Allocation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Allocation Percentage')
    plt.legend(title='Asset Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('asset_allocation_over_time_100k.png', bbox_inches='tight')
    plt.show()

    # pie chart animation of asset class weights (monthly)
    fig, ax = plt.subplots(figsize=(8, 8))

    # convert dates to pandas datetime and get monthly data
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

    ani = FuncAnimation(fig, update, frames=len(monthly_dates), interval=2000, repeat=False)
    ani.save('asset_class_weights_animation_100k.gif', writer='pillow')
    
if __name__ == "__main__":
    analyze_model_performance()
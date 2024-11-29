import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from env import PortfolioEnv
import gymnasium as gym

def make_env(rank, data, window_size, rebalance_frequency, additional_features=True):
    """
    Utility function for creating a PortfolioEnv environment with provided data.
    
    Args:
        rank (int): Unique identifier for the environment.
        data (pd.DataFrame): The dataset to be used in the environment.
        window_size (int): The size of the observation window.
        rebalance_frequency (str): Frequency of rebalancing (e.g., 'monthly').
        additional_features (bool): Whether to include additional features.
    """
    def _init():
        env = PortfolioEnv(
            data=data,
            window_size=window_size,
            rebalance_frequency=rebalance_frequency,
            trading_cost=0.001,
            risk_free_rate=0.02,
            cvar_alpha=0.05,
            initial_balance=1_000_000,
            cache_data=True,
            additional_features=additional_features
        )
        return env
    return _init

def analyze_model_performance(model_path="models/final_model_10k",
                            vec_normalize_path="models/vec_normalize_10k.pkl",
                            data_path='data/processed/merged_data.csv'):
    """
    Analyze the trained model's performance using only test data.
    """
    # config
    WINDOW_SIZE = 365
    REBALANCE_FREQUENCY = 'monthly'
    N_ENVS = 4

    # load and sort data
    merged_data = pd.read_csv(data_path, parse_dates=['Date'])
    merged_data.sort_values('Date', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # get only test data using the same split function from train.py
    _, _, test_data = create_rolling_day_splits(
        merged_data, 
        train_days=6,
        val_days=2,
        test_days=2
    )

    # test environment
    test_env = SubprocVecEnv([make_env(i, test_data, WINDOW_SIZE, REBALANCE_FREQUENCY) 
                             for i in range(N_ENVS)])

    # load the normalization stats from training
    vec_normalize = VecNormalize.load(vec_normalize_path, test_env)
    vec_normalize.training = False  # Disable training mode
    vec_normalize.norm_reward = False  # Disable reward normalization

    # load the trained model
    model = PPO.load(model_path)

    # eval only on test dataset
    print("\nTest Set Performance:")
    evaluate_dataset(model, vec_normalize, test_env, test_data, "test")

    # close environment
    test_env.close()

def evaluate_dataset(model, vec_normalize, env, data, dataset_name):
    """
    Evaluate model performance on test dataset
    """
    try:
        # reset the environment
        obs = env.reset()

        # lists to store metrics
        portfolio_values = []
        sp500_values = []
        asset_weights = []
        dates = []

        scale_factor = 1.0
        last_portfolio_value = None

        for i in range(len(data)):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)

            # portfolio value
            portfolio_value = env.get_attr('portfolio_value')[0]

            # environment reset
            if dones[0]:
                print(f"Environment reset at index {i}, date {data.loc[i, 'Date']}")
                obs = env.reset()

                # update scale factor to maintain continuity
                if last_portfolio_value is not None:
                    scale_factor *= last_portfolio_value / 1000000
                continue

            # scale the portfolio value
            scaled_portfolio_value = portfolio_value * scale_factor
            last_portfolio_value = scaled_portfolio_value

            date = data.loc[i, 'Date']
            dates.append(date)
            portfolio_values.append(scaled_portfolio_value)

            sp500 = data.loc[i, '^GSPC_Close']
            sp500_values.append(sp500)

            weights = env.get_attr('current_weights')[0]
            asset_weights.append(weights)

        if len(portfolio_values) == 0:
            raise ValueError("No portfolio values were collected during evaluation")

        # calculate and display metrics
        calculate_and_display_metrics(portfolio_values, sp500_values, dates, asset_weights, data, dataset_name)

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def create_rolling_day_splits(data, train_days=6, val_days=2, test_days=2):
    """
    Split the data into training, validation, and test sets using a rolling window of days.
    """
    data = data.sort_values('Date')
    
    # init empty lists for indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    # total days in one complete cycle
    cycle_length = train_days + val_days + test_days
    
    # split the data
    for i in range(0, len(data), cycle_length):
        # get indices for this cycle
        cycle_end = min(i + cycle_length, len(data))
        train_end = min(i + train_days, cycle_end)
        val_end = min(train_end + val_days, cycle_end)
        
        train_indices.extend(range(i, train_end))
        val_indices.extend(range(train_end, val_end))
        test_indices.extend(range(val_end, cycle_end))
    
    # datasets train-val-test
    train_data = data.iloc[train_indices].sort_values('Date').reset_index(drop=True)
    val_data = data.iloc[val_indices].sort_values('Date').reset_index(drop=True)
    test_data = data.iloc[test_indices].sort_values('Date').reset_index(drop=True)
    
    # statistics
    print(f"\nSplit Statistics:")
    print(f"Train set ratio: {len(train_data)/len(data):.2%}")
    print(f"Validation set ratio: {len(val_data)/len(data):.2%}")
    print(f"Test set ratio: {len(test_data)/len(data):.2%}")
    
    return train_data, val_data, test_data

def calculate_and_display_metrics(portfolio_values, sp500_values, dates, asset_weights, data, dataset_name):
    """
    Calculate and display metrics for a specific dataset
    """
    # convert to series
    portfolio_series = pd.Series(portfolio_values, index=dates)
    sp500_series = pd.Series(sp500_values, index=dates)

    # calculate returns
    portfolio_returns = portfolio_series.pct_change().fillna(0)
    sp500_returns = sp500_series.pct_change().fillna(0)

    # calculate metrics
    metrics = calculate_metrics(portfolio_returns, sp500_returns)

    # print metrics
    print(f"\n{dataset_name.capitalize()} Set Metrics:")
    print("----------")
    print(f"Total Portfolio Profit: {metrics['Total Portfolio Profit']:.2f}%")
    print(f"Annual Return       : {metrics['Annual Return']:>10.2f}%")
    print(f"Annual Volatility   : {metrics['Annual Volatility']:>10.2f}%")
    print(f"Sharpe Ratio        : {metrics['Sharpe Ratio']:>10.2f}")
    print(f"Sortino Ratio       : {metrics['Sortino Ratio']:>10.2f}")
    print(f"Max Drawdown        : {metrics['Max Drawdown']:>10.2f}%")
    print(f"Win Rate            : {metrics['Win Rate']:>10.2f}%")
    print(f"Best Month          : {metrics['Best Month']:>10.2f}%")
    print(f"Worst Month         : {metrics['Worst Month']:>10.2f}%")

    # save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'portfolio_metrics_{dataset_name}.csv', index=False)

    create_performance_plots(portfolio_series, sp500_series, asset_weights, dates, data, dataset_name)

def calculate_metrics(returns, benchmark_returns, risk_free_rate=0.02):
    """
    Calculate financial performance metrics.
    """
    # calculate cumulative returns first
    cumulative_portfolio = (1 + returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()

    # Total profit
    total_return = (cumulative_portfolio.iloc[-1] - 1) * 100

    # Monthly returns for best/worst month calculation
    monthly_returns = returns.resample('ME').agg(lambda x: (1 + x).prod() - 1) * 100

    # Win Rate calculation (monthly basis)
    win_rate = (monthly_returns > 0).mean() * 100

    # Other metrics
    annual_returns = ((1 + returns.mean()) ** 252 - 1) * 100
    annual_vol = returns.std() * np.sqrt(252) * 100

    # Sharpe & Sortino
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = (np.sqrt(252) * excess_returns.mean() / returns.std())

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = ((annual_returns/100 - risk_free_rate) / downside_std) if downside_std != 0 else np.nan

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

def create_performance_plots(portfolio_series, sp500_series, asset_weights, dates, data, dataset_name):
    """
    Create and save performance visualization plots
    """
    # debug prints
    print("Shape of asset_weights:", np.array(asset_weights).shape)
    print("Number of columns in asset_weights_df:", len(asset_weights[0]))
    
    asset_classes = {
        'us_tech': [0, 1, 2, 3],  # MSFT, AMZN, GOOGL, TSLA
        'us_large_cap': [4, 5],  # SPY, QQQ
        'commodities': [6, 7, 8],  # GC=F, CL=F, SI=F
        'real_estate': [9, 10, 11],  # VNQ, SCHH, IYR
        'crypto_major': [12, 13],  # BTC, ETH
        'crypto_alt': [14]  # LTC
    }

    # directory for plots if it doesn't exist
    plots_dir = f'plots/{dataset_name}'
    os.makedirs(plots_dir, exist_ok=True)

    # debug prints to check weights
    print("\nChecking asset weights:")
    print("Shape of asset_weights:", np.array(asset_weights).shape)
    
    # convert weights to DataFrame and check sums
    asset_weights_df = pd.DataFrame(asset_weights, index=dates)
    daily_sums = asset_weights_df.sum(axis=1)
    print("Daily weight sums (should be close to 1):")
    print(daily_sums.describe())
    
    # convert individual asset weights to asset class weights
    asset_class_weights = pd.DataFrame(index=dates)
    for class_name, indices in asset_classes.items():
        class_weight = asset_weights_df[indices].sum(axis=1)
        asset_class_weights[class_name] = class_weight
    
    # normalize asset class weights to sum to 1
    asset_class_weights = asset_class_weights.div(asset_class_weights.sum(axis=1), axis=0)
    
    # verify the normalization
    class_sums = asset_class_weights.sum(axis=1)
    print("\nNormalized asset class weight sums (should all be 1.0):")
    print(class_sums.describe())

    # plot the normalized weights
    plt.figure(figsize=(15, 7))
    asset_class_weights.plot(
        kind='area', 
        stacked=True,
        linewidth=0,
        alpha=0.8
    )
    plt.title('Asset Class Allocation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/asset_allocation_10k.png')
    plt.close()

    # Portfolio Value vs S&P500
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series.index, portfolio_series/portfolio_series.iloc[0], label='Portfolio')
    plt.plot(sp500_series.index, sp500_series/sp500_series.iloc[0], label='S&P500')
    plt.title('Portfolio Performance vs S&P500')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_dir}/performance_comparison_10k.png')
    plt.close()

    # Monthly Returns Distribution
    monthly_returns = portfolio_series.pct_change().resample('ME').agg(lambda x: (1 + x).prod() - 1)
    plt.figure(figsize=(10, 6))
    monthly_returns.hist(bins=50)
    plt.title('Distribution of Monthly Returns')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.savefig(f'{plots_dir}/returns_distribution_10k.png')
    plt.close()

    # Drawdown Plot
    cum_returns = (1 + portfolio_series.pct_change()).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns/rolling_max - 1) * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(drawdowns.index, drawdowns)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(f'{plots_dir}/drawdown_10k.png')
    plt.close()

    # save the data
    results_df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_series.values,
        'SP500_Value': sp500_series.values
    })
    
    # add asset class weights to results
    for class_name in asset_classes.keys():
        results_df[f'{class_name}_weight'] = asset_class_weights[class_name].values
        
    results_df.to_csv(f'{plots_dir}/performance_data.csv', index=False)

if __name__ == "__main__":
    analyze_model_performance()
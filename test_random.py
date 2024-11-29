import numpy as np
from env import PortfolioEnv

def test_portfolio_env():
    # init the environment
    env = PortfolioEnv(
        data_path='data/processed/merged_data.csv',
        window_size=365,
        trading_cost=0.001,
        risk_free_rate=0.02,
        cvar_alpha=0.05,
        initial_balance=1000000,
        rebalance_frequency='weekly',
        additional_features=True
    )
    
    # available assets
    print("\n=== Available Assets ===")
    print(f"Number of assets: {env.num_assets}")
    for i, asset in enumerate(env.assets):
        print(f"{i}: {asset}")
    
    # reset environment
    obs = env.reset()
    
    # run a few steps to generate some history
    num_steps = 30
    for _ in range(num_steps):
        # gen random action (uniform distribution)
        action = np.random.uniform(0, 1, env.num_assets)
        action = action / np.sum(action)  # Normalize to sum to 1
        
        # step
        next_obs, reward, done, info = env.step(action)
        
        if done:
            break
    
    # final allocation
    print("\n=== Final Portfolio Weights ===")
    for i, (asset, weight) in enumerate(zip(env.assets, env.current_weights)):
        if weight > 0.01:  # Only show weights above 1%
            print(f"{asset:<15}: {weight:7.2%}")
    
    # render the environment
    env.render()
    
    # portfolio metrics
    print("\n=== Portfolio Metrics ===")
    print(f"Portfolio Value: ${env.portfolio_value:,.2f}")
    print(f"Total Return: {((env.portfolio_value / env.initial_balance) - 1) * 100:.2f}%")
    if info:
        print("\nDetailed Metrics:")
        for key, value in info.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    test_portfolio_env()
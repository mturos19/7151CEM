# test_random.py
import numpy as np
from env import PortfolioEnv
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message='DataFrame.fillna with.*')

def test_portfolio_env():
    # initialize the environment
    env = PortfolioEnv(
        data_path='data/processed/merged_data.csv',
        window_size=50,
        trading_cost=0.001,
        risk_free_rate=0.02,
        cvar_alpha=0.05,
        initial_balance=1000000
    )
    
    print(f"Number of assets: {env.num_assets}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # test reset
    print("\nTesting reset...")
    initial_observation = env.reset()
    print(f"Initial observation shape: {initial_observation.shape}")
    print(f"Initial portfolio value: ${env.portfolio_value:,.2f}")
    
    # test a few random steps
    print("\nTesting random actions...")
    for i in range(5):
        # generate a random valid action (portfolio weights)
        random_weights = np.random.random(env.num_assets)
        random_weights = random_weights / np.sum(random_weights)  # normalize to sum to 1
        
        # take a step
        observation, reward, done, info = env.step(random_weights)
        
        print(f"\nStep {i+1}:")
        print(f"Observation shape: {observation.shape}")
        print(f"Reward: {reward:.4f}")
        print(f"Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"Portfolio return: {info['portfolio_return']:.4f}")
        print(f"Portfolio volatility: {info['portfolio_volatility']:.4f}")
        print(f"CVaR: {info['cvar']:.4f}")
        print(f"Done: {done}")
        
        if done:
            print("Episode finished early")
            break
    
    # test render
    print("\nTesting render...")
    env.render()

if __name__ == "__main__":
    try:
        test_portfolio_env()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from env import PortfolioEnv
import pandas as pd

def make_env(rank, data_path, window_size, rebalance_frequency, additional_features=True):
    """
    utility function for multiprocessed env.
    """
    def _init():
        env = PortfolioEnv(
            data_path=data_path,
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

def main():
    # configuration
    DATA_PATH = 'data/processed/merged_data.csv'
    WINDOW_SIZE = 365
    REBALANCE_FREQUENCY = 'monthly'
    
    # load the merged data
    merged_data = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    merged_data.sort_values('Date', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # print the date range to debug
    print("Date range in merged data:", merged_data['Date'].min(), "to", merged_data['Date'].max())
    
    # gpu settings
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # number of parallel environments (adjust based on your gpu memory)
    n_envs = 4  # ensure this matches during analysis
    
    # create vectorized environment
    env = SubprocVecEnv([make_env(i, DATA_PATH, WINDOW_SIZE, REBALANCE_FREQUENCY) 
                         for i in range(n_envs)])
    
    # normalize the environment
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # optimize policy network architecture for gpu
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )
    
    # initialize ppo with optimized parameters
    model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,        # number of steps to run for each environment per update
    batch_size=512,      # minibatch size for each gradient update
    n_epochs=10,         # number of times to reuse each collected trajectory
    learning_rate=3e-4,  # standard learning rate for Adam optimizer
    gamma=0.99,          # discount factor for future rewards
    gae_lambda=0.95,     # factor for Generalized Advantage Estimation
    clip_range=0.2,      # PPO clipping parameter
    clip_range_vf=0.2,   # value function clipping parameter
    ent_coef=0.005,      # entropy coefficient for exploration
    vf_coef=0.5,         # value function coefficient in loss
    max_grad_norm=0.5,   # gradient clipping threshold
    use_sde=False,
    #sde_sample_freq=8,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./logs/",
    verbose=1,
    device=device
    )
    
    # verify model device
    print(f"Model device: {next(model.policy.parameters()).device}")
    
    # custom callback for gpu memory monitoring
    class GPUMonitorCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(GPUMonitorCallback, self).__init__(verbose)
            self.iteration = 0
        
        def _on_step(self):
            if self.iteration % 100 == 0:  # monitor every 100 steps
                print(f"\nGPU Utilization:")
                print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
                print(f"Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
                
                # add periodic memory clearing
                if self.iteration % 1000 == 0:
                    torch.cuda.empty_cache()
                
            self.iteration += 1
            return True
    
    try:
        # train with larger number of steps and gpu monitoring
        model.learn(
            total_timesteps=100_000,
            callback=[GPUMonitorCallback()],
            progress_bar=True
        )
        
        # save the model and VecNormalize
        model.save("models/final_model_100k")
        env.save("models/vec_normalize_100k.pkl")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
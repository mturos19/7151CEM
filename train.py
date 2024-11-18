import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from env import PortfolioEnv

def make_env(rank, data_path, window_size, rebalance_frequency, additional_features=True):
    """
    Utility function for multiprocessed env.
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
    # Configuration
    DATA_PATH = 'data/processed/merged_data.csv'
    WINDOW_SIZE = 365
    REBALANCE_FREQUENCY = 'monthly'
    
    # GPU settings
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Number of parallel environments (adjust based on your GPU memory)
    n_envs = 4  # Ensure this matches during analysis
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i, DATA_PATH, WINDOW_SIZE, REBALANCE_FREQUENCY) 
                         for i in range(n_envs)])
    
    # Normalize the environment
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Optimize policy network architecture for GPU
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )
    
    # Initialize PPO with optimized parameters
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,                # Increased batch size
        batch_size=2048,             # Larger batch size for GPU
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/",
        verbose=1,
        device=device
    )
    
    # Verify model device
    print(f"Model device: {next(model.policy.parameters()).device}")
    
    # Custom callback for GPU memory monitoring
    class GPUMonitorCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(GPUMonitorCallback, self).__init__(verbose)
            self.iteration = 0
        
        def _on_step(self):
            if self.iteration % 100 == 0:  # Monitor every 100 steps
                print(f"\nGPU Utilization:")
                print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
                print(f"Memory Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
            self.iteration += 1
            return True
    
    try:
        # Train with larger number of steps and GPU monitoring
        model.learn(
            total_timesteps=10_000,
            callback=[GPUMonitorCallback()],
            progress_bar=True
        )
        
        # Save the model and VecNormalize
        model.save("models/final_model")
        env.save("models/vec_normalize.pkl")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
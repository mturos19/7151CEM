import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from env import PortfolioEnv
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import time

def make_env(data_path, window_size, rebalance_frequency, seed=0):
    """Create a wrapped, monitored environment."""
    def _init():
        env = PortfolioEnv(
            data_path=data_path,
            window_size=window_size,
            rebalance_frequency=rebalance_frequency,
            additional_features=True
        )
        env = Monitor(env)
        env = FlattenObservation(env)
        return env
    set_random_seed(seed)
    return _init

def main():
    # Configuration
    DATA_PATH = 'data/processed/merged_data.csv'
    WINDOW_SIZE = 365
    REBALANCE_FREQUENCY = 'monthly'
    MODEL_DIR = 'models/'
    LOG_DIR = 'logs/'
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Learning rate schedule
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(DATA_PATH, WINDOW_SIZE, REBALANCE_FREQUENCY)])
    
    # Normalize observations and rewards with modified parameters
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=100.,  # Increased from 10 to handle larger observation ranges
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-08,
        training=True  # Explicitly set training mode
    )
    
    # Obs space debug
    # observation = env.reset()
    # print(f"Initial Observation Shape: {observation.shape}")
    
    # Network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128]   # Value network
        )
    )
    
    # Initialize the PPO agent with optimized hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(2.5e-4),  # Decreasing learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,  # Stochastic Differential Equations for exploration
        sde_sample_freq=4,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device="cuda"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODEL_DIR,
        name_prefix='ppo_portfolio_model',
        save_vecnormalize=True  # Save normalization statistics
    )
    
    # Separate environment for evaluation
    eval_env = DummyVecEnv([make_env(DATA_PATH, WINDOW_SIZE, REBALANCE_FREQUENCY, seed=42)])
    eval_env = Monitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.,
        clip_reward=10.,
        gamma=0.99,
        epsilon=1e-08
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/best_model",
        log_path=LOG_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Custom callback for learning rate adjustment
    class CustomCallback(BaseCallback):
        def __init__(self, timesteps: int, verbose=0):
            super(CustomCallback, self).__init__(verbose)
            self.training_start = time.time()
            self.timesteps = timesteps
        
        def _on_step(self):
            if (self.n_calls + 1) % 10000 == 0:
                elapsed_time = time.time() - self.training_start
                fps = (self.n_calls + 1) / elapsed_time
                remaining_steps = self.timesteps - (self.n_calls + 1)
                estimated_remaining_time = remaining_steps / fps
                
                print(f"FPS: {fps:.2f}")
                print(f"Progress: {(self.n_calls + 1)/self.timesteps*100:.1f}%")
                print(f"Estimated remaining time: {estimated_remaining_time/3600:.2f} hours")
                print("-" * 50)
            return True
    
    # Train the agent with increased timesteps
    TIMESTEPS = 100_000
    training_successful = False
    try:
        custom_callback = CustomCallback(timesteps=TIMESTEPS)
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=[checkpoint_callback, eval_callback, custom_callback],
            tb_log_name="ppo_portfolio",
            reset_num_timesteps=False
        )
        training_successful = True
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        # Save the final model and normalization stats
        model.save(f"{MODEL_DIR}/final_model")
        env.save(f"{MODEL_DIR}/vec_normalize.pkl")
        print("Training completed and model saved.")
        
        # Only try to plot if tensorboard logs exist
        try:
            plot_training_results(LOG_DIR)
        except Exception as e:
            print("Could not plot training results:", e)
            print("This is normal if tensorboard logging failed.")

def plot_training_results(log_dir):
    """Plot training metrics using tensorboard data"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib.pyplot as plt
    
    # Load tensorboard data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    rewards = event_acc.Scalars('rollout/ep_rew_mean')
    plt.plot([r.step for r in rewards], [r.value for r in rewards])
    plt.title('Average Episode Reward During Training')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.savefig(f"{log_dir}/training_rewards.png")
    plt.close()

if __name__ == "__main__":
    main()
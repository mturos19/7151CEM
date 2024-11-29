import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from env import PortfolioEnv
import pandas as pd
from sklearn.model_selection import train_test_split
from stable_baselines3.common.callbacks import EvalCallback
import nvidia_smi

# custom callback to monitor GPU memory usage during training
class GPUMonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def _on_step(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        print(f"GPU Memory Used: {info.used / 1024**2:.2f} MB")
        return True

def make_env(rank, data, window_size, rebalance_frequency, additional_features=True):
    """
    Utility function for creating a PortfolioEnv environment with provided data.
    
    Args:
        rank (int): Unique identifier for the environment.
        data (pd.DataFrame): The dataset to be used in the environment.
        window_size (int): The size of the observation window.
        rebalance_frequency (str): Frequency of rebalancing (e.g., 'monthly').
        additional_features (bool): Whether to include additional features.
    
    Returns:
        Callable: A function that initializes the environment.
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

def create_rolling_day_splits(data, train_days=6, val_days=2, test_days=2):
    """
    Split the data into training, validation, and test sets using a rolling window of days.
    Example pattern: 6 days train, 2 days validation, 2 days test, repeat.
    
    Args:
        data (pd.DataFrame): DataFrame with 'Date' column
        train_days (int): Number of consecutive days for training
        val_days (int): Number of consecutive days for validation
        test_days (int): Number of consecutive days for testing
    """
    # sort data by date
    data = data.sort_values('Date')
    
    # initialize empty lists for indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    # calculate total days in one complete cycle
    cycle_length = train_days + val_days + test_days
    
    # split the data
    for i in range(0, len(data), cycle_length):
        # indices for this cycle
        cycle_end = min(i + cycle_length, len(data))
        train_end = min(i + train_days, cycle_end)
        val_end = min(train_end + val_days, cycle_end)
        
        # train indices
        train_indices.extend(range(i, train_end))
        
        # validation indices
        val_indices.extend(range(train_end, val_end))
        
        # test indices
        test_indices.extend(range(val_end, cycle_end))
    
    # create train, validation, and test datasets
    train_data = data.iloc[train_indices].sort_values('Date')
    val_data = data.iloc[val_indices].sort_values('Date')
    test_data = data.iloc[test_indices].sort_values('Date')
    
    # print statistics
    print(f"\nSplit Statistics:")
    print(f"Train set ratio: {len(train_data)/len(data):.2%}")
    print(f"Validation set ratio: {len(val_data)/len(data):.2%}")
    print(f"Test set ratio: {len(test_data)/len(data):.2%}")
    print(f"\nDate Ranges:")
    print(f"Full data: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Train data samples: {len(train_indices)} days")
    print(f"Validation data samples: {len(val_indices)} days")
    print(f"Test data samples: {len(test_indices)} days")
    
    # print example cycles
    print("\nFirst few cycles of train/val/test split:")
    for i in range(2):
        cycle_start = data['Date'].iloc[i * cycle_length]
        cycle_end = data['Date'].iloc[min((i + 1) * cycle_length - 1, len(data) - 1)]
        print(f"\nCycle {i+1}: {cycle_start.date()} to {cycle_end.date()}")
        print(f"Train: {train_data[(train_data['Date'] >= cycle_start) & (train_data['Date'] <= cycle_end)]['Date'].dt.date.tolist()}")
        print(f"Val: {val_data[(val_data['Date'] >= cycle_start) & (val_data['Date'] <= cycle_end)]['Date'].dt.date.tolist()}")
        print(f"Test: {test_data[(test_data['Date'] >= cycle_start) & (test_data['Date'] <= cycle_end)]['Date'].dt.date.tolist()}")
    
    return train_data, val_data, test_data

def main():
    # config
    DATA_PATH = 'data/processed/merged_data.csv'
    WINDOW_SIZE = 365
    REBALANCE_FREQUENCY = 'monthly'
    TEST_SIZE = 0.2  # 20% for testing
    RANDOM_STATE = 42
    N_ENVS = 4

    # load and sort the merged data
    merged_data = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    merged_data.sort_values('Date', inplace=True)
    merged_data.reset_index(drop=True, inplace=True)

    # date range debug
    #print("Date range in merged data:", merged_data['Date'].min(), "to", merged_data['Date'].max())

    train_data, val_data, test_data = create_rolling_day_splits(
        merged_data, 
        train_days=6,
        val_days=2,
        test_days=2
    )

    print(f"Training data range: {train_data['Date'].min()} to {train_data['Date'].max()}")
    print(f"Testing data range: {test_data['Date'].min()} to {test_data['Date'].max()}")

    # GPU settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # environments for both training and validation
    train_env = SubprocVecEnv([make_env(i, train_data, WINDOW_SIZE, REBALANCE_FREQUENCY) 
                              for i in range(N_ENVS)])
    val_env = SubprocVecEnv([make_env(i, val_data, WINDOW_SIZE, REBALANCE_FREQUENCY) 
                            for i in range(N_ENVS)])

    # normalize the training environment
    vec_normalize = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # normalized validation environment
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    # copy normalization stats from training to validation env
    val_env.obs_rms = vec_normalize.obs_rms
    val_env.ret_rms = vec_normalize.ret_rms
    val_env.training = False
    val_env.norm_reward = False

    # optimize policy network architecture for GPU
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )

    # initialize PPO with optimized parameters
    model = PPO(
        "MlpPolicy",
        vec_normalize,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/",
        verbose=1,
        device=device
    )

    # evaluation callback with validation environment
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path="models/best_model",
        log_path="logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    try:
        # train the model with both callbacks
        model.learn(
            total_timesteps=100_000,
            callback=eval_callback,
            progress_bar=True
        )

        # save model and VecNormalize statistics
        model.save("models/final_model_100k")
        vec_normalize.save("models/vec_normalize_100k.pkl")
        print("Model and VecNormalize statistics have been saved successfully.")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        train_env.close()
        val_env.close()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
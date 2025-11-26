import asyncio
import logging
import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Trainer")

class SimpleProgressCallback(BaseCallback):
    """
    Callback to print progress to console safely.
    """
    def __init__(self, check_freq: int, total_timesteps: int, verbose=1):
        super(SimpleProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            progress = (self.num_timesteps / self.total_timesteps) * 100
            logger.info(f"Progress: {progress:.2f}% ({self.num_timesteps}/{self.total_timesteps})")
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log "True Reward" (Un-normalized)
        # With VecEnv, infos is a list of dicts
        for info in self.locals['infos']:
            if 'episode' in info:
                self.logger.record('rollout/ep_rew_mean_true', info['episode']['r'])
                self.logger.record('rollout/ep_len_mean', info['episode']['l'])
            # Log Trades Taken (if available in info)
            if 'trades_taken' in info:
                self.logger.record('rollout/trades_taken', info['trades_taken'])
            if 'portfolio_value' in info:
                self.logger.record('rollout/portfolio_value', info['portfolio_value'])
            if 'stop_loss_triggered' in info and info['stop_loss_triggered']:
                 self.logger.record('rollout/stop_loss_events', 1)
        return True

def run_sanity_check(df):
    """
    Run a quick sanity check on a small slice of data to ensure Env works.
    """
    logger.info("Running Sanity Check (1000 steps)...")
    small_df = df.iloc[:1000].copy()
    env = TradingEnv(small_df)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        if done:
            logger.info(f"Sanity Check: Episode finished at step {steps}. Reward: {reward}")
            break
    logger.info("Sanity Check Passed.")

async def main():
    logger.info("Starting Rescue Mission Training Pipeline...")
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using Device: {device.upper()}")

    # 1. Fetch High-Quality Data
    # Use 'binanceusdm' for Binance USDT-M Futures
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream() 
    
    logger.info("Fetching 200,000 candles (5 Months) for training...")
    df = await data_service.fetch_ohlcv(limit=200000)
    await data_service.close()
    
    if df.empty or len(df) < 5000:
        logger.error(f"Insufficient data fetched: {len(df)} candles. Exiting.")
        return
    
    logger.info(f"Successfully fetched {len(df)} candles.")

    # 2. Feature Engineering
    logger.info("Calculating Advanced Microstructure Features (Rescue Mission Edition)...")
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df) 
    
    # 3. Sanity Check
    run_sanity_check(df)

    # 4. Walk-Forward Split (70/15/15)
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    logger.info(f"Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 5. Environment Setup with Parallelization (Training)
    logger.info("Setting up Parallel Environments (16 Cores)...")
    
    env_kwargs = {'df': train_df}
    
    env = make_vec_env(
        TradingEnv, 
        n_envs=16, 
        seed=42, 
        vec_env_cls=SubprocVecEnv, 
        env_kwargs=env_kwargs,
        wrapper_class=Monitor
    )
    
    # WRAPPER: VecNormalize
    # Critical for PPO convergence.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 6. Model Architecture (Stabilized)
    # Learning Rate: 1e-4 (Lowered from 3e-4)
    model_path = "ppo_apex_model_production.zip"
    stats_path = "vec_normalize.pkl"
    
    logger.info("Initializing PPO Agent (Comprehensive Bias Correction)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048, 
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95, # Standard PPO efficiency
        clip_range=0.2,
        ent_coef=0.001, # Strong dampening of exploration
        verbose=1,
        device="cpu",
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 7. Training Loop
    TOTAL_TIMESTEPS = 10_000_000 
    logger.info(f"Starting Training for {TOTAL_TIMESTEPS} timesteps...")
    
    tb_callback = TensorboardCallback()
    progress_callback = SimpleProgressCallback(check_freq=6250, total_timesteps=TOTAL_TIMESTEPS)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./checkpoints/',
        name_prefix='ppo_apex_rescue'
    )
    
    callback = CallbackList([tb_callback, progress_callback, checkpoint_callback])
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)
        logger.info("Training Complete.")
        
        # 8. Save Model AND Normalization Stats
        logger.info(f"Saving model to {model_path}...")
        model.save(model_path)
        
        logger.info(f"Saving normalization stats to {stats_path}...")
        env.save(stats_path)
        
        # 9. Validation (Quick Check)
        logger.info("Running Validation on Out-of-Sample Data...")
        # Create a single env for validation
        val_env = TradingEnv(val_df)
        # We need to wrap it in DummyVecEnv and then apply the SAVED normalization
        from stable_baselines3.common.vec_env import DummyVecEnv
        val_vec_env = DummyVecEnv([lambda: val_env])
        val_vec_env = VecNormalize.load(stats_path, val_vec_env)
        val_vec_env.training = False # Don't update stats during validation
        val_vec_env.norm_reward = False # Don't normalize reward for evaluation
        
        mean_reward, std_reward = evaluate_policy(model, val_vec_env, n_eval_episodes=5)
        logger.info(f"Validation Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        logger.info("Pipeline Finished Successfully.")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted. Saving current state...")
        model.save(model_path)
        env.save(stats_path)

if __name__ == "__main__":
    # Windows support for multiprocessing
    asyncio.run(main())

import asyncio
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv

# 1. CONFIGURATION
MODEL_PATH = "ppo_apex_model_production"  # .zip is added automatically by load
STATS_PATH = "vec_normalize.pkl"
# We will fetch fresh data via DataIngestionService

async def main():
    print(f"--- Loading Model from {MODEL_PATH} ---")
    
    # 2. Fetch Data (Validation Set)
    # We'll use the same service but maybe a different range if possible, 
    # or just the same data to verify 'training' performance first.
    # The user said "Validation" but gave "your_data.csv". 
    # Let's fetch 1000 candles for a quick but meaningful test.
    print("Fetching validation data...")
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream()
    df = await data_service.fetch_ohlcv(limit=2000) # 2000 candles
    await data_service.close()

    if df.empty:
        print("Error: No data fetched.")
        return

    # Feature Engineering
    print("Calculating features...")
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df)

    # 3. Recreate the Environment
    # Must wrap in DummyVecEnv to match the training shape
    env = TradingEnv(df, initial_balance=10000.0)
    env = DummyVecEnv([lambda: env])
    
    # 4. Load Normalization Stats (CRITICAL)
    print(f"Loading normalization stats from {STATS_PATH}...")
    env = VecNormalize.load(STATS_PATH, env)
    
    # 5. Turn off Training Mode
    env.training = False     # Do not update stats (use frozen stats from training)
    env.norm_reward = False  # We want to see REAL dollars, not normalized scores

    # 6. Load the Brain
    model = PPO.load(MODEL_PATH)

    print("--- Starting Deterministic Backtest (God Mode) ---")
    obs = env.reset()
    
    total_reward = 0.0
    trades = 0
    
    # Run until done
    while True:
        # deterministic=True is the secret sauce. 
        action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, infos = env.step(action)
        
        # In VecEnv, rewards is an array [reward_env_0, ...]
        # With norm_reward=False, this should be the raw reward (Equity Change / 100)
        # So Real Profit = Reward * 100
        step_reward = rewards[0]
        total_reward += step_reward
        
        # Access info from the environment
        # VecEnv infos is a list of dicts
        info = infos[0]
        if info.get('trades_taken', 0) > trades:
            trades = info['trades_taken']
            
        if dones[0]:
            break

    print("\n" + "="*30)
    print(f"FINAL RESULTS")
    # Our reward is Equity/100, so Profit = Reward * 100
    realized_profit = total_reward * 100.0
    print(f"Total Realized Profit: ${realized_profit:.2f}")
    print(f"Total Trades Taken:    {trades}")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(main())

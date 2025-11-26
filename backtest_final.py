import asyncio
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv

# --- CONFIGURATION ---
MODEL_PATH = "ppo_apex_model_production" # .zip added by load
STATS_PATH = "vec_normalize.pkl"

async def run_backtest():
    print("--- 🚀 STARTING FINAL BACKTEST ---")
    
    # 1. Fetch High-Quality Data (Validation)
    print("Fetching validation data...")
    # Fetching 2000 candles to ensure we have enough data for a meaningful test
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream()
    df = await data_service.fetch_ohlcv(limit=2000)
    await data_service.close()
    
    if df.empty:
        print("Error: No data fetched.")
        return

    # 2. Feature Engineering
    print("Calculating features...")
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df)

    # 3. Setup Environment
    # Initialize Env with the fetched dataframe
    env = TradingEnv(df, initial_balance=10000.0)
    
    # 4. Wrap in Vector and Normalize
    # We use DummyVecEnv for single-threaded testing
    env = DummyVecEnv([lambda: env])
    
    # 5. Load the Saved Normalization Statistics
    # This tells the bot what "High Volume" looks like based on training history
    print(f"Loading normalization stats from {STATS_PATH}...")
    env = VecNormalize.load(STATS_PATH, env)
    
    # 6. CRITICAL: Turn off training updates
    env.training = False 
    env.norm_reward = False # We want to see Real $$, not normalized points

    # 7. Load the Brain
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    # 8. Run the Simulation
    obs = env.reset()
    total_profit = 0
    trades = 0
    
    print("Running simulation...")
    try:
        while True:
            # DETERMINISTIC = TRUE
            # This turns off the entropy/noise. The bot plays its best move only.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            # Access the 'info' from the environment to get real stats
            info = infos[0] 
            
            if dones[0]:
                final_value = info.get('portfolio_value', 0)
                initial = 10000 # Assuming 10k start
                profit = final_value - initial
                trades = info.get('trades_taken', 0)
                
                print("\n" + "="*40)
                print(f"🏁 EPISODE FINISHED")
                print(f"💰 Final Portfolio: ${final_value:.2f}")
                print(f"📈 Total Profit:    ${profit:.2f} ({profit/initial*100:.2f}%)")
                print(f"🔫 Total Trades:    {trades}")
                print("="*40)
                break
                
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    asyncio.run(run_backtest())

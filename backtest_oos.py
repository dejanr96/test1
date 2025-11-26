import asyncio
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import pandas as pd
from src.data.ingestion import DataIngestionService
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv

# --- CONFIGURATION ---
MODEL_PATH = "ppo_apex_model_production" 
STATS_PATH = "vec_normalize.pkl"

async def run_backtest():
    print("--- 🚀 STARTING OUT-OF-SAMPLE BACKTEST (Sanity Check) ---")
    
    # 1. Fetch Historical Data (30 Days Ago)
    # 30 days * 24h * 60m * 60s * 1000ms
    thirty_days_ms = 30 * 24 * 60 * 60 * 1000
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - thirty_days_ms
    
    print(f"Fetching 10,000 candles starting from 30 days ago (Timestamp: {since_ms})...")
    
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream()
    
    # Pass 'since' to fetch historical data
    df = await data_service.fetch_ohlcv(limit=10000, since=since_ms)
    await data_service.close()
    
    if df.empty:
        print("Error: No data fetched.")
        return
        
    print(f"Data Fetched: {len(df)} candles.")
    print(f"Start Date: {df['timestamp'].iloc[0]}")
    print(f"End Date:   {df['timestamp'].iloc[-1]}")

    # 2. Feature Engineering
    print("Calculating features...")
    fe = FeatureEngineer()
    df = fe.add_advanced_microstructure_features(df)

    # 3. Setup Environment
    env = TradingEnv(df, initial_balance=10000.0)
    
    # 4. Wrap in Vector and Normalize
    env = DummyVecEnv([lambda: env])
    
    # 5. Load the Saved Normalization Statistics
    print(f"Loading normalization stats from {STATS_PATH}...")
    env = VecNormalize.load(STATS_PATH, env)
    
    # 6. CRITICAL: Turn off training updates
    env.training = False 
    env.norm_reward = False 

    # 7. Load the Brain
    print(f"Loading model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    # 8. Run the Simulation
    obs = env.reset()
    trades = 0
    
    print("Running simulation...")
    try:
        step_count = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            info = infos[0] 
            step_count += 1
            
            if step_count % 1000 == 0:
                print(f"Step {step_count}...")
            
            if dones[0]:
                final_value = info.get('portfolio_value', 0)
                initial = 10000 
                profit = final_value - initial
                trades = info.get('trades_taken', 0)
                
                print("\n" + "="*40)
                print(f"🏁 OOS BACKTEST FINISHED")
                print(f"📅 Period: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
                print(f"💰 Final Portfolio: ${final_value:.2f}")
                print(f"📈 Total Profit:    ${profit:.2f} ({profit/initial*100:.2f}%)")
                print(f"🔫 Total Trades:    {trades}")
                print("="*40)
                break
                
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__ == "__main__":
    asyncio.run(run_backtest())
